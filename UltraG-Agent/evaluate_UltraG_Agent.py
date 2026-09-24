import argparse
import copy
import concurrent.futures
import csv
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DEFAULT_AGENT_SCRIPT = str(
    Path(__file__).resolve().with_name("inference_agent_grounded.py")
)
DEFAULT_SAM3_CODE_DIR = "/home/Data2/zhuquanhao/US-SAM3/US-SAM3/code"
DEFAULT_CONFIG_PATH = "/home/Data2/zhuquanhao/US-SAM3/US-SAM3/config/config.yaml"
DEFAULT_CHECKPOINT_PATH = "/home/Data2/zhuquanhao/US-SAM3/US-SAM3_weight/US-SAM3.pt"

BANNED_TERMS = ["annotation", "annotated", "dataset", "mask id", "metadata"]
SPATIAL_FINDING_HINTS = (
    "presence",
    "location",
    "size",
    "extent",
    "area",
    "pixel",
    "boundary",
    "count",
)
NON_SPATIAL_FINDING_HINTS = (
    "not_applicable",
    "scope",
    "impression",
    "summary",
    "label",
    "status",
    "distribution",
    "unavailable",
    "absence",
    "no_target",
)
UNSUPPORTED_US_DESCRIPTORS = [
    "hypoechoic",
    "hyperechoic",
    "anechoic",
    "heterogeneous",
    "homogeneous",
    "irregular",
    "circumscribed",
    "well-defined",
    "ill-defined",
    "vascular",
    "vascularity",
    "calcification",
    "suspicious",
    "appears normal",
    "abnormal",
    "pi-rads",
    "pirads",
    "bi-rads",
    "birads",
    "gleason",
    "psa",
    "biopsy",
    "pathology",
    "tissue diagnosis",
    "recommend",
    "follow-up",
    "cm",
    "mm",
    "ml",
]


PREDICTION_FIELDS = [
    "eval_id",
    "dataset",
    "task",
    "sample_id",
    "image_id",
    "split",
    "image_file",
    "image_path",
    "model_backend",
    "model_name",
    "prompt",
    "reference_json",
    "prediction_raw",
    "prediction_json",
    "error",
    "elapsed_sec",
]


METRIC_FIELDS = [
    "eval_id",
    "dataset",
    "task",
    "sample_id",
    "image_id",
    "split",
    "model_backend",
    "model_name",
    "format_ok",
    "error",
    "iou",
    "dice",
    "empty_prediction",
    "seg_token_ok",
    "answer_correct",
    "grounded_answer_correct",
    "bleu_1",
    "bleu_2",
    "bleu_3",
    "bleu_4",
    "meteor_exact",
    "rouge_l",
    "finding_type_recall",
    "finding_type_precision",
    "finding_type_f1",
    "report_grounding_iou",
    "report_grounding_recall_50",
    "presence_ok",
    "category_ok",
    "location_ok",
    "size_ok",
    "extent_ok",
    "semantic_accuracy",
    "unsupported_descriptor_count",
    "banned_term_hits",
]


def mask_to_uncompressed_rle(mask):
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    counts = []
    previous = 0
    run = 0
    for value in flat:
        value = int(value != 0)
        if value == previous:
            run += 1
        else:
            counts.append(run)
            previous = value
            run = 1
    counts.append(run)
    return {"type": "rle", "size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def infer_finding_type(text):
    norm = normalize_text(text)
    if re.search(
        r"\bno\s+(?:(?:requested|visible|labeled|labelled)\s+)?"
        r"(?:[a-z0-9]+\s+){0,4}(?:target|structure|organ|lesion|mass|tumor|nodule)\b",
        norm,
    ):
        return "no_target_impression" if "segmentation" in norm else "absence"
    if re.search(r"\bno\s+(?:[a-z0-9]+\s+){0,4}boundary\b", norm):
        return "location_not_applicable"
    if re.search(r"\bno\s+(?:[a-z0-9]+\s+){0,4}extent\b", norm):
        return "size_not_applicable"
    if re.search(r"\bno\s+(?:target\s+)?(?:anatomy|structure|organ)\b", norm):
        return "no_target"
    if "not applicable" in norm or "cannot be measured" in norm:
        if any(term in norm for term in ("size", "extent", "measure")):
            return "size_not_applicable"
        if any(term in norm for term in ("location", "localization", "boundary")):
            return "location_not_applicable"
        return "no_target"
    if any(term in norm for term in ("supports localization", "delineation only", "spatial assessment only")):
        return "grounding_scope"
    if "target for segmentation" in norm or "segmentation target" in norm:
        return "segmentation_impression"
    if any(term in norm for term in ("area", "occupies", "cover", "percent", "percentage")):
        return "organ_extent"
    if any(term in norm for term in ("size", "measure", "dimension", "span", "pixel", "visible extent")):
        return "organ_size"
    if any(term in norm for term in ("visible", "present", "observed", "identified")):
        return "organ_presence"
    if any(term in norm for term in ("center", "location", "located", "upper", "middle", "lower")):
        return "organ_location"
    if any(term in norm for term in ("target", "organ", "structure", "lesion", "tumor", "nodule")):
        return "organ_presence"
    return "other"


def load_agent_module(agent_script: str):
    path = Path(agent_script).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Agent inference script not found: {path}")
    spec = importlib.util.spec_from_file_location("us_sam3_grounded_agent", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import agent inference script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class USSam3AgentClient:
    def __init__(self, args):
        self.agent = load_agent_module(args.agent_script)
        self.categories = self.agent.parse_categories(
            args.categories, args.categories_file
        )
        print(
            f"[agent] loaded {Path(args.agent_script).resolve()} "
            f"categories={len(self.categories)}",
            flush=True,
        )
        self.model_object, self.device, self.load_info = self.agent.setup_model(
            config_path=str(Path(args.config_path).expanduser().resolve()),
            checkpoint_path=str(Path(args.checkpoint_path).expanduser().resolve()),
            sam3_code_dir=str(Path(args.sam3_code_dir).expanduser().resolve()),
        )
        self.model = "US-SAM3-Agent"
        self.current_image_path = None
        self.prompt_cache = {}
        print(f"[model] US-SAM3 agent ready device={self.device}", flush=True)

    def _run_targets_cached(self, image_path, plan, run_args):
        if image_path != self.current_image_path:
            self.current_image_path = image_path
            self.prompt_cache.clear()

        target_runs = []
        for target in plan.get("targets", []):
            cache_key = (
                str(target["sam3_prompt"]).strip().lower(),
                int(run_args.min_area),
                float(run_args.max_area_ratio),
                bool(run_args.remove_overlap),
            )
            cached = self.prompt_cache.get(cache_key)
            if cached is None:
                serialized = self.agent.sam3_predict_image(
                    model=self.model_object,
                    device=self.device,
                    image_path=image_path,
                    text_prompt=target["sam3_prompt"],
                    remove_overlap=run_args.remove_overlap,
                )
                candidates = self.agent.collect_candidates(
                    serialized,
                    target["sam3_prompt"],
                    run_args.min_area,
                    run_args.max_area_ratio,
                )
                height = int(serialized["orig_img_h"])
                width = int(serialized["orig_img_w"])
                top1 = candidates[0] if candidates else None
                mask = (
                    self.agent.decode_rle_mask(top1["rle"], height, width)
                    if top1 is not None
                    else np.zeros((height, width), dtype=np.uint8)
                )
                cached = {
                    "serialized": serialized,
                    "candidates": candidates,
                    "top1": top1,
                    "mask": mask,
                    "geometry": self.agent.mask_geometry(mask, width, height),
                }
                self.prompt_cache[cache_key] = cached
            run = dict(cached)
            run["target"] = target
            target_runs.append(run)
        return target_runs

    def _mask_rle(self, target_runs, target_ids):
        union = self.agent.union_mask_for_target_ids(target_runs, target_ids)
        return None if union is None else mask_to_uncompressed_rle(union)

    def _structure_prediction(self, record, response, target_runs):
        task_type = record["task"]
        if task_type in ("grounded_segmentation", "grounded_vqa"):
            valid_ids = self.agent.valid_mask_target_ids(target_runs)
            target_ids = self.agent.normalize_response_target_ids(
                response.get("target_ids"), valid_ids, valid_ids
            )
            predicted_mask = self._mask_rle(target_runs, target_ids)
            return {
                "answer": self.agent.ensure_seg_token(
                    response.get("answer", ""), predicted_mask is not None
                ),
                "predicted_mask": predicted_mask,
            }

        valid_ids = self.agent.valid_mask_target_ids(target_runs)
        report = []
        no_predicted_target = not any(
            bool((run.get("geometry") or {}).get("has_mask")) for run in target_runs
        )
        for sentence in response.get("report", []):
            if not isinstance(sentence, dict):
                continue
            target_ids = self.agent.normalize_response_target_ids(
                sentence.get("target_ids"), valid_ids, []
            )
            text = str(sentence.get("text") or "").strip()
            if no_predicted_target and normalize_organ_name(record.get("organ")) != "prostate":
                text = re.sub(
                    r"\bprostate(?:\s+gland)?\b",
                    organ_display_name(record),
                    text,
                    flags=re.IGNORECASE,
                )
            report.append({
                "section": str(sentence.get("section") or "Findings"),
                "text": text,
                "finding_type": str(
                    sentence.get("finding_type") or infer_finding_type(
                        text
                    )
                ),
                "predicted_mask": self._mask_rle(target_runs, target_ids),
            })
        return {"report": report}

    def predict(self, record, instruction, args):
        run_args = copy.copy(args)
        run_args.task = record["task"]
        plan, raw_plan = self.agent.agent_parse_question(
            record["image_path"], instruction, self.categories, run_args
        )
        plan["task_type"] = record["task"]
        target_runs = self._run_targets_cached(
            record["image_path"], plan, run_args
        )
        response, raw_final = self.agent.compose_grounded_response(
            image_path=record["image_path"],
            instruction=instruction,
            plan=plan,
            target_runs=target_runs,
            args=run_args,
        )
        prediction = self._structure_prediction(record, response, target_runs)
        raw = {
            "plan": plan,
            "response": response,
            "raw_plan_response": raw_plan,
            "raw_final_response": raw_final,
            "sam_prompt_cache_size": len(self.prompt_cache),
        }
        return prediction, raw


def build_agent_client(args):
    return USSam3AgentClient(args)


def slug(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_") or "dataset"


def normalize_organ_name(value):
    return slug(value).lower()


def organ_display_name(record):
    image = record.get("image") if isinstance(record.get("image"), dict) else {}
    value = image.get("organ") or record.get("organ") or "organ"
    return str(value).replace("_", " ").strip()


def is_spatial_finding_type(value):
    normalized = normalize_organ_name(value)
    if any(hint in normalized for hint in NON_SPATIAL_FINDING_HINTS):
        return False
    return any(hint in normalized for hint in SPATIAL_FINDING_HINTS)


def reference_spatial_finding_types(report):
    return {
        sentence.get("finding_type")
        for sentence in report or []
        if isinstance(sentence, dict)
        and sentence.get("finding_type")
        and sentence.get("evidence_mask_ids")
        and is_spatial_finding_type(sentence.get("finding_type"))
    }


def annotation_categories(path, split="test"):
    categories = set()
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != split:
                continue
            for mask in record.get("masks", []):
                if not isinstance(mask, dict):
                    continue
                category = mask.get("canonical_region") or mask.get("category")
                if category:
                    categories.add(str(category).strip())
    return sorted(categories, key=lambda value: value.lower())


def load_full_records(
    path,
    data_root,
    organ,
    limit_images=None,
    max_per_task=None,
    split="test",
):
    tasks = []
    image_count = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("split") != split:
                continue
            image_count += 1
            if limit_images is not None and image_count > limit_images:
                break
            dataset = item.get("dataset")
            if not dataset:
                raise ValueError(f"Missing dataset field in sample {item.get('sample_id')}")
            image_path = Path(data_root) / dataset / item["split"] / item["image"]["file_name"]
            base = {
                "image_id": item["sample_id"],
                "split": item["split"],
                "dataset": dataset,
                "source": dataset,
                "organ": organ,
                "image": item["image"],
                "image_path": str(image_path),
                "masks": item.get("masks", []),
            }
            for seg in item.get("annotation", {}).get("segmentation_tasks", []):
                rec = dict(base)
                rec.update(seg)
                rec["sample_id"] = seg.get("task_id")
                rec["task"] = "grounded_segmentation"
                tasks.append(rec)
            for qa in item.get("annotation", {}).get("grounded_vqa", []):
                rec = dict(base)
                rec.update(qa)
                rec["sample_id"] = qa.get("qa_id")
                rec["task"] = "grounded_vqa"
                tasks.append(rec)
            report = item.get("annotation", {}).get("grounded_report", {})
            rec = dict(base)
            rec.update({
                "sample_id": item["sample_id"] + "_report",
                "task": "grounded_report_generation",
                "prompt": report.get(
                    "prompt",
                    f"Generate a structured {str(organ).replace('_', ' ')} ultrasound description.",
                ),
                "report": report.get("sentences", []),
            })
            tasks.append(rec)
    if max_per_task is not None:
        kept = []
        counts = Counter()
        for task in tasks:
            if counts[task["task"]] < max_per_task:
                kept.append(task)
                counts[task["task"]] += 1
        tasks = kept
    return tasks


def find_coco_file(split_dir, annotation_name):
    preferred = split_dir / annotation_name
    if preferred.exists():
        return preferred
    candidates = sorted(split_dir.glob("*.json"))
    return candidates[0] if len(candidates) == 1 else None


def load_coco_index(
    data_root,
    annotation_name="_annotations.coco.json",
    split="test",
):
    index = {}
    data_root = Path(data_root)
    for dataset_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        dataset_id = slug(dataset_dir.name)
        for selected_split in [split]:
            split_dir = dataset_dir / selected_split
            if not split_dir.is_dir():
                continue
            coco_path = find_coco_file(split_dir, annotation_name)
            if coco_path is None:
                print(f"[coco-index] skip ambiguous or missing JSON: {split_dir}", flush=True)
                continue
            data = json.loads(coco_path.read_text(encoding="utf-8-sig"))
            images = {img["id"]: img for img in data.get("images", [])}
            cats = {cat["id"]: cat.get("name", str(cat["id"])) for cat in data.get("categories", [])}
            for ann in data.get("annotations", []):
                if ann.get("iscrowd", 0) or ann.get("image_id") not in images:
                    continue
                img = images[ann["image_id"]]
                mask_id = f"{dataset_id}_{selected_split}_img{img['id']}_ann{ann['id']}"
                index[mask_id] = {
                    "mask_id": mask_id,
                    "dataset": dataset_dir.name,
                    "split": selected_split,
                    "image_id": f"{dataset_id}_{selected_split}_{img['id']}",
                    "width": img["width"],
                    "height": img["height"],
                    "category": cats.get(ann.get("category_id"), str(ann.get("category_id"))),
                    "bbox_xywh": ann.get("bbox", []),
                    "segmentation": ann.get("segmentation", []),
                }
    return index


def extract_json(text):
    text = str(text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def empty_mask(width, height):
    return np.zeros((height, width), dtype=bool)


def ensure_mask_shape(mask, width, height):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (height, width):
        resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (width, height), Image.Resampling.NEAREST
        )
        mask = np.asarray(resized) > 0
    return mask


def bbox_pixels(bbox, width, height):
    mask = empty_mask(width, height)
    if not bbox or len(bbox) != 4:
        return mask
    x, y, w, h = [float(v) for v in bbox]
    x0 = max(0, min(width, math.floor(x)))
    y0 = max(0, min(height, math.floor(y)))
    x1 = max(0, min(width, math.ceil(x + w)))
    y1 = max(0, min(height, math.ceil(y + h)))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def polygon_pixels(points, width, height):
    if not points or len(points) < 3:
        return empty_mask(width, height)
    polygon = [float(value) for point in points for value in point[:2]]
    return coco_seg_pixels([polygon], width, height)


def coco_seg_pixels(segmentation, width, height):
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError(
            "Fast mask scoring requires pycocotools: pip install pycocotools"
        ) from exc

    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
    else:
        polygons = [poly for poly in (segmentation or []) if poly and len(poly) >= 6]
        if not polygons:
            return empty_mask(width, height)
        rles = mask_utils.frPyObjects(polygons, height, width)
        rle = mask_utils.merge(rles) if isinstance(rles, list) else rles

    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return ensure_mask_shape(decoded, width, height)


def looks_like_points(value):
    return (
        isinstance(value, list)
        and len(value) >= 3
        and all(isinstance(p, list) and len(p) >= 2 for p in value)
    )


def normalize_pred_mask(pred_mask):
    if not pred_mask:
        return None
    if isinstance(pred_mask, dict):
        return pred_mask
    if looks_like_points(pred_mask):
        return {"type": "polygon", "points": [[p[0], p[1]] for p in pred_mask]}
    if isinstance(pred_mask, list) and len(pred_mask) == 4 and all(isinstance(v, (int, float)) for v in pred_mask):
        return {"type": "bbox", "bbox_xywh": pred_mask}
    if isinstance(pred_mask, list) and len(pred_mask) >= 6 and len(pred_mask) % 2 == 0 and all(isinstance(v, (int, float)) for v in pred_mask):
        return {"type": "polygon", "points": list(zip(pred_mask[0::2], pred_mask[1::2]))}
    return None


def pred_mask_pixels(pred_mask, width, height):
    pred_mask = normalize_pred_mask(pred_mask)
    if not pred_mask:
        return empty_mask(width, height)
    mtype = str(pred_mask.get("type", "")).lower()
    if mtype == "rle":
        rle = {
            "size": pred_mask.get("size", [height, width]),
            "counts": pred_mask.get("counts", []),
        }
        return coco_seg_pixels(rle, width, height)
    if mtype == "bbox":
        return bbox_pixels(pred_mask.get("bbox_xywh") or pred_mask.get("bbox"), width, height)
    if mtype == "polygon":
        return polygon_pixels(pred_mask.get("points") or pred_mask.get("polygon") or [], width, height)
    return empty_mask(width, height)


def lru_get(cache, key):
    if cache is None or key not in cache:
        return None
    value = cache.pop(key)
    cache[key] = value
    return value


def lru_put(cache, key, value, max_items):
    if cache is None or max_items <= 0:
        return
    if key in cache:
        cache.pop(key)
    cache[key] = value
    while len(cache) > max_items:
        cache.popitem(last=False)


def gt_pixels(
    mask_ids,
    coco_index,
    record,
    mask_cache=None,
    union_cache=None,
    cache_size=64,
):
    width = record["image"]["width"]
    height = record["image"]["height"]
    valid_items = []
    for mask_id in mask_ids or []:
        item = coco_index.get(mask_id)
        if item:
            valid_items.append((mask_id, item))
    if valid_items:
        width = valid_items[0][1]["width"]
        height = valid_items[0][1]["height"]

    union_key = (tuple(mask_id for mask_id, _ in valid_items), width, height)
    cached_union = lru_get(union_cache, union_key)
    if cached_union is not None:
        return cached_union, width, height

    pixels = empty_mask(width, height)
    for mask_id, item in valid_items:
        single = lru_get(mask_cache, mask_id)
        if single is None:
            single = coco_seg_pixels(
                item.get("segmentation", []), item["width"], item["height"]
            )
            lru_put(mask_cache, mask_id, single, cache_size)
        if single.shape != pixels.shape:
            single = ensure_mask_shape(single, width, height)
        pixels |= single

    lru_put(union_cache, union_key, pixels, cache_size)
    return pixels, width, height


def iou_dice(pred_pixels, ref_pixels):
    pred_pixels = np.asarray(pred_pixels, dtype=bool)
    ref_pixels = np.asarray(ref_pixels, dtype=bool)
    pred_count = int(np.count_nonzero(pred_pixels))
    ref_count = int(np.count_nonzero(ref_pixels))
    if pred_count == 0 and ref_count == 0:
        return 1.0, 1.0
    if pred_count == 0 or ref_count == 0:
        return 0.0, 0.0
    inter = int(np.count_nonzero(pred_pixels & ref_pixels))
    union = pred_count + ref_count - inter
    return inter / union if union else 0.0, 2 * inter / (pred_count + ref_count)


def normalize_text(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def text_tokens(text):
    return normalize_text(text).split()


def yes_no(text):
    norm = normalize_text(text)
    if norm.startswith("yes") or " yes " in f" {norm} ":
        return "yes"
    if norm.startswith("no") or " no " in f" {norm} ":
        return "no"
    return None


def no_target_answer(text):
    norm = normalize_text(text)
    patterns = (
        r"\bno\s+(?:(?:requested|visible|labeled|labelled|supplied)\s+){0,2}"
        r"(?:[a-z0-9]+\s+){0,4}(?:target|boundary|extent|structure|organ|lesion|mass|tumor|nodule)\b",
        r"\bno\s+target\s+(?:is\s+)?(?:available|identified|present|visible)\b",
        r"\b(?:target|structure|organ|lesion|mass|tumor|nodule)\s+"
        r"(?:is\s+)?(?:not available|not visible|not identified|absent)\b",
    )
    return any(re.search(pattern, norm) for pattern in patterns)


def vqa_answer_correct(pred, ref, answer_type):
    pred_norm = normalize_text(pred)
    ref_norm = normalize_text(ref)
    atype = str(answer_type or "").lower()
    if no_target_answer(ref):
        return no_target_answer(pred)
    if "yes_no" in atype:
        return yes_no(pred) == yes_no(ref)
    if "location" in atype:
        pred_locs = extract_grid_locations(pred)
        ref_locs = extract_grid_locations(ref)
        return bool(ref_locs) and pred_locs == ref_locs
    if "size" in atype or "measurement" in atype:
        numeric_match = size_matches(pred, ref, tolerance=0.2)
        if numeric_match is not None:
            return numeric_match
        qualitative_match = qualitative_extent_matches(pred, ref)
        if qualitative_match is not None:
            return qualitative_match
        return False
    if any(term in atype for term in ("area", "ratio", "percent", "extent", "proportion")):
        numeric_match = area_ratio_matches(pred, ref)
        if numeric_match is not None:
            return numeric_match
        qualitative_match = qualitative_extent_matches(pred, ref)
        if qualitative_match is not None:
            return qualitative_match
        return False
    if "count" in atype:
        pred_count = count_value(pred)
        ref_count = count_value(ref)
        if ref_count is not None:
            return pred_count == ref_count
    ref_tokens = set(ref_norm.split())
    return bool(ref_tokens) and len(ref_tokens & set(pred_norm.split())) / len(ref_tokens) >= 0.5


def ngrams(tokens, n):
    return [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]


def bleu_n(pred_text, ref_text, n):
    """Cumulative sentence BLEU-N with add-one smoothing."""
    pred = text_tokens(pred_text)
    ref = text_tokens(ref_text)
    if not pred or not ref:
        return 0.0
    precisions = []
    for order in range(1, n + 1):
        pred_ngrams = ngrams(pred, order)
        ref_counts = Counter(ngrams(ref, order))
        matches = 0
        for ng in pred_ngrams:
            if ref_counts[ng] > 0:
                matches += 1
                ref_counts[ng] -= 1
        precisions.append((matches + 1) / (len(pred_ngrams) + 1))
    bp = 1.0 if len(pred) > len(ref) else math.exp(1 - len(ref) / max(len(pred), 1))
    return bp * math.exp(sum(math.log(p) for p in precisions) / n)


def meteor_exact(pred_text, ref_text):
    """METEOR-style exact-token score with fragmentation penalty.

    This intentionally excludes stemming and synonym matching, hence the
    explicit metric name.
    """
    pred = text_tokens(pred_text)
    ref = text_tokens(ref_text)
    if not pred or not ref:
        return 0.0
    available = defaultdict(list)
    for index, token in enumerate(ref):
        available[token].append(index)
    aligned = []
    for token in pred:
        if available[token]:
            aligned.append(available[token].pop(0))
    matches = len(aligned)
    if matches == 0:
        return 0.0
    precision = matches / len(pred)
    recall = matches / len(ref)
    fmean = (10 * precision * recall) / (recall + 9 * precision) if (recall + 9 * precision) else 0.0
    chunks = 1 + sum(1 for left, right in zip(aligned, aligned[1:]) if right != left + 1)
    penalty = 0.5 * (chunks / matches) ** 3
    return fmean * (1 - penalty)


def extract_grid_locations(text):
    norm = normalize_text(text)
    return {
        f"{row}-{col}"
        for row, col in re.findall(r"\b(upper|middle|lower)\s+(left|central|right)\b", norm)
    }


def record_category_terms(record):
    terms = set()
    for mask in record.get("masks", []):
        if not isinstance(mask, dict):
            continue
        for key in ("category", "canonical_region"):
            normalized = normalize_text(mask.get(key, ""))
            if normalized:
                terms.add(normalized)
    return terms


def mentioned_categories(text, record):
    norm = normalize_text(text)
    return {
        category
        for category in record_category_terms(record)
        if re.search(rf"\b{re.escape(category)}\b", norm)
    }


def category_matches(pred_text, ref_text, record):
    ref_categories = mentioned_categories(ref_text, record)
    if not ref_categories:
        return None
    return ref_categories.issubset(mentioned_categories(pred_text, record))


def target_presence_value(text, record):
    norm = normalize_text(text)
    negative_patterns = (
        r"\bappears normal\b",
        r"\bno\s+(?:(?:discrete|definite|focal|visible|labeled|labelled|supplied)\s+){0,3}"
        r"(?:[a-z0-9]+\s+){0,4}(?:lesion|mass|tumor|abnormality|nodule|target|structure|organ)\b",
        r"\b(?:target|structure|organ|lesion|mass|tumor|nodule)\s+"
        r"(?:is\s+)?(?:not visible|absent|unavailable)\b",
    )
    if any(re.search(pattern, norm) for pattern in negative_patterns):
        return False
    target_terms = record_category_terms(record)
    image_organ = normalize_text(record.get("image", {}).get("organ", ""))
    if image_organ:
        target_terms.add(image_organ)
    if any(re.search(rf"\b{re.escape(term)}\b", norm) for term in target_terms):
        return True
    if set(norm.split()) & {
        "lesion",
        "mass",
        "tumor",
        "abnormality",
        "nodule",
        "target",
        "structure",
    }:
        return True
    return None


def size_pair(text):
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:x|by)\s*(\d+(?:\.\d+)?)\s*(?:pixels?|px)?\b", str(text).lower())
    return (float(match.group(1)), float(match.group(2))) if match else None


def size_matches(pred_text, ref_text, tolerance=0.2):
    pred = size_pair(pred_text)
    ref = size_pair(ref_text)
    if ref is None:
        return None
    if pred is None:
        return False
    direct = all(abs(p - r) <= max(1.0, tolerance * r) for p, r in zip(pred, ref))
    swapped = all(abs(p - r) <= max(1.0, tolerance * r) for p, r in zip(pred, reversed(ref)))
    return direct or swapped


def percentage_value(text):
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:%|percent\b)", str(text).lower())
    return float(match.group(1)) if match else None


def normalized_area_ratio(text):
    percent = percentage_value(text)
    if percent is not None:
        return percent / 100.0
    patterns = (
        r"\b(?:about|approximately|roughly|around)?\s*(0(?:\.\d+)|1(?:\.0+)?)\s+of\s+the\s+image",
        r"\b(?:ratio|proportion|fraction)\s*(?:is|of|=|:)?\s*(0(?:\.\d+)|1(?:\.0+)?)\b",
    )
    lower = str(text).lower()
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            return float(match.group(1))
    return None


def area_ratio_matches(pred_text, ref_text, relative_tolerance=0.15, absolute_tolerance=0.01):
    pred = normalized_area_ratio(pred_text)
    ref = normalized_area_ratio(ref_text)
    if ref is None:
        return None
    if pred is None:
        return False
    return abs(pred - ref) <= max(
        absolute_tolerance,
        relative_tolerance * max(abs(ref), 0.01),
    )


def qualitative_extent_value(text):
    norm = normalize_text(text)
    if any(term in norm.split() for term in ("small", "limited", "narrow", "minor")):
        return "small"
    if any(
        term in norm.split()
        for term in ("large", "broad", "broadly", "wide", "widely", "extensive", "substantial")
    ):
        return "broad"
    if any(term in norm.split() for term in ("moderate", "intermediate")):
        return "moderate"
    return None


def qualitative_extent_matches(pred_text, ref_text):
    ref = qualitative_extent_value(ref_text)
    if ref is None:
        return None
    return qualitative_extent_value(pred_text) == ref


def percentage_matches(pred_text, ref_text, relative_tolerance=0.15, absolute_tolerance=1.0):
    pred = percentage_value(pred_text)
    ref = percentage_value(ref_text)
    if ref is None:
        return None
    if pred is None:
        return False
    return abs(pred - ref) <= max(absolute_tolerance, relative_tolerance * max(abs(ref), 1.0))


def count_value(text):
    norm = normalize_text(text)
    if "multiple" in norm or re.search(r"\b(?:two|three|four|five|six|[2-9])\b", norm):
        return "multiple"
    if "single" in norm or re.search(r"\b(?:one|1)\b", norm):
        return "single"
    return None


def lcs_len(a, b):
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            old = dp[j]
            if x == y:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = old
    return dp[-1]


def rouge_l(pred_text, ref_text):
    pred = text_tokens(pred_text)
    ref = text_tokens(ref_text)
    if not pred or not ref:
        return 0.0
    lcs = lcs_len(pred, ref)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0


def report_text(report):
    return " ".join(str(s.get("text", "")) for s in report if isinstance(s, dict))


def unsupported_hits(text):
    lower = str(text).lower()
    hits = []
    for term in UNSUPPORTED_US_DESCRIPTORS:
        for match in re.finditer(re.escape(term), lower):
            context = lower[max(0, match.start() - 40):match.start()]
            if not re.search(r"\b(no|not|without|cannot|unable|unavailable|unassessed)\b[^.!?]{0,30}$", context):
                hits.append(term)
                break
    return hits


def banned_hits(text):
    lower = str(text).lower()
    return [term for term in BANNED_TERMS if term in lower]


def segmentation_prompt(record):
    return str(record["instruction"]).strip()


def vqa_prompt(record):
    return str(record["question"]).strip()


def report_prompt(record):
    return str(
        record.get("prompt")
        or f"Generate a grounded {str(record.get('organ') or 'organ').replace('_', ' ')} ultrasound report for this image."
    )


def finding_type_roles(value):
    normalized = normalize_organ_name(value)
    if not normalized:
        return set()
    if "no_target" in normalized or "absence" in normalized:
        return {"absence"}
    if "not_applicable" in normalized or "unavailable" in normalized:
        return {"not_applicable"}
    roles = set()
    if "scope" in normalized:
        roles.add("scope")
    if "impression" in normalized:
        roles.add("impression")
    if "summary" in normalized or "distribution" in normalized:
        roles.add("summary")
    if any(term in normalized for term in ("category", "label", "status", "class")):
        roles.add("category")
    if "location" in normalized or "boundary" in normalized:
        roles.add("location")
    if "presence" in normalized or "count" in normalized:
        roles.add("presence")
    if "pixel_extent" in normalized or "size" in normalized:
        roles.add("size")
    if (
        "area" in normalized
        or ("extent" in normalized and "pixel_extent" not in normalized)
    ):
        roles.add("extent")
    return roles or {normalized}


def report_role_set(report):
    return {
        role
        for sentence in report
        if isinstance(sentence, dict)
        for role in finding_type_roles(sentence.get("finding_type"))
    }


def best_predicted_sentence(ref_sentence, pred_report):
    ref_roles = finding_type_roles(ref_sentence.get("finding_type"))
    ref_tokens = set(text_tokens(ref_sentence.get("text", "")))
    best = {}
    best_score = (-1, -1)
    for sentence in pred_report:
        if not isinstance(sentence, dict):
            continue
        pred_roles = finding_type_roles(sentence.get("finding_type"))
        role_overlap = len(ref_roles & pred_roles)
        if not role_overlap:
            continue
        exact = int(
            normalize_organ_name(sentence.get("finding_type"))
            == normalize_organ_name(ref_sentence.get("finding_type"))
        )
        token_overlap = len(ref_tokens & set(text_tokens(sentence.get("text", ""))))
        score = (exact * 100 + role_overlap, token_overlap)
        if score > best_score:
            best_score = score
            best = sentence
    return best


def prompt_for(record):
    if record["task"] == "grounded_segmentation":
        return segmentation_prompt(record)
    if record["task"] == "grounded_vqa":
        return vqa_prompt(record)
    if record["task"] == "grounded_report_generation":
        return report_prompt(record)
    raise ValueError(record["task"])


def score_prediction(
    record,
    pred,
    coco_index,
    mask_cache=None,
    union_cache=None,
    cache_size=64,
):
    metrics = {k: "" for k in METRIC_FIELDS}
    metrics.update({
        "task": record.get("task"),
        "dataset": record.get("dataset") or record.get("source"),
        "sample_id": record.get("sample_id"),
        "image_id": record.get("image_id"),
        "split": record.get("split"),
        "format_ok": False,
        "error": "",
    })
    if not isinstance(pred, dict):
        metrics["error"] = "prediction_not_json_object"
        return metrics
    metrics["format_ok"] = True
    if record["task"] == "grounded_segmentation":
        ref_pixels, width, height = gt_pixels(
            record.get("target_mask_ids", []), coco_index, record,
            mask_cache, union_cache, cache_size,
        )
        pred_pixels = pred_mask_pixels(pred.get("predicted_mask"), width, height)
        iou, dice = iou_dice(pred_pixels, ref_pixels)
        answer = str(pred.get("answer", ""))
        metrics.update({
            "iou": iou,
            "dice": dice,
            "empty_prediction": int(not np.any(pred_pixels)),
            "seg_token_ok": int(("[SEG]" in answer) == bool(np.any(ref_pixels))),
            "banned_term_hits": len(banned_hits(answer)),
        })
    elif record["task"] == "grounded_vqa":
        ref_pixels, width, height = gt_pixels(
            record.get("evidence_mask_ids", []), coco_index, record,
            mask_cache, union_cache, cache_size,
        )
        pred_pixels = pred_mask_pixels(pred.get("predicted_mask"), width, height)
        iou, dice = iou_dice(pred_pixels, ref_pixels)
        answer = str(pred.get("answer", ""))
        correct = vqa_answer_correct(answer, record.get("answer", ""), record.get("answer_type", ""))
        requires_grounding = record.get("requires_grounding") is True
        metrics.update({
            "iou": iou if requires_grounding else "",
            "dice": dice if requires_grounding else "",
            "empty_prediction": int(not np.any(pred_pixels)),
            "answer_correct": int(correct),
            "grounded_answer_correct": int(correct and iou > 0.5) if requires_grounding else "",
            "banned_term_hits": len(banned_hits(answer)),
        })
    elif record["task"] == "grounded_report_generation":
        pred_report = pred.get("report", pred.get("sentences", []))
        ref_report = record.get("report", [])
        if not isinstance(pred_report, list):
            metrics["format_ok"] = False
            metrics["error"] = "report_not_list"
            return metrics
        pred_text = report_text(pred_report if isinstance(pred_report, list) else [])
        ref_text = report_text(ref_report)
        pred_types = report_role_set(pred_report)
        ref_types = report_role_set(ref_report)
        type_intersection = len(pred_types & ref_types)
        type_precision = type_intersection / len(pred_types) if pred_types else 0.0
        type_recall = type_intersection / len(ref_types) if ref_types else 0.0
        type_f1 = 2 * type_precision * type_recall / (type_precision + type_recall) if (type_precision + type_recall) else 0.0

        ref_presence = target_presence_value(ref_text, record)
        pred_presence = target_presence_value(pred_text, record)
        presence_ok = None if ref_presence is None else pred_presence == ref_presence
        category_ok = category_matches(pred_text, ref_text, record)
        ref_locs = extract_grid_locations(ref_text)
        pred_locs = extract_grid_locations(pred_text)
        location_ok = None if not ref_locs else pred_locs == ref_locs
        size_ok = size_matches(pred_text, ref_text)
        extent_ok = percentage_matches(pred_text, ref_text)
        semantic_values = [
            value
            for value in (presence_ok, category_ok, location_ok, size_ok, extent_ok)
            if value is not None
        ]

        spatial_types = reference_spatial_finding_types(ref_report)
        grounding_scores = []
        for ref_sentence in ref_report:
            if not isinstance(ref_sentence, dict):
                continue
            finding_type = ref_sentence.get("finding_type")
            if finding_type not in spatial_types:
                continue
            evidence_ids = ref_sentence.get("evidence_mask_ids", [])
            if not evidence_ids:
                continue
            ref_pixels, width, height = gt_pixels(
                evidence_ids, coco_index, record,
                mask_cache, union_cache, cache_size,
            )
            pred_sentence = best_predicted_sentence(ref_sentence, pred_report)
            pred_pixels = pred_mask_pixels(pred_sentence.get("predicted_mask"), width, height)
            grounding_scores.append(iou_dice(pred_pixels, ref_pixels)[0])
        metrics.update({
            "bleu_1": bleu_n(pred_text, ref_text, 1),
            "bleu_2": bleu_n(pred_text, ref_text, 2),
            "bleu_3": bleu_n(pred_text, ref_text, 3),
            "bleu_4": bleu_n(pred_text, ref_text, 4),
            "meteor_exact": meteor_exact(pred_text, ref_text),
            "rouge_l": rouge_l(pred_text, ref_text),
            "finding_type_recall": type_recall,
            "finding_type_precision": type_precision,
            "finding_type_f1": type_f1,
            "report_grounding_iou": statistics.mean(grounding_scores) if grounding_scores else "",
            "report_grounding_recall_50": statistics.mean(score > 0.5 for score in grounding_scores) if grounding_scores else "",
            "presence_ok": int(presence_ok) if presence_ok is not None else "",
            "category_ok": int(category_ok) if category_ok is not None else "",
            "location_ok": int(location_ok) if location_ok is not None else "",
            "size_ok": int(size_ok) if size_ok is not None else "",
            "extent_ok": int(extent_ok) if extent_ok is not None else "",
            "semantic_accuracy": statistics.mean(semantic_values) if semantic_values else "",
            "unsupported_descriptor_count": len(unsupported_hits(pred_text)),
            "banned_term_hits": len(banned_hits(pred_text)),
        })
    return metrics


def write_csv_rows(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv_row(path, row, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def prepare_resume_state(predictions_csv, resume, retry_errors):
    done = set()
    rows = []
    path = Path(predictions_csv)
    if not resume or not path.exists():
        return done, 1
    rows = read_csv_rows(path)
    if retry_errors:
        rows = [row for row in rows if not row.get("error")]
        write_csv_rows(path, rows, PREDICTION_FIELDS)
    for row in rows:
        if row.get("sample_id"):
            done.add(row["sample_id"])
    return done, len(rows) + 1


def prediction_row(eval_id, record, client, args):
    prompt = prompt_for(record)
    row = {
        "eval_id": eval_id,
        "dataset": record["dataset"],
        "task": record["task"],
        "sample_id": record["sample_id"],
        "image_id": record["image_id"],
        "split": record["split"],
        "image_file": record["image"]["file_name"],
        "image_path": record["image_path"],
        "model_backend": "us_sam3_agent",
        "model_name": client.model,
        "prompt": prompt,
        "reference_json": json.dumps(record, ensure_ascii=False),
        "prediction_raw": "",
        "prediction_json": "",
        "error": "",
        "elapsed_sec": "",
    }
    t0 = time.time()
    try:
        pred, raw = client.predict(record, prompt, args)
        row["prediction_raw"] = json.dumps(raw, ensure_ascii=False)
        row["prediction_json"] = json.dumps(pred, ensure_ascii=False)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    row["elapsed_sec"] = round(time.time() - t0, 3)
    return row


def pending_prediction_jobs(tasks, done, start_idx):
    jobs = []
    for source_index, record in enumerate(tasks, 1):
        if record["sample_id"] in done:
            continue
        jobs.append((source_index, record["sample_id"], record))
    return jobs


def check_consecutive_errors(row, consecutive_errors, args):
    if row["error"]:
        consecutive_errors += 1
        if args.stop_after_consecutive_errors and consecutive_errors >= args.stop_after_consecutive_errors:
            raise RuntimeError(
                f"Stopped after {consecutive_errors} consecutive prediction errors. "
                "Check the local model, GPU memory, and input images, then rerun with --resume --retry-errors."
            )
    else:
        consecutive_errors = 0
    return consecutive_errors


def show_prediction_progress(completed_run, pending_total, initial_done, total, source_index, record, row, started_at):
    overall_done = min(total, initial_done + completed_run)
    remaining = max(0, total - overall_done)
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    rate = completed_run / elapsed
    eta = remaining / rate if rate > 0 else math.inf
    print(
        f"[predict overall={overall_done}/{total} source={source_index}/{total} remaining={remaining} | "
        f"this_run={completed_run}/{pending_total} | rate={rate:.2f}/s ETA={format_duration(eta)}] "
        f"dataset={record.get('dataset')} task={record['task']} sample={record['sample_id']} "
        f"error={bool(row['error'])}",
        flush=True,
    )


def run_predictions_sequential(jobs, total, initial_done, client, args):
    consecutive_errors = 0
    started_at = time.perf_counter()
    pending_total = len(jobs)
    for completed_run, (source_index, eval_id, record) in enumerate(jobs, 1):
        row = prediction_row(eval_id, record, client, args)
        append_csv_row(args.predictions_csv, row, PREDICTION_FIELDS)
        show_prediction_progress(
            completed_run, pending_total, initial_done, total, source_index, record, row, started_at
        )
        consecutive_errors = check_consecutive_errors(row, consecutive_errors, args)


def run_predictions_threaded(jobs, total, initial_done, client, args):
    consecutive_errors = 0
    next_job = 0
    completed_run = 0
    pending_total = len(jobs)
    started_at = time.perf_counter()
    active = {}

    def submit_next(executor):
        nonlocal next_job
        if next_job >= len(jobs):
            return
        source_index, eval_id, record = jobs[next_job]
        next_job += 1
        future = executor.submit(prediction_row, eval_id, record, client, args)
        active[future] = (source_index, eval_id, record)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for _ in range(min(args.workers, len(jobs))):
            submit_next(executor)
        while active:
            done_futures, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done_futures:
                source_index, eval_id, record = active.pop(future)
                row = future.result()
                append_csv_row(args.predictions_csv, row, PREDICTION_FIELDS)
                completed_run += 1
                show_prediction_progress(
                    completed_run, pending_total, initial_done, total, source_index, record, row, started_at
                )
                try:
                    consecutive_errors = check_consecutive_errors(row, consecutive_errors, args)
                except RuntimeError:
                    for pending in active:
                        pending.cancel()
                    raise
                submit_next(executor)


def run_predictions(args):
    tasks = load_full_records(
        args.input,
        args.data_root,
        args.organ,
        args.limit_images,
        args.max_per_task,
        args.split,
    )
    missing_images = sorted({record["image_path"] for record in tasks if not Path(record["image_path"]).is_file()})
    if missing_images:
        examples = "\n".join(missing_images[:10])
        raise FileNotFoundError(f"{len(missing_images)} referenced images were not found. Examples:\n{examples}")
    if not args.categories and not args.categories_file:
        detected_categories = annotation_categories(args.input, args.split)
        if not detected_categories:
            raise ValueError("No mask categories were found in the annotation JSONL.")
        args.categories = ",".join(detected_categories)
        print(
            f"[preflight] auto_categories={detected_categories}",
            flush=True,
        )
    spatial_types = sorted({
        finding_type
        for record in tasks
        if record["task"] == "grounded_report_generation"
        for finding_type in reference_spatial_finding_types(record.get("report", []))
    })
    print(
        f"[preflight] organ={args.organ} split={args.split} "
        f"input={args.input} data_root={args.data_root} "
        f"tasks={len(tasks)} images={len({record['image_id'] for record in tasks})} "
        f"datasets={len({record['dataset'] for record in tasks})}",
        flush=True,
    )
    print(f"[preflight] report_spatial_finding_types={spatial_types}", flush=True)
    if args.workers != 1:
        raise ValueError(
            "US-SAM3 agent inference requires --workers 1 because one GPU model is shared."
        )
    client = build_agent_client(args)
    done, start_idx = prepare_resume_state(args.predictions_csv, args.resume, args.retry_errors)
    if Path(args.predictions_csv).exists() and not args.resume:
        Path(args.predictions_csv).unlink()
        start_idx = 1
    jobs = pending_prediction_jobs(tasks, done, start_idx)
    initial_done = len(tasks) - len(jobs)
    print(
        f"[predict] total={len(tasks)} already_done={initial_done} "
        f"pending={len(jobs)} resume={args.resume} retry_errors={args.retry_errors}",
        flush=True,
    )
    if args.workers <= 1:
        run_predictions_sequential(jobs, len(tasks), initial_done, client, args)
    else:
        run_predictions_threaded(jobs, len(tasks), initial_done, client, args)


def format_duration(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def show_score_progress(current, total, row, errors, started_at, force=False, every=50):
    if total <= 0:
        return
    interactive = sys.stdout.isatty()
    if not force and not interactive and current % max(1, every) != 0:
        return
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    rate = current / elapsed
    eta = (total - current) / rate if rate > 0 else math.inf
    fraction = current / total
    width = 28
    filled = min(width, int(width * fraction))
    bar = "=" * filled + (">" if filled < width else "") + "." * max(0, width - filled - 1)
    task = str(row.get("task", ""))[:24]
    sample = str(row.get("sample_id", ""))[:36]
    message = (
        f"[score] [{bar}] {current}/{total} ({fraction * 100:6.2f}%) "
        f"errors={errors} rate={rate:.1f}/s ETA={format_duration(eta)} "
        f"task={task} sample={sample}"
    )
    if interactive:
        print("\r" + message.ljust(170), end="\n" if force else "", flush=True)
    else:
        print(message, flush=True)


def score_predictions(args):
    coco_index = load_coco_index(
        args.data_root,
        args.annotation_name,
        args.split,
    )
    pred_rows = [
        row
        for row in read_csv_rows(args.predictions_csv)
        if row.get("split") == args.split
    ]
    mask_cache = OrderedDict()
    union_cache = OrderedDict()
    referenced_masks = set()
    for row in pred_rows:
        try:
            reference = json.loads(row.get("reference_json", "{}"))
        except json.JSONDecodeError:
            continue
        referenced_masks.update(reference.get("target_mask_ids", []) or [])
        referenced_masks.update(reference.get("evidence_mask_ids", []) or [])
        for sentence in reference.get("report", []) or []:
            if isinstance(sentence, dict):
                referenced_masks.update(sentence.get("evidence_mask_ids", []) or [])
    missing_masks = sorted(referenced_masks - set(coco_index))
    if missing_masks:
        examples = "\n".join(missing_masks[:10])
        raise KeyError(f"{len(missing_masks)} referenced mask IDs are absent from the COCO index. Examples:\n{examples}")
    empty_segmentations = sorted(
        mask_id for mask_id in referenced_masks if not coco_index[mask_id].get("segmentation")
    )
    if empty_segmentations:
        examples = "\n".join(empty_segmentations[:10])
        raise ValueError(
            f"{len(empty_segmentations)} referenced COCO masks have empty segmentation fields. Examples:\n{examples}"
        )
    print(
        f"[preflight] coco_masks={len(coco_index)} referenced_masks={len(referenced_masks)} "
        f"mask_cache_size={args.score_mask_cache_size}",
        flush=True,
    )
    metric_rows = []
    total = len(pred_rows)
    started_at = time.perf_counter()
    score_errors = 0
    if not args.no_score_progress:
        print(f"[score] starting metric computation for {total} prediction rows", flush=True)
    for index, row in enumerate(pred_rows, 1):
        metric = {k: "" for k in METRIC_FIELDS}
        metric.update({
            "eval_id": row.get("eval_id"),
            "dataset": row.get("dataset"),
            "task": row.get("task"),
            "sample_id": row.get("sample_id"),
            "image_id": row.get("image_id"),
            "split": row.get("split"),
            "model_backend": row.get("model_backend"),
            "model_name": row.get("model_name"),
            "format_ok": False,
            "error": row.get("error", ""),
        })
        if row.get("error"):
            metric_rows.append(metric)
            score_errors += 1
            if not args.no_score_progress:
                show_score_progress(index, total, row, score_errors, started_at,
                                    force=index == total, every=args.score_progress_every)
            continue
        try:
            record = json.loads(row["reference_json"])
            pred = json.loads(row["prediction_json"])
            metric = score_prediction(
                record,
                pred,
                coco_index,
                mask_cache,
                union_cache,
                args.score_mask_cache_size,
            )
            metric.update({
                "eval_id": row.get("eval_id"),
                "dataset": row.get("dataset") or record.get("dataset") or record.get("source"),
                "model_backend": row.get("model_backend"),
                "model_name": row.get("model_name"),
            })
        except Exception as exc:
            metric["error"] = str(exc)
            score_errors += 1
        metric_rows.append(metric)
        if not args.no_score_progress:
            show_score_progress(index, total, row, score_errors, started_at,
                                force=index == total, every=args.score_progress_every)
    write_csv_rows(args.metrics_csv, metric_rows, METRIC_FIELDS)
    return metric_rows


def append_metric_section(lines, title, rows):
    lines.append(title)
    lines.append(f"- records: {len(rows)}")
    lines.append(f"- unique_images: {len({r.get('image_id') for r in rows})}")
    errors = sum(1 for r in rows if r.get("error"))
    lines.append(f"- errors: {errors}")
    lines.append(f"- format_success_rate: {statistics.mean(float(bool(r.get('format_ok'))) for r in rows):.4f}")
    excluded = {"eval_id", "dataset", "task", "sample_id", "image_id", "split", "model_backend", "model_name", "error"}
    for key in [field for field in METRIC_FIELDS if field not in excluded]:
        values_by_image = defaultdict(list)
        for row in rows:
            value = row.get(key, "")
            if value == "":
                continue
            try:
                values_by_image[row.get("image_id")].append(float(value))
            except (TypeError, ValueError):
                pass
        image_values = [statistics.mean(values) for values in values_by_image.values() if values]
        if image_values:
            lines.append(f"- {key}_image_macro: {statistics.mean(image_values):.4f} (n={len(image_values)})")
    lines.append("")


def summarize_metrics(metric_rows, summary_path, organ):
    by_task = defaultdict(list)
    for row in metric_rows:
        by_task[row["task"]].append(row)
    lines = [f"# {organ} Full Multimodal Evaluation", "", "# Overall", ""]
    for task, rows in by_task.items():
        append_metric_section(lines, f"## {task}", rows)
    lines.extend(["# Per Dataset", ""])
    datasets = sorted({row.get("dataset") or "unknown" for row in metric_rows})
    for dataset in datasets:
        lines.append(f"## {dataset}")
        lines.append("")
        dataset_rows = [row for row in metric_rows if (row.get("dataset") or "unknown") == dataset]
        for task in sorted({row["task"] for row in dataset_rows}):
            append_metric_section(lines, f"### {task}", [row for row in dataset_rows if row["task"] == task])
    Path(summary_path).write_text("\n".join(lines), encoding="utf-8")


def available_organ_directories(grounded_us_root):
    grounded_us_root = Path(grounded_us_root)
    if not grounded_us_root.is_dir():
        return []
    return sorted(
        path
        for path in grounded_us_root.iterdir()
        if path.is_dir() and any(path.glob("*.jsonl"))
    )


def resolve_named_directory(parent, requested, candidates=None):
    parent = Path(parent)
    candidates = list(candidates) if candidates is not None else [
        path for path in parent.iterdir() if path.is_dir()
    ]
    wanted = normalize_organ_name(requested)
    matches = [path for path in candidates if normalize_organ_name(path.name) == wanted]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        available = ", ".join(path.name for path in candidates)
        raise ValueError(f"Unknown organ {requested!r}. Available organs: {available}")
    raise ValueError(
        f"Ambiguous organ {requested!r}: {', '.join(path.name for path in matches)}"
    )


def resolve_server_paths(args, parser):
    server_root = Path(args.server_root)
    grounded_us_root = (
        Path(args.grounded_us_root)
        if args.grounded_us_root
        else server_root / "Grounded-US"
    )
    organ_dirs = available_organ_directories(grounded_us_root)
    if args.list_organs:
        if not organ_dirs:
            parser.error(f"No organ JSONL directories found under {grounded_us_root}")
        for organ_dir in organ_dirs:
            jsonls = ", ".join(path.name for path in sorted(organ_dir.glob("*.jsonl")))
            print(f"{organ_dir.name}\t{jsonls}")
        raise SystemExit(0)

    explicit_input = Path(args.input) if args.input else None
    if args.grounded_root:
        organ_root = Path(args.grounded_root)
        if not args.organ:
            args.organ = organ_root.name
    elif args.organ:
        try:
            organ_root = resolve_named_directory(
                grounded_us_root, args.organ, organ_dirs
            )
        except ValueError as exc:
            parser.error(str(exc))
        args.organ = organ_root.name
    elif explicit_input is not None:
        inferred_name = explicit_input.parent.name
        try:
            organ_root = resolve_named_directory(
                grounded_us_root, inferred_name, organ_dirs
            )
        except ValueError:
            available = ", ".join(path.name for path in organ_dirs)
            parser.error(
                f"Could not infer organ from input parent {inferred_name!r}. "
                f"Pass --organ. Available organs: {available}"
            )
        args.organ = organ_root.name
    else:
        available = ", ".join(path.name for path in organ_dirs)
        parser.error(
            "Pass --organ or --input. "
            f"Available organs with one or more JSONLs: {available}"
        )

    if explicit_input is not None:
        input_path = explicit_input
    else:
        candidates = sorted(organ_root.glob(args.annotation_glob))
        if not candidates:
            parser.error(
                f"No JSONL matching {args.annotation_glob!r} under {organ_root}; "
                "pass --input"
            )
        if len(candidates) > 1:
            names = ", ".join(path.name for path in candidates)
            parser.error(
                f"Multiple annotation JSONLs found ({names}); pass --input explicitly"
            )
        input_path = candidates[0]

    if args.data_root:
        data_root = Path(args.data_root)
    else:
        datasets_root = (
            Path(args.datasets_root)
            if args.datasets_root
            else server_root / "datasets"
        )
        dataset_organ_dirs = (
            [path for path in datasets_root.iterdir() if path.is_dir()]
            if datasets_root.is_dir()
            else []
        )
        try:
            dataset_organ_root = resolve_named_directory(
                datasets_root, args.organ, dataset_organ_dirs
            )
        except ValueError as exc:
            parser.error(str(exc))
        data_root = dataset_organ_root / "Datasets"

    args.input = str(input_path)
    args.data_root = str(data_root)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else organ_root / "evaluation" / "us_sam3_agent"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = slug(args.organ).lower() + "_us_sam3_agent_full_eval"
    args.predictions_csv = args.predictions_csv or str(output_dir / f"{prefix}_predictions.csv")
    args.metrics_csv = args.metrics_csv or str(output_dir / f"{prefix}_metrics.csv")
    args.summary = args.summary or str(output_dir / f"{prefix}_summary.md")
    if not Path(args.data_root).is_dir():
        parser.error(f"Dataset root does not exist: {args.data_root}")
    if not Path(args.input).is_file():
        parser.error(f"Annotation JSONL does not exist: {args.input}")
    if not args.score_only:
        for value, label in (
            (args.agent_script, "agent inference script"),
            (args.sam3_code_dir, "SAM3 code directory"),
            (args.config_path, "SAM3 config"),
            (args.checkpoint_path, "US-SAM3 checkpoint"),
        ):
            if not Path(value).expanduser().exists():
                parser.error(f"{label} does not exist: {value}")
        if not args.no_api and not args.api_key:
            print(
                "[WARN] No API key was provided; the agent will use local fallback rules.",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(
        description="US-SAM3 agent Grounded-US multi-organ multitask evaluator."
    )
    parser.add_argument(
        "--organ",
        default=None,
        help=(
            "Organ name, matched case-insensitively and ignoring punctuation. "
            "May be omitted when --input is directly inside its Grounded-US organ folder."
        ),
    )
    parser.add_argument("--server-root", default="/home/Data2/zhuquanhao")
    parser.add_argument(
        "--grounded-us-root",
        default=None,
        help="Grounded-US parent directory; defaults to <server-root>/Grounded-US.",
    )
    parser.add_argument(
        "--datasets-root",
        default=None,
        help="Datasets parent directory; defaults to <server-root>/datasets.",
    )
    parser.add_argument(
        "--list-organs",
        action="store_true",
        help="List discovered organ folders and JSONLs, then exit.",
    )
    parser.add_argument("--input", default=None, help="Annotation JSONL; auto-discovered under Grounded-US/<organ> by default.")
    parser.add_argument("--annotation-glob", default="*.jsonl")
    parser.add_argument("--data-root", default=None, help="COCO dataset parent; defaults to datasets/<organ>/Datasets.")
    parser.add_argument("--grounded-root", default=None, help="Explicit directory for one organ under Grounded-US.")
    parser.add_argument("--annotation-name", default="_annotations.coco.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--agent-script", default=DEFAULT_AGENT_SCRIPT)
    parser.add_argument("--sam3-code-dir", default=DEFAULT_SAM3_CODE_DIR)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--categories", default=None)
    parser.add_argument("--categories-file", default=None)
    parser.add_argument("--max-targets", type=int, default=8)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY") or "token-abc123",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get(
            "OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"
        ),
    )
    parser.add_argument(
        "--api-model", default=os.environ.get("OPENAI_MODEL", "/home/Data2/zhuquanhao/Grounded-US/Models/InternVL3_5-8B")
    )
    parser.add_argument("--api-timeout", type=int, default=180)
    parser.add_argument("--api-max-tokens", type=int, default=512)
    parser.add_argument("--api-response-max-tokens", type=int, default=1024)
    parser.add_argument("--api-image-max-side", type=int, default=768)
    parser.add_argument("--api-temperature", type=float, default=0.0)
    parser.add_argument(
        "--vqa-generation-mode",
        choices=["hybrid", "deterministic", "api"],
        default="hybrid",
        help=(
            "Use API answers with mask-geometry constraints (default), "
            "geometry-only answers, or unconstrained API answers."
        ),
    )
    parser.add_argument(
        "--report-generation-mode",
        choices=["hybrid", "deterministic", "api"],
        default="hybrid",
        help=(
            "Use an API report writer constrained and validated by predicted-mask "
            "facts (default), geometry templates, or an unconstrained API composer."
        ),
    )
    parser.add_argument(
        "--report-validation-retries",
        type=int,
        default=1,
        help="Number of API correction attempts after a hybrid report fails validation.",
    )
    parser.add_argument(
        "--api-fallback-on-error", action="store_true", default=True
    )
    parser.add_argument(
        "--no-api-fallback-on-error",
        dest="api_fallback_on_error",
        action="store_false",
    )
    parser.add_argument("--no-api", action="store_true")
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--max-area-ratio", type=float, default=1.0)
    parser.add_argument("--remove-overlap", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Must be 1 because one US-SAM3 GPU model is shared.",
    )
    parser.add_argument("--stop-after-consecutive-errors", type=int, default=0, help="Stop prediction after this many consecutive errors. Default: 0 (disabled).")
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--max-per-task", type=int, default=None)
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="Annotation split to evaluate. Default: test.",
    )
    parser.add_argument("--predictions-csv", default=None)
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--score-only", action="store_true", help="Only compute metrics from an existing predictions CSV.")
    parser.add_argument("--no-score-progress", action="store_true", help="Disable metric-scoring progress output.")
    parser.add_argument("--score-progress-every", type=int, default=50, help="When stdout is redirected, print scoring progress every N rows.")
    parser.add_argument(
        "--score-mask-cache-size",
        type=int,
        default=64,
        help="Maximum decoded GT masks and mask unions cached during scoring; use 0 to disable.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip sample_ids already present in predictions CSV.")
    parser.add_argument("--retry-errors", action="store_true", help="With --resume, remove failed rows from predictions CSV and rerun them.")
    args = parser.parse_args()
    resolve_server_paths(args, parser)

    if not args.score_only:
        run_predictions(args)
    metric_rows = score_predictions(args)
    summarize_metrics(metric_rows, args.summary, args.organ)
    print(json.dumps({"organ": args.organ, "model": "US-SAM3-Agent", "input": args.input, "data_root": args.data_root,
                      "predictions_csv": args.predictions_csv, "metrics_csv": args.metrics_csv,
                      "summary": args.summary, "rows": len(metric_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
