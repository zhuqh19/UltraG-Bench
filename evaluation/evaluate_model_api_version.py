import argparse
import base64
import concurrent.futures
import csv
import io
import json
import math
import os
import random
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path


DEFAULT_API_KEY = "token-abc123"
DEFAULT_BASE_URL = "http://183.62.69.210:9002/v1"
DEFAULT_MODEL = "/data2/ZhuQuanhao/models/Lingshu-32B"

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
    "irregular",
    "oval",
    "round",
    "circumscribed",
    "well-defined",
    "ill-defined",
    "spiculated",
    "angular",
    "parallel",
    "non-parallel",
    "posterior shadowing",
    "posterior enhancement",
    "vascular",
    "vascularity",
    "suspicious",
    "malignant-appearing",
    "benign-appearing",
    "consistent with",
    "bi-rads",
    "birads",
    "tissue diagnosis",
    "recommended",
    "recommend",
    "cm",
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
    "semantic_accuracy",
    "unsupported_descriptor_count",
    "banned_term_hits",
]


def encode_image_data_url(path):
    path = Path(path)
    suffix = path.suffix.lower()
    direct_mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    if suffix in direct_mime:
        mime = direct_mime[suffix]
        image_bytes = path.read_bytes()
    else:
        try:
            from PIL import Image, ImageOps
        except ImportError:
            Image = ImageOps = None
        if Image is not None:
            try:
                with Image.open(path) as image:
                    image.seek(0)
                    image = ImageOps.exif_transpose(image)
                    if image.mode not in {"L", "RGB", "RGBA"}:
                        image = image.convert("RGB")
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    image_bytes = buffer.getvalue()
            except Exception as exc:
                raise RuntimeError(f"Could not decode/convert image {path} (suffix={suffix!r}): {exc}") from exc
        else:
            try:
                import cv2
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError("cv2.imread returned None")
                ok, encoded = cv2.imencode(".png", image)
                if not ok:
                    raise ValueError("cv2.imencode returned false")
                image_bytes = encoded.tobytes()
            except ImportError as exc:
                raise RuntimeError(
                    f"Converting {suffix or 'extensionless'} images requires Pillow or OpenCV. "
                    "Install one with: pip install pillow"
                ) from exc
            except Exception as exc:
                raise RuntimeError(f"Could not decode/convert image {path} (suffix={suffix!r}): {exc}") from exc
        mime = "image/png"
    data = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{data}"


class OpenAICompatibleVisionClient:
    def __init__(self, api_key, base_url, model, timeout=180, retries=2,
                 retry_base_sleep=2, retry_max_sleep=60):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for the openai_compatible backend. "
                "Install it with: pip install -U openai"
            ) from exc
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.retry_base_sleep = retry_base_sleep
        self.retry_max_sleep = retry_max_sleep
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )

    def retry_delay(self, exc, attempt):
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after:
                try:
                    return min(self.retry_max_sleep, max(self.retry_base_sleep, float(retry_after)))
                except (TypeError, ValueError):
                    pass
        exponential = min(self.retry_max_sleep, self.retry_base_sleep * (2 ** attempt))
        return exponential + random.uniform(0, min(1.0, exponential * 0.1))

    def predict(self, image_path, prompt):
        image_data_url = encode_image_data_url(image_path)
        last_error = "unknown API error"
        for attempt in range(self.retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_data_url}},
                            ],
                        }
                    ],
                    temperature=0,
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("API returned empty message content")
                return content
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                last_error = f"{type(exc).__name__}: {exc}"
                retryable = status_code is None or status_code in {408, 409, 429} or status_code >= 500
                if attempt >= self.retries or not retryable:
                    break
                delay = self.retry_delay(exc, attempt)
                print(
                    f"[api retry {attempt + 1}/{self.retries}] {last_error[:500]}; "
                    f"waiting {delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError(f"OpenAI-compatible API call failed: {last_error}")


class CommandVisionClient:
    def __init__(self, command_template, model_name="command_model", timeout=300):
        self.command_template = command_template
        self.model = model_name
        self.timeout = timeout

    def predict(self, image_path, prompt):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as pf:
            pf.write(prompt)
            prompt_path = pf.name
        try:
            command = self.command_template.format(
                image_path=str(image_path),
                prompt_path=prompt_path,
                prompt=shlex.quote(prompt),
            )
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or f"Command failed with code {completed.returncode}")
            return completed.stdout.strip()
        finally:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass


def build_client(args):
    if args.backend == "command":
        if not args.command_template:
            raise ValueError("--command-template is required when --backend command")
        return CommandVisionClient(args.command_template, args.model or "command_model", args.timeout)
    if args.backend == "openai_compatible":
        api_key = args.api_key
        base_url = args.base_url
        model = args.model
        if not api_key or not base_url or not model:
            raise ValueError(
                "Missing api_key/base_url/model. Set DEFAULT_API_KEY, DEFAULT_BASE_URL, "
                "and DEFAULT_MODEL in this script, or pass --api-key/--base-url/--model."
            )
        return OpenAICompatibleVisionClient(
            api_key, base_url, model, args.timeout, args.retries,
            args.retry_base_sleep, args.retry_max_sleep,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


def slug(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_") or "dataset"


def normalize_organ_name(value):
    return slug(value).lower()


def organ_display_name(record):
    image_organ = record.get("image", {}).get("organ")
    return str(image_organ or record.get("organ") or "target organ").replace("_", " ")


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


def load_full_records(
    path,
    data_root,
    organ,
    limit_images=None,
    max_per_task=None,
    splits=("test",),
):
    tasks = []
    image_count = 0
    selected_splits = set(splits)
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("split") not in selected_splits:
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
                    f"Generate a structured {str(organ).replace('_', ' ')} ultrasound report.",
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
    splits=("test",),
):
    index = {}
    data_root = Path(data_root)
    for dataset_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        dataset_id = slug(dataset_dir.name)
        for split in splits:
            split_dir = dataset_dir / split
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
                mask_id = f"{dataset_id}_{split}_img{img['id']}_ann{ann['id']}"
                index[mask_id] = {
                    "mask_id": mask_id,
                    "dataset": dataset_dir.name,
                    "split": split,
                    "image_id": f"{dataset_id}_{split}_{img['id']}",
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


def bbox_pixels(bbox, width, height):
    import numpy as np
    pixels = np.zeros((height, width), dtype=bool)
    if not bbox or len(bbox) != 4:
        return pixels
    x, y, w, h = [float(v) for v in bbox]
    x0 = max(0, min(width, math.floor(x)))
    y0 = max(0, min(height, math.floor(y)))
    x1 = max(0, min(width, math.ceil(x + w)))
    y1 = max(0, min(height, math.ceil(y + h)))
    if x1 > x0 and y1 > y0:
        pixels[y0:y1, x0:x1] = True
    return pixels


def point_in_poly(x, y, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def polygon_pixels(points, width, height):
    import numpy as np
    pixels = np.zeros((height, width), dtype=bool)
    if not points or len(points) < 3:
        return pixels
    pts = [(float(x), float(y)) for x, y in points]
    min_x = max(0, math.floor(min(x for x, _ in pts)))
    max_x = min(width, math.ceil(max(x for x, _ in pts)))
    min_y = max(0, math.floor(min(y for _, y in pts)))
    max_y = min(height, math.ceil(max(y for _, y in pts)))
    if max_x <= min_x or max_y <= min_y:
        return pixels
    # Preserve the original ray-crossing rule at pixel centers.
    xs = np.arange(min_x, max_x, dtype=float) + 0.5
    for y0 in range(min_y, max_y, 128):
        y1 = min(y0 + 128, max_y)
        ys = np.arange(y0, y1, dtype=float) + 0.5
        block = pixels[y0:y1, min_x:max_x]
        xj, yj = pts[-1]
        for xi, yi in pts:
            rows = np.flatnonzero((yi > ys) != (yj > ys))
            if rows.size:
                crossings = (xj - xi) * (ys[rows] - yi) / ((yj - yi) or 1e-9) + xi
                block[rows] ^= xs[None, :] < crossings[:, None]
            xj, yj = xi, yi
    return pixels


def coco_seg_pixels(segmentation, width, height):
    import numpy as np
    if isinstance(segmentation, dict):
        try:
            import numpy as np
            from pycocotools import mask as mask_utils
        except ImportError as exc:
            raise RuntimeError("RLE masks require numpy and pycocotools: pip install numpy pycocotools") from exc
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        if decoded.shape != (height, width):
            raise ValueError("RLE dimensions differ from image dimensions")
        return decoded.astype(bool)
    pixels = np.zeros((height, width), dtype=bool)
    for poly in segmentation or []:
        if not poly or len(poly) < 6:
            continue
        pixels |= polygon_pixels(list(zip(poly[0::2], poly[1::2])), width, height)
    return pixels


def is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def point_xy(value):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x, y = value[0], value[1]
    elif isinstance(value, dict):
        x, y = value.get("x"), value.get("y")
    else:
        return None
    if not is_finite_number(x) or not is_finite_number(y):
        return None
    return [float(x), float(y)]


def normalize_points(value):
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if all(is_finite_number(v) for v in value):
        if len(value) < 6 or len(value) % 2:
            return None
        return [[float(value[i]), float(value[i + 1])] for i in range(0, len(value), 2)]
    points = [point_xy(point) for point in value]
    return points if all(point is not None for point in points) else None


def looks_like_points(value):
    points = normalize_points(value)
    return points is not None and len(points) >= 3


def xyxy_to_xywh(values):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    if not all(is_finite_number(value) for value in values):
        return None
    x0, y0, x1, y1 = [float(value) for value in values]
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    return [left, top, right - left, bottom - top]


def corners_to_xywh(values):
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return None
    points = [point_xy(value) for value in values]
    if any(point is None for point in points):
        return None
    return xyxy_to_xywh([points[0][0], points[0][1], points[1][0], points[1][1]])


def corner_box_to_xywh(value):
    while (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    return corners_to_xywh(value) or xyxy_to_xywh(value)


def dict_bbox_xywh(pred_mask):
    value = pred_mask.get("bbox_xywh")
    if isinstance(value, (list, tuple)) and len(value) == 4 and all(is_finite_number(v) for v in value):
        return [float(v) for v in value]

    # Keep the evaluator's original interpretation of a plain "bbox" as xywh.
    value = pred_mask.get("bbox")
    if isinstance(value, (list, tuple)) and len(value) == 4 and all(is_finite_number(v) for v in value):
        return [float(v) for v in value]
    bbox = corner_box_to_xywh(value)
    if bbox is not None:
        return bbox

    for width_key, height_key in (("width", "height"), ("w", "h")):
        values = [
            pred_mask.get("x"),
            pred_mask.get("y"),
            pred_mask.get(width_key),
            pred_mask.get(height_key),
        ]
        if all(is_finite_number(v) for v in values):
            return [float(v) for v in values]

    for keys in (
        ("x_min", "y_min", "x_max", "y_max"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("x0", "y0", "x1", "y1"),
        ("x1", "y1", "x2", "y2"),
        ("left", "top", "right", "bottom"),
    ):
        values = [pred_mask.get(key) for key in keys]
        if all(is_finite_number(v) for v in values):
            return xyxy_to_xywh(values)

    bbox = corner_box_to_xywh(pred_mask.get("points"))
    if bbox is not None:
        return bbox

    # "coordinates", "coords", and "box" are treated as two-corner/xyxy forms.
    for key in ("coordinates", "coords", "box"):
        bbox = corner_box_to_xywh(pred_mask.get(key))
        if bbox is not None:
            return bbox
    return None


def is_plain_xy_point(value):
    return (
        isinstance(value, dict)
        and point_xy(value) is not None
        and not any(
            key in value
            for key in (
                "type", "bbox", "bbox_xywh", "box", "coordinates", "coords",
                "w", "h", "width", "height", "x0", "y0", "x1", "y1",
                "x2", "y2", "x_min", "y_min", "x_max", "y_max",
            )
        )
    )


def normalize_pred_mask(pred_mask):
    if pred_mask is None or pred_mask == "":
        return None

    if isinstance(pred_mask, dict):
        mask_type = str(pred_mask.get("type", "")).strip().lower()
        if mask_type in {"null", "none", "empty"}:
            return None

        if mask_type in {"multi", "multiple"} or isinstance(pred_mask.get("masks"), list):
            masks = [
                normalized
                for item in pred_mask.get("masks", [])
                if (normalized := normalize_pred_mask(item)) is not None
            ]
            return {"type": "multi", "masks": masks} if masks else None

        polygon_value = None
        if mask_type in {"polygon", "poly"}:
            polygon_value = (
                pred_mask.get("points")
                or pred_mask.get("polygon")
                or pred_mask.get("coordinates")
                or pred_mask.get("coords")
            )
        elif "polygon" in pred_mask:
            polygon_value = pred_mask.get("polygon")
        points = normalize_points(polygon_value)
        if points is not None and len(points) >= 3:
            return {"type": "polygon", "points": points}

        bbox = dict_bbox_xywh(pred_mask)
        if bbox is not None:
            return {"type": "bbox", "bbox_xywh": bbox}

        # Some responses omit type but still return points/coordinates.
        for key in ("points", "coordinates", "coords"):
            points = normalize_points(pred_mask.get(key))
            if points is not None and len(points) >= 3:
                return {"type": "polygon", "points": points}
        return None

    if not isinstance(pred_mask, (list, tuple)) or not pred_mask:
        return None

    if len(pred_mask) == 4 and all(is_finite_number(v) for v in pred_mask):
        return {"type": "bbox", "bbox_xywh": [float(v) for v in pred_mask]}

    if all(is_plain_xy_point(item) for item in pred_mask):
        points = [point_xy(item) for item in pred_mask]
        if len(points) >= 3:
            return {"type": "polygon", "points": points}
        if len(points) == 2:
            return {"type": "bbox", "bbox_xywh": corners_to_xywh(points)}
        return None

    points = normalize_points(pred_mask)
    if points is not None:
        if len(points) >= 3:
            return {"type": "polygon", "points": points}
        if len(points) == 2:
            return {"type": "bbox", "bbox_xywh": corners_to_xywh(points)}

    masks = [
        normalized
        for item in pred_mask
        if (normalized := normalize_pred_mask(item)) is not None
    ]
    return {"type": "multi", "masks": masks} if masks else None


def pred_mask_pixels(pred_mask, width, height):
    import numpy as np
    pred_mask = normalize_pred_mask(pred_mask)
    if not pred_mask:
        return np.zeros((height, width), dtype=bool)
    mtype = str(pred_mask.get("type", "")).lower()
    if mtype == "multi":
        pixels = np.zeros((height, width), dtype=bool)
        for item in pred_mask.get("masks", []):
            pixels |= pred_mask_pixels(item, width, height)
        return pixels
    if mtype == "bbox":
        return bbox_pixels(pred_mask.get("bbox_xywh"), width, height)
    if mtype == "polygon":
        return polygon_pixels(pred_mask.get("points") or [], width, height)
    return np.zeros((height, width), dtype=bool)


def gt_pixels(mask_ids, coco_index, record, mask_cache=None):
    import numpy as np
    width = record["image"]["width"]
    height = record["image"]["height"]
    pixels = np.zeros((height, width), dtype=bool)
    for mask_id in mask_ids or []:
        item = coco_index.get(mask_id)
        if item:
            width, height = item["width"], item["height"]
            if pixels.shape != (height, width):
                raise ValueError("COCO mask dimensions differ from reference image")
            single = mask_cache.get(mask_id) if mask_cache is not None else None
            if single is None:
                single = coco_seg_pixels(item.get("segmentation", []), width, height)
                if mask_cache is not None:
                    mask_cache[mask_id] = single
                    # Bound decoded cache memory and entry count.
                    while len(mask_cache) > 64 or sum(v.nbytes for v in mask_cache.values()) > 128 * 1024**2:
                        mask_cache.popitem(last=False)
            elif mask_cache is not None:
                mask_cache.move_to_end(mask_id)
            pixels |= single
    return pixels, width, height


def iou_dice(pred_pixels, ref_pixels):
    import numpy as np
    pred_area = int(np.count_nonzero(pred_pixels))
    ref_area = int(np.count_nonzero(ref_pixels))
    if not pred_area and not ref_area:
        return 1.0, 1.0
    if not pred_area or not ref_area:
        return 0.0, 0.0
    inter = int(np.count_nonzero(pred_pixels & ref_pixels))
    union = pred_area + ref_area - inter
    return inter / union if union else 0.0, 2 * inter / (pred_area + ref_area)


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


def vqa_answer_correct(pred, ref, answer_type):
    pred_norm = normalize_text(pred)
    ref_norm = normalize_text(ref)
    atype = str(answer_type or "").lower()
    if "yes_no" in atype:
        return yes_no(pred) == yes_no(ref)
    if "location" in atype:
        pred_locs = extract_grid_locations(pred)
        ref_locs = extract_grid_locations(ref)
        return bool(ref_locs) and pred_locs == ref_locs
    if "classification" in atype or "category" in atype or "open" in atype:
        if "malignant" in ref_norm or "benign" in ref_norm:
            return ("malignant" in ref_norm and "malignant" in pred_norm) or ("benign" in ref_norm and "benign" in pred_norm)
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
    pred_categories = mentioned_categories(pred_text, record)
    return ref_categories.issubset(pred_categories)


def target_presence_value(text, record):
    norm = normalize_text(text)
    negative_patterns = (
        r"\bappears normal\b",
        r"\bno\s+(?:(?:discrete|definite|focal|visible|labeled|labelled|supplied)\s+){0,3}"
        r"(?:[a-z0-9]+\s+){0,4}(?:lesion|mass|tumor|abnormality|nodule|target|structure|organ)\b",
        r"\b(?:target|structure|organ|lesion|mass|tumor|nodule)\s+(?:is\s+)?(?:not visible|absent|unavailable)\b",
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
    return (
        "You are given a clinical ultrasound image and a segmentation instruction.\n"
        "Return valid JSON only with keys: answer, predicted_mask.\n"
        "predicted_mask must be {\"type\":\"polygon\",\"points\":[[x,y],...]}, {\"type\":\"bbox\",\"bbox_xywh\":[x,y,w,h]}, or null.\n"
        "Use original-image pixel coordinates. Include [SEG] in answer if a target is present.\n"
        "Do not use annotation, annotated, dataset, mask id, or metadata.\n\n"
        + json.dumps({"image_width": record["image"]["width"], "image_height": record["image"]["height"], "instruction": record["instruction"]}, ensure_ascii=False)
    )


def vqa_prompt(record):
    return (
        "You are given a clinical ultrasound image and a question.\n"
        "Return valid JSON only with keys: answer, predicted_mask.\n"
        "If the answer depends on a visible target, organ, or structure, provide predicted_mask as polygon or bbox in original-image pixel coordinates; otherwise use null.\n"
        "Do not use annotation, annotated, dataset, mask id, or metadata.\n\n"
        + json.dumps({
            "organ": organ_display_name(record),
            "image_width": record["image"]["width"],
            "image_height": record["image"]["height"],
            "question": record["question"],
        }, ensure_ascii=False)
    )


def report_prompt(record):
    structure = sorted({
        sentence.get("finding_type") for sentence in record.get("report", [])
        if isinstance(sentence, dict) and sentence.get("finding_type")
    })
    spatial_types = sorted(reference_spatial_finding_types(record.get("report", [])))
    spatial_instruction = (
        "For these spatial finding_type values, predicted_mask must localize the supporting "
        "target, organ, or structure as a polygon or bbox in original-image pixel coordinates: "
        + ", ".join(spatial_types)
        + ".\n"
        if spatial_types
        else ""
    )
    return (
        "You are given a clinical ultrasound image.\n"
        "Return valid JSON only with key report.\n"
        "report must be a list of sentence objects with section, text, finding_type, predicted_mask.\n"
        f"{spatial_instruction}"
        "Use null predicted_mask for non-spatial statements.\n"
        f"Use this finding_type structure when applicable: {', '.join(structure)}.\n"
        "Use concise benchmark-style wording. Do not invent ultrasound descriptors if uncertain.\n"
        "Do not invent diagnostic grades, pathology classes, or treatment recommendations, and do not use cm unless a scale marker is visible.\n"
        "If estimating size, use pixel coordinates or pixel extent only.\n"
        "Do not use annotation, annotated, dataset, mask id, or metadata.\n\n"
        + json.dumps({
            "organ": organ_display_name(record),
            "image_width": record["image"]["width"],
            "image_height": record["image"]["height"],
            "prompt": record.get("prompt", ""),
        }, ensure_ascii=False)
    )


def prompt_for(record):
    if record["task"] == "grounded_segmentation":
        return segmentation_prompt(record)
    if record["task"] == "grounded_vqa":
        return vqa_prompt(record)
    if record["task"] == "grounded_report_generation":
        return report_prompt(record)
    raise ValueError(record["task"])


def score_prediction(record, pred, coco_index, mask_cache=None):
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
        ref_pixels, width, height = gt_pixels(record.get("target_mask_ids", []), coco_index, record, mask_cache)
        pred_pixels = pred_mask_pixels(pred.get("predicted_mask"), width, height)
        iou, dice = iou_dice(pred_pixels, ref_pixels)
        answer = str(pred.get("answer", ""))
        metrics.update({
            "iou": iou,
            "dice": dice,
            "empty_prediction": int(not pred_pixels.any()),
            "seg_token_ok": int(("[SEG]" in answer) == bool(ref_pixels.any())),
            "banned_term_hits": len(banned_hits(answer)),
        })
    elif record["task"] == "grounded_vqa":
        ref_pixels, width, height = gt_pixels(record.get("evidence_mask_ids", []), coco_index, record, mask_cache)
        pred_pixels = pred_mask_pixels(pred.get("predicted_mask"), width, height)
        iou, dice = iou_dice(pred_pixels, ref_pixels)
        answer = str(pred.get("answer", ""))
        correct = vqa_answer_correct(answer, record.get("answer", ""), record.get("answer_type", ""))
        requires_grounding = record.get("requires_grounding") is True
        metrics.update({
            "iou": iou if requires_grounding else "",
            "dice": dice if requires_grounding else "",
            "empty_prediction": int(not pred_pixels.any()),
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
        pred_types = {s.get("finding_type") for s in pred_report if isinstance(s, dict) and s.get("finding_type")}
        ref_types = {s.get("finding_type") for s in ref_report if isinstance(s, dict) and s.get("finding_type")}
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
        semantic_values = [value for value in (presence_ok, category_ok, location_ok, size_ok) if value is not None]

        spatial_types = reference_spatial_finding_types(ref_report)
        pred_by_type = {
            sentence.get("finding_type"): sentence
            for sentence in pred_report if isinstance(sentence, dict) and sentence.get("finding_type")
        }
        grounding_scores = []
        for ref_sentence in ref_report:
            if not isinstance(ref_sentence, dict) or ref_sentence.get("finding_type") not in spatial_types:
                continue
            evidence_ids = ref_sentence.get("evidence_mask_ids", [])
            if not evidence_ids:
                continue
            ref_pixels, width, height = gt_pixels(evidence_ids, coco_index, record, mask_cache)
            pred_sentence = pred_by_type.get(ref_sentence.get("finding_type"), {})
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
        "model_backend": args.backend,
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
        raw = client.predict(record["image_path"], prompt)
        row["prediction_raw"] = raw
        pred = extract_json(raw)
        row["prediction_json"] = json.dumps(pred, ensure_ascii=False)
    except Exception as exc:
        row["error"] = str(exc)
    row["elapsed_sec"] = round(time.time() - t0, 3)
    if args.sleep_between > 0:
        time.sleep(args.sleep_between)
    return row


def pending_prediction_jobs(tasks, done, start_idx):
    jobs = []
    for record in tasks:
        if record["sample_id"] in done:
            continue
        # Stable IDs avoid collisions after threaded runs and retry-error resume.
        jobs.append((record["sample_id"], record))
    return jobs


def check_consecutive_errors(row, consecutive_errors, args):
    if row["error"]:
        consecutive_errors += 1
        if args.stop_after_consecutive_errors and consecutive_errors >= args.stop_after_consecutive_errors:
            raise RuntimeError(
                f"Stopped after {consecutive_errors} consecutive prediction errors. "
                "The API may be temporarily unavailable. Rerun later with --resume --retry-errors."
            )
    else:
        consecutive_errors = 0
    return consecutive_errors


def show_prediction_progress(completed_run, pending_total, initial_done, total, record, row, started_at):
    overall_done = min(total, initial_done + completed_run)
    remaining = max(0, total - overall_done)
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    rate = completed_run / elapsed
    eta = remaining / rate if rate > 0 else math.inf
    error_text = str(row.get("error", "")).replace("\n", " ")
    error_suffix = f" detail={error_text[:240]}" if error_text else ""
    print(
        f"[predict overall={overall_done}/{total} remaining={remaining} | "
        f"this_run={completed_run}/{pending_total} | rate={rate:.2f}/s ETA={format_duration(eta)}] "
        f"dataset={record.get('dataset')} task={record['task']} sample={record['sample_id']} "
        f"error={bool(row['error'])}{error_suffix}",
        flush=True,
    )


def run_predictions_sequential(jobs, total, initial_done, client, args):
    consecutive_errors = 0
    started_at = time.perf_counter()
    pending_total = len(jobs)
    for completed_run, (eval_id, record) in enumerate(jobs, 1):
        row = prediction_row(eval_id, record, client, args)
        append_csv_row(args.predictions_csv, row, PREDICTION_FIELDS)
        show_prediction_progress(completed_run, pending_total, initial_done, total, record, row, started_at)
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
        eval_id, record = jobs[next_job]
        next_job += 1
        future = executor.submit(prediction_row, eval_id, record, client, args)
        active[future] = (eval_id, record)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for _ in range(min(args.workers, len(jobs))):
            submit_next(executor)
        while active:
            done_futures, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done_futures:
                eval_id, record = active.pop(future)
                row = future.result()
                append_csv_row(args.predictions_csv, row, PREDICTION_FIELDS)
                completed_run += 1
                show_prediction_progress(completed_run, pending_total, initial_done, total, record, row, started_at)
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
        args.splits,
    )
    missing_images = sorted({record["image_path"] for record in tasks if not Path(record["image_path"]).is_file()})
    if missing_images:
        examples = "\n".join(missing_images[:10])
        raise FileNotFoundError(f"{len(missing_images)} referenced images were not found. Examples:\n{examples}")
    spatial_types = sorted({
        finding_type
        for record in tasks
        if record["task"] == "grounded_report_generation"
        for finding_type in reference_spatial_finding_types(record.get("report", []))
    })
    print(
        f"[preflight] organ={args.organ} splits={','.join(args.splits)} "
        f"input={args.input} data_root={args.data_root} "
        f"tasks={len(tasks)} images={len({record['image_id'] for record in tasks})} "
        f"datasets={len({record['dataset'] for record in tasks})}",
        flush=True,
    )
    print(f"[preflight] report_spatial_finding_types={spatial_types}", flush=True)
    client = build_client(args)
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
    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError("Fast scoring requires numpy: pip install numpy") from exc
    mask_cache = OrderedDict()
    coco_index = load_coco_index(
        args.data_root,
        args.annotation_name,
        args.splits,
    )
    selected_splits = set(args.splits)
    pred_rows = [
        row
        for row in read_csv_rows(args.predictions_csv)
        if row.get("split") in selected_splits
    ]
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
    print(f"[preflight] coco_masks={len(coco_index)} referenced_masks={len(referenced_masks)}", flush=True)
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
            metric = score_prediction(record, pred, coco_index, mask_cache)
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
        else organ_root / "evaluation" / slug(args.model)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = slug(args.organ).lower() + "_full_eval"
    args.predictions_csv = args.predictions_csv or str(output_dir / f"{prefix}_predictions.csv")
    args.metrics_csv = args.metrics_csv or str(output_dir / f"{prefix}_metrics.csv")
    args.summary = args.summary or str(output_dir / f"{prefix}_summary.md")
    if not Path(args.data_root).is_dir():
        parser.error(f"Dataset root does not exist: {args.data_root}")
    if not Path(args.input).is_file():
        parser.error(f"Annotation JSONL does not exist: {args.input}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one Grounded-US organ with segmentation, grounded VQA, and report metrics."
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
    parser.add_argument("--backend", choices=["openai_compatible", "command"], default="openai_compatible")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--command-template", default=None, help="For local/open-source models. Use {image_path}, {prompt_path}, or {prompt}. Command must print JSON.")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent prediction workers. Use 2-4 for API evaluation.")
    parser.add_argument("--retry-base-sleep", type=float, default=2.0, help="Initial seconds for exponential API retry backoff.")
    parser.add_argument("--retry-max-sleep", type=float, default=60.0, help="Maximum seconds between API retries.")
    parser.add_argument("--sleep-between", type=float, default=0.0, help="Seconds to sleep after each prediction row.")
    parser.add_argument("--stop-after-consecutive-errors", type=int, default=0, help="Stop prediction after this many consecutive errors. Default: 0 (disabled).")
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--max-per-task", type=int, default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "test"],
        default=["train", "test"],
        help=(
            "Annotation splits to evaluate. Default: test only. "
            "Use --splits train test for both."
        ),
    )
    parser.add_argument("--predictions-csv", default=None)
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--score-only", action="store_true", help="Only compute metrics from an existing predictions CSV.")
    parser.add_argument("--no-score-progress", action="store_true", help="Disable metric-scoring progress output.")
    parser.add_argument("--score-progress-every", type=int, default=50, help="When stdout is redirected, print scoring progress every N rows.")
    parser.add_argument("--resume", action="store_true", help="Skip sample_ids already present in predictions CSV.")
    parser.add_argument("--retry-errors", action="store_true", help="With --resume, remove failed rows from predictions CSV and rerun them.")
    args = parser.parse_args()
    resolve_server_paths(args, parser)

    if not args.score_only:
        run_predictions(args)
    metric_rows = score_predictions(args)
    summarize_metrics(metric_rows, args.summary, args.organ)
    print(json.dumps({"organ": args.organ, "input": args.input, "data_root": args.data_root,
                      "predictions_csv": args.predictions_csv, "metrics_csv": args.metrics_csv,
                      "summary": args.summary, "rows": len(metric_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
