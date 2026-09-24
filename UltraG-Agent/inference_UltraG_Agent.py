#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-image ultrasound instruction -> agent-planned US-SAM3 grounded output.

  1) load SAM3-family checkpoint with Hydra config;
  2) classify the instruction as segmentation, grounded VQA, or grounded report;
  3) let the front-end agent select one or more category-grounded SAM3 prompts;
  4) run direct SAM3 inference for every unique target prompt;
  5) compose an answer/report from the image and measured mask geometry;
  6) save per-target masks, overlays, candidates, and one structured result JSON.

Typical usage:
  python inference_agent.py \
    --image /path/to/case002.png \
    --instruction "Please segment the thyroid nodule in this ultrasound image." \
    --api-key $OPENAI_API_KEY
    --api-url your_api_url

If no API key is available, use --no-api to fall back to a simple local prompt
rewrite based on the question text.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import base64
import gc
import json
import re
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import pycocotools.mask as maskUtils


# ========================= Default local paths =========================
# Modify these three defaults if your code/config/checkpoint locations differ.
DEFAULT_SAM3_CODE_DIR = "/home/Data2/zhuquanhao/US-SAM3/US-SAM3/code"
DEFAULT_CONFIG_PATH = "/home/Data2/zhuquanhao/US-SAM3/US-SAM3/config/config.yaml"
DEFAULT_CHECKPOINT_PATH = "/home/Data2/zhuquanhao/US-SAM3/US-SAM3_weight/US-SAM3.pt"
DEFAULT_OUTPUT_ROOT = "/home/Data2/zhuquanhao/US-SAM3/US-SAM3/inference/sam3_agent_single_image_outputs"

# A small built-in ultrasound category pool. You can override with --categories.
DEFAULT_CATEGORY_OPTIONS = [
    "thyroid nodule",
    "thyroid gland",
    "breast lesion",
    "tumor",
    "lesion",
    "liver tumor",
    "kidney",
    "common carotid artery",
    "carotid intima-media region",
    "left ventricular cavity",
    "left atrium",
    "right ventricle",
    "right atrium",
    "myocardium",
    "fetal head",
    "fetal abdomen",
    "prostate",
    "nerve",
    "biceps brachii muscle",
    "gastrocnemius muscle",
    "tibialis anterior muscle",
]

CATEGORY_PROMPT_ALIASES = {
    "BB": "biceps brachii muscle",
    "BB_Healthy": "healthy biceps brachii muscle",
    "BB_Pathological": "pathological biceps brachii muscle",
    "GM": "gastrocnemius muscle",
    "GM_Healthy": "healthy gastrocnemius muscle",
    "GM_Pathological": "pathological gastrocnemius muscle",
    "TA": "tibialis anterior muscle",
    "TA_Healthy": "healthy tibialis anterior muscle",
    "TA_Pathological": "pathological tibialis anterior muscle",
    "CCA": "common carotid artery",
    "CCAUI": "common carotid artery intima-media region",
    "CUBS": "carotid ultrasound boundary structure",
    "TN3K": "thyroid nodule",
    "TG3K": "thyroid gland",
    "LV": "left ventricular cavity",
    "LA": "left atrium",
    "RV": "right ventricle",
    "RA": "right atrium",
    "MYO": "myocardium",
    "LVID": "left ventricular internal diameter",
    "IVS": "interventricular septum",
    "LVPW": "left ventricular posterior wall",
}

TASK_TYPES = (
    "grounded_segmentation",
    "grounded_vqa",
    "grounded_report_generation",
)

OVERLAY_COLORS = [
    (255, 64, 64),
    (32, 180, 255),
    (255, 196, 32),
    (64, 210, 128),
    (196, 96, 255),
    (255, 128, 32),
]


# ========================= Utility functions =========================
def clean_category_prompt(category_name: str) -> str:
    if category_name in CATEGORY_PROMPT_ALIASES:
        return CATEGORY_PROMPT_ALIASES[category_name]
    s = str(category_name).strip()
    s = CATEGORY_PROMPT_ALIASES.get(s, s)
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s or "object"


def parse_categories(categories_arg: Optional[str], categories_file: Optional[str]) -> List[Dict[str, Any]]:
    """Return a fixed list of valid target categories for the front agent."""
    raw: List[str] = []
    if categories_file:
        p = Path(categories_file).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"categories file not found: {p}")
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json":
            obj = json.loads(text)
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, str):
                        raw.append(item)
                    elif isinstance(item, dict):
                        raw.append(str(item.get("meaning") or item.get("name") or item.get("category") or ""))
        else:
            raw.extend([x.strip() for x in re.split(r"[,\n]", text) if x.strip()])
    if categories_arg:
        raw.extend([x.strip() for x in categories_arg.split(",") if x.strip()])
    if not raw:
        raw = list(DEFAULT_CATEGORY_OPTIONS)

    # De-duplicate by cleaned meaning, keep a stable option_id.
    seen = set()
    out = []
    for name in raw:
        meaning = clean_category_prompt(name)
        key = meaning.lower()
        if not meaning or key in seen:
            continue
        seen.add(key)
        out.append({
            "option_id": str(len(out) + 1),
            "name": name,
            "meaning": meaning,
        })
    return out


def load_image_as_rgb(image_path: str) -> Image.Image:
    with Image.open(image_path) as im:
        im.load()
        return im.convert("RGB")


def pil_to_data_url(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    buf = BytesIO()
    img = img.convert("RGB")
    img.save(buf, format=fmt, quality=quality, optimize=True)
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def image_path_to_data_url(image_path: str, max_side: int = 768) -> str:
    with Image.open(image_path) as im:
        im.load()
        im = im.convert("RGB")
        if max_side and max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.BILINEAR)
        return pil_to_data_url(im, fmt="JPEG", quality=90)


def extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def infer_task_type(instruction: str, forced_task: str = "auto") -> str:
    if forced_task and forced_task != "auto":
        return forced_task
    text = str(instruction or "").strip().lower()
    if (
        re.search(r"\b(report|findings|impression|description)\b", text)
        or text.startswith(("describe ", "summarize "))
    ):
        return "grounded_report_generation"
    if re.search(r"\b(segment|delineate|outline|mask|contour)\b", text):
        return "grounded_segmentation"
    question_starts = (
        "is ", "are ", "does ", "do ", "can ", "what ", "which ", "where ",
        "how ", "whether ", "has ", "have ",
    )
    if text.endswith("?") or text.startswith(question_starts):
        return "grounded_vqa"
    return "grounded_segmentation"


def normalize_task_type(value: Any, fallback: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "segmentation": "grounded_segmentation",
        "segment": "grounded_segmentation",
        "grounded_segmentation": "grounded_segmentation",
        "vqa": "grounded_vqa",
        "question_answering": "grounded_vqa",
        "grounded_question_answering": "grounded_vqa",
        "grounded_vqa": "grounded_vqa",
        "report": "grounded_report_generation",
        "report_generation": "grounded_report_generation",
        "grounded_report": "grounded_report_generation",
        "grounded_report_generation": "grounded_report_generation",
    }
    return aliases.get(text, fallback if fallback in TASK_TYPES else "grounded_segmentation")


def location_grid_from_center(cx: float, cy: float, width: int, height: int) -> str:
    column = "left" if cx < width / 3 else ("right" if cx >= 2 * width / 3 else "central")
    row = "upper" if cy < height / 3 else ("lower" if cy >= 2 * height / 3 else "middle")
    return f"{row}-{column}"


def json_safe_rle(rle: Optional[dict]) -> Optional[dict]:
    if not isinstance(rle, dict):
        return None
    result = dict(rle)
    if isinstance(result.get("counts"), bytes):
        result["counts"] = result["counts"].decode("utf-8")
    if "size" in result:
        result["size"] = [int(value) for value in result["size"]]
    return result


def encode_binary_mask(mask: np.ndarray) -> dict:
    encoded = maskUtils.encode(np.asfortranarray((mask > 0).astype(np.uint8)))
    return json_safe_rle(encoded)


def mask_geometry(mask: np.ndarray, width: int, height: int) -> Dict[str, Any]:
    mask_bool = mask > 0
    area = int(mask_bool.sum())
    if area <= 0:
        return {
            "has_mask": False,
            "area_px": 0,
            "area_ratio": 0.0,
            "bbox_xywh": [0, 0, 0, 0],
            "long_axis_px": 0,
            "short_axis_px": 0,
            "center_xy": None,
            "location_grid": None,
        }
    ys, xs = np.where(mask_bool)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    box_w, box_h = x1 - x0 + 1, y1 - y0 + 1
    cx, cy = float(xs.mean()), float(ys.mean())
    return {
        "has_mask": True,
        "area_px": area,
        "area_ratio": round(area / float(max(1, width * height)), 6),
        "bbox_xywh": [x0, y0, box_w, box_h],
        "long_axis_px": int(max(box_w, box_h)),
        "short_axis_px": int(min(box_w, box_h)),
        "center_xy": [round(cx, 2), round(cy, 2)],
        "location_grid": location_grid_from_center(cx, cy, width, height),
    }


def safe_filename(value: str, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return text[:80] or fallback


def colorize_overlay(image_rgb: Image.Image, mask: np.ndarray, alpha: float = 0.45, color=(255, 0, 0)) -> Image.Image:
    arr = np.asarray(image_rgb.convert("RGB")).copy()
    mask_bool = (mask > 0)
    overlay = arr.copy()
    overlay[mask_bool] = (np.array(color) * alpha + overlay[mask_bool] * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(overlay)


def decode_rle_mask(rle: dict, h: int, w: int) -> np.ndarray:
    if isinstance(rle, dict) and isinstance(rle.get("counts"), list):
        rle = maskUtils.frPyObjects(rle, h, w)
    m = maskUtils.decode(rle)
    if m.ndim == 3:
        m = np.max(m, axis=2)
    if m.shape[:2] != (h, w):
        import cv2
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


# ========================= SAM3 model + direct inference path =========================
def add_sam3_to_path(sam3_code_dir: str):
    sam3_code_dir = os.path.abspath(os.path.expanduser(sam3_code_dir))
    if sam3_code_dir not in sys.path:
        sys.path.insert(0, sam3_code_dir)


def setup_model(config_path: str, checkpoint_path: str, sam3_code_dir: str):
    """Adapted from eval_agent_all_datasets_v2.py::setup_model."""
    add_sam3_to_path(sam3_code_dir)
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from sam3.train.utils.train_utils import register_omegaconf_resolvers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        register_omegaconf_resolvers()
    except Exception:
        pass

    config_dir = os.path.dirname(os.path.abspath(config_path))
    config_name = os.path.basename(config_path)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=config_dir, version_base="1.2")
    cfg = compose(config_name=config_name)
    cfg.trainer.model.checkpoint_path = None
    cfg.trainer.model.load_from_HF = False

    print("[INFO] Loading SAM3 model...")
    model = instantiate(cfg.trainer.model)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "model_state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    # Strip common distributed/trainer prefixes if present.
    if isinstance(ckpt, dict):
        stripped = {}
        for k, v in ckpt.items():
            nk = k
            for prefix in ["module.", "model.", "_orig_mod."]:
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            stripped[nk] = v
        ckpt = stripped
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"[INFO] Loaded checkpoint with missing={len(missing)}, unexpected={len(unexpected)}")
    model.to(device)
    model.eval()
    return model, device, {"missing_keys": len(missing), "unexpected_keys": len(unexpected)}


def build_batch_for_sam3(image_pil: Image.Image, text_prompt: str, device: str):
    """Adapted from eval_agent_all_datasets_v2.py::build_batch_for_sam3."""
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.data.sam3_image_dataset import (
        Datapoint,
        Image as SAMImage,
        FindQueryLoaded,
        InferenceMetadata,
    )

    orig_w, orig_h = image_pil.size
    image_resized = image_pil.resize((1008, 1008), resample=Image.BILINEAR)
    image_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    image_normalized = (image_tensor - mean) / std

    find_query = FindQueryLoaded(
        query_text=text_prompt,
        image_id=0,
        object_ids_output=[],
        is_exhaustive=True,
        inference_metadata=InferenceMetadata(
            coco_image_id=0,
            original_image_id=0,
            original_category_id=0,
            original_size=(orig_h, orig_w),
            object_id=0,
            frame_index=0,
        ),
    )
    image_obj = SAMImage(data=image_normalized, objects=[], size=(1008, 1008))
    datapoint = Datapoint(find_queries=[find_query], images=[image_obj], raw_images=None)
    batch_input = collate_fn_api([datapoint], dict_key="all")["all"]

    if hasattr(batch_input, "img_batch"):
        batch_input.img_batch = batch_input.img_batch.to(device)
    for stage in batch_input.find_inputs:
        stage.input_boxes = stage.input_boxes.to(device)
        stage.input_boxes_mask = stage.input_boxes_mask.to(device)
        stage.input_boxes_label = stage.input_boxes_label.to(device)
        stage.input_points = stage.input_points.to(device)
        stage.input_points_mask = stage.input_points_mask.to(device)
        stage.img_ids = stage.img_ids.to(device)
    return batch_input, orig_h, orig_w


def extract_logits_masks(outputs, device: str):
    """Adapted from eval_agent_all_datasets_v2.py::extract_logits_masks."""
    output = outputs[0]
    if "find_stages" in output:
        last_stage = output["find_stages"][-1]
        if isinstance(last_stage, list):
            pred_logits = last_stage[0]["pred_logits"]
            pred_masks = last_stage[0]["pred_masks"]
        else:
            pred_logits = last_stage["pred_logits"][0]
            pred_masks = last_stage["pred_masks"][0]
    elif "pred_logits" in output:
        pred_logits = output["pred_logits"]
        pred_masks = output["pred_masks"]
    else:
        pred_logits = torch.empty((0, 1), device=device)
        pred_masks = torch.empty((0, 1008, 1008), device=device)
    return pred_logits, pred_masks


def masks_to_serialized(pred_logits, pred_masks, orig_h: int, orig_w: int) -> Dict[str, Any]:
    """Adapted from eval_agent_all_datasets_v2.py::masks_to_serialized."""
    if pred_logits.numel() > 0:
        if pred_logits.shape[-1] == 1:
            scores = torch.sigmoid(pred_logits).view(-1)
        else:
            scores = torch.softmax(pred_logits, dim=-1)[..., 0].view(-1)
    else:
        scores = torch.empty((0,), device=pred_masks.device if hasattr(pred_masks, "device") else "cpu")

    if pred_masks.numel() > 0:
        if pred_masks.ndim == 3:
            pred_masks = pred_masks.unsqueeze(1)
        elif pred_masks.ndim > 4:
            pred_masks = pred_masks.view(-1, 1, pred_masks.shape[-2], pred_masks.shape[-1])
        pred_masks = torch.nn.functional.interpolate(
            pred_masks.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False
        ).squeeze(1)
        if pred_masks.ndim > 3:
            pred_masks = pred_masks.view(-1, orig_h, orig_w)
        elif pred_masks.ndim == 2:
            pred_masks = pred_masks.unsqueeze(0)
        pred_masks_np = (pred_masks > 0).detach().cpu().numpy().astype(np.uint8)
    else:
        pred_masks_np = np.empty((0, orig_h, orig_w), dtype=np.uint8)

    scores_np = scores.detach().cpu().numpy().astype(float).tolist()
    boxes, rles, areas = [], [], []
    for m in pred_masks_np:
        ys, xs = np.where(m > 0)
        area = int(m.sum())
        areas.append(area)
        if len(xs) > 0:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            boxes.append([x0, y0, x1 - x0 + 1, y1 - y0 + 1])
        else:
            boxes.append([0, 0, 0, 0])
        rle = maskUtils.encode(np.asfortranarray(m))
        rle["counts"] = rle["counts"].decode("utf-8")
        rles.append(rle)
    return {
        "orig_img_h": int(orig_h),
        "orig_img_w": int(orig_w),
        "pred_boxes": boxes,
        "pred_masks": rles,
        "pred_scores": scores_np,
        "pred_areas": areas,
    }


def maybe_remove_overlaps(serialized: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from sam3.agent.helpers.mask_overlap_removal import remove_overlapping_masks
    except Exception:
        return serialized
    h, w = serialized["orig_img_h"], serialized["orig_img_w"]
    tmp = {
        "orig_img_h": h,
        "orig_img_w": w,
        "pred_boxes": serialized.get("pred_boxes", []),
        "pred_masks": [r["counts"] if isinstance(r, dict) else r for r in serialized.get("pred_masks", [])],
        "pred_scores": serialized.get("pred_scores", []),
    }
    tmp = remove_overlapping_masks(tmp)
    masks, areas = [], []
    for counts in tmp.get("pred_masks", []):
        rle = {"size": [h, w], "counts": counts}
        masks.append(rle)
        try:
            areas.append(int(maskUtils.area(rle)))
        except Exception:
            areas.append(0)
    serialized["pred_boxes"] = tmp.get("pred_boxes", [])
    serialized["pred_masks"] = masks
    serialized["pred_scores"] = tmp.get("pred_scores", [])
    serialized["pred_areas"] = areas
    return serialized


def sam3_predict_image(model, device: str, image_path: str, text_prompt: str, remove_overlap: bool = False) -> Dict[str, Any]:
    image_pil = load_image_as_rgb(image_path)
    batch_input, orig_h, orig_w = build_batch_for_sam3(image_pil, text_prompt, device)
    try:
        with torch.inference_mode():
            outputs = model(batch_input)
        pred_logits, pred_masks = extract_logits_masks(outputs, device)
        serialized = masks_to_serialized(pred_logits, pred_masks, orig_h, orig_w)
        if remove_overlap:
            serialized = maybe_remove_overlaps(serialized)
        return serialized
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def collect_candidates(serialized: Dict[str, Any], prompt: str, min_area: int, max_area_ratio: float) -> List[Dict[str, Any]]:
    candidates = []
    h, w = int(serialized["orig_img_h"]), int(serialized["orig_img_w"])
    max_area = int(h * w * max_area_ratio) if max_area_ratio and max_area_ratio > 0 else None
    masks = serialized.get("pred_masks", [])
    scores = serialized.get("pred_scores", [0.0] * len(masks))
    areas = serialized.get("pred_areas", [0] * len(masks))
    boxes = serialized.get("pred_boxes", [[0, 0, 0, 0]] * len(masks))
    for mask_idx, rle in enumerate(masks):
        area = int(areas[mask_idx])
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        candidates.append({
            "prompt": prompt,
            "mask_idx": int(mask_idx),
            "score": float(scores[mask_idx]),
            "area": area,
            "box": boxes[mask_idx],
            "rle": rle,
            "h": h,
            "w": w,
        })
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


# ========================= Front-end agent =========================
def resolve_chat_completions_url(api_url: str) -> str:
    """Accept either an OpenAI base URL or a full chat-completions endpoint."""
    url = str(api_url or "").strip().rstrip("/")
    if not url:
        raise ValueError("API URL is empty")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"


def call_chat_api(messages: List[Dict[str, Any]], args, max_tokens: Optional[int] = None) -> str:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}
    payload = {
        "model": args.api_model,
        "messages": messages,
        "temperature": args.api_temperature,
        "max_tokens": max_tokens or args.api_max_tokens,
    }
    url = resolve_chat_completions_url(args.api_url)
    resp = requests.post(url, headers=headers, json=payload, timeout=args.api_timeout)
    if not resp.ok:
        body = resp.text.strip()
        raise RuntimeError(
            f"API HTTP {resp.status_code} for {url}: "
            f"{body[:2000] if body else '<empty response body>'}"
        )
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def local_fallback_prompt(question: str, categories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple no-API fallback: choose category by substring, otherwise clean question."""
    q = question.lower()
    best = None
    for c in categories:
        meaning = c["meaning"].lower()
        name = str(c["name"]).lower()
        if meaning in q or name in q:
            best = c
            break
    if best is not None:
        prompt = best["meaning"]
        return {
            "chosen_option_id": best["option_id"],
            "chosen_category_name": best["name"],
            "sam3_prompt": prompt,
            "reason": "local fallback matched category text in question",
            "status": "local_fallback_category_match",
        }
    # Remove common instruction words; keep a short noun-ish phrase.
    prompt = re.sub(r"(?i)\b(please|segment|identify|find|show|mask|outline|this|the|in|image|ultrasound|us|for|me)\b", " ", question)
    prompt = re.sub(r"[^A-Za-z0-9\-_/ ]+", " ", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip()
    if not prompt:
        prompt = question.strip()
    prompt = " ".join(prompt.split()[:10])
    return {
        "chosen_option_id": "free_fallback",
        "chosen_category_name": prompt,
        "sam3_prompt": prompt,
        "reason": "local fallback cleaned the question because no category matched",
        "status": "local_fallback_clean_question",
    }


def local_fallback_plan(
    instruction: str,
    categories: List[Dict[str, Any]],
    forced_task: str = "auto",
    status: Optional[str] = None,
) -> Dict[str, Any]:
    target = local_fallback_prompt(instruction, categories)
    task_type = infer_task_type(instruction, forced_task)
    return {
        "task_type": task_type,
        "targets": [{
            "target_id": "target_1",
            "chosen_option_id": target["chosen_option_id"],
            "chosen_category_name": target["chosen_category_name"],
            "sam3_prompt": target["sam3_prompt"],
            "reason": target.get("reason", ""),
        }],
        "draft_answer": "",
        "report_outline": [],
        "status": status or target.get("status", "local_fallback"),
    }


def normalize_agent_plan(
    obj: Dict[str, Any],
    instruction: str,
    categories: List[Dict[str, Any]],
    args,
) -> Dict[str, Any]:
    fallback_task = infer_task_type(instruction, args.task)
    task_type = normalize_task_type(obj.get("task_type") or obj.get("task"), fallback_task)
    if args.task != "auto":
        task_type = args.task

    raw_targets = obj.get("targets")
    if not isinstance(raw_targets, list):
        raw_targets = [obj] if obj.get("sam3_prompt") or obj.get("prompt") else []

    by_option = {c["option_id"]: c for c in categories}
    by_name = {str(c["name"]).strip().lower(): c for c in categories}
    by_meaning = {str(c["meaning"]).strip().lower(): c for c in categories}
    targets = []
    seen_prompts = set()
    for raw_target in raw_targets[: max(1, args.max_targets)]:
        if not isinstance(raw_target, dict):
            continue
        chosen_option = str(raw_target.get("chosen_option_id") or raw_target.get("option_id") or "").strip()
        chosen_name = str(
            raw_target.get("chosen_category_name")
            or raw_target.get("category_name")
            or raw_target.get("category")
            or ""
        ).strip()
        prompt = str(
            raw_target.get("sam3_prompt")
            or raw_target.get("prompt")
            or raw_target.get("simple_prompt")
            or ""
        ).strip()
        chosen_cat = by_option.get(chosen_option)
        if chosen_cat is None and chosen_name:
            chosen_cat = (
                by_name.get(chosen_name.lower())
                or by_meaning.get(clean_category_prompt(chosen_name).lower())
            )
        if chosen_cat is not None:
            chosen_option = chosen_cat["option_id"]
            chosen_name = chosen_cat["name"]
            if not prompt:
                prompt = chosen_cat["meaning"]
        if not prompt:
            continue
        prompt = re.sub(r"\s+", " ", prompt).strip()
        prompt_key = prompt.lower()
        if prompt_key in seen_prompts:
            continue
        seen_prompts.add(prompt_key)
        targets.append({
            "target_id": f"target_{len(targets) + 1}",
            "chosen_option_id": chosen_option or "free_text",
            "chosen_category_name": chosen_name or prompt,
            "sam3_prompt": prompt,
            "reason": str(raw_target.get("reason", "")),
        })

    if not targets:
        return local_fallback_plan(
            instruction,
            categories,
            forced_task=args.task,
            status="local_fallback_empty_agent_targets",
        )

    report_outline = obj.get("report_outline", obj.get("report", []))
    if not isinstance(report_outline, list):
        report_outline = []
    return {
        "task_type": task_type,
        "targets": targets,
        "draft_answer": str(obj.get("draft_answer") or obj.get("answer") or "").strip(),
        "report_outline": report_outline,
        "status": "api",
    }


def agent_parse_question(image_path: str, question: str, categories: List[Dict[str, Any]], args) -> Tuple[Dict[str, Any], str]:
    if args.no_api:
        parsed = local_fallback_plan(question, categories, args.task)
        return parsed, ""
    if not args.api_key:
        parsed = local_fallback_plan(
            question, categories, args.task, status="local_fallback_missing_api_key"
        )
        return parsed, ""

    target_payload = [
        {
            "option_id": c["option_id"],
            "name": c["name"],
            "meaning": c["meaning"],
        }
        for c in categories
    ]
    system_text = (
        "You are a front-end task planner for an ultrasound-grounded US-SAM3 system. "
        "Classify the instruction as grounded_segmentation, grounded_vqa, or "
        "grounded_report_generation. Identify only the anatomy/pathology targets that must be "
        "segmented to answer the instruction. For each target, choose a category option when a "
        "clear match exists and produce one concise SAM3 noun-phrase prompt. Preserve the option's "
        "category meaning; do not invent categories, diagnoses, measurements, or unseen findings. "
        "Most instructions need one target. Use multiple targets only when the instruction clearly "
        "requires distinct structures. Return ONLY valid JSON with keys: task_type, targets, "
        "draft_answer, report_outline. Each targets item must contain chosen_option_id, "
        "chosen_category_name, sam3_prompt, reason. report_outline may contain objects with "
        "section, text, finding_type, target_ids."
    )
    user_text = (
        f"User ultrasound instruction: {question!r}\n"
        f"Requested task override: {args.task!r}\n\n"
        "Valid target categories as JSON array:\n"
        f"{json.dumps(target_payload, ensure_ascii=False)}\n\n"
        f"Return at most {args.max_targets} targets. Write each sam3_prompt as a concise noun phrase "
        "under 14 words. Prefer concrete medical/ultrasound terms instead of vague words like 'it'. "
        "draft_answer and report_outline are optional drafts; masks and measured geometry will be "
        "added after US-SAM3 inference."
    )
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_path_to_data_url(image_path, args.api_image_max_side)}},
        ]},
    ]
    try:
        raw = call_chat_api(messages, args, max_tokens=args.api_max_tokens)
        obj = extract_json_object(raw)
        if not isinstance(obj, dict):
            parsed = local_fallback_plan(
                question, categories, args.task, status="local_fallback_invalid_agent_json"
            )
            parsed["raw_agent_response"] = raw[:1000]
            return parsed, raw
        parsed = normalize_agent_plan(obj, question, categories, args)
        return parsed, raw
    except Exception as e:
        if not args.api_fallback_on_error:
            raise
        parsed = local_fallback_plan(
            question, categories, args.task, status="local_fallback_api_error"
        )
        parsed["error"] = str(e)
        return parsed, f"ERROR: {e}"


def run_planned_targets(model, device: str, image_path: str, plan: Dict[str, Any], args) -> List[Dict[str, Any]]:
    target_runs = []
    for index, target in enumerate(plan.get("targets", []), 1):
        prompt = target["sam3_prompt"]
        print(
            f"[INFO] US-SAM3 target {index}/{len(plan['targets'])}: "
            f"{target['target_id']} prompt={prompt!r}"
        )
        serialized = sam3_predict_image(
            model=model,
            device=device,
            image_path=image_path,
            text_prompt=prompt,
            remove_overlap=args.remove_overlap,
        )
        candidates = collect_candidates(
            serialized, prompt, args.min_area, args.max_area_ratio
        )
        h, w = int(serialized["orig_img_h"]), int(serialized["orig_img_w"])
        top1 = candidates[0] if candidates else None
        mask = (
            decode_rle_mask(top1["rle"], h, w)
            if top1 is not None
            else np.zeros((h, w), dtype=np.uint8)
        )
        geometry = mask_geometry(mask, w, h)
        target_runs.append({
            "target": target,
            "serialized": serialized,
            "candidates": candidates,
            "top1": top1,
            "mask": mask,
            "geometry": geometry,
        })
        print(
            f"[INFO] {target['target_id']} candidates={len(candidates)} "
            f"score={None if top1 is None else round(float(top1['score']), 6)} "
            f"area={geometry['area_px']}"
        )
    return target_runs


def target_facts_for_agent(target_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    facts = []
    for run in target_runs:
        target = run["target"]
        geometry = run["geometry"]
        top1 = run["top1"]
        facts.append({
            "target_id": target["target_id"],
            "category_name": canonical_display_category(
                target["chosen_category_name"]
            ),
            "sam3_prompt": target["sam3_prompt"],
            "has_mask": geometry["has_mask"],
            "score": None if top1 is None else round(float(top1["score"]), 6),
            "bbox_xywh": geometry["bbox_xywh"],
            "long_axis_px": geometry["long_axis_px"],
            "short_axis_px": geometry["short_axis_px"],
            "area_px": geometry["area_px"],
            "area_ratio_percent": round(geometry["area_ratio"] * 100, 2),
            "location_grid": geometry["location_grid"],
        })
    return facts


def valid_mask_target_ids(target_runs: List[Dict[str, Any]]) -> List[str]:
    return [
        run["target"]["target_id"]
        for run in target_runs
        if run["geometry"]["has_mask"]
    ]


def ensure_seg_token(answer: str, has_grounding: bool) -> str:
    answer = re.sub(r"\s+", " ", str(answer or "")).strip()
    if has_grounding and "[SEG]" not in answer:
        answer = f"{answer} [SEG]".strip()
    if not has_grounding:
        answer = answer.replace("[SEG]", "").strip()
    return answer


def canonical_display_category(category: str) -> str:
    category = re.sub(r"\s+", " ", str(category or "")).strip()
    if category.lower() in {"prostate", "prostate organ"}:
        return "prostate gland"
    return category or "requested target"


def qualitative_extent_label(area_ratio: float) -> str:
    if area_ratio < 0.10:
        return "small"
    if area_ratio >= 0.25:
        return "broad"
    return "moderate"


def deterministic_vqa_answer(
    instruction: str,
    plan: Dict[str, Any],
    target_runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    valid_runs = [run for run in target_runs if run["geometry"]["has_mask"]]
    target_ids = [run["target"]["target_id"] for run in valid_runs]
    if not valid_runs:
        return {"answer": "No requested target was identified.", "target_ids": []}

    primary = valid_runs[0]
    category = canonical_display_category(
        primary["target"]["chosen_category_name"]
    )
    geometry = primary["geometry"]
    text = str(instruction or "").lower()
    yes_no_question = text.strip().startswith(
        ("is ", "are ", "does ", "do ", "can ", "has ", "have ")
    )
    if any(term in text for term in ("where", "location", "located", "position")):
        answer = f"The {category} is located in the {geometry['location_grid']} image region."
    elif any(term in text for term in ("how many", "number", "count", "multiple", "single")):
        count = len(valid_runs)
        if count == 1:
            answer = "One target is present."
        else:
            answer = f"{count} targets are present."
    elif (
        any(term in text for term in ("small", "large", "broad", "narrow"))
        and any(term in text for term in ("area", "portion", "extent", "occup"))
    ):
        label = qualitative_extent_label(float(geometry["area_ratio"]))
        if label == "small":
            answer = f"The {category} occupies a small portion of the image."
        elif label == "broad":
            answer = f"The {category} occupies a broad area of the image."
        else:
            answer = f"The {category} occupies a moderate portion of the image."
    elif any(
        term in text
        for term in (
            "how much",
            "area ratio",
            "percentage",
            "percent",
            "proportion",
            "fraction",
            "covered",
            "occupy",
        )
    ):
        ratio = float(geometry["area_ratio"])
        answer = (
            f"The visible {category} occupies approximately {ratio:.6f} "
            f"of the image area ({ratio * 100:.2f}%)."
        )
    elif yes_no_question:
        answer = f"Yes, the {category} is visible as a segmentation target."
    elif any(term in text for term in ("size", "dimension", "measure", "extent", "span")):
        answer = (
            f"The visible {category} measures approximately "
            f"{geometry['long_axis_px']} by {geometry['short_axis_px']} pixels."
        )
    elif any(term in text for term in ("what", "which structure", "which anatomy", "identify")):
        answer = f"The identified target is the {category}."
    elif any(term in text for term in ("visible", "present", "boundary")):
        answer = f"Yes, the {category} is visible as a segmentation target."
    else:
        answer = plan.get("draft_answer") or f"The {category} is visible in the image."
    return {
        "answer": ensure_seg_token(answer, True),
        "target_ids": target_ids,
    }


def should_enforce_geometry_vqa(instruction: str, has_mask: bool) -> bool:
    text = str(instruction or "").lower()
    if not has_mask:
        return True
    if any(term in text for term in ("how many", "number", "count", "multiple", "single")):
        return True
    if any(
        term in text
        for term in ("percentage", "percent", "proportion", "fraction", "area ratio")
    ):
        return True
    if any(term in text for term in ("size", "dimension", "measure", "span")) and not any(
        term in text for term in ("small", "large", "broad", "narrow")
    ):
        return True
    yes_no_question = text.strip().startswith(
        ("is ", "are ", "does ", "do ", "can ", "has ", "have ")
    )
    return yes_no_question and any(
        term in text
        for term in ("visible", "present", "boundary", "outline", "segment", "target")
    )


def deterministic_report(target_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    report = []
    valid_runs = [run for run in target_runs if run["geometry"]["has_mask"]]
    if not valid_runs:
        return [
            {
                "section": "Findings",
                "text": "No prostate target is available in this ultrasound image.",
                "finding_type": "absence",
                "target_ids": [],
            },
            {
                "section": "Findings",
                "text": "No prostate boundary is available for localization.",
                "finding_type": "location_not_applicable",
                "target_ids": [],
            },
            {
                "section": "Findings",
                "text": "No prostate extent can be measured in the image.",
                "finding_type": "size_not_applicable",
                "target_ids": [],
            },
            {
                "section": "Assessment",
                "text": "No target anatomy is available for spatial assessment.",
                "finding_type": "no_target",
                "target_ids": [],
            },
            {
                "section": "Impression",
                "text": "No prostate target is available for segmentation.",
                "finding_type": "no_target_impression",
                "target_ids": [],
            },
        ]
    for run in valid_runs:
        target = run["target"]
        geometry = run["geometry"]
        target_ids = [target["target_id"]]
        category = canonical_display_category(target["chosen_category_name"])
        short_category = re.sub(r"\s+gland$", "", category, flags=re.IGNORECASE)
        report.extend([
            {
                "section": "Findings",
                "text": (
                    f"The {category} is visible in the "
                    f"{geometry['location_grid']} image region."
                ),
                "finding_type": "organ_presence",
                "target_ids": target_ids,
            },
            {
                "section": "Findings",
                "text": (
                    f"The {short_category} center lies in the "
                    f"{geometry['location_grid']} image region."
                ),
                "finding_type": "organ_location",
                "target_ids": target_ids,
            },
            {
                "section": "Findings",
                "text": (
                    f"Its visible extent is approximately "
                    f"{float(geometry['long_axis_px']):.1f} x "
                    f"{float(geometry['short_axis_px']):.1f} pixels."
                ),
                "finding_type": "organ_size",
                "target_ids": target_ids,
            },
            {
                "section": "Findings",
                "text": (
                    f"The visible {short_category} area occupies approximately "
                    f"{geometry['area_ratio'] * 100:.2f}% of the image."
                ),
                "finding_type": "organ_extent",
                "target_ids": target_ids,
            },
            {
                "section": "Assessment",
                "text": (
                    f"The available evidence supports {short_category} localization "
                    "and boundary delineation only."
                ),
                "finding_type": "grounding_scope",
                "target_ids": target_ids,
            },
            {
                "section": "Impression",
                "text": f"Visible {category} target for segmentation.",
                "finding_type": "segmentation_impression",
                "target_ids": target_ids,
            },
        ])
    return report


def normalize_response_target_ids(value: Any, valid_ids: List[str], fallback: List[str]) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        value = []
    selected = []
    for target_id in value:
        target_id = str(target_id)
        if target_id in valid_ids and target_id not in selected:
            selected.append(target_id)
    return selected or list(fallback)


REPORT_SECTIONS = {
    "organ_presence": "Findings",
    "organ_location": "Findings",
    "organ_size": "Findings",
    "organ_extent": "Findings",
    "grounding_scope": "Assessment",
    "segmentation_impression": "Impression",
}

NO_TARGET_REPORT_SECTIONS = {
    "absence": "Findings",
    "location_not_applicable": "Findings",
    "size_not_applicable": "Findings",
    "no_target": "Assessment",
    "no_target_impression": "Impression",
}


def report_prompt_payload(
    instruction: str,
    plan: Dict[str, Any],
    target_runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    facts = target_facts_for_agent(target_runs)
    valid_facts = [fact for fact in facts if fact["has_mask"]]
    if valid_facts:
        required_structure = [
            {
                "section": section,
                "finding_type": finding_type,
                "required_target_ids": [fact["target_id"] for fact in valid_facts],
            }
            for finding_type, section in REPORT_SECTIONS.items()
        ]
        wording_rules = [
            "Write exactly one concise sentence for each required finding_type, in the listed order.",
            "organ_presence: state that category_name is visible in location_grid.",
            "organ_location: state that the organ center lies in location_grid.",
            "organ_size: copy long_axis_px and short_axis_px exactly and report them as A x B pixels.",
            "organ_extent: copy area_ratio_percent exactly and report it as a percentage of the image.",
            "grounding_scope: limit the evidence to localization and boundary delineation.",
            "segmentation_impression: identify category_name as the visible segmentation target.",
            "Use category_name rather than a generic word such as anatomy, structure, object, or region.",
            "Do not merge finding types into one sentence and do not add extra report objects.",
        ]
    else:
        required_structure = [
            {
                "section": section,
                "finding_type": finding_type,
                "required_target_ids": [],
            }
            for finding_type, section in NO_TARGET_REPORT_SECTIONS.items()
        ]
        wording_rules = [
            "State only that no target, location, measurable extent, or segmentation target is available.",
        ]
    return {
        "task_type": "grounded_report_generation",
        "instruction": instruction,
        "authoritative_mask_facts": facts,
        "required_report_structure": required_structure,
        "wording_rules": wording_rules,
        "output_contract": {
            "top_level_key": "report",
            "report_item_keys": [
                "section",
                "text",
                "finding_type",
                "target_ids",
            ],
            "json_only": True,
        },
        "planner_outline_for_context_only": plan.get("report_outline", []),
    }


def normalize_api_report_items(raw_report: Any, valid_ids: List[str]) -> List[Dict[str, Any]]:
    if not isinstance(raw_report, list):
        return []
    allowed_sections = REPORT_SECTIONS if valid_ids else NO_TARGET_REPORT_SECTIONS
    report = []
    seen = set()
    for item in raw_report:
        if not isinstance(item, dict):
            continue
        finding_type = str(item.get("finding_type") or "").strip()
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        if finding_type not in allowed_sections or not text:
            continue
        target_ids = normalize_response_target_ids(
            item.get("target_ids"), valid_ids, []
        )
        key = (finding_type, tuple(target_ids))
        if key in seen:
            continue
        seen.add(key)
        report.append({
            "section": allowed_sections[finding_type],
            "text": text,
            "finding_type": finding_type,
            "target_ids": target_ids,
        })
    order = {name: index for index, name in enumerate(allowed_sections)}
    report.sort(key=lambda item: order[item["finding_type"]])
    return report


def report_validation_errors(
    report: List[Dict[str, Any]],
    target_runs: List[Dict[str, Any]],
) -> List[str]:
    errors = []
    valid_runs = [run for run in target_runs if run["geometry"]["has_mask"]]
    required = list(REPORT_SECTIONS if valid_runs else NO_TARGET_REPORT_SECTIONS)
    actual = [item.get("finding_type") for item in report]
    missing = [finding_type for finding_type in required if finding_type not in actual]
    extra = [finding_type for finding_type in actual if finding_type not in required]
    if missing:
        errors.append("missing finding_type values: " + ", ".join(missing))
    if extra:
        errors.append("unexpected finding_type values: " + ", ".join(extra))
    if len(report) != len(required):
        errors.append(f"expected exactly {len(required)} report objects, got {len(report)}")

    if not valid_runs:
        if any(item.get("target_ids") for item in report):
            errors.append("no-target report objects must use empty target_ids")
        return errors

    valid_ids = set(valid_mask_target_ids(target_runs))
    for item in report:
        if not item.get("target_ids"):
            errors.append(f"{item.get('finding_type')} must contain a target_id")
        elif not set(item["target_ids"]).issubset(valid_ids):
            errors.append(f"{item.get('finding_type')} contains an invalid target_id")

    primary = valid_runs[0]
    geometry = primary["geometry"]
    by_type = {item["finding_type"]: item["text"].lower() for item in report}
    location = str(geometry["location_grid"]).lower()
    if location not in by_type.get("organ_presence", ""):
        errors.append(f"organ_presence must contain location_grid={location}")
    if location not in by_type.get("organ_location", ""):
        errors.append(f"organ_location must contain location_grid={location}")

    size_numbers = [
        float(value)
        for value in re.findall(r"\d+(?:\.\d+)?", by_type.get("organ_size", ""))
    ]
    expected_size = sorted([
        float(geometry["long_axis_px"]),
        float(geometry["short_axis_px"]),
    ])
    if len(size_numbers) < 2 or any(
        abs(left - right) > 0.05
        for left, right in zip(sorted(size_numbers[:2]), expected_size)
    ):
        errors.append(
            "organ_size must use the exact long_axis_px and short_axis_px values"
        )

    extent_numbers = [
        float(value)
        for value in re.findall(r"\d+(?:\.\d+)?", by_type.get("organ_extent", ""))
    ]
    expected_extent = round(float(geometry["area_ratio"]) * 100, 2)
    if not extent_numbers or abs(extent_numbers[0] - expected_extent) > 0.01:
        errors.append(
            f"organ_extent must use area_ratio_percent={expected_extent:.2f}"
        )
    return errors


def compose_hybrid_api_report(
    image_path: str,
    instruction: str,
    plan: Dict[str, Any],
    target_runs: List[Dict[str, Any]],
    args,
) -> Tuple[Dict[str, Any], str]:
    payload = report_prompt_payload(instruction, plan, target_runs)
    system_text = (
        "You are the report-writing component of an evidence-constrained ultrasound system. "
        "US-SAM3 has already produced the authoritative mask facts in the user payload. "
        "Write the report from those facts; the image is context only and must not override "
        "location_grid, pixel measurements, area percentages, target presence, or target IDs. "
        "Return valid JSON only with top-level key report. The report must contain exactly the "
        "required finding types, sections, and target IDs from required_report_structure. "
        "Use every numeric value exactly as supplied. Write concise clinical-style English, but "
        "do not invent diagnoses, pathology, echogenicity, vascularity, morphology, physical-unit "
        "measurements, recommendations, or any finding absent from authoritative_mask_facts. "
        "Do not mention SAM3, masks, annotations, metadata, datasets, prompts, or confidence scores."
    )
    user_text = (
        "Generate the grounded report according to this JSON specification:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_path_to_data_url(
                        image_path, args.api_image_max_side
                    )
                },
            },
        ]},
    ]

    attempts = []
    errors = []
    max_attempts = 1 + max(0, int(getattr(args, "report_validation_retries", 1)))
    for attempt in range(max_attempts):
        raw = call_chat_api(
            messages,
            args,
            max_tokens=args.api_response_max_tokens,
        )
        attempts.append(raw)
        obj = extract_json_object(raw)
        raw_report = obj.get("report") if isinstance(obj, dict) else None
        report = normalize_api_report_items(
            raw_report,
            valid_mask_target_ids(target_runs),
        )
        errors = report_validation_errors(report, target_runs)
        if not errors:
            return {
                "task_type": "grounded_report_generation",
                "report": report,
                "response_status": (
                    "hybrid_api_grounded"
                    if attempt == 0
                    else "hybrid_api_grounded_after_revision"
                ),
                "report_api_attempts": attempt + 1,
                "report_validation_errors": [],
            }, json.dumps(
                {"attempts": attempts},
                ensure_ascii=False,
            )

        if attempt + 1 < max_attempts:
            messages.extend([
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "The report failed factual or structural validation. Correct the JSON "
                        "without changing any authoritative fact. Return the complete report "
                        "again, not a patch. Validation errors:\n- "
                        + "\n- ".join(errors)
                    ),
                },
            ])

    fallback = {
        "task_type": "grounded_report_generation",
        "report": deterministic_report(target_runs),
        "response_status": "fallback_invalid_hybrid_api_report",
        "report_api_attempts": len(attempts),
        "report_validation_errors": errors,
    }
    return fallback, json.dumps(
        {"attempts": attempts, "validation_errors": errors},
        ensure_ascii=False,
    )


def compose_grounded_response(
    image_path: str,
    instruction: str,
    plan: Dict[str, Any],
    target_runs: List[Dict[str, Any]],
    args,
) -> Tuple[Dict[str, Any], str]:
    task_type = plan["task_type"]
    valid_ids = valid_mask_target_ids(target_runs)
    if task_type == "grounded_segmentation":
        if valid_ids:
            names = [
                run["target"]["chosen_category_name"]
                for run in target_runs
                if run["target"]["target_id"] in valid_ids
            ]
            answer = ensure_seg_token(
                "Segmented " + ", ".join(str(name) for name in names) + ".", True
            )
        else:
            answer = "No requested target was identified."
        return {"task_type": task_type, "answer": answer, "target_ids": valid_ids}, ""

    if (
        task_type == "grounded_vqa"
        and getattr(args, "vqa_generation_mode", "hybrid") == "deterministic"
    ):
        response = deterministic_vqa_answer(instruction, plan, target_runs)
        response.update({
            "task_type": task_type,
            "response_status": "deterministic_geometry",
        })
        return response, ""

    if (
        task_type == "grounded_report_generation"
        and getattr(args, "report_generation_mode", "hybrid") == "deterministic"
    ):
        return {
            "task_type": task_type,
            "report": deterministic_report(target_runs),
            "response_status": "deterministic_geometry",
        }, ""

    fallback = (
        deterministic_vqa_answer(instruction, plan, target_runs)
        if task_type == "grounded_vqa"
        else {"report": deterministic_report(target_runs)}
    )
    if args.no_api or not args.api_key:
        fallback["task_type"] = task_type
        fallback["response_status"] = (
            "fallback_no_api"
            if args.no_api
            else "fallback_missing_api_key"
        )
        return fallback, ""

    if (
        task_type == "grounded_report_generation"
        and getattr(args, "report_generation_mode", "hybrid") == "hybrid"
    ):
        try:
            return compose_hybrid_api_report(
                image_path=image_path,
                instruction=instruction,
                plan=plan,
                target_runs=target_runs,
                args=args,
            )
        except Exception as exc:
            if not args.api_fallback_on_error:
                raise
            fallback.update({
                "task_type": task_type,
                "response_status": "fallback_hybrid_report_api_error",
                "response_error": str(exc),
            })
            return fallback, f"ERROR: {exc}"

    system_text = (
        "You compose the final output for an ultrasound grounded model after US-SAM3 inference. "
        "The supplied mask-derived target facts are authoritative. Do not change pixel measurements, "
        "area ratios, locations, target IDs, or claim that an absent target is present. Do not invent "
        "diagnoses, pathology, echogenicity, vascularity, physical-unit measurements, or recommendations "
        "that are not provided in the facts. For grounded_vqa return JSON with answer and target_ids. "
        "For grounded_report_generation return JSON with report, where report is a list of objects with "
        "section, text, finding_type, target_ids. Every target_ids value must use the supplied IDs. "
        "Return valid JSON only."
    )
    user_text = (
        f"Task type: {task_type}\n"
        f"Original instruction: {instruction!r}\n"
        f"Agent draft answer: {plan.get('draft_answer', '')!r}\n"
        f"Agent report outline: {json.dumps(plan.get('report_outline', []), ensure_ascii=False)}\n"
        "Authoritative target facts:\n"
        f"{json.dumps(target_facts_for_agent(target_runs), ensure_ascii=False, indent=2)}"
    )
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_path_to_data_url(
                        image_path, args.api_image_max_side
                    )
                },
            },
        ]},
    ]
    try:
        raw = call_chat_api(messages, args, max_tokens=args.api_response_max_tokens)
        obj = extract_json_object(raw)
        if not isinstance(obj, dict):
            fallback["task_type"] = task_type
            fallback["response_status"] = "fallback_invalid_response_json"
            return fallback, raw
        if task_type == "grounded_vqa":
            answer = str(obj.get("answer") or fallback["answer"]).strip()
            target_ids = normalize_response_target_ids(
                obj.get("target_ids"), valid_ids, fallback.get("target_ids", valid_ids)
            )
            response_status = "api"
            if (
                getattr(args, "vqa_generation_mode", "hybrid") == "hybrid"
                and should_enforce_geometry_vqa(instruction, bool(valid_ids))
            ):
                answer = fallback["answer"]
                target_ids = fallback.get("target_ids", valid_ids)
                response_status = "hybrid_geometry_constraint"
            return {
                "task_type": task_type,
                "answer": ensure_seg_token(answer, bool(target_ids)),
                "target_ids": target_ids,
                "response_status": response_status,
            }, raw

        raw_report = obj.get("report")
        if not isinstance(raw_report, list):
            raw_report = []
        report = []
        for item in raw_report:
            if not isinstance(item, dict) or not str(item.get("text", "")).strip():
                continue
            report.append({
                "section": str(item.get("section") or "Findings"),
                "text": re.sub(r"\s+", " ", str(item["text"])).strip(),
                "finding_type": str(item.get("finding_type") or "grounded_finding"),
                "target_ids": normalize_response_target_ids(
                    item.get("target_ids"), valid_ids, valid_ids[:1]
                ),
            })
        if not report:
            report = fallback["report"]
        return {
            "task_type": task_type,
            "report": report,
            "response_status": "api",
        }, raw
    except Exception as exc:
        if not args.api_fallback_on_error:
            raise
        fallback["task_type"] = task_type
        fallback["response_status"] = "fallback_api_error"
        fallback["response_error"] = str(exc)
        return fallback, f"ERROR: {exc}"


# ========================= Save outputs =========================
def save_outputs(
    image_path: str,
    question: str,
    parsed: Dict[str, Any],
    raw_agent_response: str,
    candidates: List[Dict[str, Any]],
    serialized: Dict[str, Any],
    args,
    load_info: Dict[str, Any],
) -> Dict[str, str]:
    stem = Path(image_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root).expanduser().resolve() / f"{stem}_agent_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    image = load_image_as_rgb(image_path)
    h, w = int(serialized["orig_img_h"]), int(serialized["orig_img_w"])
    if image.size != (w, h):
        image = image.resize((w, h), Image.BILINEAR)

    top1 = candidates[0] if candidates else None
    if top1 is None:
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        mask = decode_rle_mask(top1["rle"], h, w)

    mask_path = run_dir / f"{stem}_agent_top1_mask.png"
    overlay_path = run_dir / f"{stem}_agent_top1_overlay.png"
    candidates_path = run_dir / f"{stem}_agent_candidates.json"
    meta_path = run_dir / f"{stem}_agent_meta.json"
    raw_agent_path = run_dir / f"{stem}_agent_raw_response.txt"

    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(mask_path)
    colorize_overlay(image, mask).save(overlay_path)

    json_safe_candidates = []
    for c in candidates:
        item = dict(c)
        item["rle"] = dict(item["rle"])
        if isinstance(item["rle"].get("counts"), bytes):
            item["rle"]["counts"] = item["rle"]["counts"].decode("utf-8")
        json_safe_candidates.append(item)
    candidates_path.write_text(json.dumps(json_safe_candidates, indent=2, ensure_ascii=False), encoding="utf-8")
    raw_agent_path.write_text(raw_agent_response or "", encoding="utf-8")

    meta = {
        "image": str(Path(image_path).expanduser().resolve()),
        "question": question,
        "agent_status": parsed.get("status", ""),
        "agent_chosen_option_id": parsed.get("chosen_option_id", ""),
        "agent_chosen_category_name": parsed.get("chosen_category_name", ""),
        "sam3_prompt": parsed.get("sam3_prompt", ""),
        "agent_reason": parsed.get("reason", ""),
        "checkpoint": str(Path(args.checkpoint_path).expanduser().resolve()),
        "config_path": str(Path(args.config_path).expanduser().resolve()),
        "sam3_code_dir": str(Path(args.sam3_code_dir).expanduser().resolve()),
        "num_raw_masks": len(serialized.get("pred_masks", [])),
        "num_candidates_after_filter": len(candidates),
        "top1_score": None if top1 is None else float(top1["score"]),
        "top1_area_pixels": int(mask.sum()),
        "top1_box": None if top1 is None else top1.get("box"),
        "min_area": args.min_area,
        "max_area_ratio": args.max_area_ratio,
        "remove_overlap": bool(args.remove_overlap),
        "model_load_info": load_info,
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "candidates_path": str(candidates_path),
        "raw_agent_response_path": str(raw_agent_path),
        "run_dir": str(run_dir),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if top1 is None:
        print("[WARN] No valid candidate after filtering. Saved an empty mask.")
    else:
        print(f"[INFO] Agent SAM3 prompt: {meta['sam3_prompt']!r}")
        print(f"[INFO] Raw masks={meta['num_raw_masks']}, candidates={len(candidates)}, top1 score={meta['top1_score']:.6f}, area={meta['top1_area_pixels']} pixels")

    return {
        "run_dir": str(run_dir),
        "mask": str(mask_path),
        "overlay": str(overlay_path),
        "meta": str(meta_path),
        "candidates": str(candidates_path),
        "raw_agent_response": str(raw_agent_path),
    }


def union_mask_for_target_ids(
    target_runs: List[Dict[str, Any]],
    target_ids: List[str],
) -> Optional[np.ndarray]:
    selected = [
        run["mask"]
        for run in target_runs
        if run["target"]["target_id"] in set(target_ids)
        and run["geometry"]["has_mask"]
    ]
    if not selected:
        return None
    union = np.zeros_like(selected[0], dtype=np.uint8)
    for mask in selected:
        union |= (mask > 0).astype(np.uint8)
    return union


def save_multitask_outputs(
    image_path: str,
    instruction: str,
    plan: Dict[str, Any],
    raw_plan_response: str,
    response: Dict[str, Any],
    raw_final_response: str,
    target_runs: List[Dict[str, Any]],
    args,
    load_info: Dict[str, Any],
) -> Dict[str, str]:
    stem = Path(image_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.output_root).expanduser().resolve()
        / f"{stem}_{plan['task_type']}_{ts}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    targets_dir = run_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    image = load_image_as_rgb(image_path)
    combined_overlay = image.copy()
    target_payloads = []
    for index, run in enumerate(target_runs):
        target = run["target"]
        target_id = target["target_id"]
        label = safe_filename(
            f"{target_id}_{target['chosen_category_name']}", target_id
        )
        mask = run["mask"]
        color = OVERLAY_COLORS[index % len(OVERLAY_COLORS)]
        mask_path = targets_dir / f"{label}_mask.png"
        overlay_path = targets_dir / f"{label}_overlay.png"
        candidates_path = targets_dir / f"{label}_candidates.json"
        Image.fromarray((mask > 0).astype(np.uint8) * 255).save(mask_path)
        colorize_overlay(image, mask, color=color).save(overlay_path)
        if run["geometry"]["has_mask"]:
            combined_overlay = colorize_overlay(
                combined_overlay, mask, alpha=0.35, color=color
            )

        candidates_json = []
        for candidate in run["candidates"]:
            item = dict(candidate)
            item["rle"] = json_safe_rle(item.get("rle"))
            candidates_json.append(item)
        candidates_path.write_text(
            json.dumps(candidates_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        top1 = run["top1"]
        target_payloads.append({
            "target_id": target_id,
            "chosen_option_id": target["chosen_option_id"],
            "chosen_category_name": target["chosen_category_name"],
            "sam3_prompt": target["sam3_prompt"],
            "reason": target.get("reason", ""),
            "score": None if top1 is None else float(top1["score"]),
            **run["geometry"],
            "predicted_mask": (
                None if top1 is None else json_safe_rle(top1["rle"])
            ),
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "candidates_path": str(candidates_path),
            "num_candidates": len(run["candidates"]),
        })

    combined_overlay_path = run_dir / f"{stem}_combined_overlay.png"
    combined_overlay.save(combined_overlay_path)
    raw_plan_path = run_dir / f"{stem}_raw_plan_response.txt"
    raw_final_path = run_dir / f"{stem}_raw_final_response.txt"
    plan_path = run_dir / f"{stem}_task_plan.json"
    result_path = run_dir / f"{stem}_grounded_result.json"
    meta_path = run_dir / f"{stem}_meta.json"
    raw_plan_path.write_text(raw_plan_response or "", encoding="utf-8")
    raw_final_path.write_text(raw_final_response or "", encoding="utf-8")
    plan_path.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    task_type = plan["task_type"]
    structured_result: Dict[str, Any] = {
        "task_type": task_type,
        "instruction": instruction,
        "image": str(Path(image_path).expanduser().resolve()),
        "image_width": image.width,
        "image_height": image.height,
        "targets": target_payloads,
    }
    if task_type in ("grounded_segmentation", "grounded_vqa"):
        target_ids = normalize_response_target_ids(
            response.get("target_ids"),
            [item["target_id"] for item in target_payloads if item["has_mask"]],
            [],
        )
        union = union_mask_for_target_ids(target_runs, target_ids)
        structured_result.update({
            "answer": ensure_seg_token(
                response.get("answer", ""), union is not None
            ),
            "target_ids": target_ids,
            "predicted_mask": (
                None if union is None else encode_binary_mask(union)
            ),
        })
    else:
        valid_ids = [
            item["target_id"] for item in target_payloads if item["has_mask"]
        ]
        grounded_report = []
        for sentence in response.get("report", []):
            if not isinstance(sentence, dict):
                continue
            target_ids = normalize_response_target_ids(
                sentence.get("target_ids"), valid_ids, []
            )
            union = union_mask_for_target_ids(target_runs, target_ids)
            grounded_report.append({
                "section": str(sentence.get("section") or "Findings"),
                "text": str(sentence.get("text") or "").strip(),
                "finding_type": str(
                    sentence.get("finding_type") or "grounded_finding"
                ),
                "target_ids": target_ids,
                "predicted_mask": (
                    None if union is None else encode_binary_mask(union)
                ),
            })
        structured_result["report"] = grounded_report

    result_path.write_text(
        json.dumps(structured_result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    meta = {
        "task_type": task_type,
        "agent_status": plan.get("status", ""),
        "num_planned_targets": len(plan.get("targets", [])),
        "num_targets_with_mask": sum(
            int(run["geometry"]["has_mask"]) for run in target_runs
        ),
        "checkpoint": str(Path(args.checkpoint_path).expanduser().resolve()),
        "config_path": str(Path(args.config_path).expanduser().resolve()),
        "sam3_code_dir": str(Path(args.sam3_code_dir).expanduser().resolve()),
        "min_area": args.min_area,
        "max_area_ratio": args.max_area_ratio,
        "remove_overlap": bool(args.remove_overlap),
        "model_load_info": load_info,
        "task_plan_path": str(plan_path),
        "result_path": str(result_path),
        "combined_overlay_path": str(combined_overlay_path),
        "raw_plan_response_path": str(raw_plan_path),
        "raw_final_response_path": str(raw_final_path),
        "run_dir": str(run_dir),
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "run_dir": str(run_dir),
        "result": str(result_path),
        "plan": str(plan_path),
        "meta": str(meta_path),
        "combined_overlay": str(combined_overlay_path),
    }


# ========================= CLI =========================
def parse_args():
    parser = argparse.ArgumentParser(
        "Single-image US-SAM3 agent inference for segmentation, grounded VQA, and grounded reports"
    )
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument(
        "--instruction",
        "--question",
        dest="instruction",
        required=True,
        help="Natural-language segmentation, grounded VQA, or report instruction",
    )
    parser.add_argument(
        "--task",
        choices=["auto", *TASK_TYPES],
        default="auto",
        help="Force a task type or let the agent infer it",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=4,
        help="Maximum distinct category targets planned for one instruction",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--sam3-code-dir", default=DEFAULT_SAM3_CODE_DIR)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional, e.g. 0. Must be set before model import.")

    parser.add_argument("--categories", default=None, help="Comma-separated valid categories shown to the agent, e.g. 'thyroid nodule,thyroid gland'")
    parser.add_argument("--categories-file", default=None, help="Optional .txt/.json category list shown to the agent")

    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY"),
    )
    parser.add_argument("--api-url", default=os.environ.get("OPENAI_BASE_URL", "https://zyapi.tuluo.top:8888/v1"))
    parser.add_argument("--api-model", default=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--api-timeout", type=int, default=180)
    parser.add_argument("--api-max-tokens", type=int, default=512)
    parser.add_argument(
        "--api-response-max-tokens",
        type=int,
        default=1024,
        help="Token limit for the post-segmentation answer/report composer",
    )
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
    parser.add_argument("--api-fallback-on-error", action="store_true", default=True)
    parser.add_argument("--no-api-fallback-on-error", dest="api_fallback_on_error", action="store_false")
    parser.add_argument("--no-api", action="store_true", help="Skip API agent and use local fallback prompt extraction")

    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--max-area-ratio", type=float, default=1.0)
    parser.add_argument("--remove-overlap", action="store_true", help="Apply SAM3 overlap-removal helper if available")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    for p, label in [
        (args.image, "image"),
        (args.sam3_code_dir, "sam3 code dir"),
        (args.config_path, "config"),
        (args.checkpoint_path, "checkpoint"),
    ]:
        if not os.path.exists(os.path.expanduser(p)):
            raise FileNotFoundError(f"{label} not found: {p}")

    categories = parse_categories(args.categories, args.categories_file)
    print(f"[INFO] Agent category options: {len(categories)}")

    plan, raw_plan = agent_parse_question(
        args.image, args.instruction, categories, args
    )
    print(f"[INFO] Agent status: {plan.get('status')}")
    print(f"[INFO] Task type: {plan.get('task_type')}")
    for target in plan.get("targets", []):
        print(
            f"[INFO] Planned {target['target_id']}: "
            f"category={target['chosen_category_name']!r}, "
            f"prompt={target['sam3_prompt']!r}"
        )

    model, device, load_info = setup_model(
        config_path=os.path.abspath(os.path.expanduser(args.config_path)),
        checkpoint_path=os.path.abspath(os.path.expanduser(args.checkpoint_path)),
        sam3_code_dir=os.path.abspath(os.path.expanduser(args.sam3_code_dir)),
    )

    target_runs = run_planned_targets(
        model=model,
        device=device,
        image_path=args.image,
        plan=plan,
        args=args,
    )
    response, raw_final = compose_grounded_response(
        image_path=args.image,
        instruction=args.instruction,
        plan=plan,
        target_runs=target_runs,
        args=args,
    )
    outputs = save_multitask_outputs(
        image_path=args.image,
        instruction=args.instruction,
        plan=plan,
        raw_plan_response=raw_plan,
        response=response,
        raw_final_response=raw_final,
        target_runs=target_runs,
        args=args,
        load_info=load_info,
    )

    print("\n[DONE] Single-image grounded inference finished.")
    print(f"[DONE] Task:             {plan['task_type']}")
    print(f"[DONE] Result JSON:      {outputs['result']}")
    print(f"[DONE] Combined overlay: {outputs['combined_overlay']}")
    print(f"[DONE] Task plan:        {outputs['plan']}")
    print(f"[DONE] Meta:             {outputs['meta']}")
    print(f"[DONE] Run dir:          {outputs['run_dir']}")


if __name__ == "__main__":
    main()
