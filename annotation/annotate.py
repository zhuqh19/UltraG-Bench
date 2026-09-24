#!/usr/bin/env python3
"""Build grounded multitask JSONL from thyroid ultrasound COCO datasets.

The script scans DATA_ROOT/<dataset>/<split>/_annotations.coco.json and writes
segmentation, grounded VQA, and grounded report annotations. It merges duplicate
thyroid-nodule masks, maps ``thyroid tumor`` to ``thyroid nodule``, preserves a
supplied benign/malignant nodule label as an attribute, and retains the thyroid,
carotid artery, and jugular vein classes from Segthy.

VQA is generated with an OpenAI-compatible multimodal endpoint through the
official OpenAI Python SDK. No urllib request code is used. Segmentation tasks
and reports are deterministic so their claims remain traceable to COCO labels
and mask geometry. Images without retained masks are skipped by default.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from pathlib import Path


CATEGORY_ORDER = ("thyroid nodule", "thyroid", "carotid artery", "jugular vein")

DISPLAY_NAMES = {
    "thyroid nodule": "thyroid nodule",
    "thyroid": "thyroid gland",
    "carotid artery": "carotid artery",
    "jugular vein": "jugular vein",
}

MENTION_PATTERNS = {
    "thyroid nodule": r"\b(?:thyroid\s+)?(?:nodule|nodules|tumou?r|tumou?rs)\b",
    "thyroid": r"\b(?:thyroid gland|whole thyroid|thyroid parenchyma)\b",
    "carotid artery": r"\b(?:carotid|carotid artery|carotid arteries)\b",
    "jugular vein": r"\b(?:jugular|jugular vein|jugular veins)\b",
}

EXPECTED_RAW_CATEGORIES = {
    "ddti_coco": {"thyroid nodule", "benign thyroid nodule", "malignant thyroid nodule"},
    "kfgnet_coco": {"thyroid nodule", "benign thyroid nodule", "malignant thyroid nodule"},
    "segthy_coco": {"thyroid", "carotid artery", "jugular vein"},
    "tg3k_coco": {"thyroid nodule", "thyroid tumor"},
    "thyroid_us_cineclip_coco": {
        "thyroid nodule", "benign thyroid nodule", "malignant thyroid nodule",
    },
    "tn3k_coco": {"thyroid nodule", "thyroid tumor"},
}

FORBIDDEN_USER_TERMS = (
    "annotation",
    "annotated",
    "dataset",
    "mask id",
    "metadata",
    "file name",
    "filename",
    "case number",
    "slice number",
    "image id",
    "train split",
    "test split",
)

NON_CLINICAL_QUESTION_TERMS = (
    "file",
    "filename",
    "file name",
    "path",
    "case id",
    "case number",
    "slice id",
    "slice number",
    "image id",
    "dataset",
    "source",
    "train",
    "test",
    "date",
)

UNSUPPORTED_CLINICAL_RE = re.compile(
    r"\b(?:cancer|carcinoma|metasta\w*|papillary carcinoma|follicular carcinoma|medullary carcinoma|"
    r"anaplastic|lymphoma|tirads|ti-rads|acr|bethesda|biopsy|cytolog\w*|histolog\w*|"
    r"lymph node|extrathyroidal extension|invasion|stage|staging|diagnos\w*|treat\w*|"
    r"surgery|ablation|follow[- ]?up|recommend\w*|consistent with|suggestive of|"
    r"compatible with|indicative of)\b",
    re.IGNORECASE,
)

LOW_VALUE_QUESTION_PATTERNS = (
    r"\bfile\b",
    r"\bcase\b",
    r"\bslice\b",
    r"\bdataset\b",
    r"\bsource\b",
    r"\benough (?:information|evidence).*(?:diagnos|cancer|malignan|benign|patholog|ti-rads)",
    r"\b(?:determine|infer|diagnose).*(?:diagnos|cancer|malignan|benign|patholog|ti-rads)",
    r"\b(?:cancer|diagnos|malignan|benign|patholog|ti-rads).*(?:determine|infer|diagnose|evidence)",
    r"\b(?:most\s+)?consistent with\b",
    r"\b(?:suggestive|compatible|indicative) of\b",
    r"\b(left|right|bilateral)\s+(?:thyroid|lobe|nodule|carotid|jugular)\b",
    r"\b\d+(?:\.\d+)?\s*(?:mm|cm|ml|cc)\b",
)

_CLIENT_LOCAL = threading.local()


def slug(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_") or "dataset"


def location_grid(cx, cy, width, height):
    columns = ("left", "central", "right")
    rows = ("upper", "middle", "lower")
    xi = min(2, max(0, int(cx / max(width, 1) * 3)))
    yi = min(2, max(0, int(cy / max(height, 1) * 3)))
    return f"{rows[yi]}-{columns[xi]}"


def canonical_region(category):
    text = re.sub(r"[^a-z]+", " ", str(category or "").lower()).strip()
    aliases = {
        "thyroid nodule": "thyroid nodule",
        "benign thyroid nodule": "thyroid nodule",
        "malignant thyroid nodule": "thyroid nodule",
        "thyroid tumor": "thyroid nodule",
        "thyroid tumour": "thyroid nodule",
        "thyroid": "thyroid",
        "thyroid gland": "thyroid",
        "carotid": "carotid artery",
        "carotid artery": "carotid artery",
        "jugular": "jugular vein",
        "jugular vein": "jugular vein",
    }
    return aliases.get(text, text)


def pathology_label(category):
    text = str(category or "").strip().lower()
    if text == "benign thyroid nodule":
        return "benign"
    if text == "malignant thyroid nodule":
        return "malignant"
    return "unspecified"


def label_priority(category):
    text = str(category or "").strip().lower()
    if text in {"benign thyroid nodule", "malignant thyroid nodule"}:
        return 3
    if text == "thyroid nodule":
        return 2
    if text in {"thyroid tumor", "thyroid tumour"}:
        return 1
    return 2


def geometry_key(mask):
    return tuple(mask["bbox_xywh"]) + (mask["mask_area_px"],)


def canonical_mask(dataset_id, split, image, ann, category):
    bbox = ann.get("bbox") or [0, 0, 0, 0]
    if len(bbox) != 4:
        bbox = [0, 0, 0, 0]
    x, y, width, height = [float(value) for value in bbox]
    image_width = int(image.get("width") or 0)
    image_height = int(image.get("height") or 0)
    bbox_area = max(0.0, width * height)
    area = float(ann.get("area", bbox_area) or 0)
    region = canonical_region(category)
    return {
        "mask_id": f"{dataset_id}_{split}_img{image['id']}_ann{ann['id']}",
        "annotation_id": ann["id"],
        "category": region,
        "source_category": str(category),
        "canonical_region": region,
        "target_type": (
            "nodule" if region == "thyroid nodule"
            else "organ" if region == "thyroid"
            else "vessel"
        ),
        "pathology_label": pathology_label(category) if region == "thyroid nodule" else "not_applicable",
        "source_annotation_ids": [ann["id"]],
        "source_categories": [str(category)],
        "bbox_xywh": [round(x, 2), round(y, 2), round(width, 2), round(height, 2)],
        "bbox_area_px": round(bbox_area, 2),
        "mask_area_px": round(area, 2),
        "image_area_px": image_width * image_height,
        "area_ratio": round(area / max(image_width * image_height, 1), 6),
        "long_axis_px": round(max(width, height), 2),
        "short_axis_px": round(min(width, height), 2),
        "location_grid": location_grid(x + width / 2, y + height / 2, image_width, image_height),
    }


def build_facts(dataset_name, split, image, annotations, categories, image_path):
    dataset_id = slug(dataset_name)
    grouped_masks = {}
    merged_duplicate_count = 0
    ignored_unknown_categories = []
    pathology_conflicts = []
    expected = EXPECTED_RAW_CATEGORIES.get(dataset_name.lower())
    for ann in annotations:
        if ann.get("iscrowd", 0):
            continue
        raw_category = categories.get(ann.get("category_id"), str(ann.get("category_id")))
        raw_normalized = str(raw_category).strip().lower()
        category = canonical_region(raw_category)
        if category not in CATEGORY_ORDER or (expected is not None and raw_normalized not in expected):
            ignored_unknown_categories.append(str(raw_category))
            continue
        candidate = canonical_mask(dataset_id, split, image, ann, raw_category)
        key = (category,) + geometry_key(candidate)
        existing = grouped_masks.get(key)
        if existing is None:
            grouped_masks[key] = candidate
            continue
        merged_duplicate_count += 1
        source_ids = existing["source_annotation_ids"] + candidate["source_annotation_ids"]
        source_categories = existing["source_categories"] + candidate["source_categories"]
        pathologies = {
            label for label in (existing["pathology_label"], candidate["pathology_label"])
            if label not in {"unspecified", "not_applicable"}
        }
        if label_priority(raw_category) > label_priority(existing["source_category"]):
            candidate["source_annotation_ids"] = source_ids
            candidate["source_categories"] = source_categories
            grouped_masks[key] = candidate
            existing = candidate
        else:
            existing["source_annotation_ids"] = source_ids
            existing["source_categories"] = source_categories
        if len(pathologies) > 1:
            existing["pathology_label"] = "conflicting"
            existing["pathology_label_conflict"] = sorted(pathologies)
            pathology_conflicts.append({
                "geometry": list(key[1:]),
                "labels": sorted(pathologies),
            })

    masks = list(grouped_masks.values())
    masks.sort(key=lambda mask: (CATEGORY_ORDER.index(mask["canonical_region"]), mask["annotation_id"]))

    category_evidence = defaultdict(list)
    category_geometry = defaultdict(set)
    pathology_evidence = defaultdict(list)
    for mask in masks:
        category_evidence[mask["canonical_region"]].append(mask["mask_id"])
        category_geometry[mask["canonical_region"]].add(geometry_key(mask))
        if mask["pathology_label"] in {"benign", "malignant"}:
            pathology_evidence[mask["pathology_label"]].append(mask["mask_id"])
    present_categories = [category for category in CATEGORY_ORDER if category_evidence.get(category)]
    return {
        "dataset": dataset_name,
        "split": split,
        "image_id": f"{dataset_id}_{split}_{image['id']}",
        "source_image_id": image["id"],
        "file_name": image.get("file_name"),
        "image_path": str(image_path),
        "width": int(image.get("width") or 0),
        "height": int(image.get("height") or 0),
        "modality": "thyroid ultrasound",
        "organ": "thyroid",
        "masks": masks,
        "has_thyroid_targets": bool(masks),
        "present_categories": present_categories,
        "present_pathology_labels": [
            label for label in ("benign", "malignant") if pathology_evidence.get(label)
        ],
        "primary_evidence_mask_ids": [mask["mask_id"] for mask in masks],
        "category_evidence_mask_ids": dict(category_evidence),
        "pathology_evidence_mask_ids": dict(pathology_evidence),
        "category_target_counts": {
            category: len(category_geometry.get(category, set())) for category in CATEGORY_ORDER
        },
        "merged_duplicate_annotations": merged_duplicate_count,
        "pathology_label_conflicts": pathology_conflicts,
        "ignored_unknown_categories": sorted(set(ignored_unknown_categories)),
        "notes": [
            "Use the image only for directly visible appearance and supplied masks for category and geometry.",
            "Thyroid tumor is normalized to thyroid nodule and duplicate nodule geometries are merged.",
            "A benign or malignant label may be stated only when it is supplied by the COCO category.",
            "Do not infer TI-RADS, histology, cytology, cancer subtype, invasion, stage, treatment, or prognosis.",
            "Do not infer patient left/right laterality, physical size, Doppler flow, or whole-exam findings.",
            "Do not ask about file names, case/slice numbers, image IDs, datasets, or splits.",
        ],
    }


def prompt_facts(facts):
    excluded = {"dataset", "split", "source_image_id", "file_name", "image_path"}
    return {key: value for key, value in facts.items() if key not in excluded}


def make_prompt(facts):
    category_text = ", ".join(CATEGORY_ORDER)
    return (
        "Create grounded VQA for one thyroid ultrasound image. Return valid JSON only.\n"
        f"The only permitted canonical categories are: {category_text}. Treat thyroid tumor as exactly the same "
        "segmentation category as thyroid nodule; never output thyroid tumor as a separate category. Preserve the "
        "thyroid gland, carotid artery, and jugular vein as distinct anatomy classes.\n"
        "Use the attached image for directly visible appearance and supplied mask facts for category, localization, "
        "count, and pixel geometry. A benign or malignant nodule label may be used only when that exact supplied "
        "pathology_label is attached to the cited mask; never infer it from appearance. Do not invent thyroid cancer, "
        "histologic or cytologic subtype, TI-RADS/ACR score, Bethesda class, lymph-node disease, extrathyroidal "
        "extension, invasion, stage, pathology, prognosis, treatment, or recommendations.\n"
        "Do not infer patient left/right laterality, thyroid lobe, physical dimensions, Doppler flow, temporal behavior, "
        "or whole-exam conclusions. Never ask whether cancer, a diagnosis, or benign/malignant status can be determined. "
        "Do not use diagnostic phrases such as 'consistent with', 'suggestive of', 'compatible with', or 'indicative of', "
        "even when only asking about visible shape or appearance; ask directly what is visible instead.\n"
        "Never ask about or mention file names, case/slice numbers, image IDs, dataset/source names, split names, "
        "paths, dates, annotation, annotated, mask id, or metadata in user-facing text.\n"
        "Every question must be relevant to visible thyroid ultrasound targets: supplied class or pathology label, "
        "presence, image-region location, contour, shape, visible echogenicity, pixel extent, area ratio, target count, "
        "or boundary availability. Do not ask about an unlabeled structure.\n"
        "Every positive category, pathology, appearance, or geometry answer must set requires_grounding=true and cite only matching "
        "evidence_mask_ids. Use evidence_source=mask_geometry for category/location/count/extent and "
        "evidence_source=image_and_mask for directly visible appearance. Mask IDs may appear only in evidence_mask_ids.\n\n"
        "Return exactly one top-level key: grounded_vqa. It must contain 4-6 objects with qa_id, question, answer, "
        "answer_type, requires_grounding, evidence_mask_ids, reasoning_type, and evidence_source. Do not return "
        "segmentation tasks, reports, Markdown, or explanations.\n\nFacts:\n"
        + json.dumps(prompt_facts(facts), ensure_ascii=False)
    )


def get_openai_client(api_key, base_url, timeout):
    cache_key = (api_key, base_url.rstrip("/"), timeout)
    if getattr(_CLIENT_LOCAL, "cache_key", None) == cache_key:
        return _CLIENT_LOCAL.client
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required: pip install -U openai") from exc
    _CLIENT_LOCAL.cache_key = cache_key
    _CLIENT_LOCAL.client = OpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        timeout=timeout,
        max_retries=0,
    )
    return _CLIENT_LOCAL.client


def retry_delay(exc, attempt, base_seconds, max_seconds):
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                return min(max_seconds, max(base_seconds, float(retry_after)))
            except (TypeError, ValueError):
                pass
    exponential = min(max_seconds, base_seconds * (2 ** attempt))
    return exponential + random.uniform(0, min(1.0, exponential * 0.1))


def image_data_url(image_path):
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    if suffix in mime_types:
        payload = path.read_bytes()
        mime_type = mime_types[suffix]
    else:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                f"Pillow is required to convert {suffix or '<no extension>'}: pip install Pillow"
            ) from exc
        with Image.open(path) as image:
            converted = image.convert("RGB")
            buffer = io.BytesIO()
            converted.save(buffer, format="PNG")
        payload = buffer.getvalue()
        mime_type = "image/png"
    return f"data:{mime_type};base64,{base64.b64encode(payload).decode('ascii')}"


def api_call(api_key, base_url, model, prompt, image_path, timeout, retries, temperature,
             retry_base_seconds, retry_max_seconds):
    client = get_openai_client(api_key, base_url, timeout)
    last_error = "unknown API error"
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                    ],
                }],
                temperature=temperature,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("API returned empty message content")
            return content
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            last_error = f"{type(exc).__name__}: {exc}"
            retryable = status_code is None or status_code == 429 or status_code >= 500
            if attempt >= retries or not retryable:
                break
            delay = retry_delay(exc, attempt, retry_base_seconds, retry_max_seconds)
            print(f"[api retry {attempt + 1}/{retries}] {last_error}; waiting {delay:.1f}s", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"API call failed: {last_error}")


def extract_json(text):
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def clean_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def mentioned_categories(text):
    return {
        category for category, pattern in MENTION_PATTERNS.items()
        if re.search(pattern, str(text or ""), flags=re.IGNORECASE)
    }


def mentioned_pathologies(text):
    lowered = str(text or "").lower()
    found = set()
    if re.search(r"\bbenign\b", lowered):
        found.add("benign")
    if re.search(r"\bmalignan\w*\b", lowered):
        found.add("malignant")
    return found


def is_high_value_question(question, answer=""):
    combined = f"{question} {answer}"
    lowered = combined.lower()
    if any(term in lowered for term in FORBIDDEN_USER_TERMS + NON_CLINICAL_QUESTION_TERMS):
        return False
    if UNSUPPORTED_CLINICAL_RE.search(combined):
        return False
    if any(re.search(pattern, combined, flags=re.IGNORECASE) for pattern in LOW_VALUE_QUESTION_PATTERNS):
        return False
    useful = (
        "visible", "present", "located", "location", "region", "extent", "pixel", "area",
        "occup", "boundary", "margin", "contour", "shape", "echotexture", "echogenicity",
        "composition", "segment", "target", "single", "multiple", "how many", "count",
        "category", "structure", "label", "benign", "malignant",
    )
    return bool(mentioned_categories(combined)) and any(term in lowered for term in useful)


def category_masks(facts, category):
    return [mask for mask in facts["masks"] if mask["canonical_region"] == category]


def category_evidence(facts, category):
    return list(facts.get("category_evidence_mask_ids", {}).get(category, []))


def format_locations(masks):
    locations = []
    for mask in masks:
        if mask["location_grid"] not in locations:
            locations.append(mask["location_grid"])
    if len(locations) == 1:
        return f"the {locations[0]} image region"
    if len(locations) == 2:
        return f"the {locations[0]} and {locations[1]} image regions"
    return "the " + ", ".join(locations[:-1]) + f", and {locations[-1]} image regions"


def deterministic_segmentation(facts):
    sid = facts["image_id"]
    all_ids = facts["primary_evidence_mask_ids"]
    if not all_ids:
        return [{
            "task_id": sid + "_seg_1",
            "instruction": "Segment the labeled thyroid ultrasound targets.",
            "answer": "No labeled thyroid ultrasound target is available.",
            "target_mask_ids": [],
            "answerability": "no_target",
        }]
    tasks = [{
        "instruction": "Segment all labeled thyroid ultrasound targets.",
        "answer": "They are [SEG]. Labeled thyroid ultrasound targets.",
        "target_mask_ids": all_ids,
    }]
    for category in facts["present_categories"]:
        name = DISPLAY_NAMES[category]
        refs = category_evidence(facts, category)
        tasks.append({
            "instruction": f"Segment the visible {name} target{'s' if len(refs) != 1 else ''}.",
            "answer": f"{'They are' if len(refs) != 1 else 'It is'} [SEG]. Visible {name} target{'s' if len(refs) != 1 else ''}.",
            "target_mask_ids": refs,
        })
    return [
        dict(task, task_id=f"{sid}_seg_{index}", answerability="answerable")
        for index, task in enumerate(tasks, 1)
    ]


def report_category_sentence(facts, category):
    masks = category_masks(facts, category)
    count = facts["category_target_counts"][category]
    name = DISPLAY_NAMES[category]
    noun = "target" if count == 1 else "targets"
    verb = "is" if count == 1 else "are"
    return {
        "section": "Findings",
        "text": f"{count} visible {name} {noun} {verb} centered in {format_locations(masks)}.",
        "evidence_mask_ids": category_evidence(facts, category),
        "finding_type": f"{category}_presence_location",
        "evidence_source": "mask_geometry",
    }


def pathology_report_sentences(facts):
    sentences = []
    for label in facts["present_pathology_labels"]:
        refs = list(facts["pathology_evidence_mask_ids"].get(label, []))
        count = len(refs)
        sentences.append({
            "section": "Findings",
            "text": (
                f"The supplied category labels this thyroid nodule as {label}."
                if count == 1 else
                f"The supplied categories label {count} thyroid nodule targets as {label}."
            ),
            "evidence_mask_ids": refs,
            "finding_type": f"supplied_{label}_nodule_label",
            "evidence_source": "mask_geometry",
        })
    return sentences


def deterministic_report(facts):
    all_ids = facts["primary_evidence_mask_ids"]
    if not all_ids:
        sentences = [
            {"section": "Findings", "text": "No labeled thyroid ultrasound target is available in this image.",
             "evidence_mask_ids": [], "finding_type": "absence", "evidence_source": "none"},
            {"section": "Assessment", "text": "No labeled thyroid ultrasound target is available for spatial assessment.",
             "evidence_mask_ids": [], "finding_type": "no_target", "evidence_source": "none"},
            {"section": "Impression", "text": "No thyroid ultrasound target is available for segmentation.",
             "evidence_mask_ids": [], "finding_type": "no_target_impression", "evidence_source": "none"},
        ]
    else:
        sentences = [report_category_sentence(facts, category) for category in facts["present_categories"]]
        sentences.extend(pathology_report_sentences(facts))
        largest = max(facts["masks"], key=lambda mask: mask["mask_area_px"])
        if len(sentences) < 4:
            sentences.append({
                "section": "Findings",
                "text": (
                    f"The largest labeled target is the {DISPLAY_NAMES[largest['canonical_region']]} with a "
                    f"pixel extent of approximately {largest['long_axis_px']} x {largest['short_axis_px']} pixels."
                ),
                "evidence_mask_ids": [largest["mask_id"]],
                "finding_type": "largest_target_extent",
                "evidence_source": "mask_geometry",
            })
        names = [DISPLAY_NAMES[category] for category in facts["present_categories"]]
        name_text = ", ".join(names)
        sentences.extend([
            {
                "section": "Assessment",
                "text": f"The labeled thyroid ultrasound targets in this image are: {name_text}.",
                "evidence_mask_ids": all_ids,
                "finding_type": "anatomic_summary",
                "evidence_source": "mask_geometry",
            },
            {
                "section": "Impression",
                "text": f"Visible thyroid ultrasound targets available for segmentation: {name_text}.",
                "evidence_mask_ids": all_ids,
                "finding_type": "segmentation_impression",
                "evidence_source": "mask_geometry",
            },
        ])
    return {"prompt": "Generate a structured thyroid ultrasound description.", "sentences": sentences}


def fallback_vqa(facts):
    sid = facts["image_id"]
    if not facts["masks"]:
        items = [
            ("Is a labeled thyroid ultrasound target available?", "No labeled thyroid ultrasound target is available.", "yes_no", False, [], "absence", "none"),
            ("Is a thyroid ultrasound target boundary available for segmentation?", "No labeled boundary is available.", "yes_no", False, [], "absence", "none"),
            ("Can a target location be provided?", "No target is available for localization.", "location", False, [], "not_applicable", "none"),
            ("Can a target extent be provided?", "No target is available for extent measurement.", "size", False, [], "not_applicable", "none"),
        ]
    else:
        categories = facts["present_categories"]
        names = ", ".join(DISPLAY_NAMES[category] for category in categories)
        all_ids = facts["primary_evidence_mask_ids"]
        items = [
            ("Which labeled thyroid ultrasound targets are visible?", f"The visible labeled targets are: {names}.", "category", True, all_ids, "recognition", "mask_geometry"),
        ]
        for category in categories[:3]:
            masks = category_masks(facts, category)
            refs = category_evidence(facts, category)
            name = DISPLAY_NAMES[category]
            items.append((
                f"Where is the {name} located in the image?",
                f"The {name} is centered in {format_locations(masks)}.",
                "location", True, refs, "location", "mask_geometry",
            ))
        items.append((
            "Are boundaries available for the labeled thyroid ultrasound targets?",
            "Yes, boundaries are available for the labeled structures.",
            "yes_no", True, all_ids, "segmentation_target", "mask_geometry",
        ))
        for label in facts["present_pathology_labels"]:
            refs = list(facts["pathology_evidence_mask_ids"].get(label, []))
            items.append((
                f"How many thyroid nodule targets carry the supplied {label} label?",
                f"There {'is' if len(refs) == 1 else 'are'} {len(refs)} thyroid nodule "
                f"{'target' if len(refs) == 1 else 'targets'} with the supplied {label} label.",
                "classification", True, refs, "label_lookup", "mask_geometry",
            ))
        if len(items) < 6:
            total = sum(facts["category_target_counts"].values())
            items.append((
                "How many distinct labeled thyroid ultrasound targets are visible?",
                (
                    "There is one distinct labeled thyroid ultrasound target."
                    if total == 1 else
                    f"There are {total} distinct labeled thyroid ultrasound targets."
                ),
                "count", True, all_ids, "count", "mask_geometry",
            ))
    return [
        {
            "qa_id": f"{sid}_vqa_{index}", "question": question, "answer": answer,
            "answer_type": answer_type, "requires_grounding": grounded,
            "evidence_mask_ids": refs, "reasoning_type": reasoning, "evidence_source": source,
        }
        for index, (question, answer, answer_type, grounded, refs, reasoning, source)
        in enumerate(items[:6], 1)
    ]


def fallback_annotation(facts):
    return {
        "segmentation_tasks": deterministic_segmentation(facts),
        "grounded_vqa": fallback_vqa(facts),
        "grounded_report": deterministic_report(facts),
    }


def refs_support_mentions(refs, text, facts):
    mentions = mentioned_categories(text)
    if not mentions:
        return False
    id_to_category = {mask["mask_id"]: mask["canonical_region"] for mask in facts["masks"]}
    id_to_pathology = {mask["mask_id"]: mask.get("pathology_label") for mask in facts["masks"]}
    referenced_categories = {id_to_category.get(ref) for ref in refs}
    pathologies = mentioned_pathologies(text)
    referenced_pathologies = {id_to_pathology.get(ref) for ref in refs}
    return mentions.issubset(referenced_categories) and pathologies.issubset(referenced_pathologies)


def normalize_annotation(obj, facts):
    if not isinstance(obj, dict):
        raise ValueError("API output is not a JSON object")
    allowed = set(facts["primary_evidence_mask_ids"])
    sample_id = facts["image_id"]
    vqa = obj.get("grounded_vqa")
    if not isinstance(vqa, list) or not 4 <= len(vqa) <= 6:
        vqa = fallback_vqa(facts)
    cleaned = []
    seen_questions = set()
    for item in vqa:
        if not isinstance(item, dict):
            continue
        question = clean_text(item.get("question"))
        answer = clean_text(item.get("answer"))
        refs = item.get("evidence_mask_ids", [])
        if (
            not question or not answer or not is_high_value_question(question, answer)
            or not isinstance(refs, list) or not refs or any(ref not in allowed for ref in refs)
            or not refs_support_mentions(refs, f"{question} {answer}", facts)
        ):
            continue
        normalized_question = question.lower()
        if normalized_question in seen_questions:
            continue
        seen_questions.add(normalized_question)
        requested_source = str(item.get("evidence_source", "")).strip().lower()
        source = requested_source if requested_source in {"mask_geometry", "image_and_mask"} else "mask_geometry"
        cleaned.append({
            "qa_id": f"{sample_id}_vqa_{len(cleaned) + 1}",
            "question": question,
            "answer": answer,
            "answer_type": str(item.get("answer_type", "open")),
            "requires_grounding": True,
            "evidence_mask_ids": refs,
            "reasoning_type": str(item.get("reasoning_type", "visual_reasoning")),
            "evidence_source": source,
        })
    return {
        "segmentation_tasks": deterministic_segmentation(facts),
        "grounded_vqa": cleaned if 4 <= len(cleaned) <= 6 else fallback_vqa(facts),
        "grounded_report": deterministic_report(facts),
    }


def find_coco_file(split_dir, annotation_name):
    preferred = split_dir / annotation_name
    if preferred.exists():
        return preferred
    candidates = sorted(split_dir.glob("*.json"))
    return candidates[0] if len(candidates) == 1 else None


def iter_facts(data_root, datasets, splits, annotation_name, limit):
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    dataset_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if datasets:
        requested = set(datasets)
        dataset_dirs = [path for path in dataset_dirs if path.name in requested]
        missing = requested - {path.name for path in dataset_dirs}
        if missing:
            raise FileNotFoundError(f"Dataset directories not found: {sorted(missing)}")
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset directories found under: {root}")
    found_split = False
    for dataset_dir in dataset_dirs:
        for split in splits:
            split_dir = dataset_dir / split
            if not split_dir.is_dir():
                continue
            coco_path = find_coco_file(split_dir, annotation_name)
            if coco_path is None:
                print(f"[skip] no unambiguous COCO JSON: {split_dir}", flush=True)
                continue
            found_split = True
            data = json.loads(coco_path.read_text(encoding="utf-8-sig"))
            categories = {cat["id"]: cat.get("name", str(cat["id"])) for cat in data.get("categories", [])}
            by_image = defaultdict(list)
            for ann in data.get("annotations", []):
                by_image[ann.get("image_id")].append(ann)
            images = sorted(data.get("images", []), key=lambda item: str(item.get("id")))
            if limit is not None:
                images = images[:limit]
            for image in images:
                image_path = split_dir / str(image.get("file_name") or "")
                if not image_path.is_file():
                    raise FileNotFoundError(f"Image file not found for COCO image {image.get('id')}: {image_path}")
                yield build_facts(
                    dataset_dir.name, split, image, by_image.get(image["id"], []), categories, image_path
                )
    if not found_split:
        raise FileNotFoundError(f"No requested COCO splits found under: {root}")


def existing_vqa_needs_regeneration(record):
    masks = record.get("masks")
    if not isinstance(masks, list) or not masks:
        return False
    annotation = record.get("annotation")
    vqa = annotation.get("grounded_vqa") if isinstance(annotation, dict) else None
    if not isinstance(vqa, list) or not 4 <= len(vqa) <= 6:
        return True
    facts = {"masks": masks}
    allowed = {mask.get("mask_id") for mask in masks if isinstance(mask, dict)}
    seen_questions = set()
    for item in vqa:
        if not isinstance(item, dict):
            return True
        question = clean_text(item.get("question"))
        answer = clean_text(item.get("answer"))
        refs = item.get("evidence_mask_ids")
        normalized_question = question.lower()
        combined = f"{question} {answer}"
        lowered = combined.lower()
        has_forbidden_wording = (
            any(term in lowered for term in FORBIDDEN_USER_TERMS + NON_CLINICAL_QUESTION_TERMS)
            or UNSUPPORTED_CLINICAL_RE.search(combined) is not None
            or any(
                re.search(pattern, combined, flags=re.IGNORECASE)
                for pattern in LOW_VALUE_QUESTION_PATTERNS
            )
        )
        has_specific_mentions = bool(mentioned_categories(combined))
        if (
            not question
            or not answer
            or normalized_question in seen_questions
            or has_forbidden_wording
            or not isinstance(refs, list)
            or not refs
            or any(ref not in allowed for ref in refs)
            or (has_specific_mentions and not refs_support_mentions(refs, combined, facts))
        ):
            return True
        seen_questions.add(normalized_question)
    return False


def load_existing(path, retry_errors, include_empty):
    path = Path(path)
    if not path.exists():
        return set(), 0, 0
    kept = []
    done = set()
    dropped_empty = 0
    dropped_invalid_vqa = 0
    changed = False
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid existing JSONL at line {line_no}: {exc}") from exc
        masks = record.get("masks")
        if not include_empty and (not isinstance(masks, list) or not masks):
            dropped_empty += 1
            changed = True
            continue
        has_error = bool(record.get("annotation", {}).get("api_error"))
        if retry_errors and has_error:
            changed = True
            continue
        if existing_vqa_needs_regeneration(record):
            dropped_invalid_vqa += 1
            changed = True
            continue
        kept.append(record)
        done.add(record.get("sample_id"))
    if changed:
        temp = path.with_suffix(path.suffix + ".resume.tmp")
        temp.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in kept), encoding="utf-8")
        temp.replace(path)
    return done, dropped_empty, dropped_invalid_vqa


def annotate_one(facts, args):
    source = "fallback"
    try:
        if args.no_api:
            annotation = fallback_annotation(facts)
        else:
            if args.request_delay > 0:
                time.sleep(args.request_delay)
            for format_attempt in range(args.format_retries + 1):
                raw = api_call(
                    args.api_key, args.base_url, args.model, make_prompt(facts), facts["image_path"],
                    args.timeout, args.retries, args.temperature,
                    args.retry_base_seconds, args.retry_max_seconds,
                )
                try:
                    annotation = normalize_annotation(extract_json(raw), facts)
                    source = "api"
                    break
                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    if format_attempt >= args.format_retries:
                        raise
                    delay = min(args.retry_max_seconds, args.retry_base_seconds * (2 ** format_attempt))
                    print(
                        f"[format retry {format_attempt + 1}/{args.format_retries}] "
                        f"{type(exc).__name__}: {exc}; waiting {delay:.1f}s",
                        flush=True,
                    )
                    time.sleep(delay)
    except Exception as exc:
        annotation = fallback_annotation(facts)
        annotation["api_error"] = f"{type(exc).__name__}: {exc}"
    return {
        "sample_id": facts["image_id"],
        "dataset": facts["dataset"],
        "split": facts["split"],
        "image_path": facts["image_path"],
        "image": {
            "file_name": facts["file_name"], "width": facts["width"], "height": facts["height"],
            "organ": facts["organ"], "modality": facts["modality"],
        },
        "masks": facts["masks"],
        "merged_duplicate_annotations": facts["merged_duplicate_annotations"],
        "pathology_label_conflicts": facts["pathology_label_conflicts"],
        "ignored_unknown_categories": facts["ignored_unknown_categories"],
        "annotation_source": source,
        "annotation": annotation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--datasets", nargs="*", help="Dataset directory names; default scans all directories")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--annotation-name", default="_annotations.coco.json")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY"),
    )
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://zyapi.tuluo.top:8888/v1"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--format-retries", type=int, default=2)
    parser.add_argument("--retry-base-seconds", type=float, default=5.0)
    parser.add_argument("--retry-max-seconds", type=float, default=60.0)
    parser.add_argument("--request-delay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--limit-per-dataset-split", type=int)
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Keep images without retained masks as negative samples; default skips them",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--no-api", action="store_true", help="Generate deterministic fallback annotations only")
    args = parser.parse_args()

    if not args.no_api and not all((args.api_key, args.base_url, args.model)):
        parser.error("Set OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL or pass matching arguments")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    done, dropped_existing_empty, dropped_invalid_vqa = (
        load_existing(output, args.retry_errors, args.include_empty)
        if args.resume else (set(), 0, 0)
    )
    if output.exists() and not args.resume:
        output.unlink()
    candidate_facts = [
        facts for facts in iter_facts(
            args.data_root, args.datasets, args.splits, args.annotation_name, args.limit_per_dataset_split
        ) if facts["image_id"] not in done
    ]
    skipped_empty = sum(not facts["masks"] for facts in candidate_facts)
    facts_list = (
        candidate_facts if args.include_empty
        else [facts for facts in candidate_facts if facts["masks"]]
    )
    total = len(facts_list)
    print(json.dumps({
        "pending": total,
        "already_done": len(done),
        "skipped_empty": 0 if args.include_empty else skipped_empty,
        "dropped_existing_empty": dropped_existing_empty,
        "dropped_invalid_vqa": dropped_invalid_vqa,
        "output": str(output),
    }, ensure_ascii=False), flush=True)

    completed = api_ok = fallback = 0
    with output.open("a", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            for record in executor.map(lambda facts: annotate_one(facts, args), facts_list):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                completed += 1
                if record["annotation_source"] == "api":
                    api_ok += 1
                else:
                    fallback += 1
                remaining = total - completed
                error = str(record.get("annotation", {}).get("api_error", "")).replace("\n", " ")
                suffix = f" error={error[:240]}" if error else ""
                print(
                    f"[{completed}/{total} remaining={remaining}] {record['sample_id']} "
                    f"source={record['annotation_source']}{suffix}",
                    flush=True,
                )
    print(json.dumps({
        "completed": completed, "api_ok": api_ok, "fallback": fallback,
        "already_done": len(done), "out": str(output),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
