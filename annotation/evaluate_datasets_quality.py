#!/usr/bin/env python3
"""Rule-based quality audit for grounded thyroid ultrasound JSONL.

The audit checks COCO-derived mask integrity, nodule/tumor normalization,
duplicate-geometry merging, benign/malignant label grounding, category/evidence
alignment, segmentation/VQA/report schemas, and unsupported clinical claims.
It writes per-record and per-issue CSV files plus Markdown and JSON summaries.
It does not replace expert review of image-label or polygon correctness.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


CATEGORY_ORDER = ("thyroid nodule", "thyroid", "carotid artery", "jugular vein")
ALLOWED_CATEGORIES = set(CATEGORY_ORDER)
REPORT_REQUIRED_SECTIONS = {"Findings", "Impression"}
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

MENTION_PATTERNS = {
    "thyroid nodule": r"\b(?:thyroid\s+)?(?:nodule|nodules|tumou?r|tumou?rs)\b",
    "thyroid": r"\b(?:thyroid gland|whole thyroid|thyroid parenchyma)\b",
    "carotid artery": r"\b(?:carotid|carotid artery|carotid arteries)\b",
    "jugular vein": r"\b(?:jugular|jugular vein|jugular veins)\b",
}

FORBIDDEN_USER_TERMS = (
    "annotation", "annotated", "dataset", "mask id", "metadata",
    "file name", "filename", "case number", "slice number", "image id",
    "train split", "test split",
)

NON_CLINICAL_QUESTION_TERMS = (
    "file", "filename", "file name", "path", "case id", "case number",
    "slice id", "slice number", "image id", "dataset", "source",
    "train", "test", "date",
)

GROUNDING_CUES = (
    "visible", "present", "labeled", "located", "centered", "region",
    "extent", "pixel", "area", "occup", "boundary", "contour", "target",
    "segmentation", "structure", "benign", "malignant",
)

SUPPORTED_REPORT_CUES = GROUNDING_CUES + (
    "thyroid", "nodule", "carotid", "jugular", "gland", "anatomy",
)

DIAGNOSTIC_OVERREACH_PATTERNS = (
    (r"\b(cancer|carcinoma|metasta\w*|papillary carcinoma|follicular carcinoma|medullary carcinoma|anaplastic|lymphoma)\b", "unsupported thyroid cancer or subtype claim"),
    (r"\b(ti[- ]?rads|tirads|acr category|bethesda|cytolog\w*|histolog\w*)\b", "unsupported thyroid risk score or pathology claim"),
    (r"\b(lymph node|adenopathy|extrathyroidal extension|capsular invasion|vascular invasion|stage|staging)\b", "unsupported extension or staging claim"),
    (r"\b(thyroiditis|goiter|graves|hashimoto|hyperthyroid\w*|hypothyroid\w*)\b", "unsupported thyroid diagnosis"),
    (r"\b(normal|abnormal|unremarkable|healthy|diseased)\b", "unsupported normality or abnormality assessment"),
    (r"\b(diagnos\w*|patholog\w*|etiolog\w*|consistent with|suggestive of|rules out)\b", "unsupported diagnostic interpretation"),
    (r"\b(treat\w*|therapy|surgery|ablation|recommend\w*|follow[- ]?up|biopsy)\b", "unsupported management recommendation"),
)

PHYSICAL_MEASUREMENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(mm|cm|ml|cc)\b", re.IGNORECASE)
ANATOMIC_LATERALITY_RE = re.compile(
    r"\b(left|right|bilateral)\s+(thyroid|thyroid lobe|lobe|nodule|carotid|jugular)\b",
    re.IGNORECASE,
)
UNSUPPORTED_SUBSTRUCTURE_RE = re.compile(
    r"\b(thyroid lobe|upper pole|mid pole|middle pole|lower pole|isthmus|capsule|"
    r"recurrent laryngeal nerve|trachea|esophagus|lymph node)\b",
    re.IGNORECASE,
)
WHOLE_EXAM_RE = re.compile(
    r"\b(entire thyroid|whole thyroid examination|complete examination|no other abnormalit\w*|"
    r"no other nodules|all cervical structures)\b",
    re.IGNORECASE,
)
THYROID_TUMOR_RE = re.compile(r"\bthyroid tumou?r\b", re.IGNORECASE)


def issue(sample_id, dataset, split, scope, category, severity, message, text=""):
    return {
        "sample_id": sample_id,
        "dataset": dataset,
        "split": split,
        "scope": scope,
        "category": category,
        "severity": severity,
        "message": message,
        "text": text,
    }


def read_jsonl(path, max_records=None):
    records = []
    parse_issues = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if max_records is not None and len(records) >= max_records:
                break
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                parse_issues.append(issue(
                    f"line_{line_no}", "", "", "record", "json_parse", "critical",
                    f"Invalid JSON line: {exc}", line[:200],
                ))
    return records, parse_issues


def text_has_any(text, terms):
    lowered = str(text or "").lower()
    return any(term in lowered for term in terms)


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


def is_scope_limitation(text):
    lowered = str(text or "").lower()
    return any(cue in lowered for cue in (
        "cannot", "can't", "does not establish", "does not prove", "not enough",
        "insufficient", "unable to determine", "cannot be determined", "no conclusion",
    ))


def valid_refs(refs, allowed):
    return isinstance(refs, list) and all(ref in allowed for ref in refs)


def mask_geometry_key(mask):
    bbox = mask.get("bbox_xywh") if isinstance(mask, dict) else None
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        return tuple(round(float(value), 2) for value in bbox) + (
            round(float(mask.get("mask_area_px", 0)), 2),
        )
    except (TypeError, ValueError):
        return None


def unique_geometry_count(masks):
    return len({key for key in (mask_geometry_key(mask) for mask in masks) if key is not None})


def id_category_map(masks):
    return {
        mask.get("mask_id"): str(mask.get("canonical_region") or mask.get("category") or "").strip().lower()
        for mask in masks if isinstance(mask, dict)
    }


def id_pathology_map(masks):
    return {
        mask.get("mask_id"): str(mask.get("pathology_label") or "").strip().lower()
        for mask in masks if isinstance(mask, dict)
    }


def audit_generation(record):
    sample_id = record.get("sample_id", "")
    dataset = record.get("dataset", "")
    split = record.get("split", "")
    annotation = record.get("annotation")
    if not isinstance(annotation, dict):
        return [issue(sample_id, dataset, split, "record", "missing_annotation", "critical", "annotation is missing or not an object")]
    out = []
    source = str(record.get("annotation_source", "unknown"))
    if source != "api":
        out.append(issue(sample_id, dataset, split, "record", "non_api_annotation", "info",
                         f"Annotation source is {source}; VQA may be deterministic fallback"))
    if annotation.get("api_error"):
        out.append(issue(sample_id, dataset, split, "record", "api_failure_fallback", "minor",
                         "API generation failed and fallback VQA was used", str(annotation.get("api_error"))))
    unknown = record.get("ignored_unknown_categories", [])
    if isinstance(unknown, list) and unknown:
        out.append(issue(sample_id, dataset, split, "record", "ignored_unknown_category", "major",
                         "COCO annotations with unknown categories were excluded", ", ".join(map(str, unknown))))
    conflicts = record.get("pathology_label_conflicts", [])
    if isinstance(conflicts, list) and conflicts:
        out.append(issue(sample_id, dataset, split, "record", "pathology_label_conflict", "critical",
                         "The same nodule geometry has conflicting benign and malignant labels",
                         json.dumps(conflicts, ensure_ascii=False)))
    return out


def audit_masks(record):
    sample_id = record.get("sample_id", "")
    dataset = record.get("dataset", "")
    split = record.get("split", "")
    masks = record.get("masks") if isinstance(record.get("masks"), list) else []
    image = record.get("image") if isinstance(record.get("image"), dict) else {}
    width = int(image.get("width") or 0)
    height = int(image.get("height") or 0)
    expected_source_categories = EXPECTED_RAW_CATEGORIES.get(str(dataset).lower())
    out = []
    seen_ids = set()
    geometry_groups = defaultdict(list)
    if not masks:
        out.append(issue(sample_id, dataset, split, "masks", "no_specific_masks", "major",
                         "Record has no retained thyroid ultrasound masks"))
    for index, mask in enumerate(masks, 1):
        scope = f"mask_{index}"
        if not isinstance(mask, dict):
            out.append(issue(sample_id, dataset, split, scope, "bad_mask_schema", "critical", "Mask is not an object"))
            continue
        mask_id = mask.get("mask_id")
        if not mask_id or mask_id in seen_ids:
            out.append(issue(sample_id, dataset, split, scope, "duplicate_or_missing_mask_id", "critical", "mask_id is missing or duplicated"))
        seen_ids.add(mask_id)
        category = str(mask.get("canonical_region") or mask.get("category") or "").strip().lower()
        if category not in ALLOWED_CATEGORIES:
            out.append(issue(sample_id, dataset, split, scope, "unexpected_category", "critical",
                             f"Unexpected thyroid mask category: {category or 'empty'}"))
        expected_type = (
            "nodule" if category == "thyroid nodule"
            else "organ" if category == "thyroid"
            else "vessel"
        )
        if str(mask.get("target_type", "")).strip().lower() != expected_type:
            out.append(issue(sample_id, dataset, split, scope, "unexpected_target_type", "major",
                             f"{category or 'Unknown'} target_type must be {expected_type}"))
        pathology = str(mask.get("pathology_label") or "").strip().lower()
        allowed_pathology = {"benign", "malignant", "unspecified"} if category == "thyroid nodule" else {"not_applicable"}
        if pathology not in allowed_pathology:
            out.append(issue(sample_id, dataset, split, scope, "invalid_pathology_label", "critical",
                             f"Invalid pathology_label={pathology or 'empty'} for {category or 'unknown category'}"))
        source_category = str(mask.get("source_category") or "").strip().lower()
        source_categories = mask.get("source_categories")
        if not isinstance(source_categories, list) or not source_categories:
            out.append(issue(sample_id, dataset, split, scope, "missing_source_categories", "major",
                             "source_categories must preserve the merged COCO category labels"))
            source_categories = [source_category] if source_category else []
        normalized_sources = {str(value).strip().lower() for value in source_categories}
        if expected_source_categories is not None:
            unexpected_sources = normalized_sources - expected_source_categories
            if unexpected_sources:
                out.append(issue(sample_id, dataset, split, scope, "dataset_category_mismatch", "critical",
                                 f"Source categories are not valid for {dataset}: {sorted(unexpected_sources)}"))
        if pathology in {"benign", "malignant"}:
            expected_specific = f"{pathology} thyroid nodule"
            if expected_specific not in normalized_sources:
                out.append(issue(sample_id, dataset, split, scope, "pathology_source_mismatch", "critical",
                                 f"pathology_label={pathology} lacks source category {expected_specific}"))
        if pathology == "unspecified" and normalized_sources.intersection({
            "benign thyroid nodule", "malignant thyroid nodule",
        }):
            out.append(issue(sample_id, dataset, split, scope, "lost_pathology_label", "critical",
                             "A supplied benign/malignant source category was reduced to unspecified"))
        if source_category in {"thyroid tumor", "thyroid tumour"} and category != "thyroid nodule":
            out.append(issue(sample_id, dataset, split, scope, "tumor_not_normalized", "critical",
                             "A thyroid tumor source label was not normalized to thyroid nodule"))
        if THYROID_TUMOR_RE.search(str(mask.get("category") or "")) or THYROID_TUMOR_RE.search(category):
            out.append(issue(sample_id, dataset, split, scope, "tumor_category_retained", "critical",
                             "thyroid tumor must not be retained as an output category"))
        key = mask_geometry_key(mask)
        if key is None:
            out.append(issue(sample_id, dataset, split, scope, "invalid_geometry", "critical", "bbox_xywh or mask_area_px is invalid"))
            continue
        x, y, box_w, box_h, area = key
        if box_w <= 0 or box_h <= 0 or area <= 0:
            out.append(issue(sample_id, dataset, split, scope, "nonpositive_geometry", "critical", "Mask bbox or area is non-positive"))
        if width > 0 and height > 0 and (x < 0 or y < 0 or x + box_w > width + 1 or y + box_h > height + 1):
            out.append(issue(sample_id, dataset, split, scope, "bbox_out_of_bounds", "major", "Mask bbox extends outside image bounds"))
        try:
            ratio = float(mask.get("area_ratio"))
            expected_ratio = area / max(width * height, 1)
            if not 0 < ratio <= 1:
                raise ValueError
            if width > 0 and height > 0 and abs(ratio - expected_ratio) > 1e-4:
                out.append(issue(sample_id, dataset, split, scope, "area_ratio_mismatch", "major", "area_ratio is inconsistent with mask and image area"))
        except (TypeError, ValueError):
            out.append(issue(sample_id, dataset, split, scope, "invalid_area_ratio", "major", "area_ratio must be numeric in (0, 1]"))
        geometry_groups[key].append((mask_id, category))
    for group in geometry_groups.values():
        if len(group) > 1:
            categories = sorted(category for _, category in group)
            out.append(issue(sample_id, dataset, split, "masks", "duplicate_geometry", "critical",
                             "Identical geometry is retained more than once instead of being merged", ", ".join(categories)))
    return out


def audit_clinical_text(record, scope, text, refs, id_to_category, id_to_pathology, require_relevance=True):
    sample_id = record.get("sample_id", "")
    dataset = record.get("dataset", "")
    split = record.get("split", "")
    out = []
    if text_has_any(text, FORBIDDEN_USER_TERMS):
        out.append(issue(sample_id, dataset, split, scope, "metadata_leak", "major",
                         "User-facing text mentions file/dataset/annotation metadata", text))
    mentions = mentioned_categories(text)
    referenced_categories = {id_to_category.get(ref) for ref in refs}
    unsupported_mentions = mentions - referenced_categories
    if refs and unsupported_mentions:
        out.append(issue(sample_id, dataset, split, scope, "unsupported_category_reference", "critical",
                         f"Text mentions categories not supported by cited masks: {sorted(unsupported_mentions)}", text))
    pathology_mentions = mentioned_pathologies(text)
    referenced_pathologies = {id_to_pathology.get(ref) for ref in refs}
    unsupported_pathologies = pathology_mentions - referenced_pathologies
    if unsupported_pathologies:
        out.append(issue(sample_id, dataset, split, scope, "unsupported_pathology_label", "critical",
                         f"Benign/malignant wording is not supported by cited nodule labels: {sorted(unsupported_pathologies)}", text))
    if THYROID_TUMOR_RE.search(text):
        out.append(issue(sample_id, dataset, split, scope, "noncanonical_tumor_term", "major",
                         "Use thyroid nodule, not thyroid tumor, in output text", text))
    for pattern, label in DIAGNOSTIC_OVERREACH_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            if is_scope_limitation(text):
                out.append(issue(sample_id, dataset, split, scope, "diagnostic_scope_statement", "minor",
                                 "Safe limitation wording is clinically low-value if repeated", text))
            else:
                out.append(issue(sample_id, dataset, split, scope, "diagnostic_overreach", "critical", label, text))
    if PHYSICAL_MEASUREMENT_RE.search(text):
        out.append(issue(sample_id, dataset, split, scope, "unsupported_physical_measurement", "major",
                         "Physical size is unsupported without pixel spacing", text))
    if ANATOMIC_LATERALITY_RE.search(text):
        out.append(issue(sample_id, dataset, split, scope, "unsupported_laterality", "major",
                         "Patient left/right laterality is unsupported by image-grid location", text))
    if UNSUPPORTED_SUBSTRUCTURE_RE.search(text):
        out.append(issue(sample_id, dataset, split, scope, "unsupported_substructure", "major",
                         "Whole-organ masks do not support the named substructure", text))
    if WHOLE_EXAM_RE.search(text):
        out.append(issue(sample_id, dataset, split, scope, "whole_exam_overreach", "major",
                         "A single labeled frame cannot support a whole-exam conclusion", text))
    if require_relevance and not text_has_any(text, SUPPORTED_REPORT_CUES):
        out.append(issue(sample_id, dataset, split, scope, "low_relevance", "minor",
                         "Text has weak connection to supported thyroid mask evidence", text))
    return out


def audit_report(record):
    sample_id = record.get("sample_id", "")
    dataset = record.get("dataset", "")
    split = record.get("split", "")
    masks = record.get("masks") if isinstance(record.get("masks"), list) else []
    allowed = {mask.get("mask_id") for mask in masks if isinstance(mask, dict)}
    id_to_category = id_category_map(masks)
    id_to_pathology = id_pathology_map(masks)
    report = record.get("annotation", {}).get("grounded_report")
    if not isinstance(report, dict):
        return [issue(sample_id, dataset, split, "report", "missing_report", "critical", "Missing grounded_report")]
    sentences = report.get("sentences")
    if not isinstance(sentences, list):
        return [issue(sample_id, dataset, split, "report", "bad_report_schema", "critical", "grounded_report.sentences is not a list")]
    out = []
    if not 3 <= len(sentences) <= 10:
        out.append(issue(sample_id, dataset, split, "report", "sentence_count", "major",
                         f"Report has {len(sentences)} sentences; expected 3-10 for a multi-organ image"))
    sections = {str(item.get("section", "")).strip() for item in sentences if isinstance(item, dict)}
    missing = sorted(REPORT_REQUIRED_SECTIONS - sections)
    if missing:
        out.append(issue(sample_id, dataset, split, "report", "missing_section", "major",
                         f"Missing report sections: {', '.join(missing)}"))
    report_mentions = set()
    for index, sentence in enumerate(sentences, 1):
        scope = f"report_sentence_{index}"
        if not isinstance(sentence, dict):
            out.append(issue(sample_id, dataset, split, scope, "bad_sentence_schema", "critical", "Report sentence is not an object"))
            continue
        text = str(sentence.get("text", "")).strip()
        refs = sentence.get("evidence_mask_ids", [])
        source = str(sentence.get("evidence_source", "")).strip().lower()
        if not text:
            out.append(issue(sample_id, dataset, split, scope, "empty_sentence", "critical", "Report sentence is empty"))
            continue
        if not valid_refs(refs, allowed):
            out.append(issue(sample_id, dataset, split, scope, "invalid_evidence", "critical", "Report cites unknown mask IDs", text))
            refs = []
        if masks and text_has_any(text, GROUNDING_CUES) and not refs:
            out.append(issue(sample_id, dataset, split, scope, "missing_grounding", "major", "Positive spatial/anatomic report claim lacks evidence IDs", text))
        if refs and source not in {"mask_geometry", "image_and_mask"}:
            out.append(issue(sample_id, dataset, split, scope, "invalid_evidence_source", "major", "Grounded report sentence has an invalid evidence_source", text))
        report_mentions.update(mentioned_categories(text))
        out.extend(audit_clinical_text(record, scope, text, refs, id_to_category, id_to_pathology))
    present = set(id_to_category.values())
    omitted = present - report_mentions
    if omitted:
        out.append(issue(sample_id, dataset, split, "report", "category_omission", "major",
                         f"Report does not mention labeled categories: {sorted(omitted)}"))
    return out


def audit_vqa(record):
    sample_id = record.get("sample_id", "")
    dataset = record.get("dataset", "")
    split = record.get("split", "")
    masks = record.get("masks") if isinstance(record.get("masks"), list) else []
    allowed = {mask.get("mask_id") for mask in masks if isinstance(mask, dict)}
    id_to_category = id_category_map(masks)
    id_to_pathology = id_pathology_map(masks)
    vqa = record.get("annotation", {}).get("grounded_vqa")
    if not isinstance(vqa, list):
        return [issue(sample_id, dataset, split, "vqa", "missing_vqa", "critical", "grounded_vqa is missing or not a list")]
    out = []
    if not 4 <= len(vqa) <= 6:
        out.append(issue(sample_id, dataset, split, "vqa", "vqa_count", "major", f"grounded_vqa has {len(vqa)} items; expected 4-6"))
    seen_questions = set()
    for index, item in enumerate(vqa, 1):
        scope = f"vqa_{index}"
        if not isinstance(item, dict):
            out.append(issue(sample_id, dataset, split, scope, "bad_vqa_schema", "critical", "VQA item is not an object"))
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        combined = f"{question} {answer}"
        refs = item.get("evidence_mask_ids", [])
        grounded = bool(item.get("requires_grounding"))
        source = str(item.get("evidence_source", "")).strip().lower()
        if not question or not answer:
            out.append(issue(sample_id, dataset, split, scope, "empty_vqa", "critical", "Question or answer is empty"))
        normalized_question = re.sub(r"\s+", " ", question.lower())
        if normalized_question in seen_questions:
            out.append(issue(sample_id, dataset, split, scope, "duplicate_question", "minor", "Duplicate VQA question", question))
        seen_questions.add(normalized_question)
        if text_has_any(question, NON_CLINICAL_QUESTION_TERMS):
            out.append(issue(sample_id, dataset, split, scope, "non_clinical_question", "major", "Question asks about file/case/slice/dataset metadata", question))
        if not valid_refs(refs, allowed):
            out.append(issue(sample_id, dataset, split, scope, "invalid_evidence", "critical", "VQA cites unknown mask IDs", question))
            refs = []
        if grounded and not refs:
            out.append(issue(sample_id, dataset, split, scope, "missing_grounding", "major", "requires_grounding=true but evidence IDs are empty", question))
        if not grounded and refs:
            out.append(issue(sample_id, dataset, split, scope, "unneeded_grounding", "minor", "requires_grounding=false but evidence IDs are present", question))
        if grounded and source not in {"mask_geometry", "image_and_mask"}:
            out.append(issue(sample_id, dataset, split, scope, "invalid_evidence_source", "major", "Grounded VQA has an invalid evidence_source", question))
        if not grounded and source not in {"", "none"}:
            out.append(issue(sample_id, dataset, split, scope, "unexpected_evidence_source", "minor", "Ungrounded VQA should use evidence_source=none", question))
        out.extend(audit_clinical_text(record, scope, combined, refs, id_to_category, id_to_pathology))
    return out


def audit_segmentation(record):
    sample_id = record.get("sample_id", "")
    dataset = record.get("dataset", "")
    split = record.get("split", "")
    masks = record.get("masks") if isinstance(record.get("masks"), list) else []
    allowed = {mask.get("mask_id") for mask in masks if isinstance(mask, dict)}
    id_to_category = id_category_map(masks)
    id_to_pathology = id_pathology_map(masks)
    segmentation = record.get("annotation", {}).get("segmentation_tasks")
    if not isinstance(segmentation, list):
        return [issue(sample_id, dataset, split, "segmentation", "missing_segmentation", "critical", "segmentation_tasks is missing or not a list")]
    out = []
    expected_max = len(set(id_to_category.values())) + 1 if masks else 1
    if not 1 <= len(segmentation) <= max(1, expected_max):
        out.append(issue(sample_id, dataset, split, "segmentation", "segmentation_count", "major",
                         f"segmentation_tasks has {len(segmentation)} items; expected at most {expected_max}"))
    covered_categories = set()
    for index, item in enumerate(segmentation, 1):
        scope = f"segmentation_{index}"
        if not isinstance(item, dict):
            out.append(issue(sample_id, dataset, split, scope, "bad_segmentation_schema", "critical", "Segmentation item is not an object"))
            continue
        instruction = str(item.get("instruction", ""))
        answer = str(item.get("answer", ""))
        text = f"{instruction} {answer}"
        refs = item.get("target_mask_ids", [])
        if not valid_refs(refs, allowed):
            out.append(issue(sample_id, dataset, split, scope, "invalid_target_masks", "critical", "target_mask_ids contain unknown masks", text))
            refs = []
        if refs and "[SEG]" not in answer:
            out.append(issue(sample_id, dataset, split, scope, "missing_seg_token", "critical", "Target masks are present but answer lacks [SEG]", answer))
        if not refs and "[SEG]" in answer:
            out.append(issue(sample_id, dataset, split, scope, "unexpected_seg_token", "critical", "Answer contains [SEG] without target masks", answer))
        mentions = mentioned_categories(text)
        referenced = {id_to_category.get(ref) for ref in refs}
        if mentions - referenced:
            out.append(issue(sample_id, dataset, split, scope, "mixed_category_targets", "critical",
                             f"Segmentation wording is not supported by cited categories: {sorted(mentions - referenced)}", text))
        if len(mentions) == 1 and mentions == referenced:
            covered_categories.update(mentions)
        out.extend(audit_clinical_text(
            record, scope, text, refs, id_to_category, id_to_pathology, require_relevance=False
        ))
    missing_category_tasks = set(id_to_category.values()) - covered_categories
    if masks and missing_category_tasks:
        out.append(issue(sample_id, dataset, split, "segmentation", "missing_category_task", "major",
                         f"No category-specific segmentation task for: {sorted(missing_category_tasks)}"))
    return out


def score_record(record_issues):
    weights = {"critical": 25, "major": 10, "minor": 3, "info": 0}
    return max(0, 100 - sum(weights.get(item["severity"], 0) for item in record_issues))


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records, all_issues, per_record_rows):
    severity_counts = Counter(item["severity"] for item in all_issues)
    category_counts = Counter(item["category"] for item in all_issues)
    source_counts = Counter(str(record.get("annotation_source", "unknown")) for record in records)
    dataset_counts = Counter(record.get("dataset", "") for record in records)
    dataset_issue_counts = Counter(item["dataset"] for item in all_issues)
    mask_category_counts = Counter(
        str(mask.get("canonical_region") or mask.get("category") or "")
        for record in records for mask in record.get("masks", []) if isinstance(mask, dict)
    )
    pathology_counts = Counter(
        str(mask.get("pathology_label") or "")
        for record in records for mask in record.get("masks", [])
        if isinstance(mask, dict) and str(mask.get("canonical_region") or "") == "thyroid nodule"
    )
    scores = [row["score"] for row in per_record_rows]
    clinical_pass = sum(row["clinical_report_pass"] == "yes" for row in per_record_rows)
    overall_pass = sum(row["data_quality_pass"] == "yes" for row in per_record_rows)
    lines = [
        "# Thyroid Grounded Dataset Quality Summary", "",
        f"- Records: {len(records)}",
        f"- Average score: {(sum(scores) / len(scores) if scores else 0):.2f}/100",
        f"- Clinical report pass: {clinical_pass}/{len(per_record_rows)} ({clinical_pass / max(len(per_record_rows), 1) * 100:.1f}%)",
        f"- Overall data-quality pass: {overall_pass}/{len(per_record_rows)} ({overall_pass / max(len(per_record_rows), 1) * 100:.1f}%)",
        f"- Issues: critical={severity_counts.get('critical', 0)}, major={severity_counts.get('major', 0)}, minor={severity_counts.get('minor', 0)}, info={severity_counts.get('info', 0)}",
        f"- Annotation sources: {', '.join(f'{key}={value}' for key, value in sorted(source_counts.items()))}",
        "", "## Retained Mask Categories", "",
    ]
    for category in CATEGORY_ORDER:
        lines.append(f"- {category}: {mask_category_counts.get(category, 0)}")
    lines.extend(["", "## Thyroid Nodule Labels", ""])
    for label in ("benign", "malignant", "unspecified", "conflicting"):
        lines.append(f"- {label}: {pathology_counts.get(label, 0)}")
    lines.extend(["", "## Dataset Breakdown", ""])
    for dataset, count in sorted(dataset_counts.items()):
        lines.append(f"- {dataset or 'unknown'}: records={count}, issues={dataset_issue_counts.get(dataset, 0)}")
    lines.extend(["", "## Top Issue Categories", ""])
    for category, count in category_counts.most_common(15):
        lines.append(f"- {category}: {count}")
    lines.extend(["", "## Representative Issues", ""])
    for item in all_issues[:30]:
        text = str(item.get("text", "")).replace("\n", " ")
        if len(text) > 180:
            text = text[:177] + "..."
        lines.append(
            f"- [{item['severity']}] {item['sample_id']} {item['scope']} {item['category']}: "
            f"{item['message']}" + (f" | {text}" if text else "")
        )
    if not all_issues:
        lines.append("- No rule-based issues found.")
    lines.extend([
        "", "## Interpretation", "",
        "This audit verifies structure, grounding consistency, and conservative clinical scope. "
        "It does not verify polygon accuracy, image-label correctness, or replace blinded review by thyroid-ultrasound experts.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input annotation JSONL")
    parser.add_argument("--out-prefix", help="Default: <input stem>_quality_eval beside input")
    parser.add_argument("--max-records", type=int, help="Audit only the first N valid records")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_prefix = Path(args.out_prefix) if args.out_prefix else input_path.with_name(input_path.stem + "_quality_eval")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    records, parse_issues = read_jsonl(input_path, args.max_records)
    issues_by_sample = defaultdict(list)
    all_issues = list(parse_issues)
    total = len(records)
    for index, record in enumerate(records, 1):
        sample_id = record.get("sample_id", "")
        record_issues = []
        record_issues.extend(audit_generation(record))
        record_issues.extend(audit_masks(record))
        record_issues.extend(audit_segmentation(record))
        record_issues.extend(audit_vqa(record))
        record_issues.extend(audit_report(record))
        issues_by_sample[sample_id].extend(record_issues)
        all_issues.extend(record_issues)
        if index == 1 or index == total or index % 100 == 0:
            print(f"[audit {index}/{total} remaining={total - index}] {sample_id}", flush=True)

    per_record_rows = []
    for record in records:
        sample_id = record.get("sample_id", "")
        record_issues = issues_by_sample.get(sample_id, [])
        counts = Counter(item["severity"] for item in record_issues)
        report_bad = any(
            item["scope"].startswith("report") and item["severity"] in {"critical", "major"}
            for item in record_issues
        )
        data_bad = any(item["severity"] in {"critical", "major"} for item in record_issues)
        masks = record.get("masks", []) if isinstance(record.get("masks"), list) else []
        report_sentences = record.get("annotation", {}).get("grounded_report", {}).get("sentences", [])
        categories = sorted(set(id_category_map(masks).values()))
        pathologies = sorted({
            value for value in id_pathology_map(masks).values()
            if value and value != "not_applicable"
        })
        conflicts = record.get("pathology_label_conflicts", [])
        per_record_rows.append({
            "sample_id": sample_id,
            "dataset": record.get("dataset", ""),
            "split": record.get("split", ""),
            "annotation_source": record.get("annotation_source", "unknown"),
            "mask_count": len(masks),
            "unique_geometry_count": unique_geometry_count(masks),
            "categories": "|".join(categories),
            "pathology_labels": "|".join(pathologies),
            "merged_duplicate_annotations": record.get("merged_duplicate_annotations", ""),
            "pathology_label_conflict_count": len(conflicts) if isinstance(conflicts, list) else "",
            "report_sentence_count": len(report_sentences) if isinstance(report_sentences, list) else 0,
            "score": score_record(record_issues),
            "critical": counts.get("critical", 0),
            "major": counts.get("major", 0),
            "minor": counts.get("minor", 0),
            "info": counts.get("info", 0),
            "clinical_report_pass": "no" if report_bad else "yes",
            "data_quality_pass": "no" if data_bad else "yes",
        })

    issue_rows = sorted(all_issues, key=lambda item: (
        {"critical": 0, "major": 1, "minor": 2, "info": 3}.get(item["severity"], 9),
        item["sample_id"], item["scope"],
    ))
    write_csv(out_prefix.with_suffix(".records.csv"), per_record_rows, [
        "sample_id", "dataset", "split", "annotation_source", "mask_count", "unique_geometry_count",
        "categories", "pathology_labels", "merged_duplicate_annotations",
        "pathology_label_conflict_count", "report_sentence_count", "score",
        "critical", "major", "minor", "info", "clinical_report_pass", "data_quality_pass",
    ])
    write_csv(out_prefix.with_suffix(".issues.csv"), issue_rows, [
        "sample_id", "dataset", "split", "scope", "category", "severity", "message", "text",
    ])
    out_prefix.with_suffix(".summary.md").write_text(
        summarize(records, issue_rows, per_record_rows), encoding="utf-8"
    )
    summary_json = {
        "records": len(records),
        "issues": dict(Counter(item["severity"] for item in all_issues)),
        "issue_categories": dict(Counter(item["category"] for item in all_issues)),
        "clinical_report_pass": sum(row["clinical_report_pass"] == "yes" for row in per_record_rows),
        "data_quality_pass": sum(row["data_quality_pass"] == "yes" for row in per_record_rows),
        "outputs": {
            "records_csv": str(out_prefix.with_suffix(".records.csv")),
            "issues_csv": str(out_prefix.with_suffix(".issues.csv")),
            "summary_md": str(out_prefix.with_suffix(".summary.md")),
        },
    }
    out_prefix.with_suffix(".summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary_json, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
