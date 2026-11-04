from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .manifest import ManifestItem, stable_id


def _read_text(path: Path, max_chars: int = 20000) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]


def clean_report_text(text: str) -> str:
    """Light cleanup for OCR/report text without destroying clinical content."""
    text = text.replace("\x0c", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return cleaned.strip()


def clean_annotation_summary(text: str) -> str:
    """Annotation files are often a quoted one-paragraph summary."""
    text = clean_report_text(text)
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    text = text.replace('\\"', '"')
    return " ".join(text.split())


def _tcga_case_from_slide(path: Path) -> str:
    # TCGA-B6-A0RN-01Z-00-DX1.UUID.svs -> TCGA-B6-A0RN
    parts = path.name.split(".")[0].split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return path.stem


def _index_svs_by_case(root: Path) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for svs_path in sorted(root.rglob("*.svs")):
        case_id = _tcga_case_from_slide(svs_path)
        index.setdefault(case_id, []).append(str(svs_path))
    return index


def build_wsi_caption_report_manifest(root: Path, max_report_chars: int = 20000) -> List[ManifestItem]:
    """Build report-supervised items from Wsi-Caption-master/datasets."""
    root = root.resolve()
    svs_by_case = _index_svs_by_case(root)
    items: List[ManifestItem] = []
    for report_path in sorted(root.rglob("Report.txt")):
        case_dir = report_path.parent
        case_id = case_dir.name
        cancer_type = case_dir.parent.name
        image_paths = sorted(str(p) for p in case_dir.glob("images_*.png"))
        pdf_paths = sorted(str(p) for p in case_dir.glob("*.PDF")) + sorted(str(p) for p in case_dir.glob("*.pdf"))
        annotation_paths = []
        summary_path = None
        for name in ("annotation", "annotations.txt"):
            p = case_dir / name
            if p.exists():
                annotation_paths.append(str(p))
                if p.name == "annotation":
                    summary_path = p

        slide_paths = svs_by_case.get(case_id, [])
        usable_for = ["report_generation", "report_eval"]
        if image_paths:
            usable_for.append("image_grounded_report")
        if slide_paths:
            usable_for.append("wsi_grounded_report")

        items.append(
            ManifestItem(
                item_id=stable_id("Wsi-Caption", case_id, report_path),
                dataset_name="Wsi-Caption",
                case_id=case_id,
                item_type="pathology_report",
                path=str(report_path),
                report_path=str(report_path),
                report_text=clean_report_text(_read_text(report_path, max_report_chars)),
                annotation_paths=annotation_paths,
                metadata={
                    "cancer_type": cancer_type,
                    "summary_report_path": str(summary_path) if summary_path else None,
                    "summary_report_text": (
                        clean_annotation_summary(_read_text(summary_path, max_report_chars))
                        if summary_path else None
                    ),
                    "full_report_path": str(report_path),
                    "full_report_text": clean_report_text(_read_text(report_path, max_report_chars)),
                    "available_targets": [
                        target for target, ok in (
                            ("summary", summary_path is not None),
                            ("full", report_path.exists()),
                        )
                        if ok
                    ],
                    "image_paths": image_paths,
                    "pdf_paths": pdf_paths,
                    "slide_paths": slide_paths,
                    "source_root": str(root),
                },
                usable_for=usable_for,
            )
        )
    return items


def build_wsi_caption_wsi_manifest(root: Path) -> List[ManifestItem]:
    root = root.resolve()
    report_cases = {p.parent.name for p in root.rglob("Report.txt")}
    items: List[ManifestItem] = []
    for svs_path in sorted(root.rglob("*.svs")):
        case_id = _tcga_case_from_slide(svs_path)
        cancer_type = _infer_cancer_type(root, svs_path)
        usable_for = ["wsi_roi"]
        if case_id in report_cases:
            usable_for.append("wsi_grounded_report")
        items.append(
            ManifestItem(
                item_id=stable_id("Wsi-Caption-WSI", case_id, svs_path),
                dataset_name="Wsi-Caption",
                case_id=case_id,
                item_type="wsi",
                path=str(svs_path),
                metadata={
                    "cancer_type": cancer_type,
                    "source_root": str(root),
                    "has_report_case_dir": case_id in report_cases,
                },
                usable_for=usable_for,
            )
        )
    return items


def build_wsi_caption_image_manifest(root: Path) -> List[ManifestItem]:
    root = root.resolve()
    items: List[ManifestItem] = []
    for image_path in sorted(root.rglob("images_*.png")):
        case_dir = image_path.parent
        case_id = case_dir.name
        report_path = case_dir / "Report.txt"
        summary_path = case_dir / "annotation"
        items.append(
            ManifestItem(
                item_id=stable_id("Wsi-Caption-Image", case_id, image_path),
                dataset_name="Wsi-Caption",
                case_id=case_id,
                item_type="report_image",
                path=str(image_path),
                report_path=str(report_path) if report_path.exists() else None,
                report_text=clean_report_text(_read_text(report_path)) if report_path.exists() else None,
                metadata={
                    "cancer_type": case_dir.parent.name,
                    "summary_report_path": str(summary_path) if summary_path.exists() else None,
                    "summary_report_text": (
                        clean_annotation_summary(_read_text(summary_path)) if summary_path.exists() else None
                    ),
                    "full_report_path": str(report_path) if report_path.exists() else None,
                    "full_report_text": clean_report_text(_read_text(report_path)) if report_path.exists() else None,
                    "source_root": str(root),
                },
                usable_for=["image_grounded_report", "report_generation"],
            )
        )
    return items


def build_wsivqa_manifest(root: Path, downloads_root: Optional[Path] = None) -> List[dict]:
    """Build WSI-VQA question items, preserving official train/val split."""
    root = root.resolve()
    downloads_root = (downloads_root or root / "tcga_wsi_downloads").resolve()
    svs_by_case = _index_svs_by_case(downloads_root) if downloads_root.exists() else {}
    items: List[dict] = []
    for split_name, json_name in (("train", "WsiVQA_train.json"), ("val", "WsiVQA_val.json")):
        path = root / json_name
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        for idx, row in enumerate(rows):
            case_id = str(row.get("Id", "")).strip()
            question = row.get("Question")
            answer = row.get("Answer")
            choices = row.get("Choice") or []
            slide_paths = svs_by_case.get(case_id, [])
            items.append(
                {
                    "item_id": stable_id("WsiVQA", split_name, idx, case_id, question),
                    "dataset_name": "WsiVQA",
                    "case_id": case_id,
                    "item_type": "vqa",
                    "path": str(path),
                    "split": split_name,
                    "question": question,
                    "answer": answer,
                    "choices": choices,
                    "metadata": {
                        "slide_paths": slide_paths,
                        "has_downloaded_slide": bool(slide_paths),
                        "source_root": str(root),
                    },
                    "usable_for": ["vqa_eval", "report_field_eval", "wsi_question_answering"],
                }
            )
    return items


def manifest_items_to_dicts(items: Iterable[ManifestItem]) -> List[dict]:
    return [asdict(item) for item in items]


def _infer_cancer_type(root: Path, path: Path) -> Optional[str]:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    if rel.parts:
        return rel.parts[0]
    return None
