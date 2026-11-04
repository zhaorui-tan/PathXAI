from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def write_baseline_report(
    out_dir: Path,
    slide_item: Dict[str, object],
    slide_info: Dict[str, object],
    rois: List[Dict[str, object]],
    note: Optional[str] = None,
) -> Dict[str, object]:
    """Write a traceable dry-run report artifact before model adapters are wired."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "slide_id": slide_item.get("case_id"),
        "dataset_name": slide_item.get("dataset_name"),
        "mode": "baseline_no_model",
        "diagnosis": "Not generated - model adapters not enabled in this baseline run.",
        "summary": (
            "This baseline selected ROI crops and saved evidence metadata. "
            "Next stages should add MedSAM segmentation, cell statistics, knowledge retrieval, "
            "and LLM report generation."
        ),
        "slide_info": slide_info,
        "selected_rois": rois,
        "note": note,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    md = [
        f"# Baseline Report: {report['slide_id']}",
        "",
        f"- Dataset: {report['dataset_name']}",
        f"- Mode: {report['mode']}",
        f"- Diagnosis: {report['diagnosis']}",
        f"- ROI count: {len(rois)}",
        "",
        "## Summary",
        "",
        str(report["summary"]),
        "",
        "## Selected ROIs",
        "",
    ]
    for roi in rois:
        md.append(f"- `{roi['roi_id']}` at x={roi['x']}, y={roi['y']}, size={roi['width']}x{roi['height']}")
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report
