from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict


def evaluate_outputs(reports_dir: Path) -> Dict[str, object]:
    reports = sorted(reports_dir.rglob("report.json"))
    counts = Counter()
    examples = []
    for report_path in reports:
        counts["reports"] += 1
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            counts["invalid_json"] += 1
            continue
        mode = data.get("mode", "unknown")
        counts[f"mode:{mode}"] += 1
        roi_count = len(data.get("selected_rois") or [])
        if roi_count:
            counts["reports_with_rois"] += 1
        if len(examples) < 10:
            examples.append({
                "report_path": str(report_path),
                "slide_id": data.get("slide_id"),
                "dataset_name": data.get("dataset_name"),
                "mode": mode,
                "roi_count": roi_count,
            })
    return {"counts": dict(counts), "examples": examples}
