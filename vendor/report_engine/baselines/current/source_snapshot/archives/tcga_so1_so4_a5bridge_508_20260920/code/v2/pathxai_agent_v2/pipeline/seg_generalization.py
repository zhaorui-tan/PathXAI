"""Cross-tissue segmentation generalization analysis.

Validates that the segmentation model produces consistent, sane tissue-class
distributions across diverse cancer types — WITHOUT requiring pixel-level
ground truth on WSI.  This is a post-hoc analysis that runs on completed
pipeline results.

Validation strategy:
    1. Tumor should be the dominant class in tumor ROIs (face validity).
    2. Class distribution should vary plausibly by organ.
    3. Tumor detection rate should be high and consistent across cancer types.
    4. Dead tissue fraction should correlate with known necrotic cancer types.

This analysis is intended for the paper's generalization figure and table.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# ── Known class names (mirrors seg_visual.TISSUE_NAMES) ──────────────────────
CLASS_NAMES = {
    "0": "Background",
    "1": "Neoplastic",
    "2": "Inflammatory",
    "3": "Connective/Stroma",
    "4": "Epithelial",
    "5": "Dead/Necrotic",
    "6": "Gland",
}

# ── Cancer types known to frequently exhibit necrosis ─────────────────────────
NECROTIC_CANCERS = {
    "TCGA-GBM",   # glioblastoma — hallmark necrosis
    "TCGA-LUAD",  # lung adenocarcinoma — common necrosis
    "TCGA-LUSC",  # lung squamous — common necrosis
    "TCGA-KIRC",  # renal clear cell — necrosis common
    "TCGA-UCEC",  # endometrial — necrosis possible
}


def analyze_cross_tissue_generalization(
    run_dir: str | Path,
    *,
    min_rois_per_type: int = 3,
) -> dict[str, Any]:
    """Analyze segmentation class distributions across cancer types.

    Parameters
    ----------
    run_dir:
        Path to a completed ablation run (e.g. ``outputs/ablation/run_.../R0_full/``).
    min_rois_per_type:
        Minimum ROIs needed to include a cancer type in the analysis.

    Returns
    -------
    dict with:
        per_cancer: {cancer_type: {roi_count, class_means, tumor_detection_rate, ...}}
        summary: aggregate stats
        dead_tissue_check: whether necrotic cancers have higher dead fractions
    """
    run_dir = Path(run_dir)

    # Collect per-ROI class fractions
    per_cancer: dict[str, dict] = defaultdict(
        lambda: {"roi_count": 0, "class_fracs": defaultdict(list),
                 "case_count": 0, "cases": set()}
    )

    for result_path in sorted(run_dir.glob("**/pipeline_result.json")):
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        cancer_type = _cancer_type_from_path(result_path, run_dir)
        seg = data.get("segmentation_context") or {}
        if not seg.get("available"):
            continue

        per_cancer[cancer_type]["cases"].add(data.get("case_id", ""))
        per_cancer[cancer_type]["case_count"] = len(per_cancer[cancer_type]["cases"])

        for rr in seg.get("roi_results", []):
            per_cancer[cancer_type]["roi_count"] += 1
            for cid, frac in (rr.get("stats", {}).get("class_fractions", {}).items()):
                per_cancer[cancer_type]["class_fracs"][cid].append(float(frac))

    # Build summary
    cancer_summaries = {}
    for cancer_type, stats in sorted(per_cancer.items()):
        if stats["roi_count"] < min_rois_per_type:
            continue

        class_means = {}
        all_class_ids = set()
        for cid_fracs in stats["class_fracs"].values():
            all_class_ids.update(cid_fracs.keys() if hasattr(cid_fracs, 'keys') else [])
        all_class_ids = stats["class_fracs"].keys()

        for cid in sorted(all_class_ids, key=lambda x: int(x)):
            fracs = stats["class_fracs"][cid]
            arr = np.array(fracs)
            class_means[cid] = {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "median": round(float(np.median(arr)), 4),
                "class_name": CLASS_NAMES.get(cid, f"Class{cid}"),
            }

        # Tumor detection rate: % of ROIs with ≥10% neoplastic
        tumor_fracs = stats["class_fracs"].get("1", [])
        tumor_detection = float(np.mean([f > 0.10 for f in tumor_fracs])) if tumor_fracs else 0.0

        # Dead tissue fraction
        dead_fracs = stats["class_fracs"].get("5", [])
        avg_dead = float(np.mean(dead_fracs)) if dead_fracs else 0.0

        cancer_summaries[cancer_type] = {
            "case_count": stats["case_count"],
            "roi_count": stats["roi_count"],
            "class_means": class_means,
            "tumor_detection_rate": round(tumor_detection, 3),
            "avg_dead_pct": round(avg_dead * 100, 1),
        }

    # Aggregate
    total_cases = sum(s["case_count"] for s in cancer_summaries.values())
    total_rois = sum(s["roi_count"] for s in cancer_summaries.values())
    all_tumor_rates = [s["tumor_detection_rate"] for s in cancer_summaries.values()]

    # Check: necrotic cancers should have higher dead fractions
    necrotic_dead = [
        s["avg_dead_pct"] for ct, s in cancer_summaries.items()
        if ct in NECROTIC_CANCERS
    ]
    non_necrotic_dead = [
        s["avg_dead_pct"] for ct, s in cancer_summaries.items()
        if ct not in NECROTIC_CANCERS
    ]

    return {
        "per_cancer": cancer_summaries,
        "summary": {
            "cancer_types_analyzed": len(cancer_summaries),
            "total_cases": total_cases,
            "total_rois": total_rois,
            "mean_tumor_detection_rate": round(float(np.mean(all_tumor_rates)), 3) if all_tumor_rates else 0,
            "std_tumor_detection_rate": round(float(np.std(all_tumor_rates)), 3) if all_tumor_rates else 0,
            "min_tumor_detection_rate": round(float(np.min(all_tumor_rates)), 3) if all_tumor_rates else 0,
        },
        "dead_tissue_sanity_check": {
            "necrotic_cancers_mean_dead_pct": round(float(np.mean(necrotic_dead)), 1) if necrotic_dead else 0,
            "non_necrotic_mean_dead_pct": round(float(np.mean(non_necrotic_dead)), 1) if non_necrotic_dead else 0,
            "expected": "Necrotic cancers should have higher dead tissue fraction",
            "passes": (
                float(np.mean(necrotic_dead)) > float(np.mean(non_necrotic_dead))
                if necrotic_dead and non_necrotic_dead else None
            ),
        },
    }


def plot_cross_tissue_heatmap(
    summary: dict,
    out_path: str | Path,
    *,
    title: str = "Segmentation class distribution across cancer types\n(zero-shot, no per-organ fine-tuning)",
) -> str | None:
    """Generate a heatmap: cancer type × seg class → mean fraction.

    Returns the output path on success, None on failure.
    """
    cancer_summaries = summary.get("per_cancer", {})
    if not cancer_summaries:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    cancer_types = sorted(cancer_summaries.keys())

    # Collect all class IDs that appear
    all_cids: set[str] = set()
    for ct in cancer_types:
        all_cids.update(cancer_summaries[ct]["class_means"].keys())
    class_ids = sorted(all_cids, key=lambda x: int(x))
    class_labels = [CLASS_NAMES.get(cid, f"Class{cid}") for cid in class_ids]

    matrix = np.zeros((len(cancer_types), len(class_ids)))
    for i, ct in enumerate(cancer_types):
        for j, cid in enumerate(class_ids):
            matrix[i, j] = cancer_summaries[ct]["class_means"].get(cid, {}).get("mean", 0)

    fig, ax = plt.subplots(figsize=(12, max(5, len(cancer_types) * 0.5)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.8)
    ax.set_xticks(range(len(class_ids)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(cancer_types)))
    ax.set_yticklabels(cancer_types)
    plt.colorbar(im, ax=ax, label="Mean fraction")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_tumor_detection_bar(
    summary: dict,
    out_path: str | Path,
) -> str | None:
    """Bar chart of tumor detection rate per cancer type."""
    cancer_summaries = summary.get("per_cancer", {})
    if not cancer_summaries:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    cancer_types = sorted(cancer_summaries.keys())
    rates = [cancer_summaries[ct]["tumor_detection_rate"] for ct in cancer_types]

    fig, ax = plt.subplots(figsize=(max(8, len(cancer_types) * 0.5), 4))
    bars = ax.bar(cancer_types, rates, color="#3b82f6", edgecolor="white")
    ax.axhline(y=summary["summary"]["mean_tumor_detection_rate"],
               color="red", linestyle="--", linewidth=1, label="Mean")
    ax.set_ylabel("Tumor Detection Rate (% ROIs with ≥10% neoplastic)")
    ax.set_title("Tumor detection consistency across cancer types")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def print_summary_table(summary: dict) -> str:
    """Pretty-print the generalization summary as a text table."""
    cancer_summaries = summary.get("per_cancer", {})
    if not cancer_summaries:
        return "No data available."

    lines = []
    lines.append(f"{'Cancer Type':<14} {'Cases':>6} {'ROIs':>6} {'Tumor Det%':>10} {'Dead%':>7} {'Top Classes'}")
    lines.append("-" * 80)
    for ct, s in sorted(cancer_summaries.items()):
        top_classes = ", ".join(
            f"{CLASS_NAMES.get(cid, cid)}={m['mean']*100:.0f}%"
            for cid, m in sorted(s["class_means"].items(), key=lambda x: -x[1]["mean"])[:3]
        )
        lines.append(
            f"{ct:<14} {s['case_count']:>6} {s['roi_count']:>6} "
            f"{s['tumor_detection_rate']*100:>9.1f}% {s['avg_dead_pct']:>6.1f}% "
            f" {top_classes}"
        )
    lines.append("-" * 80)
    s = summary["summary"]
    lines.append(f"Mean tumor detection: {s['mean_tumor_detection_rate']*100:.1f}% ± {s['std_tumor_detection_rate']*100:.1f}%")
    d = summary["dead_tissue_sanity_check"]
    lines.append(f"Necrotic cancers dead%: {d['necrotic_cancers_mean_dead_pct']:.1f}% vs non-necrotic: {d['non_necrotic_mean_dead_pct']:.1f}% (passes={d['passes']})")
    return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _cancer_type_from_path(result_path: Path, run_dir: Path) -> str:
    """Extract cancer type from path like .../TCGA-ACC/TCGA-OR-A5J1/pipeline_result.json"""
    try:
        # result_path is .../TCGA-ACC/case_id/pipeline_result.json
        return result_path.parent.parent.name
    except Exception:
        return "unknown"
