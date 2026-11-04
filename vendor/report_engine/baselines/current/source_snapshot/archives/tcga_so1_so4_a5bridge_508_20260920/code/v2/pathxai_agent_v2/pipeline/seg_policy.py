"""Seg-driven evidence acquisition policy.

A lightweight, rule-based policy that checks segmentation statistics before
invoking the keyword-based reinspection planner.  The policy is designed to be
**ablatable** — toggle it via the ``seg_policy`` component in ablation labels
(``Q3_no_seg_policy`` disables it).

When active, the policy runs BEFORE the keyword-based ReinspectionStage and can
return one of:

* ``stop`` — segmentation signals are strong enough; no more evidence needed.
* ``acquire_tumor_roi`` — average tumor fraction too low for diagnosis.
* ``acquire_interface_roi`` — tumor present but no tumor-stroma interface
  detected (needed for invasion assessment).
* ``acquire_necrosis_roi`` — necrosis claimed but not detected in any ROI.
* ``zoom_current_roi`` — a visually-claimed feature contradicts seg stats
  (e.g., "necrosis present" but dead class = 0%).
* ``None`` — no clear seg-driven signal; fall through to keyword/LLM planner.

The policy is intentionally conservative: it only fires when seg stats give a
clear, interpretable signal.  Ambiguous cases delegate to the downstream planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Tissue class constants (mirrors seg_visual.TISSUE_NAMES) ─────────────────
CLASS_NEOPLASTIC    = "1"
CLASS_INFLAMMATORY  = "2"
CLASS_CONNECTIVE    = "3"
CLASS_EPITHELIAL    = "4"
CLASS_DEAD          = "5"

# ── Thresholds ────────────────────────────────────────────────────────────────
# These are deliberately conservative to avoid false-alarm acquisitions.
MIN_TUMOR_FRAC       = 0.08   # below this → acquire more tumor
GOOD_TUMOR_FRAC      = 0.20   # above this + can_report → consider stop
MIN_DEAD_FOR_NECROSIS = 0.02  # dead tissue fraction to confirm necrosis
MIN_INTERFACE_STROMA = 0.03   # stroma fraction needed for tumor-stroma interface
MIN_INTERFACE_TUMOR  = 0.08   # tumor fraction needed for interface assessment
MIN_INFLAM_FRAC      = 0.01   # inflammatory fraction threshold

# ── Minimum ROI count to consider "sufficient evidence" ───────────────────────
MIN_ROIS_FOR_STOP    = 3


@dataclass
class SegPolicyAction:
    """Structured output from the seg-driven policy."""
    action: str = ""          # "stop" | "acquire_tumor_roi" | "acquire_interface_roi" |
                              # "acquire_necrosis_roi" | "zoom_current_roi" | ""
    reason: str = ""
    priority: int = 0         # higher = more urgent
    evidence: dict = field(default_factory=dict)  # supporting seg stats

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "priority": self.priority,
            "evidence": self.evidence,
        }


def seg_driven_action(
    seg_context: dict,
    diagnosis: Any,  # DiagnosisPrediction (avoid import cycle)
    sufficiency: Any,  # EvidenceSufficiency
) -> SegPolicyAction | None:
    """Check segmentation stats and suggest an evidence-acquisition action.

    Returns a ``SegPolicyAction`` if seg signals are unambiguous, or ``None``
    if seg has no clear recommendation (delegate to downstream planner).

    The returned action can be fed into the existing reinspection machinery
    (``ReinspectionStage`` for ROI selection details) or used directly to
    short-circuit LLM calls.

    Parameters
    ----------
    seg_context:
        ``SegmentationOutput.to_dict()`` — must contain ``roi_results``.
    diagnosis:
        ``DiagnosisPrediction`` instance with ``necrosis``, ``invasion``,
        ``primary_diagnosis`` fields.
    sufficiency:
        ``EvidenceSufficiency`` instance with ``can_confidently_report``.
    """
    roi_results: list[dict] = (seg_context.get("roi_results") or []) if seg_context else []
    if not roi_results:
        return None  # no seg data → can't make seg-driven decisions

    # ── Compute per-class fractions across all ROIs ───────────────────────
    tumor_fracs   = _class_fracs(roi_results, CLASS_NEOPLASTIC)
    dead_fracs    = _class_fracs(roi_results, CLASS_DEAD)
    stroma_fracs  = _class_fracs(roi_results, CLASS_CONNECTIVE)
    inflam_fracs  = _class_fracs(roi_results, CLASS_INFLAMMATORY)

    avg_tumor  = sum(tumor_fracs) / len(tumor_fracs) if tumor_fracs else 0.0
    max_tumor  = max(tumor_fracs) if tumor_fracs else 0.0
    avg_dead   = sum(dead_fracs) / len(dead_fracs) if dead_fracs else 0.0
    max_dead   = max(dead_fracs) if dead_fracs else 0.0

    # ── Count ROIs with tumor-stroma interface ────────────────────────────
    interface_rois = sum(
        1 for t, s in zip(tumor_fracs, stroma_fracs)
        if t >= MIN_INTERFACE_TUMOR and s >= MIN_INTERFACE_STROMA
    )

    can_report = getattr(sufficiency, "can_confidently_report", False)

    # ── Rule 1: tumor fraction critically low ─────────────────────────────
    if max_tumor < MIN_TUMOR_FRAC:
        return SegPolicyAction(
            action="acquire_tumor_roi",
            reason=(
                f"Maximum tumor fraction across ROIs is {max_tumor:.1%} "
                f"(threshold: {MIN_TUMOR_FRAC:.0%}). "
                "Insufficient neoplastic tissue for reliable diagnosis."
            ),
            priority=90,
            evidence={"max_tumor_pct": round(max_tumor * 100, 1), "roi_count": len(roi_results)},
        )

    # ── Rule 2: tumor fraction low — may need reinforcement ───────────────
    if avg_tumor < MIN_TUMOR_FRAC and not can_report:
        return SegPolicyAction(
            action="acquire_tumor_roi",
            reason=(
                f"Average tumor fraction {avg_tumor:.1%} across {len(roi_results)} ROIs "
                f"(threshold: {MIN_TUMOR_FRAC:.0%}). "
                "Acquiring more tumor-rich ROIs is recommended."
            ),
            priority=70,
            evidence={"avg_tumor_pct": round(avg_tumor * 100, 1), "roi_count": len(roi_results)},
        )

    # ── Rule 3: necrosis claimed but not detected → contradiction ─────────
    diag_necrosis = str(getattr(diagnosis, "necrosis", "") or "").lower()
    claims_necrosis = (
        diag_necrosis
        and "absent" not in diag_necrosis
        and "no necro" not in diag_necrosis
        and any(t in diag_necrosis for t in ("necrosis", "necrotic", "dead"))
    )
    if claims_necrosis and max_dead < MIN_DEAD_FOR_NECROSIS:
        return SegPolicyAction(
            action="zoom_current_roi",
            reason=(
                f"Diagnosis claims necrosis ('{diag_necrosis[:80]}') but "
                f"maximum dead tissue fraction is {max_dead:.1%} across all ROIs. "
                "Zoom to verify — necrosis may be focal or misclassified."
            ),
            priority=85,
            evidence={"max_dead_pct": round(max_dead * 100, 1), "necrosis_claim": diag_necrosis[:120]},
        )

    # ── Rule 4: invasion-related claim but no tumor-stroma interface ──────
    diag_text = str(getattr(diagnosis, "primary_diagnosis", "") or "").lower()
    diag_invasion = str(getattr(diagnosis, "invasion", "") or "").lower()
    claims_invasion = (
        ("invasive" in diag_text or "carcinoma" in diag_text or "invasion" in diag_invasion)
        and "in situ" not in diag_text
        and "non-invasive" not in diag_text
        and "noninvasive" not in diag_text
    )
    if claims_invasion and interface_rois == 0:
        return SegPolicyAction(
            action="acquire_interface_roi",
            reason=(
                f"Invasion-related diagnosis ('{diag_text[:60]}') but "
                f"no ROI shows clear tumor-stroma interface "
                f"(need ≥{MIN_INTERFACE_TUMOR:.0%} tumor AND ≥{MIN_INTERFACE_STROMA:.0%} stroma). "
                "Acquiring an interface ROI is recommended for invasion assessment."
            ),
            priority=80,
            evidence={
                "interface_rois": interface_rois,
                "total_rois": len(roi_results),
            },
        )

    # ── Rule 5: all signals strong → suggest stop ─────────────────────────
    # Stop when seg stats are strong AND diagnosis is confident.
    # Do NOT require can_confidently_report (LLM too conservative) or zero gaps
    # (LLM always finds something missing).
    is_confident = getattr(diagnosis, "confidence", "low") == "high"
    if (
        len(roi_results) >= MIN_ROIS_FOR_STOP
        and avg_tumor >= GOOD_TUMOR_FRAC
        and is_confident
    ):
        return SegPolicyAction(
            action="stop",
            reason=(
                f"Strong signals: {len(roi_results)} ROIs, "
                f"avg tumor {avg_tumor:.1%}, confidence={getattr(diagnosis, 'confidence', '?')}. "
                "No seg-driven gaps detected."
            ),
            priority=50,
            evidence={
                "avg_tumor_pct": round(avg_tumor * 100, 1),
                "max_tumor_pct": round(max_tumor * 100, 1),
                "interface_rois": interface_rois,
                "roi_count": len(roi_results),
            },
        )

    # ── No clear signal → delegate ────────────────────────────────────────
    return None


def seg_policy_summary(
    seg_context: dict,
    diagnosis: Any,
    sufficiency: Any,
) -> dict:
    """Return a human-readable summary of seg stats for logging/debugging.

    Does NOT make decisions — purely informational.  Useful for comparing
    seg-driven vs keyword-driven decision traces.
    """
    roi_results: list[dict] = (seg_context.get("roi_results") or []) if seg_context else []
    if not roi_results:
        return {"available": False}

    tumor_fracs = _class_fracs(roi_results, CLASS_NEOPLASTIC)
    dead_fracs  = _class_fracs(roi_results, CLASS_DEAD)
    stroma_fracs = _class_fracs(roi_results, CLASS_CONNECTIVE)
    inflam_fracs = _class_fracs(roi_results, CLASS_INFLAMMATORY)

    interface_rois = sum(
        1 for t, s in zip(tumor_fracs, stroma_fracs)
        if t >= MIN_INTERFACE_TUMOR and s >= MIN_INTERFACE_STROMA
    )

    return {
        "available": True,
        "roi_count": len(roi_results),
        "avg_tumor_pct": round(sum(tumor_fracs) / len(tumor_fracs) * 100, 1) if tumor_fracs else 0,
        "max_tumor_pct": round(max(tumor_fracs) * 100, 1) if tumor_fracs else 0,
        "avg_dead_pct": round(sum(dead_fracs) / len(dead_fracs) * 100, 1) if dead_fracs else 0,
        "max_dead_pct": round(max(dead_fracs) * 100, 1) if dead_fracs else 0,
        "interface_rois": interface_rois,
        "avg_stroma_pct": round(sum(stroma_fracs) / len(stroma_fracs) * 100, 1) if stroma_fracs else 0,
        "avg_inflam_pct": round(sum(inflam_fracs) / len(inflam_fracs) * 100, 1) if inflam_fracs else 0,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _class_fracs(roi_results: list[dict], class_id: str) -> list[float]:
    """Extract per-ROI fraction for a single class."""
    return [
        float(r.get("stats", {}).get("class_fractions", {}).get(class_id, 0))
        for r in roi_results
    ]
