"""Claim-level segmentation verification.

Maps every diagnostic claim to segmentation class statistics and produces
per-claim structured support/contradiction scores.  This is the **structured
visual evidence backbone** of the agent — claims cannot pass unless seg stats
support them.

Design principle:
    Segmentation statistics are the GROUND TRUTH layer.
    MedGemma and KEEP are supplementary — they cannot override seg.

Ablation:
    This module is always active when segmentation is available.
    To ablate: use ``E1_visual_only`` (no seg) vs ``E2_visual_seg`` (seg on).
"""

from __future__ import annotations

import numpy as np
from typing import Any

# ── Tissue class constants (mirrors seg_visual.TISSUE_NAMES) ─────────────────
CLASS_NEOPLASTIC   = "1"
CLASS_INFLAMMATORY = "2"
CLASS_CONNECTIVE   = "3"
CLASS_DEAD         = "5"

# ── Claim → Seg class mapping ─────────────────────────────────────────────────
# Each rule maps a claim field to: class_id(s), class_name(s), check_fn, threshold.
# check_fn(claim_text: str, frac_or_fracs: float | dict) → float in [-1, 1]
#   +1 = strongly supported, 0 = not visible, -1 = contradicted

def _necrosis_score(claim_text: str, dead_frac: float) -> float:
    has_claim = any(t in claim_text for t in
        ("necrosis", "necrotic", "dead", "infarction", "ghost cells"))
    claims_absent = any(t in claim_text for t in
        ("no necrosis", "no necrotic", "absent", "no coagulative", "no tumor necrosis"))
    if claims_absent and dead_frac < 0.01:
        return 1.0
    elif claims_absent and dead_frac > 0.05:
        return -1.0
    elif has_claim and dead_frac > 0.05:
        return 1.0
    elif has_claim and dead_frac < 0.01:
        return -0.5
    return 0.0


def _interface_score(fracs: dict) -> float:
    tumor = fracs.get(CLASS_NEOPLASTIC, 0)
    stroma = fracs.get(CLASS_CONNECTIVE, 0)
    if tumor > 0.10 and stroma > 0.05:
        return min(1.0, (tumor * stroma) / 0.05)
    elif tumor > 0.10:
        return 0.3
    return -0.5


def _stroma_inflam_score(_claim_text: str, fracs: dict) -> float:
    stroma = fracs.get(CLASS_CONNECTIVE, 0)
    inflam = fracs.get(CLASS_INFLAMMATORY, 0)
    score = 0.0
    if stroma > 0.05:
        score += 0.3
    if inflam > 0.01:
        score += 0.4
    return min(1.0, score)


def _tumor_support(tumor_frac: float, min_pct: float = 0.10, ceiling: float = 0.30) -> float:
    if tumor_frac > ceiling:
        return 1.0
    elif tumor_frac > min_pct:
        return (tumor_frac - min_pct) / (ceiling - min_pct)
    else:
        return -0.5


def _grade_support(fracs: dict) -> float:
    tumor = fracs.get(CLASS_NEOPLASTIC, 0)
    necrosis = fracs.get(CLASS_DEAD, 0)
    if tumor > 0.15:
        return 0.5 + min(0.5, necrosis * 5)
    return -0.3


CLAIM_SEG_RULES: dict[str, dict] = {
    "necrosis": {
        "class_ids": ["5"], "class_names": ["Dead/Necrotic"],
        "check_fn": lambda claim, f: _necrosis_score(claim, f.get("5", 0)),
        "description": "Matches necrosis claim to dead tissue fraction",
    },
    "invasion": {
        "class_ids": ["1", "3"], "class_names": ["Neoplastic", "Connective/Stroma"],
        "check_fn": lambda claim, f: _interface_score(f),
        "description": "Requires tumor-stroma interface for invasion assessment",
    },
    "invasion_status": {
        "class_ids": ["1", "3"], "class_names": ["Neoplastic", "Connective/Stroma"],
        "check_fn": lambda claim, f: _interface_score(f),
        "description": "Invasion status requires tumor-stroma interface",
    },
    "stroma_inflammation": {
        "class_ids": ["3", "2"], "class_names": ["Connective/Stroma", "Inflammatory"],
        "check_fn": lambda claim, f: _stroma_inflam_score(claim, f),
        "description": "Stroma + inflammatory cell presence",
    },
    "tumor_architecture": {
        "class_ids": ["1"], "class_names": ["Neoplastic"],
        "check_fn": lambda claim, f: _tumor_support(f.get("1", 0), 0.10, 0.30),
        "description": "Requires >=10% neoplastic tissue",
    },
    "cytology": {
        "class_ids": ["1"], "class_names": ["Neoplastic"],
        "check_fn": lambda claim, f: _tumor_support(f.get("1", 0), 0.05, 0.20),
        "description": "Requires >=5% neoplastic tissue for cytologic assessment",
    },
    "primary_diagnosis": {
        "class_ids": ["1"], "class_names": ["Neoplastic"],
        "check_fn": lambda claim, f: _tumor_support(f.get("1", 0), 0.10, 0.30),
        "description": "Primary diagnosis requires >=10% tumor",
    },
    "broad_type": {
        "class_ids": ["1"], "class_names": ["Neoplastic"],
        "check_fn": lambda claim, f: 0.5 if f.get("1", 0) > 0.05 else -0.5,
        "description": "Broad type requires >=5% tumor",
    },
    "subtype": {
        "class_ids": ["1"], "class_names": ["Neoplastic"],
        "check_fn": lambda claim, f: _tumor_support(f.get("1", 0), 0.15, 0.40),
        "description": "Subtype requires >=15% tumor (preferably >=40%)",
    },
    "malignancy": {
        "class_ids": ["1"], "class_names": ["Neoplastic"],
        "check_fn": lambda claim, f: _tumor_support(f.get("1", 0), 0.10, 0.30),
        "description": "Malignancy assessment supported by tumor fraction",
    },
    "malignancy_status": {
        "class_ids": ["1"], "class_names": ["Neoplastic"],
        "check_fn": lambda claim, f: _tumor_support(f.get("1", 0), 0.10, 0.30),
        "description": "Malignancy status requires tumor presence",
    },
    "grade": {
        "class_ids": ["1", "5"], "class_names": ["Neoplastic", "Dead/Necrotic"],
        "check_fn": lambda claim, f: _grade_support(f),
        "description": "Grade supported by tumor + necrosis fractions",
    },
    "site": {
        "class_ids": [], "class_names": [],
        "check_fn": lambda claim, f: 1.0,  # known input
        "description": "Site is known input — always supported",
    },
    "ihc_molecular": {
        "class_ids": [], "class_names": [],
        "check_fn": lambda claim, f: 0.0,  # never seg-verifiable
        "description": "IHC/molecular results cannot be verified from H&E segmentation",
    },
    "differential_diagnosis": {
        "class_ids": [], "class_names": [],
        "check_fn": lambda claim, f: 0.0,
        "description": "Differential is reasoning metadata, not a verifiable claim",
    },
    "confidence": {
        "class_ids": [], "class_names": [],
        "check_fn": lambda claim, f: 0.0,
        "description": "Confidence is metadata, not a verifiable claim",
    },
}


def verify_claim_with_seg(
    claim: dict,
    seg_roi_results: list[dict],
) -> dict:
    """Compute structured seg support for a single diagnostic claim.

    Parameters
    ----------
    claim:
        Dict with ``field`` (str) and ``claim`` (str).
    seg_roi_results:
        List of per-ROI results from ``SegmentationOutput.roi_results``.
        Each must have ``stats.class_fractions``.

    Returns
    -------
    dict with:
        status: "supported" | "weak" | "contradicted" | "not_visible" | "not_assessable"
        alignment_score: float in [-1, 1]
        supporting_rois: list of roi_ids
        contradicting_rois: list of roi_ids
        per_roi_scores: {roi_id: score}
        aggregate_class_fractions: {class_name: max_pct}
        reason: str
    """
    field = claim.get("field", "")
    claim_text = str(claim.get("claim", "")).lower()

    rule = CLAIM_SEG_RULES.get(field)
    if not rule or not rule.get("class_ids"):
        return {
            "status": "not_assessable",
            "alignment_score": 0.0,
            "supporting_rois": [],
            "contradicting_rois": [],
            "per_roi_scores": {},
            "aggregate_class_fractions": {},
            "reason": f"no seg verification rule for field: {field}",
        }

    class_ids = rule["class_ids"]
    check_fn = rule["check_fn"]
    per_roi_scores: dict[str, float] = {}
    for rr in seg_roi_results:
        rid = rr.get("roi_id", "?")
        fracs = {
            cid: float(rr.get("stats", {}).get("class_fractions", {}).get(cid, 0))
            for cid in class_ids
        }
        per_roi_scores[rid] = check_fn(claim_text, fracs)

    supporting = [rid for rid, s in per_roi_scores.items() if s > 0.3]
    contradicting = [rid for rid, s in per_roi_scores.items() if s < -0.3]
    scores = list(per_roi_scores.values())
    alignment = float(np.mean(scores)) if scores else 0.0

    if alignment > 0.5:
        status = "supported"
    elif alignment > 0.0:
        status = "weak"
    elif alignment > -0.5:
        status = "not_visible"
    else:
        status = "contradicted"

    agg_fracs = _aggregate_max_fractions(seg_roi_results, rule)

    return {
        "status": status,
        "alignment_score": round(alignment, 4),
        "supporting_rois": supporting,
        "contradicting_rois": contradicting,
        "per_roi_scores": {rid: round(s, 4) for rid, s in per_roi_scores.items()},
        "aggregate_class_fractions": agg_fracs,
        "reason": _build_reason(status, alignment, len(supporting), len(contradicting), rule),
    }


def verify_all_claims(
    claims: list[dict],
    seg_roi_results: list[dict],
    *,
    fallback_verifier: bool = True,
) -> tuple[list[dict], dict]:
    """Verify all claims against segmentation.

    Parameters
    ----------
    claims:
        List of claim dicts from ``_slots_to_claims()``.
    seg_roi_results:
        Segmentation ROI results.
    fallback_verifier:
        If True, claims not covered by seg rules are marked "not_assessable"
        but included. If False, they are skipped (for ablation comparison).

    Returns
    -------
    (enriched_claims, summary)
        enriched_claims: claims with ``seg_verification`` key added.
        summary: {"total": N, "supported": N, "weak": N, "contradicted": N,
                   "not_visible": N, "not_assessable": N}
    """
    enriched = []
    summary = {
        "total": len(claims), "supported": 0, "weak": 0,
        "contradicted": 0, "not_visible": 0, "not_assessable": 0,
    }
    for claim in claims:
        result = verify_claim_with_seg(claim, seg_roi_results)
        enriched.append({**claim, "seg_verification": result})
        summary[result["status"]] = summary.get(result["status"], 0) + 1
    return enriched, summary


# ── Internal helpers ──────────────────────────────────────────────────────────

def _aggregate_max_fractions(roi_results: list[dict], rule: dict) -> dict:
    """Max per-class fraction across all ROIs, as percentages."""
    result = {}
    for cid, cname in zip(rule.get("class_ids", []), rule.get("class_names", [])):
        max_f = max(
            (float(r.get("stats", {}).get("class_fractions", {}).get(cid, 0))
             for r in roi_results),
            default=0.0,
        )
        result[cname] = round(max_f * 100, 1)
    return result


def _build_reason(status: str, alignment: float, n_sup: int, n_con: int, rule: dict) -> str:
    desc = rule.get("description", "")
    if status == "supported":
        return f"{desc}. Aligned: {n_sup}/{n_sup + n_con} ROIs (score={alignment:.2f})."
    elif status == "contradicted":
        return f"{desc}. Contradicted: {n_con} ROIs show opposite pattern (score={alignment:.2f})."
    elif status == "not_visible":
        return f"{desc}. Not detected in any ROI (score={alignment:.2f})."
    else:
        return f"{desc}. Score={alignment:.2f}."
