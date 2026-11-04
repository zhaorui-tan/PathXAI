"""HTML visualization — renders the rich v1 evidence-review workspace for a v2 case.

Rather than maintaining a parallel (inferior) renderer, this module builds a
payload compatible with the v1 review-workspace renderer
(``scripts/render_review_workspace_html.py`` in the PathXAI_Agent v1 repo) from
the v2 ``pipeline_result.json`` and delegates the actual HTML generation to it.

That renderer produces the tabbed workspace the user designed: Result / Evidence
/ Reasoning / Verification / Audit, with ROI + segmentation overlays, WSI
overviews, per-claim evidence tracing, zoom-in lineage, and the full decision
chain.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any


def _v1_scripts_dir() -> Path:
    """Locate the v1 ``scripts`` directory that holds the review renderer."""
    v1_src = os.environ.get("V1_SRC_ROOT", "/local/PathXAI_Agent/src")
    return Path(v1_src).parent / "scripts"


def _load_json(path: Path) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def _slide_id_from_roi(roi_id: str) -> str:
    rid = str(roi_id or "")
    for marker in ("_foundation_", "_zoom_", "_roi_"):
        if marker in rid:
            return rid.split(marker, 1)[0]
    return rid


def _norm_status(s: str) -> str:
    """Normalise status strings to the values render_audit expects for CSS colors."""
    s = str(s or "").lower()
    # match → green
    if s in ("exact_match", "exact", "strong_match", "implied_by_subtype", "supported"):
        return "match"
    # partial → amber
    if s in ("partial", "partial_match", "synonym_exact", "contained", "token_contained",
             "word_overlap", "hyponym", "lineage_partial", "lineage_entity_partial",
             "subtype_partial", "broad_only", "adjacent", "weak"):
        return "partial"
    # mismatch → red
    if s in ("mismatch", "miss", "missing", "contradicted", "unsupported"):
        return "mismatch"
    # neutral → grey (not_applicable, not_evaluated, not_assessable)
    return "neutral"


def _build_progress_trace(case_dir: Path, result: dict) -> dict:
    """Build a progress_trace dict with stages from v2 checkpoint files."""
    # Map: (checkpoint filename, v1 stage name for render_decisions label matching)
    CKPT_MAP = [
        ("ckpt_01_roi_selection.json",         "01_roi_selection"),
        ("ckpt_02_site_prediction.json",        "02_site_prediction"),
        ("ckpt_03_segmentation.json",           "04_segmentation_qc"),
        ("ckpt_04_morphology.json",             "05_medgemma_morphology"),
        ("ckpt_06_diagnosis.json",              "06_preliminary_diagnosis_sufficiency"),
        ("ckpt_07_evidence_sufficiency.json",   "07_reinspection"),
        ("ckpt_09_evidence_graph.json",         "09_slots_claim_verification"),
        ("ckpt_10_report.json",                 "10_report"),
    ]
    stages = []
    for filename, stage_name in CKPT_MAP:
        p = case_dir / filename
        if p.exists():
            stages.append({"stage": stage_name, "path": str(p)})

    # Insert reinspection rounds between evidence-sufficiency and slots
    for rnd_dir in sorted(case_dir.glob("reinspect_*/"), key=lambda d: d.name):
        rnd_num = rnd_dir.name.split("_")[-1]
        reinspect_ckpt = rnd_dir / "ckpt_reinspect.json"
        if reinspect_ckpt.exists():
            # Insert after 07_reinspection position
            stages.append({"stage": f"07_reinspection_r{rnd_num}", "path": str(reinspect_ckpt)})

    return {"stages": stages}


def _build_diagnostic_hierarchy(case_dir: Path, result: dict) -> dict:
    """Build diagnostic_hierarchy from structured facts (per-report GT), not tiered TCGA GT.

    Tiered scores (from tcga_ground_truth.py) are recorded in eval.json but
    may conflict with per-case reference report facts.  The HTML Audit tab
    shows the structured facts for transparency.
    """
    ev = _load_json(case_dir / "eval.json") or {}
    field_scores_raw = ev.get("field_scores") or {}
    field_scores = {}
    for field, row in field_scores_raw.items():
        if not isinstance(row, dict):
            continue
        field_scores[field] = {
            "score": row.get("score", 0),
            "status": _norm_status(str(row.get("status", ""))),
            "prediction": row.get("prediction", ""),
            "reference": row.get("reference", ""),
            "weight": row.get("weight", 0),
            "applicable": row.get("applicable", True),
        }
    overall = ev.get("hierarchy_score", ev.get("diagnosis_score"))
    return {"overall_score": overall, "field_scores": field_scores}


def _build_eval_with_metrics(case_dir: Path) -> dict:
    """Load eval.json and inject primary_metrics + handle multi-ref evals."""
    ev = _load_json(case_dir / "eval.json") or {}
    if not ev:
        return {}

    # Prefer image_only scores as primary, fallback to flat
    ref_scores = ev.get("reference_scores") or {}
    primary = ref_scores.get("image_only") or ref_scores.get("full") or ev

    primary_metrics = {
        "primary_score": primary.get("composite_score"),
        "image_only_core_score": primary.get("diagnosis_score"),
        "pathology_report_score": primary.get("morph_f1"),
        "full_report_quality_score": primary.get("bleu_4"),
    }
    return {**primary, "primary_metrics": primary_metrics, "reference_scores": ref_scores}


def _enrich_selected_rois(selected: list[dict], seg: dict) -> list[dict]:
    """Merge segmentation roi_results (overlay, composite, mask, stats) into selected ROIs."""
    seg_by_roi: dict[str, dict] = {}
    for rr in (seg.get("roi_results") or []):
        rid = rr.get("roi_id") or ""
        if rid:
            seg_by_roi[rid] = rr
    enriched = []
    for roi in selected:
        rid = roi.get("roi_id") or ""
        sr = seg_by_roi.get(rid) or {}
        enriched.append({
            **roi,
            "overlay_path":   sr.get("overlay_path")   or roi.get("overlay_path"),
            "composite_path": sr.get("composite_path") or roi.get("composite_path"),
            "mask_path":      sr.get("mask_path")      or roi.get("mask_path"),
            "stats":          sr.get("stats")          or roi.get("stats"),
            "seg_qc":         sr.get("seg_qc")         or roi.get("seg_qc"),
        })
    return enriched


def _selected_rois_from_segmentation(seg: dict) -> list[dict]:
    """Recover final selected ROI records from segmentation output.

    Older v2 runs only stored the initial 3 ROIs under ``roi_selection`` even
    after reinspection expanded the case to more ROIs.  The segmentation stage
    sees the final ROI set, so it is the best lossless fallback for HTML
    regeneration.
    """
    selected: list[dict] = []
    for rr in seg.get("roi_results") or []:
        rid = rr.get("roi_id") or ""
        if not rid:
            continue
        coords = _coords_from_roi_id(rid)
        selected.append({
            "roi_id": rid,
            "image_path": rr.get("image_path") or "",
            **coords,
            "score": rr.get("score", 0.0),
            "source": "segmentation_context",
            "metadata": {
                "recovered_from": "segmentation_context.roi_results",
            },
            "overlay_path": rr.get("overlay_path"),
            "composite_path": rr.get("composite_path"),
            "mask_path": rr.get("mask_path"),
            "stats": rr.get("stats"),
        })
    return selected


def _coords_from_roi_id(roi_id: str) -> dict[str, int]:
    m = re.search(r"_x(\d+)_y(\d+)_l(\d+)_w(\d+)_h(\d+)", str(roi_id or ""))
    if not m:
        return {}
    x, y, level, width, height = (int(v) for v in m.groups())
    return {"x": x, "y": y, "level": level, "width": width, "height": height}


def _tumor_rich_refs(selected: list[dict], max_refs: int = 4) -> list[str]:
    ranked: list[tuple[float, int]] = []
    for i, roi in enumerate(selected, start=1):
        stats = roi.get("stats") or {}
        tumor = 0.0
        try:
            tumor = float((stats.get("class_fractions") or {}).get("1", 0.0))
        except Exception:
            tumor = 0.0
        ranked.append((tumor, i))
    ranked.sort(reverse=True)
    refs = [f"FIG-{i}" for _, i in ranked[:max_refs]]
    return refs or [f"FIG-{i}" for i in range(1, min(len(selected), max_refs) + 1)]


def _roi_ids_to_fig_refs(selected: list[dict], roi_ids: list[Any]) -> list[str]:
    refs: list[str] = []
    for roi_id in roi_ids or []:
        rid = str(roi_id or "")
        if not rid:
            continue
        for idx, roi in enumerate(selected, start=1):
            current = str(roi.get("roi_id") or "")
            if rid == current or rid.endswith(current) or current.endswith(rid):
                refs.append(f"FIG-{idx}")
                break
    return refs


def _morph_refs(morph: dict) -> list[str]:
    rounds = morph.get("rounds") or []
    return [f"MORPH-R{i}" for i in range(1, len(rounds) + 1)]


def _knowledge_refs(knowledge: dict, limit: int = 5) -> list[str]:
    if isinstance(knowledge.get("per_hypothesis"), list) and knowledge.get("per_hypothesis"):
        n = len(knowledge.get("per_hypothesis") or [])
    else:
        hits = (knowledge.get("site_subtype_candidates") or []) + (knowledge.get("disease_hits") or [])
        n = len(hits)
    return [f"KEEP-{i}" for i in range(1, min(n, limit) + 1)]


def _clean_claim_reason(text: Any) -> str:
    """Hide non-clickable anonymized ROI hashes from visible claim prose."""
    s = str(text or "")
    s = re.sub(r"\broi::\d+\b", "linked ROI evidence", s)
    return s


def _normalise_claim_evidence_refs(
    row: dict,
    selected: list[dict],
    knowledge: dict,
    morph: dict,
    tumor_refs: list[str],
) -> list[str]:
    """Convert abstract/LLM evidence IDs into clickable workspace refs."""
    morph_refs = _morph_refs(morph)
    keep_refs = _knowledge_refs(knowledge)
    out: list[str] = []

    def add(ref: str | None) -> None:
        if ref and ref not in out:
            out.append(ref)

    for ev in row.get("evidence") or []:
        ev_s = str(ev or "")
        if not ev_s:
            continue
        if ev_s.startswith("FIG-") or ev_s.startswith("MORPH-R") or ev_s.startswith("KEEP-"):
            add(ev_s)
        elif ev_s == "evidence::site":
            add("SITE")
        elif ev_s == "evidence::diagnosis":
            add("DX")
        elif ev_s == "evidence::subtype_sufficiency":
            add("SUBTYPE")
        elif ev_s == "evidence::broad_hypotheses":
            add("BROAD")
        elif ev_s == "evidence::sufficiency":
            add("SUFF")
        elif ev_s.startswith("evidence::morphology") or ev_s.startswith("evidence::visual_description"):
            for ref in morph_refs:
                add(ref)
        elif ev_s.startswith("evidence::knowledge"):
            for ref in keep_refs:
                add(ref)
        elif ev_s.startswith("evidence::segmentation"):
            add("SEG")
        else:
            for ref in _roi_ids_to_fig_refs(selected, [ev_s]):
                add(ref)

    seg_v = row.get("seg_verification") or {}
    supporting = seg_v.get("supporting_rois") or []
    contradicting = seg_v.get("contradicting_rois") or []
    for ref in _roi_ids_to_fig_refs(selected, supporting):
        add(ref)
    for ref in _roi_ids_to_fig_refs(selected, contradicting):
        add(ref)
    if supporting or contradicting:
        add("SEG")

    field = str(row.get("field") or "").lower()
    status = str(row.get("status") or "").lower()
    if field in {
        "primary_diagnosis",
        "broad_type",
        "subtype",
        "malignancy",
        "tumor_architecture",
        "cytology",
        "necrosis",
        "invasion",
        "stroma_inflammation",
    }:
        if not any(ref.startswith("FIG-") for ref in out):
            for ref in tumor_refs:
                add(ref)
        if not any(ref.startswith("MORPH-R") for ref in out):
            for ref in morph_refs:
                add(ref)
        if field in {"primary_diagnosis", "subtype", "broad_type"} and not any(ref.startswith("KEEP-") for ref in out):
            for ref in keep_refs[:3]:
                add(ref)
    elif field == "site":
        add("SITE")
        for ref in morph_refs[:1]:
            add(ref)
    elif status in {"not_assessable", "not assessable"}:
        add("SUFF")

    return out


def _claim_evidence_refs(claims: list[dict], selected: list[dict], knowledge: dict, morph: dict) -> list[dict]:
    tumor_refs = _tumor_rich_refs(selected)
    out = []
    for claim in claims:
        row = dict(claim)
        if row.get("reason"):
            row["reason"] = _clean_claim_reason(row.get("reason"))
        row["evidence"] = _normalise_claim_evidence_refs(row, selected, knowledge, morph, tumor_refs)
        if row["evidence"]:
            out.append(row)
            continue
        field = str(row.get("field") or "").lower()
        status = str(row.get("status") or "").lower()
        claim_text = str(row.get("claim") or "").lower()
        evidence: list[str] = []
        if field == "site":
            evidence.append("SITE")
        elif (
            field in {"ihc_molecular", "grade"}
            or status in {"not_assessable", "not assessable"}
            or any(t in claim_text for t in ("not assessable", "not available", "unavailable", "not provided"))
        ):
            evidence.append("SUFF")
        elif field in {"primary_diagnosis", "broad_type", "subtype", "malignancy", "tumor_architecture", "cytology", "necrosis", "invasion", "stroma_inflammation"}:
            evidence.extend(tumor_refs)
            evidence.extend(_morph_refs(morph))
            evidence.extend(_knowledge_refs(knowledge, limit=3))
            evidence.append("SUFF")
        row["evidence"] = evidence
        out.append(row)
    return out


def _build_wsi_overviews(case_dir: Path, figures: list[dict]) -> list[dict]:
    """Build wsi_overviews payload key from annotated thumbnail files in case_dir."""
    overviews = []
    for ann_path in sorted(case_dir.glob("wsi_thumbnail_*_annotated.jpg")):
        # Filename: wsi_thumbnail_<UUID>_annotated.jpg
        stem = ann_path.stem  # e.g. wsi_thumbnail_145D832C-..._annotated
        uuid_part = stem.removeprefix("wsi_thumbnail_").removesuffix("_annotated")
        # Match figures whose slide_id ends with this UUID
        fig_nums = [
            f["roi_index"]
            for f in figures
            if uuid_part.upper() in (f.get("slide_id") or "").upper()
        ]
        overviews.append({
            "annotated_path": str(ann_path),
            "slide_id": uuid_part,
            "figure_numbers": fig_nums,
        })
    return overviews


def _build_payload(case_dir: Path, result: dict) -> dict:
    """Map a v2 ``pipeline_result.json`` onto the v1 review-workspace payload."""
    roi_selection = result.get("roi_selection") or {}
    seg = result.get("segmentation_context") or {}
    eg = result.get("evidence_graph") or {}
    kc = dict(result.get("knowledge_context") or {})
    morph = result.get("medgemma_morphology_context") or {}
    if kc.get("hypothesis_hits") and not kc.get("per_hypothesis"):
        kc["per_hypothesis"] = kc.get("hypothesis_hits")

    selected_raw = result.get("final_selected_rois") or []
    if not selected_raw:
        selected_raw = roi_selection.get("selected_rois") or []
        expected_count = int(result.get("selected_count") or 0)
        if expected_count and len(selected_raw) < expected_count:
            recovered = _selected_rois_from_segmentation(seg)
            if len(recovered) >= expected_count:
                selected_raw = recovered

    # Enrich selected ROIs with segmentation overlay/composite/mask
    selected = _enrich_selected_rois(selected_raw, seg)

    figures: list[dict] = []
    for i, roi in enumerate(selected, start=1):
        rid = roi.get("roi_id") or f"roi-{i}"
        figures.append({
            "figure": f"Figure {i}",
            "roi_id": rid,
            "roi_index": i,
            "image_path": roi.get("image_path"),
            "composite_path": roi.get("composite_path"),
            "overlay_path": roi.get("overlay_path"),
            "mask_path": roi.get("mask_path"),
            "score": roi.get("score"),
            "evidence_role": roi.get("evidence_role") or roi.get("role") or "",
            "is_zoom": "zoom" in str(rid).lower(),
            "zoom_parent_roi_id": roi.get("zoom_parent_roi_id") or "",
            "slide_id": _slide_id_from_roi(rid),
            "stats": roi.get("stats"),
        })

    verified_claims = _claim_evidence_refs(
        eg.get("verified_claims") or eg.get("claims") or [],
        selected,
        kc,
        morph,
    )
    diagnosis_slots = eg.get("filtered_slots") or eg.get("diagnosis_slots") or {}
    slides = sorted({_slide_id_from_roi(r.get("roi_id")) for r in selected if r.get("roi_id")})

    long_report = result.get("generated_long_report") or ""
    diag_pred = result.get("diagnosis_prediction") or {}
    ev_suff = result.get("evidence_sufficiency") or {}
    site_pred = result.get("site_prediction") or {}

    eval_with_metrics = _build_eval_with_metrics(case_dir)
    diagnostic_hierarchy = _build_diagnostic_hierarchy(case_dir, result)
    progress_trace = _build_progress_trace(case_dir, result)
    wsi_overviews = _build_wsi_overviews(case_dir, figures)

    # Prefer real v2 reasoning artifacts.  Older runs did not have these
    # checkpoints, so keep a small compatibility fallback for HTML regeneration.
    broad_hypotheses = result.get("broad_hypotheses") or kc.get("broad_hypotheses") or {}
    if not isinstance(broad_hypotheses, dict) or not broad_hypotheses:
        broad_hypotheses = {
            "available": False,
            "best_broad_type": diag_pred.get("broad_type") or "",
            "confidence": diag_pred.get("confidence") or "",
            "reason": "fallback from diagnosis_prediction; broad-hypothesis stage was not stored",
        }
    subtype_sufficiency = result.get("subtype_sufficiency") or kc.get("subtype_sufficiency") or {}
    if not isinstance(subtype_sufficiency, dict) or not subtype_sufficiency:
        subtype_sufficiency = {
            "available": False,
            "recommended_primary_diagnosis": diag_pred.get("subtype") or diag_pred.get("primary_diagnosis") or "",
            "best_supported_level": diag_pred.get("subtype") and "subtype" or "broad_type",
            "reason": "fallback from diagnosis_prediction; subtype-sufficiency stage was not stored",
        }

    payload: dict[str, Any] = {
        "case_id": result.get("case_id") or case_dir.name,
        "mode": result.get("mode") or "",
        "cancer_type": case_dir.parent.name if case_dir.parent else "",
        "figures": figures,
        "selected_rois": selected,
        "roi_selection": roi_selection,
        "site_prediction": site_pred,
        "diagnosis_prediction": diag_pred,
        "evidence_sufficiency": ev_suff,
        "segmentation_context": seg,
        "medgemma_morphology_context": morph,
        "knowledge_context": kc,
        "evidence_graph": eg,
        "diagnosis_slots": diagnosis_slots,
        "claim_verification": {"claims": verified_claims},
        "generated_long_report": long_report,
        "generated_long_report_rich": long_report,
        "slides_used": slides,
        "broad_hypotheses": broad_hypotheses,
        "subtype_sufficiency": subtype_sufficiency,
        "wsi_overviews": wsi_overviews,
        "_case_dir": str(case_dir),
    }

    payload["_files"] = {
        "eval": eval_with_metrics,
        "diagnostic_hierarchy": diagnostic_hierarchy,
        "diagnosis_prediction": diag_pred,
        "diagnosis_slots": diagnosis_slots,
        "claim_verification": {"claims": verified_claims},
        "evidence_sufficiency": ev_suff,
        "knowledge_context": kc,
        # medgemma: must be the morphology output with .rounds[] list
        "medgemma": morph,
        "segmentation": seg,
        "site_prediction": site_pred,
        "progress_trace": progress_trace,
        "artifact_index": {"cancer_type": payload["cancer_type"], "slides": slides},
    }
    payload["_reports"] = {
        "long": long_report,
        "rich": long_report,
        "md": "",
    }
    return payload


def render_html(case_dir: str | Path, output_path: str | Path | None = None) -> str:
    """Render the rich v1 evidence-review workspace for a single v2 case.

    Parameters
    ----------
    case_dir:
        Path to the v2 case output directory containing ``pipeline_result.json``.
    output_path:
        Where to write the HTML. Defaults to ``<case_dir>/report.html``.

    Returns the full HTML string.
    """
    case_dir = Path(case_dir).resolve()
    result = _load_json(case_dir / "pipeline_result.json") or {}
    payload = _build_payload(case_dir, result)

    scripts_dir = _v1_scripts_dir()
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from render_review_workspace_html import render_html as _v1_render  # noqa: PLC0415

    out_path = Path(output_path) if output_path else case_dir / "report.html"
    html = _v1_render(payload, out_path)
    out_path.write_text(html, encoding="utf-8")
    return html
