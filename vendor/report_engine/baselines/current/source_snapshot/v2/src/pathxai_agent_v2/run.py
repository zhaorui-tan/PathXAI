#!/usr/bin/env python3
"""PathXAI v2 — Main entry point for running the pathology report generation pipeline.

Usage:
    python -m pathxai_agent_v2.run --case-id TCGA-XX-XXXX \\
        --slide /path/to/slide.svs --output outputs/

Or programmatically:

    from pathxai_agent_v2.run import run_single_case
    result = run_single_case(settings, case_id, candidate_rois, out_dir)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pathxai_agent_v2.config import Settings, load_settings
from pathxai_agent_v2.leakage_guard import (
    DiagnosisPriors,
    assert_no_cancer_type_leak,
)
from pathxai_agent_v2.llm import LegacyQwenAdapter, LLMClient
from pathxai_agent_v2.pipeline.base import RoiInfo
from pathxai_agent_v2.pipeline.orchestrator import EnhancedOrchestrator, EnhancedPipelineResult

REF_FACTS_SCHEMA_VERSION = "ref_facts_v3_tcga_anchor_20260703"


def _roi_image_is_readable(path: str) -> tuple[bool, str]:
    p = Path(path or "")
    if not p.exists():
        return False, "missing"
    try:
        from PIL import Image
        with Image.open(p) as im:
            im.verify()
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _repair_roi_image_from_metadata(roi: dict, *, reason: str = "") -> tuple[bool, str]:
    """Repair a corrupt cached ROI PNG/JPEG by re-cropping from the source WSI."""
    if os.environ.get("PATHXAI_REPAIR_BAD_ROI_CACHE", "1").strip().lower() in {"0", "false", "no"}:
        return False, "repair disabled"
    image_path = Path(str(roi.get("image_path") or ""))
    meta = roi.get("metadata") or {}
    slide_path = Path(str(meta.get("slide_path") or ""))
    if not image_path or not slide_path.exists():
        return False, "missing image_path or slide_path"
    try:
        x = int(roi.get("x"))
        y = int(roi.get("y"))
        w = int(roi.get("width"))
        h = int(roi.get("height"))
        level = int(roi.get("level", 0) or 0)
    except Exception as exc:
        return False, f"bad ROI geometry: {exc}"
    try:
        from pathxai_agent.wsi.slide_reader import SlideReader  # type: ignore
        reader = SlideReader(str(slide_path))
        img = reader.read_region(x, y, w, h, level)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if image_path.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = image_path.with_suffix(image_path.suffix + f".broken_{stamp}")
            shutil.copy2(image_path, backup)
        tmp = image_path.with_suffix(image_path.suffix + ".tmp")
        fmt = "JPEG" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "PNG"
        save_kwargs = {"quality": 92} if fmt == "JPEG" else {}
        img.save(tmp, format=fmt, **save_kwargs)
        tmp.replace(image_path)
        ok, verify_reason = _roi_image_is_readable(str(image_path))
        if not ok:
            return False, f"repair wrote unreadable image: {verify_reason}"
        return True, f"repaired from WSI after {reason}"
    except Exception as exc:
        return False, f"repair failed: {type(exc).__name__}: {exc}"


def _filter_unreadable_candidate_rois(
    candidates: list[dict],
    *,
    case_dir: Path | None = None,
    case_id: str = "",
) -> list[dict]:
    """Drop corrupted ROI image files before any VLM stage sees them.

    A single half-written PNG in shared ROI cache otherwise aborts ROI
    selection for every ablation variant that reuses the same manifest.
    """
    good: list[dict] = []
    bad: list[dict[str, Any]] = []
    repaired: list[dict[str, Any]] = []
    for idx, roi in enumerate(candidates, start=1):
        ok, reason = _roi_image_is_readable(str(roi.get("image_path") or ""))
        if not ok:
            fixed, repair_reason = _repair_roi_image_from_metadata(roi, reason=reason)
            if fixed:
                repaired.append({
                    "candidate_index": idx,
                    "case_id": case_id,
                    "roi_id": roi.get("roi_id", ""),
                    "image_path": roi.get("image_path", ""),
                    "reason": repair_reason,
                })
                ok = True
                reason = ""
        if ok:
            good.append(roi)
        else:
            bad.append({
                "candidate_index": idx,
                "case_id": case_id,
                "roi_id": roi.get("roi_id", ""),
                "image_path": roi.get("image_path", ""),
                "reason": reason,
            })
    if (bad or repaired) and case_dir is not None:
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "bad_roi_images.json").write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "bad_count": len(bad),
                    "repaired_count": len(repaired),
                    "input_candidate_count": len(candidates),
                    "kept_candidate_count": len(good),
                    "repaired_rois": repaired,
                    "bad_rois": bad,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
    if bad:
        print(
            f"  [warn] {case_id}: filtered {len(bad)} unreadable ROI image(s); "
            f"kept {len(good)}/{len(candidates)}",
            flush=True,
        )
    if repaired:
        print(
            f"  [repair] {case_id}: repaired {len(repaired)} cached ROI image(s)",
            flush=True,
        )
    return good


def run_single_case(
    settings: Settings,
    case_id: str,
    candidate_rois: list[dict],
    out_dir: str | Path,
    *,
    diagnosis_priors: DiagnosisPriors | None = None,
    stages_llm: LLMClient | None = None,
    medgemma_client: LLMClient | None = None,
    seg_reader_client: LLMClient | None = None,
    report_llm_client: LLMClient | None = None,
    dry_run: bool = False,
    resume: bool = True,
    ablation: str = "full",
) -> EnhancedPipelineResult:
    """Run the full pipeline for a single case.

    Parameters
    ----------
    settings:
        Pipeline configuration.
    case_id:
        Unique case identifier.
    candidate_rois:
        List of ROI candidate dicts (from CONCH ranking / PathAgent).
        Each dict must have: roi_id, image_path, x, y, width, height, score.
    out_dir:
        Output directory for this case (checkpoints, report, trace).
    diagnosis_priors:
        Optional known site (must be explicitly opted-in via LeakageConfig).
    stages_llm:
        LLM client for pipeline stages. If None, creates one from settings.
    dry_run:
        Skip LLM calls, produce placeholder output.
    resume:
        Resume from saved checkpoints if available.
    """
    # Initialize LLM client if not provided
    if stages_llm is None:
        if dry_run:
            stages_llm = _create_dummy_client()
        else:
            stages_llm = _create_client(settings)
    if report_llm_client is None and not dry_run and settings.llm.report_api:
        report_llm_client = _create_report_client(settings)

    out_dir = Path(out_dir)
    candidate_rois = _filter_unreadable_candidate_rois(
        list(candidate_rois),
        case_dir=out_dir,
        case_id=case_id,
    )
    if not candidate_rois:
        raise ValueError(f"No readable candidate ROI images for case {case_id}")

    # Convert candidate dicts to RoiInfo
    rois = [RoiInfo.from_candidate(r) for r in candidate_rois]

    # Resolve diagnosis priors from leakage config
    if diagnosis_priors is None:
        diagnosis_priors = DiagnosisPriors()  # no priors by default

    # Run pipeline
    orchestrator = EnhancedOrchestrator(
        settings, stages_llm, out_dir,
        medgemma_client=medgemma_client,
        seg_reader_client=seg_reader_client,
        report_llm_client=report_llm_client,
        ablation=ablation,
    )
    result = orchestrator.run(
        case_id=case_id,
        candidate_rois=rois,
        diagnosis_priors=diagnosis_priors,
        dry_run=dry_run,
        resume=resume,
    )

    return result


def run_from_manifest(
    settings: Settings,
    manifest_path: str | Path,
    out_root: str | Path,
    *,
    stages_llm: LLMClient | None = None,
    medgemma_client: LLMClient | None = None,
    seg_reader_client: LLMClient | None = None,
    report_llm_client: LLMClient | None = None,
    dry_run: bool = False,
    max_cases: int | None = None,
    resume: bool = True,
    retry_errors: bool = False,
    ablation: str = "full",
    reference_dir: str | Path | None = None,
    round_robin_by_cancer: bool = False,
) -> list[EnhancedPipelineResult]:
    """Run the pipeline for all cases in a JSONL manifest.

    Manifest format (one JSON object per line):
        {"case_id": "TCGA-XX-XXXX", "candidate_rois": [...], "cancer_type": "TCGA-ACC", ...}
    """
    manifest_path = Path(manifest_path)
    out_root = Path(out_root)

    if stages_llm is None and not dry_run:
        stages_llm = _create_client(settings)
    elif dry_run:
        stages_llm = _create_dummy_client()
    if seg_reader_client is None and not dry_run and _ablation_uses_seg_reader(ablation):
        seg_reader_client = _create_seg_reader_client(settings)
    if report_llm_client is None and not dry_run and settings.llm.report_api:
        report_llm_client = _create_report_client(settings)

    # ── Pre-generate aligned refs + facts for all cases ────────────────────
    _prep_refs_if_needed(manifest_path, stages_llm, dry_run)

    items: list[tuple[int, dict]] = []
    with manifest_path.open(encoding="utf-8") as f:
        for raw_i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            items.append((raw_i, json.loads(line)))

    if round_robin_by_cancer:
        items = _round_robin_manifest_items(items)

    if max_cases:
        items = items[:max_cases]

    results = []
    total_items = len(items)
    for i, (_raw_i, item) in enumerate(items):
        case_id = str(item.get("case_id", f"case_{i:04d}"))
        cancer_type = str(item.get("cancer_type", ""))
        candidates = item.get("candidate_rois", [])
        candidates = _attach_slide_paths_to_candidates(candidates, settings)

        # Resolve diagnosis priors from TCGA cancer type
        priors = DiagnosisPriors()
        if settings.leakage.known_site_from == "tcga" and cancer_type:
            try:
                from pathxai_agent.datasets.tcga_sites import known_site_for_tcga as _tcga_fn  # type: ignore
                from pathxai_agent_v2.leakage_guard import resolve_diagnosis_priors
                priors = resolve_diagnosis_priors(
                    settings.leakage, cancer_type, known_site_for_tcga=_tcga_fn
                )
            except Exception as _e:
                print(f"  [warn] could not resolve TCGA site for {cancer_type}: {_e}", flush=True)

        case_dir = out_root / cancer_type / case_id if cancer_type else out_root / case_id

        # A valid completion is authoritative even if an older error marker
        # remains beside it.  Error-only and never-started cases reach the
        # normal execution path below, so resume already retries them.
        result_path = case_dir / "pipeline_result.json"
        error_path = case_dir / "error.json"
        if resume and result_path.exists():
            pipeline = _load_pipeline_json(case_dir)
            report_text = _report_text_from_pipeline(case_dir, pipeline)
            decision = _completed_case_resume_decision(
                report_text,
                error_exists=error_path.exists(),
            )
            if decision == "rerun_invalid":
                print(f"[{i+1}/{total_items}] {case_id} — resume invalid report, rerunning", flush=True)
                pass  # re-run
            else:
                status = _repair_completed_case(
                    settings=settings,
                    case_dir=case_dir,
                    cancer_type=cancer_type,
                    case_id=case_id,
                    candidates=candidates,
                    reference_dir=reference_dir,
                    stages_llm=stages_llm,
                    dry_run=dry_run,
                )
                stale_error = ", stale_error=ignored" if decision == "reuse_stale_error" else ""
                print(f"[{i+1}/{total_items}] {case_id} — resume {status}{stale_error}", flush=True)
                continue

        print(f"[{i+1}/{total_items}] {case_id} ({cancer_type})", flush=True)

        try:
            result = run_single_case(
                settings=settings,
                case_id=case_id,
                candidate_rois=candidates,
                out_dir=case_dir,
                diagnosis_priors=priors,
                stages_llm=stages_llm,
                medgemma_client=medgemma_client,
                seg_reader_client=seg_reader_client,
                report_llm_client=report_llm_client,
                dry_run=dry_run,
                resume=resume,
                ablation=ablation,
            )
            results.append(result)

            # ── WSI thumbnails with ROI bounding boxes ─────────────────
            try:
                from pathxai_agent_v2.pipeline.wsi_thumbnail import generate_wsi_thumbnails
                _sel_rois = [r.to_dict() for r in (result.final_selected_rois or [])]
                if not _sel_rois and result.roi_selection:
                    _sel_rois = [r.to_dict() for r in result.roi_selection.selected_rois]
                generate_wsi_thumbnails(case_dir, candidates, _sel_rois)
            except Exception as wsi_exc:
                print(f"  [wsi] skipped: {wsi_exc}", flush=True)

            # ── Per-case evaluation ────────────────────────────────────
            if reference_dir and not dry_run and result.report:
                _write_metadata_only_eval_from_result(settings, result, cancer_type, case_id, case_dir)
                _eval_case(settings, result, cancer_type, case_id, case_dir, reference_dir,
                           stages_llm=stages_llm)

            # ── HTML visualization ─────────────────────────────────────
            try:
                from pathxai_agent_v2.html_viz import render_html
                render_html(case_dir, case_dir / "report.html")
            except Exception as html_exc:
                print(f"  [html] skipped: {html_exc}", flush=True)

        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "error.json").write_text(json.dumps({"error": str(exc)}))
            results.append(EnhancedPipelineResult(
                case_id=case_id,
                errors=[str(exc)],
            ))

    return results


def _round_robin_manifest_items(items: list[tuple[int, dict]]) -> list[tuple[int, dict]]:
    """Interleave manifest cases by cancer type while preserving per-type order."""
    buckets: dict[str, list[tuple[int, dict]]] = {}
    order: list[str] = []
    for item in items:
        cancer_type = str(item[1].get("cancer_type", ""))
        if cancer_type not in buckets:
            buckets[cancer_type] = []
            order.append(cancer_type)
        buckets[cancer_type].append(item)

    out: list[tuple[int, dict]] = []
    while any(buckets.values()):
        for cancer_type in order:
            bucket = buckets[cancer_type]
            if bucket:
                out.append(bucket.pop(0))
    return out


def _repair_completed_case(
    *,
    settings: Settings,
    case_dir: Path,
    cancer_type: str,
    case_id: str,
    candidates: list[dict],
    reference_dir: str | Path | None,
    stages_llm,
    dry_run: bool,
) -> str:
    """Repair derived artifacts for a completed case without rerunning inference."""
    actions: list[str] = []

    if reference_dir and not dry_run:
        pipeline = _load_pipeline_json(case_dir)
        _write_metadata_only_eval_from_pipeline(
            settings=settings,
            case_dir=case_dir,
            cancer_type=cancer_type,
            case_id=case_id,
            pipeline=pipeline,
        )
        if not (case_dir / "eval.json").exists():
            ok = _eval_case_from_pipeline(
                settings=settings,
                case_dir=case_dir,
                cancer_type=cancer_type,
                case_id=case_id,
                reference_dir=reference_dir,
                stages_llm=stages_llm,
            )
            actions.append("eval=ok" if ok else "eval=blocked")
        else:
            actions.append("eval=exists")

    try:
        from pathxai_agent_v2.pipeline.wsi_thumbnail import generate_wsi_thumbnails
        pipeline = _load_pipeline_json(case_dir)
        selected = _selected_rois_from_pipeline(pipeline)
        if selected:
            generate_wsi_thumbnails(case_dir, candidates, selected)
            actions.append("wsi=ok")
    except Exception as exc:
        actions.append(f"wsi=skip:{exc}")

    if not dry_run and not (case_dir / "report.html").exists():
        try:
            from pathxai_agent_v2.html_viz import render_html
            render_html(case_dir, case_dir / "report.html")
            actions.append("html=ok")
        except Exception as exc:
            actions.append(f"html=skip:{exc}")
    elif (case_dir / "report.html").exists():
        actions.append("html=exists")

    return ", ".join(actions) if actions else "complete"


def _load_pipeline_json(case_dir: Path) -> dict[str, Any]:
    path = case_dir / "pipeline_result.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _report_text_from_pipeline(case_dir: Path, pipeline: dict[str, Any]) -> str:
    report_path = case_dir / "generated_long_report.txt"
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            return text
    return str(pipeline.get("generated_long_report") or "").strip()


def _is_invalid_completed_report(report_text: str) -> bool:
    text = (report_text or "").strip()
    if not text:
        return True
    lowered = text.lower()
    if "dry_run" in lowered or "no diagnosis generated" in lowered:
        return True
    return False


def _completed_case_resume_decision(report_text: str, *, error_exists: bool) -> str:
    """Return the safe resume action for a case with a result artifact.

    Successful artifacts take precedence over stale failure markers.  The
    marker is retained for provenance; only an invalid completed report is
    rerun.  Cases without ``pipeline_result.json`` never call this helper and
    are retried by the normal batch execution path.
    """
    if _is_invalid_completed_report(report_text):
        return "rerun_invalid"
    if error_exists:
        return "reuse_stale_error"
    return "reuse"


def _selected_rois_from_pipeline(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    rois = pipeline.get("final_selected_rois")
    if isinstance(rois, list) and rois:
        return [r for r in rois if isinstance(r, dict)]
    roi_selection = pipeline.get("roi_selection") or {}
    rois = roi_selection.get("selected_rois") if isinstance(roi_selection, dict) else None
    if isinstance(rois, list):
        return [r for r in rois if isinstance(r, dict)]
    return []


def _write_metadata_only_eval_from_result(
    settings: Settings,
    result: EnhancedPipelineResult,
    cancer_type: str,
    case_id: str,
    case_dir: Path,
) -> bool:
    generated = result.report.long_report if result.report else ""
    diag = result.diagnosis_prediction.to_dict() if result.diagnosis_prediction else {}
    return _write_metadata_only_eval(
        settings=settings,
        case_dir=case_dir,
        cancer_type=cancer_type,
        case_id=case_id,
        generated=generated,
        diagnosis_prediction=diag,
    )


def _write_metadata_only_eval_from_pipeline(
    *,
    settings: Settings,
    case_dir: Path,
    cancer_type: str,
    case_id: str,
    pipeline: dict[str, Any],
) -> bool:
    if not pipeline:
        return False
    return _write_metadata_only_eval(
        settings=settings,
        case_dir=case_dir,
        cancer_type=cancer_type,
        case_id=case_id,
        generated=_report_text_from_pipeline(case_dir, pipeline),
        diagnosis_prediction=pipeline.get("diagnosis_prediction") if isinstance(pipeline.get("diagnosis_prediction"), dict) else {},
    )


def _write_metadata_only_eval(
    *,
    settings: Settings,
    case_dir: Path,
    cancer_type: str,
    case_id: str,
    generated: str,
    diagnosis_prediction: dict[str, Any] | None,
) -> bool:
    """Write cohort-anchor-only eval for cases without report references.

    This is intentionally separate from eval.json.  It uses the TCGA project
    label only as the reference side and scores broad lineage, cohort subtype,
    and malignancy.  It does not score report text overlap, morphology, grade,
    necrosis, invasion, IHC, margins, or any case-specific report field.
    """
    from pathxai_agent_v2.eval.metrics import evaluate_diagnosis_tiers
    from pathxai_agent_v2.eval.tcga_ground_truth import ground_truth_for

    gt = ground_truth_for(cancer_type)
    if not gt:
        return False

    diag = diagnosis_prediction or {}
    reference_stub = (
        f"Diagnosis: {gt.get('subtype', 'not specified')}\n"
        f"Tumor/Lesion Description: {gt.get('lineage', 'not specified')}\n"
        f"Stage/Grade/Prognostic Features: Not assessable from metadata-only evaluation\n"
        f"Final Summary: {gt.get('subtype', 'not specified')}; "
        f"{gt.get('broad_type', 'not specified')}; {gt.get('malignancy', 'not specified')}."
    )
    tiers = evaluate_diagnosis_tiers(
        generated or str(diag),
        cancer_type=cancer_type,
        reference=reference_stub,
        tier_weights={
            "broad": settings.eval.tier_weights.get("broad", 0.35),
            "subtype": settings.eval.tier_weights.get("subtype", 0.35),
            "malignancy": settings.eval.tier_weights.get("malignancy", 0.15),
            "grade": 0.0,
        },
        pred_facts=diag,
    )
    tier_map = tiers.get("tiers", {})
    payload = {
        "case_id": case_id,
        "cancer_type": cancer_type,
        "status": "metadata_only",
        "reference_source": "tcga_cohort_ground_truth",
        "scope": {
            "included": ["broad_type", "cohort_subtype", "malignancy_status"],
            "excluded": [
                "text_overlap",
                "morphology_f1",
                "critical_recall",
                "site",
                "grade",
                "necrosis",
                "invasion",
                "margins",
                "IHC",
                "molecular",
                "stage",
                "case_specific_reference_report_fields",
            ],
        },
        "diagnosis_score": float(tiers.get("diagnosis_score", 0.0)),
        "broad_score": float(tier_map.get("broad", {}).get("score", 0.0)),
        "subtype_score": float(tier_map.get("subtype", {}).get("score", 0.0)),
        "malignancy_score": float(tier_map.get("malignancy", {}).get("score", 0.0)),
        "grade_score": 0.0,
        "tier_details": tier_map,
        "predicted_diagnosis": str(tier_map.get("subtype", {}).get("prediction", "")),
        "reference_diagnosis": str(tier_map.get("subtype", {}).get("reference", gt.get("subtype", ""))),
        "ground_truth": {
            "broad_type": gt.get("broad_type"),
            "subtype": gt.get("subtype"),
            "subtype_aliases": gt.get("subtype_aliases", []),
            "malignancy": gt.get("malignancy"),
            "lineage": gt.get("lineage"),
        },
        "note": "Partial cohort-level eval only; not comparable to full report-based eval.json.",
    }
    (case_dir / "eval_metadata_only.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return True


def _attach_slide_paths_to_candidates(candidates: list[dict], settings: Settings) -> list[dict]:
    """Attach slide_path metadata needed by zoom reinspection.

    Older manifests stored only the ROI image path and slide_id.  True zoom-in
    crops need the source WSI path, which is available in each slide's
    ``foundation_roi_ranking.json`` cache.
    """
    cache_roots = [
        Path(settings.paths.roi_cache_dir) / settings.roi.foundation_model,
        Path(settings.paths.shared_artifacts) / "roi_cache" / settings.roi.foundation_model,
        Path("/local/PathXAI_Agent/shared_artifacts/roi_cache") / settings.roi.foundation_model,
    ]
    slide_path_cache: dict[str, str] = {}
    out = []
    for roi in candidates or []:
        row = dict(roi)
        meta = dict(row.get("metadata") or {})
        if not meta.get("slide_path"):
            sid = str(row.get("slide_id") or "")
            if not sid:
                rid = str(row.get("roi_id") or "")
                for sep in ("_foundation_", "_zoom_", "_roi_"):
                    if sep in rid:
                        sid = rid.split(sep, 1)[0]
                        break
            if sid:
                if sid not in slide_path_cache:
                    slide_path_cache[sid] = ""
                    for root in cache_roots:
                        p = root / sid / "foundation_roi_ranking.json"
                        if p.exists():
                            try:
                                data = json.loads(p.read_text(encoding="utf-8"))
                                slide_path_cache[sid] = str(data.get("slide_path") or "")
                                break
                            except Exception:
                                pass
                if slide_path_cache.get(sid):
                    meta["slide_path"] = slide_path_cache[sid]
        row["metadata"] = meta
        out.append(row)
    return out


def _eval_case(
    settings: Settings,
    result: EnhancedPipelineResult,
    cancer_type: str,
    case_id: str,
    case_dir: Path,
    reference_dir: str | Path,
    *,
    stages_llm=None,
) -> None:
    """Evaluate a single case against its reference report."""
    from pathxai_agent_v2.eval.runner import EvalRunner

    ref_dir = Path(reference_dir)
    refs, ref_status = _ensure_references_for_eval(ref_dir, cancer_type, case_id, stages_llm)

    if not refs:
        print(f"    [eval] SKIP: no reference for {cancer_type}/{case_id}", flush=True)
        _write_eval_status(
            case_dir,
            status="blocked_no_reference",
            cancer_type=cancer_type,
            case_id=case_id,
            details={"reference_status": ref_status},
        )
        return

    generated = result.report.long_report if result.report else ""
    if len(generated.strip().split()) < 20:
        print(f"    [eval] SKIP: report too short ({len(generated.split())} words)", flush=True)
        _write_eval_status(
            case_dir,
            status="blocked_report_too_short",
            cancer_type=cancer_type,
            case_id=case_id,
            details={"word_count": len(generated.split())},
        )
        return

    runner = EvalRunner(settings)
    eval_all = {}

    for ref_name, ref_text in refs.items():
        # ── Structured facts (hierarchy scoring) ──────────────────────────
        ref_facts = _load_or_extract_ref_facts(
            case_id=case_id, cancer_type=cancer_type,
            ref_text=ref_text, stages_llm=stages_llm,
        )

        # ── Tiered + NLP metrics ──────────────────────────────────────────
        er = runner.evaluate_case(
            generated_report=generated,
            reference_report=ref_text,
            case_id=case_id,
            cancer_type=cancer_type,
            confidence=result.diagnosis_prediction.confidence if result.diagnosis_prediction else "low",
            diagnosis_prediction=result.diagnosis_prediction.to_dict() if result.diagnosis_prediction else None,
            known_site=settings.leakage.known_site_from != "none",
            ref_facts_precomputed=ref_facts,
        )

        # Build per-ref score dict
        eval_all[ref_name] = {
            # ── NLP / text-overlap metrics (vs raw reference text) ─────────
            "bleu_1": er.bleu_1,
            "bleu_4": er.bleu_4,
            "rouge_1": er.rouge_1,
            "rouge_2": er.rouge_2,
            "morph_f1": er.morph_f1,
            "critical_recall": er.critical_recall,
            "composite_score": er.composite_score,       # pathology_report_score
            "overclaim_rate": er.overclaim_rate,
            "section_sims": er.section_sims,
            # ── Hierarchy (structured field comparison) ────────────────────
            "hierarchy_score": er.hierarchy_score,
            "field_scores": er.field_scores,
            # ── Tiered diagnosis (TCGA-grounded) ───────────────────────────
            "diagnosis_score": er.diagnosis_score,
            "broad_score": er.broad_score,
            "subtype_score": er.subtype_score,
            "malignancy_score": er.malignancy_score,
            "grade_score": er.grade_score,
            "tier_details": er.tier_details,
            "subtype_hit": er.diagnosis_hit,
            # ── Metadata ───────────────────────────────────────────────────
            "predicted_diagnosis": er.predicted_diagnosis,
            "reference_diagnosis": er.reference_diagnosis,
            "errors": er.errors,
        }

    # Primary: image_only (H&E-visible only) — the main benchmark
    primary = eval_all.get("image_only") or next(iter(eval_all.values()))
    eval_path = case_dir / "eval.json"
    eval_path.write_text(json.dumps({
        "case_id": case_id, "cancer_type": cancer_type,
        **primary,
        "reference_scores": eval_all,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        from pathxai_agent_v2.eval.eval_v2 import build_eval_v2_from_metrics
        eval_v2_all = {
            ref_name: build_eval_v2_from_metrics(
                case_id=case_id,
                cancer_type=cancer_type,
                metrics=metrics,
                reference_name=ref_name,
                source="direct_final_report_vs_reference",
            )
            for ref_name, metrics in eval_all.items()
        }
        primary_v2 = eval_v2_all.get("image_only") or next(iter(eval_v2_all.values()))
        (case_dir / "eval_v2.json").write_text(json.dumps({
            **primary_v2,
            "reference_scores_v2": eval_v2_all,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        print(f"    [eval_v2] skip {cancer_type}/{case_id}: {exc}", flush=True)

    for ref_name in sorted(eval_all):
        e = eval_all[ref_name]
        print(f"    [eval/{ref_name}] composite={e['composite_score']:.3f} "
              f"diag={e['diagnosis_score']:.3f} "
              f"(broad={e['broad_score']:.2f} subtype={e['subtype_score']:.2f} "
              f"malig={e['malignancy_score']:.2f} grade={e['grade_score']:.2f})", flush=True)

    _write_eval_status(
        case_dir,
        status="complete",
        cancer_type=cancer_type,
        case_id=case_id,
        details={"references": sorted(eval_all), "reference_status": ref_status},
    )


def _eval_case_from_pipeline(
    *,
    settings: Settings,
    case_dir: Path,
    cancer_type: str,
    case_id: str,
    reference_dir: str | Path,
    stages_llm=None,
) -> bool:
    """Evaluate a completed case from saved pipeline_result.json."""
    from pathxai_agent_v2.eval.runner import EvalRunner

    pipeline = _load_pipeline_json(case_dir)
    if not pipeline:
        _write_eval_status(
            case_dir,
            status="blocked_missing_pipeline_result",
            cancer_type=cancer_type,
            case_id=case_id,
        )
        return False

    generated = _report_text_from_pipeline(case_dir, pipeline)
    word_count = len(generated.strip().split())
    if word_count < 20:
        _write_eval_status(
            case_dir,
            status="blocked_report_too_short",
            cancer_type=cancer_type,
            case_id=case_id,
            details={"word_count": word_count},
        )
        return False

    refs, ref_status = _ensure_references_for_eval(Path(reference_dir), cancer_type, case_id, stages_llm)
    if not refs:
        _write_eval_status(
            case_dir,
            status="blocked_no_reference",
            cancer_type=cancer_type,
            case_id=case_id,
            details={"reference_status": ref_status},
        )
        return False

    runner = EvalRunner(settings)
    diag = pipeline.get("diagnosis_prediction") or {}
    confidence = str(diag.get("confidence") or "low") if isinstance(diag, dict) else "low"
    known_site = settings.leakage.known_site_from != "none"
    eval_all: dict[str, dict[str, Any]] = {}

    for ref_name, ref_text in refs.items():
        ref_facts = _load_or_extract_ref_facts(
            case_id=case_id,
            cancer_type=cancer_type,
            ref_text=ref_text,
            stages_llm=stages_llm,
        )
        er = runner.evaluate_case(
            generated_report=generated,
            reference_report=ref_text,
            case_id=case_id,
            cancer_type=cancer_type,
            confidence=confidence,
            diagnosis_prediction=diag if isinstance(diag, dict) else None,
            known_site=known_site,
            ref_facts_precomputed=ref_facts,
        )
        eval_all[ref_name] = {
            "bleu_1": er.bleu_1,
            "bleu_4": er.bleu_4,
            "rouge_1": er.rouge_1,
            "rouge_2": er.rouge_2,
            "morph_f1": er.morph_f1,
            "critical_recall": er.critical_recall,
            "composite_score": er.composite_score,
            "overclaim_rate": er.overclaim_rate,
            "section_sims": er.section_sims,
            "hierarchy_score": er.hierarchy_score,
            "field_scores": er.field_scores,
            "diagnosis_score": er.diagnosis_score,
            "broad_score": er.broad_score,
            "subtype_score": er.subtype_score,
            "malignancy_score": er.malignancy_score,
            "grade_score": er.grade_score,
            "tier_details": er.tier_details,
            "subtype_hit": er.diagnosis_hit,
            "predicted_diagnosis": er.predicted_diagnosis,
            "reference_diagnosis": er.reference_diagnosis,
            "errors": er.errors,
        }

    primary = eval_all.get("image_only") or next(iter(eval_all.values()))
    (case_dir / "eval.json").write_text(json.dumps({
        "case_id": case_id,
        "cancer_type": cancer_type,
        **primary,
        "reference_scores": eval_all,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        from pathxai_agent_v2.eval.eval_v2 import build_eval_v2_from_metrics
        eval_v2_all = {
            ref_name: build_eval_v2_from_metrics(
                case_id=case_id,
                cancer_type=cancer_type,
                metrics=metrics,
                reference_name=ref_name,
                source="direct_final_report_vs_reference",
            )
            for ref_name, metrics in eval_all.items()
        }
        primary_v2 = eval_v2_all.get("image_only") or next(iter(eval_v2_all.values()))
        (case_dir / "eval_v2.json").write_text(json.dumps({
            **primary_v2,
            "reference_scores_v2": eval_v2_all,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        print(f"    [eval_v2] skip {cancer_type}/{case_id}: {exc}", flush=True)
    _write_eval_status(
        case_dir,
        status="complete",
        cancer_type=cancer_type,
        case_id=case_id,
        details={"references": sorted(eval_all), "reference_status": ref_status},
    )
    print(f"    [eval/resume] wrote eval.json for {cancer_type}/{case_id}", flush=True)
    return True


def _ensure_references_for_eval(
    ref_dir: Path,
    cancer_type: str,
    case_id: str,
    stages_llm=None,
) -> tuple[dict[str, str], str]:
    """Find aligned refs, lazily creating them from originals when possible."""
    refs = _find_references(ref_dir, cancer_type, case_id)
    if refs:
        return refs, "existing_aligned"

    if stages_llm is None:
        return {}, "missing_aligned_no_llm"

    try:
        from scripts.align_reference_reports import get_or_create_aligned
        aligned = get_or_create_aligned(cancer_type, case_id, stages_llm=stages_llm)
        if aligned:
            refs = _find_references(ref_dir, cancer_type, case_id) or aligned
            return refs, "generated_aligned"
    except Exception as exc:
        print(f"    [eval] aligned gen failed: {exc}", flush=True)
        return {}, f"align_failed:{exc}"

    return {}, "missing_original_reference"


def _write_eval_status(
    case_dir: Path,
    *,
    status: str,
    cancer_type: str,
    case_id: str,
    details: dict[str, Any] | None = None,
) -> None:
    payload = {
        "status": status,
        "case_id": case_id,
        "cancer_type": cancer_type,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if details:
        payload.update(details)
    (case_dir / "eval_status.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _find_references(_ref_dir: Path, cancer_type: str, case_id: str) -> dict[str, str]:
    """Find aligned reference reports — never fall back to unformatted originals."""
    refs = {}
    project_root = Path(__file__).resolve().parent.parent.parent
    search_bases: list[Path] = []
    if _ref_dir:
        ref_dir = Path(_ref_dir)
        if ref_dir.name == "reference_reports_aligned":
            search_bases.append(ref_dir)
        elif ref_dir.name == "reference_reports":
            search_bases.append(ref_dir.parent / "reference_reports_aligned")
        else:
            search_bases.append(ref_dir)
    search_bases.append(project_root / "shared_artifacts/reference_reports_aligned")

    seen: set[Path] = set()
    aligned_dirs: list[Path] = []
    for base in search_bases:
        base = base if base.is_absolute() else project_root / base
        if base in seen:
            continue
        seen.add(base)
        aligned_dirs.append(base / cancer_type / case_id)

    for name, filename in [("image_only", "report_image_only.txt"), ("full", "report_full.txt")]:
        for aligned_dir in aligned_dirs:
            path = aligned_dir / filename
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    refs[name] = text
                    break
    return refs


def _prep_refs_if_needed(manifest_path: Path, stages_llm, dry_run: bool) -> None:
    """Pre-generate aligned reports + structured facts for all cases in manifest."""
    if dry_run or stages_llm is None:
        return

    try:
        from scripts.align_reference_reports import get_or_create_aligned
    except ImportError:
        return

    cases: set[tuple[str, str]] = set()
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ct = str(item.get("cancer_type", ""))
            cid = str(item.get("case_id", ""))
            if ct and cid:
                cases.add((ct, cid))

    generated = 0
    existing = 0
    blocked = 0
    failed = 0
    for ct, cid in sorted(cases):
        # Check if aligned refs already exist
        aligned_dir = Path(__file__).resolve().parent.parent.parent / "shared_artifacts/reference_reports_aligned" / ct / cid
        if aligned_dir.is_dir() and (aligned_dir / "report_full.txt").exists():
            existing += 1
            continue

        try:
            aligned = get_or_create_aligned(ct, cid, stages_llm=stages_llm)
            if aligned:
                generated += 1
                print(f"  [prep] generated aligned refs: {ct}/{cid}", flush=True)
            else:
                blocked += 1
        except Exception as exc:
            failed += 1
            print(f"  [prep] FAILED {ct}/{cid}: {exc}", flush=True)

    print(
        f"  [prep] aligned refs: existing={existing} generated={generated} "
        f"missing_original={blocked} failed={failed}",
        flush=True,
    )


_REF_FACTS_PROMPT = """/no_think
Extract structured diagnostic facts from this pathology reference report.

Return ONLY this JSON:
{{
  "site": "anatomic site",
  "primary_diagnosis": "the MAIN diagnosis entity ONLY, no grade/stage/modifiers",
  "cancer_type": "one of: carcinoma | sarcoma | lymphoma | melanoma | glioma | neuroendocrine neoplasm | germ cell tumor | mesothelioma | carcino+sarcoma | thymic epithelial neoplasm | neoplasm",
  "histologic_subtype": "specific entity name ONLY, no grade/stage/modifiers",
  "malignancy_status": "malignant | borderline | benign | not assessable",
  "grade": "high grade | intermediate grade | low grade | not specified",
  "necrosis": "present | absent | not specified",
  "invasive_status": "invasive | in situ | not specified"
}}

Rules:
- ``histologic_subtype`` must be SHORT: "urothelial carcinoma" YES, "high grade muscle invasive urothelial cell carcinoma + cis" NO.
- Infer ``cancer_type`` from the subtype.
- If malignancy is uncertain, write "not assessable".

Report:
{{report_text}}"""


def _load_or_extract_ref_facts(
    case_id: str,
    cancer_type: str,
    ref_text: str,
    stages_llm=None,
) -> dict | None:
    """Load cached LLM-extracted reference facts, or generate + cache them on first use.

    Cache path: shared_artifacts/reference_facts/<cancer_type>/<case_id>/diagnostic_facts.json
    """
    ref_json = Path("shared_artifacts/reference_facts") / cancer_type / case_id / "diagnostic_facts.json"
    # Allow override via project root
    if not ref_json.is_absolute():
        ref_json = Path(__file__).resolve().parent.parent.parent / ref_json

    fallback_facts = _anchored_reference_facts(
        case_id=case_id,
        cancer_type=cancer_type,
        ref_text=ref_text,
    )

    # Load cached
    if ref_json.exists():
        try:
            facts = json.loads(ref_json.read_text(encoding="utf-8"))
            if _valid_ref_facts(facts, cancer_type):
                return facts
            print(f"    [ref-facts] stale/invalid cache ignored: {cancer_type}/{case_id}", flush=True)
        except Exception:
            pass

    # Generate via LLM (first time only — cached thereafter)
    if stages_llm is None:
        ref_json.parent.mkdir(parents=True, exist_ok=True)
        ref_json.write_text(json.dumps(fallback_facts, indent=2, ensure_ascii=False), encoding="utf-8")
        return fallback_facts
    try:
        from pathxai_agent_v2.llm import parse_llm_json
        prompt = _REF_FACTS_PROMPT.replace("{report_text}", ref_text)
        raw = stages_llm.chat_json(prompt)
        parsed = parse_llm_json(raw)
        facts = {
            "site": str(parsed.get("site", "")).strip() or "not specified",
            "primary_diagnosis": str(parsed.get("primary_diagnosis", "")).strip() or "not specified",
            "cancer_type": str(parsed.get("cancer_type", "")).strip() or "not specified",
            "histologic_subtype": str(parsed.get("histologic_subtype", "")).strip() or "not specified",
            "malignancy_status": str(parsed.get("malignancy_status", "")).strip() or "not specified",
            "grade": str(parsed.get("grade", "")).strip() or "not specified",
            "necrosis": str(parsed.get("necrosis", "")).strip() or "not specified",
            "invasive_status": str(parsed.get("invasive_status", "")).strip() or "not specified",
            "schema_version": REF_FACTS_SCHEMA_VERSION,
            "source": "llm_reference_facts",
            "cancer_type_tcga": cancer_type,
            "case_id": case_id,
        }
        if not _valid_ref_facts(facts, cancer_type):
            facts = fallback_facts
        ref_json.parent.mkdir(parents=True, exist_ok=True)
        ref_json.write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"    [ref-facts] generated + cached: {cancer_type}/{case_id}.json", flush=True)
        return facts
    except Exception as exc:
        print(f"    [ref-facts] LLM failed for {cancer_type}/{case_id}: {exc}", flush=True)
        ref_json.parent.mkdir(parents=True, exist_ok=True)
        ref_json.write_text(json.dumps(fallback_facts, indent=2, ensure_ascii=False), encoding="utf-8")
        return fallback_facts


def _anchored_reference_facts(case_id: str, cancer_type: str, ref_text: str) -> dict:
    """Build robust reference facts from TCGA anchor + reference-visible fields."""
    from pathxai_agent_v2.eval.metrics import (
        canonical_diagnosis,
        extract_diagnostic_facts_rule,
        _extract_report_sections,
    )
    from pathxai_agent_v2.eval.tcga_ground_truth import ground_truth_for

    rule = extract_diagnostic_facts_rule(ref_text)
    gt = ground_truth_for(cancer_type) or {}
    sections = _extract_report_sections(ref_text)
    diagnosis_text = sections.get("Diagnosis") or ref_text
    ref_canon = canonical_diagnosis(diagnosis_text)
    # For this TCGA benchmark, the disease-family/subtype reference is anchored
    # by cohort. The free-text report is still used for case-specific attributes
    # such as grade/necrosis/invasion, but must not override the cohort subtype
    # because some source reports contain broad or noisy wording.
    subtype = str(gt.get("subtype") or ref_canon or rule.get("histologic_subtype") or "not specified")
    broad = str(gt.get("broad_type") or rule.get("cancer_type") or "not specified")
    malignancy = str(gt.get("malignancy") or rule.get("malignancy_status") or "not specified")

    return {
        "site": rule.get("site", "not specified"),
        "primary_diagnosis": subtype,
        "cancer_type": broad,
        "histologic_subtype": subtype,
        "malignancy_status": malignancy,
        "invasive_status": rule.get("invasive_status", "not specified"),
        "grade": rule.get("grade", "not specified"),
        "necrosis": rule.get("necrosis", "not specified"),
        "schema_version": REF_FACTS_SCHEMA_VERSION,
        "source": "tcga_anchor_plus_reference_rules",
        "cancer_type_tcga": cancer_type,
        "case_id": case_id,
    }


def _valid_ref_facts(facts: dict, cancer_type: str) -> bool:
    """Reject stale or diagnosis-incompatible reference-fact caches."""
    if not isinstance(facts, dict):
        return False
    if facts.get("schema_version") != REF_FACTS_SCHEMA_VERSION:
        return False
    subtype = str(facts.get("histologic_subtype") or "").strip().lower()
    if not subtype or subtype in {"not specified", "unspecified"}:
        return False

    from pathxai_agent_v2.eval.metrics import canonical_diagnosis
    from pathxai_agent_v2.eval.tcga_ground_truth import ground_truth_for

    gt = ground_truth_for(cancer_type)
    if not gt:
        return True

    gt_subtype = str(gt.get("subtype") or "").lower()
    # v3 reference facts are intentionally canonicalized to the cohort subtype.
    # Aliases are useful for prediction matching, but cached references should be
    # exact to avoid generic labels such as "melanoma" replacing "uveal melanoma".
    if subtype == gt_subtype:
        return True

    fact_canon = canonical_diagnosis(subtype)
    gt_canon = canonical_diagnosis(gt_subtype)
    if fact_canon and gt_canon:
        return fact_canon == gt_canon

    # If the cache names a known but different canonical disease, reject it.
    if fact_canon and gt_canon and fact_canon != gt_canon:
        return False
    return False
    """Map TCGA project code to organ/site (standalone, no v1 dependency)."""
    TCGA_SITE_MAP = {
        "TCGA-ACC": "adrenal gland / adrenocortical tissue",
        "TCGA-PCPG": "adrenal medulla / paraganglia",
        "TCGA-BLCA": "bladder / urothelial tissue",
        "TCGA-BRCA": "breast tissue",
        "TCGA-CESC": "cervix / cervical tissue",
        "TCGA-COAD": "colon / colorectal mucosa",
        "TCGA-READ": "rectum / colorectal mucosa",
        "TCGA-ESCA": "esophagus",
        "TCGA-STAD": "stomach / gastric mucosa",
        "TCGA-HNSC": "head and neck squamous mucosa",
        "TCGA-KICH": "kidney / renal tissue",
        "TCGA-KIRC": "kidney / renal tissue",
        "TCGA-KIRP": "kidney / renal tissue",
        "TCGA-CHOL": "bile duct / cholangiocarcinoma tissue",
        "TCGA-LIHC": "liver / hepatic tissue",
        "TCGA-LUAD": "lung tissue",
        "TCGA-LUSC": "lung tissue",
        "TCGA-MESO": "pleura / mesothelial tissue",
        "TCGA-LGG": "brain / glial tissue",
        "TCGA-OV": "ovary / ovarian tissue",
        "TCGA-UCEC": "uterus / endometrium",
        "TCGA-UCS": "uterus / endometrium",
        "TCGA-PRAD": "prostate tissue",
        "TCGA-SKCM": "skin / melanocytic tissue",
        "TCGA-SARC": "soft tissue / sarcoma",
        "TCGA-TGCT": "testis / germ cell tissue",
        "TCGA-THCA": "thyroid tissue",
        "TCGA-THYM": "thymus tissue",
        "TCGA-UVM": "eye / uveal melanocytic tissue",
        "TCGA-DLBC": "lymphoid tissue",
        "TCGA-GBM": "brain / glial tissue",
    }
    return TCGA_SITE_MAP.get(cancer_type)


def _create_client(settings: Settings) -> LLMClient:
    """Create an LLM client from settings."""
    from pathxai_agent.llm.qwen_client import QwenVisionClient

    inner = QwenVisionClient(
        settings.llm.qwen_api,
        model=settings.llm.qwen_model,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        timeout=(settings.llm.connect_timeout, settings.llm.read_timeout),
        image_max_edge=settings.llm.image_max_edge,
        image_jpeg_quality=settings.llm.image_jpeg_quality,
        enable_thinking=False,
    )
    return LegacyQwenAdapter(inner)


def _create_seg_reader_client(settings: Settings) -> LLMClient | None:
    """Create the optional fine-tuned paired ROI+seg reader client."""
    if not settings.llm.seg_reader_api:
        print("  [warn] SEG_READER_API not set; seg-reader evidence will be unavailable.", flush=True)
        return None
    from pathxai_agent.llm.qwen_client import QwenVisionClient

    max_tokens = int(os.environ.get("SEG_READER_MAX_TOKENS", "1536"))
    inner = QwenVisionClient(
        settings.llm.seg_reader_api,
        model=settings.llm.seg_reader_model,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=(settings.llm.connect_timeout, settings.llm.read_timeout),
        image_max_edge=settings.llm.image_max_edge,
        image_jpeg_quality=settings.llm.image_jpeg_quality,
        enable_thinking=False,
    )
    return LegacyQwenAdapter(inner)


def _create_report_client(settings: Settings) -> LLMClient:
    """Create an optional final-report-only LLM client.

    This deliberately does not replace the stage/reasoning LLM. It is for
    controlled writer ablations where the evidence bundle and diagnostic state
    stay fixed while only the final narrative generator changes.
    """
    from pathxai_agent.llm.qwen_client import QwenVisionClient

    inner = QwenVisionClient(
        settings.llm.report_api,
        model=settings.llm.report_model or settings.llm.qwen_model,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.report_max_tokens or settings.llm.max_tokens,
        timeout=(settings.llm.connect_timeout, settings.llm.read_timeout),
        image_max_edge=settings.llm.image_max_edge,
        image_jpeg_quality=settings.llm.image_jpeg_quality,
        enable_thinking=False,
    )
    return LegacyQwenAdapter(inner)


def _ablation_uses_seg_reader(ablation: str) -> bool:
    return ablation in {
        "S7_image_seg_ft_reader",
        "S8_image_seg_ft_reader_keep",
        "SR2_seg_reader",
        "SR3_seg_reader_stats",
        "SR4_seg_reader_v2",
        "SR5_seg_reader_v2_stats",
        "SR6_seg_reader_v3",
        "SR7_seg_reader_v3_stats",
        "SR8_seg_reader_v3_gated_stats",
        "SR9_seg_reader_v3_stats_case_session",
        "SR10_seg_reader_v3_gated_session",
        "SR11_seg_reader_v3_stats_nl_session",
        "SR12_seg_reader_v3_stats_case_session_override",
        "SR13_seg_reader_v3_stats_nl_session_override",
    }


def _create_dummy_client() -> LLMClient:
    """Create a dummy client for dry runs."""
    class DummyClient:
        def chat_json(self, prompt, image_paths=None, system_prompt=None):
            return {"content": "DRY_RUN", "reasoning": "", "json": None, "elapsed_sec": 0}
        def chat(self, prompt, image_paths=None, system_prompt=None):
            return {"content": "DRY_RUN", "reasoning": ""}
    return DummyClient()  # type: ignore


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PathXAI v2 — Pathology Report Generation")
    sub = parser.add_subparsers(dest="command")

    # Single case
    single = sub.add_parser("single", help="Run a single case")
    single.add_argument("--case-id", required=True)
    single.add_argument("--roi-json", required=True)
    single.add_argument("--output", default="outputs/single_case")
    single.add_argument("--config", default=None)
    single.add_argument("--known-site", default=None)
    single.add_argument("--dry-run", action="store_true")
    ablation_choices = [
        "full",
        "A0_full", "A1_no_seg", "A2_no_medgemma", "A3_no_knowledge",
        "A4_no_reinspect", "A5_no_claims", "A6_no_hybrid",
        # Old ablation suite (run_20260702_191218)
        "R0_full", "R1_no_claim_filter", "R2_no_llm_claim_verifier",
        "R3_no_claim_verification", "R4_no_evidence_graph",
        "R5_no_subtype_sufficiency", "R6_no_broad_hypotheses",
        "R7_no_reinspection",
        "E1_visual_only", "E2_visual_seg", "E3_visual_morph",
        "E4_visual_keep", "E5_visual_seg_morph", "E6_visual_seg_keep",
        "E7_visual_morph_keep", "E8_no_known_site",
        # Segmentation-supported report suite (fixed ROI; no policy/reinspection)
        "S0_image_only",          # ROI image only
        "S1_image_seg",           # ROI image + segmentation
        "S2_image_morph",         # ROI image + morphology text
        "S3_image_seg_morph",     # ROI image + segmentation + morphology
        "S4_image_seg_keep",      # ROI image + segmentation + KEEP
        "S5_image_seg_morph_keep",# ROI image + segmentation + morphology + KEEP
        "S6_seg_morph_no_claim_filter", # S3 without claim-filtered rewrite
        "S7_image_seg_ft_reader", # ROI image + segmentation + fine-tuned seg reader
        "S8_image_seg_ft_reader_keep", # S7 + KEEP knowledge
        # Seg-reader isolation suite: raw ROI images remain the only final VLM images
        "SR0_image_only",         # Raw ROI image baseline
        "B0_visual_reinspect_baseline",  # SR0 + reinspection (visual-only zoom baseline)
        "B0_fixed_visual_reinspect",  # Explicit fixed visual-only + reinspection baseline
        "SR1_seg_stats",          # Raw ROI + compact segmentation stats
        "SR2_seg_reader",         # Raw ROI + fine-tuned seg-reader morphology only
        "SR3_seg_reader_stats",   # Raw ROI + seg-reader morphology + compact seg stats
        "SR4_seg_reader_v2",       # Raw ROI + v2 report-supervised seg-reader morphology only
        "SR5_seg_reader_v2_stats", # Raw ROI + v2 report-supervised seg-reader + compact stats
        "SR6_seg_reader_v3",       # Raw ROI + v3 combined seg-reader morphology only
        "SR7_seg_reader_v3_stats", # Raw ROI + v3 combined seg-reader + compact stats
        "SR8_seg_reader_v3_gated_stats", # SR7 + deterministic reader evidence guard
        "SR9_seg_reader_v3_stats_case_session", # SR7 + one case-level Qwen chat history
        "SR10_seg_reader_v3_gated_session",  # SR8 guard + SR9 chat session (both)
        "SR11_seg_reader_v3_stats_nl_session",  # SR9 chat + natural-language seg stats
        "SR12_seg_reader_v3_stats_case_session_override",  # SR9 + chat-history override rule
        "SR13_seg_reader_v3_stats_nl_session_override",  # SR11 + chat-history override rule
        "SS2_seg_claim_stats",      # SR1 evidence, final report stats as bounded claims
        "SS3_seg_hier_stats",       # SR1 evidence, final report stats as case+ROI hierarchy
        "SS4_seg_quality_stats",    # SR1 evidence, final report stats with ROI quality gating
        "SS5_positive_seg_stats",    # SR1 evidence, positive-only compact stats
        "SS6_report_only_seg_stats", # SR0 diagnosis + seg stats only for final report
        "SS7_dual_path_guarded_seg_stats", # image-only vs seg-stats diagnosis guard
        "SS8_dual_path_field_guarded_seg_stats", # SS7 + field-level grade/invasion/necrosis guard
        "SS9_dual_path_grade_necrosis_guarded_seg_stats", # SS7 + minimal grade/necrosis field guard
        "SS10_dual_path_grade_adjudicated_seg_stats", # SS7 + consistency-adjudicated grade + guarded necrosis
        "SS12_dual_path_richness_tiebreaker_seg_stats", # SS7 + subtype-richness tiebreaker
        "SS18_conflict_triggered_roi_expansion", # SS7 + conflict/uncertainty-triggered ROI expansion
        "SS19_sr1prompt_adjudicated_conflict_roi_expansion", # SR1 prompt/adjudication lineage + ROI expansion
        "SS20_strict_conflict_roi_expansion", # SS7 + strict bounded ROI expansion
        "SS21_ss20_reinspect_zoom_seg", # SS20-style guard + standard reinspection/zoom with inherited zoom seg
        "SS22_visual_zoom_stable_seg_stats", # SS21-style reinspection where only visual branch sees zoom/new ROI
        "SS23_ss21_report_positive_seg_stats", # SS21 unchanged + positive-only seg stats in final report
        "SS24_ss21_evidence_supported_residual_merge", # SS21 + visual-core, mask-supported residual merge
        "SO0_visual_seg_overlay_only", # VLM sees H&E+seg overlay/composite images only; no seg text/stats
        "SO1_ss21_seg_overlay_composite", # SS21 + H&E/seg overlay-composite image input
        "SO2_conch_grade_roi", # SO1 + cache-first CONCH grade-focused auxiliary H&E ROIs
        "SO3_grade_only_aux_report", # SO1 + auxiliary grade ROIs only in final report, not diagnosis branches
        "SO4_grade_aux_keep", # SO3 + KEEP knowledge context
        "SF0_he_keep",          # Fixed ROI H&E + KEEP knowledge, no seg/morph/reinspection
        "SF1_ss21_a5_keep",     # SS21_A5-style seg-stats + reinspection/zoom + KEEP knowledge
        "SF2_keep_he_branch_locked_report", # SS21_A5-style dual path; KEEP only on H&E branch + locked report
        "Q1_no_reinspect", "Q2_independent_only", "Q3_no_seg_policy",
        # New subtractive ablation suite (ablation2, v2 design)
        "V0_full",           # Full v2: broad-locked, seg→subtype, relaxed guard
        "V1_no_seg_policy",  # V0 − seg_policy
        "V2_no_reinspection",# V0 − reinspection (fixed 3 ROI)
        "V3_no_morph",       # V0 − MedGemma morphology
        "V4_no_keep",        # V0 − KEEP knowledge
        "V5_no_seg",         # V0 − segmentation (VLM + morph + keep)
        "V6_no_subtype_guard",# V0 − subtype guard
        "V7_vlm_only",       # Pure VLM (no seg, no morph, no keep)
        # Two-stage diagnosis ablation suite (ablation3, v3)
        # W0: two-stage + 3 fixed ROI + morph + KEEP (NO reinspection, NO seg in prompt)
        "W0_full",           # Base v3: no reinsp, no seg text/image in diagnosis
        "W2_no_morph",       # W0 − MedGemma morphology
        "W3_no_keep",        # W0 − KEEP knowledge
        "W4_no_seg",         # W0 − MedSAM segmentation (only SegPolicy remains)
        "W5_no_subtype_guard",# W0 − subtype guard
        "W6_vlm_only",       # Pure VLM (single-stage, no seg/morph/keep)
        "W7_with_reinsp",    # W0 + reinspection ON (to isolate reinsp effect)
        "W8_best_with_reinsp",# W2 (-Morph) + reinspection ON
        "W9_best_no_seg",    # W2 (-Morph) − segmentation
    ]
    single.add_argument("--ablation", default="full", choices=ablation_choices)

    # Batch from manifest
    batch = sub.add_parser("batch", help="Run batch from manifest")
    batch.add_argument("--manifest", required=True)
    batch.add_argument("--output", default="outputs/batch")
    batch.add_argument("--config", default=None)
    batch.add_argument("--max-cases", type=int, default=None)
    batch.add_argument("--dry-run", action="store_true")
    batch.add_argument("--no-resume", action="store_true")
    batch.add_argument(
        "--retry-errors",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Resume already retries error-only "
            "cases; valid completed results are never overwritten because of "
            "a stale error.json."
        ),
    )
    batch.add_argument("--reference-dir", default=None, help="Auto-eval each case against reference reports")
    batch.add_argument("--ablation", default="full", choices=ablation_choices)
    batch.add_argument("--round-robin-by-cancer", action="store_true",
                       help="Interleave manifest cases by cancer type")

    # Eval
    eval_p = sub.add_parser("eval", help="Evaluate a run directory")
    eval_p.add_argument("--run-dir", required=True)
    eval_p.add_argument("--reference-dir", required=True)
    eval_p.add_argument("--output", default="outputs/eval")
    eval_p.add_argument("--config", default=None)

    args = parser.parse_args()

    if args.command == "single":
        settings = load_settings(args.config)
        rois = json.loads(Path(args.roi_json).read_text(encoding="utf-8"))
        if isinstance(rois, dict):
            rois = rois.get("rois", rois.get("candidate_rois", []))
        rois = _attach_slide_paths_to_candidates(rois, settings)
        priors = DiagnosisPriors(known_site=args.known_site) if args.known_site else None
        result = run_single_case(
            settings, args.case_id, rois, args.output,
            diagnosis_priors=priors, dry_run=args.dry_run, ablation=args.ablation,
        )
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str))

    elif args.command == "batch":
        settings = load_settings(args.config)
        results = run_from_manifest(
            settings, args.manifest, args.output,
            dry_run=args.dry_run, max_cases=args.max_cases,
            resume=not args.no_resume, retry_errors=args.retry_errors,
            ablation=args.ablation,
            reference_dir=args.reference_dir,
            round_robin_by_cancer=args.round_robin_by_cancer,
        )
        ok = sum(1 for r in results if not r.errors)
        print(f"Completed: {ok}/{len(results)} cases")

    elif args.command == "eval":
        settings = load_settings(args.config)
        from pathxai_agent_v2.eval.runner import EvalRunner
        runner = EvalRunner(settings)
        case_results, summary = runner.evaluate_run(args.run_dir, args.reference_dir)
        runner.save_results(case_results, summary, args.output)
        print(f"Evaluated {summary.total_cases} cases")
        print(f"  Composite score: {summary.avg_composite_score:.3f}")
        print(f"  Diagnosis hit: {summary.avg_diagnosis_hit:.3f}")
        print(f"  Hierarchy score: {summary.avg_hierarchy_score:.3f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
