from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from pathxai_agent.llm.qwen_client import QwenVisionClient


@dataclass
class PatchEvidence:
    patch_id: str
    image_path: str
    x: Optional[int] = None
    y: Optional[int] = None
    level: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    source: str = "roi"
    description: str = ""
    score: float = 0.0
    extras: Dict[str, object] = field(default_factory=dict)


@dataclass
class PathAgentBaselineResult:
    slide_id: str
    question: str
    selected_patch_ids: List[str]
    report: str
    explanation: str
    process: List[dict]
    evidence: List[dict]
    reference: Optional[dict] = None
    mode: str = "pathagent_qwen_baseline"


def patch_evidence_from_cached_rois(cached_rois: Iterable[dict]) -> List[PatchEvidence]:
    patches = []
    for roi in cached_rois:
        patch_id = str(roi.get("roi_id") or Path(str(roi["image_path"])).stem)
        patches.append(
            PatchEvidence(
                patch_id=patch_id,
                image_path=str(roi["image_path"]),
                x=_int_or_none(roi.get("x")),
                y=_int_or_none(roi.get("y")),
                level=int(roi.get("level") or 0),
                width=_int_or_none(roi.get("width")),
                height=_int_or_none(roi.get("height")),
                source=str(roi.get("source") or "roi"),
                score=float(roi.get("score") or 0.0),
                extras=dict(roi.get("metadata") or {}),
            )
        )
    return patches


def attach_descriptions(
    patches: List[PatchEvidence],
    descriptions_file: str | Path | None,
    slide_id: str,
) -> None:
    if descriptions_file is None:
        return
    path = Path(descriptions_file)
    if not path.exists():
        raise FileNotFoundError(f"Patch descriptions file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    slide_desc = data.get(slide_id)
    if slide_desc is None:
        candidates = [key for key in data if str(key).startswith(slide_id) or slide_id.startswith(str(key))]
        slide_desc = data.get(candidates[0]) if candidates else None
    if not isinstance(slide_desc, dict):
        return
    for patch in patches:
        keys = _description_keys(patch)
        for key in keys:
            if key in slide_desc:
                patch.description = str(slide_desc[key])
                break


def select_initial_patches(patches: List[PatchEvidence], max_patches: int) -> List[PatchEvidence]:
    ranked = sorted(patches, key=lambda p: (p.score, bool(p.description)), reverse=True)
    return ranked[:max(1, min(max_patches, len(ranked)))]


def evidence_text(patches: Iterable[PatchEvidence], include_descriptions: bool = True) -> str:
    lines = []
    for idx, patch in enumerate(patches, start=1):
        desc = patch.description.strip() if patch.description else "No offline patch description available; inspect the attached patch image directly."
        coord = f"x={patch.x}, y={patch.y}" if patch.x is not None and patch.y is not None else "coordinates unknown"
        size = f"{patch.width}x{patch.height}" if patch.width and patch.height else "unknown size"
        line = f"[Patch {idx}] {coord}; level={patch.level}; size={size}; source=ROI crop."
        if include_descriptions:
            line += f"\nDescription: {desc}"
        lines.append(line)
    return "\n\n".join(lines)


def foundation_site_hint_text(patches: Iterable[PatchEvidence]) -> str:
    seen = set()
    hints = []
    for patch in patches:
        hint = patch.extras.get("foundation_site_prediction")
        if not isinstance(hint, dict) or not hint.get("available"):
            continue
        key = json.dumps({
            "top_site": hint.get("top_site"),
            "top_score": hint.get("top_score"),
            "margin_to_second": hint.get("margin_to_second"),
        }, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        scores = hint.get("scores") or []
        alternatives = []
        for row in scores[1:4]:
            alternatives.append(f"{row.get('site')} ({float(row.get('score') or 0.0):.3f})")
        hints.append(
            "- Top predicted site: "
            f"{hint.get('top_site')} "
            f"(confidence={hint.get('confidence')}, score={float(hint.get('top_score') or 0.0):.3f}, "
            f"margin={_format_optional_float(hint.get('margin_to_second'))}). "
            f"Top alternatives: {', '.join(alternatives) if alternatives else 'none'}."
        )
    if not hints:
        return "No foundation-model specimen/site hint available."
    return "\n".join(hints)


def run_qwen_pathagent_baseline(
    slide_id: str,
    patches: List[PatchEvidence],
    qwen_client: Optional[QwenVisionClient],
    question: str,
    out_dir: str | Path,
    reference: Optional[dict] = None,
    initial_k: int = 5,
    max_rounds: int = 3,
    add_per_round: int = 2,
    max_zoom_depth: int = 0,
    max_selected_patches: int | None = None,
    strict_minimal_evidence: bool = False,
    dry_run: bool = False,
) -> PathAgentBaselineResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not patches:
        raise ValueError("No patches were provided to the PathAgent baseline.")

    selected = select_initial_patches(patches, initial_k)
    selected_ids = {p.patch_id for p in selected}
    process: List[dict] = []
    max_selected_patches = max_selected_patches or len(patches)
    zoom_dir = out_dir / "zoom_rois"

    for round_idx in range(1, max_rounds + 1):
        prompt = _build_sufficiency_prompt(
            question,
            selected,
            reference,
            enable_zoom=max_zoom_depth > 0,
            max_selected_patches=max_selected_patches,
            strict_minimal_evidence=strict_minimal_evidence,
        )
        image_paths = [p.image_path for p in selected]
        if dry_run:
            judgement = {
                "sufficient": "Yes",
                "answer": "DRY_RUN",
                "thinking_steps": "Qwen call skipped; prompt and selected patch evidence were saved.",
                "missing_info": "",
            }
            raw = {"content": "", "reasoning": "", "json": judgement}
        else:
            if qwen_client is None:
                raise ValueError("qwen_client is required unless dry_run=True")
            raw = qwen_client.chat_json(prompt, image_paths=image_paths)
            judgement = raw.get("json") or {
                "sufficient": "Uncertain",
                "answer": "",
                "thinking_steps": raw.get("content", ""),
                "missing_info": "unclear diagnostic evidence",
            }

        process.append({
            "round": round_idx,
            "mode": "evaluate_selected_patches",
            "selected_patch_ids": [p.patch_id for p in selected],
            "input": {
                "prompt": prompt,
                "image_paths": image_paths,
            },
            "judgement": judgement,
            "raw_content": raw.get("content", ""),
            "reasoning": raw.get("reasoning", ""),
        })

        if str(judgement.get("sufficient", "")).strip().lower() == "yes" or _wants_report(judgement):
            break
        if len(selected) >= max_selected_patches:
            break

        remaining = [p for p in patches if p.patch_id not in selected_ids]
        new_patches = []
        if max_zoom_depth > 0 and _wants_zoom(judgement):
            zoom_patch = make_zoom_patch_from_judgement(
                selected=selected,
                judgement=judgement,
                zoom_dir=zoom_dir,
                max_zoom_depth=max_zoom_depth,
            )
            if zoom_patch is not None and zoom_patch.patch_id not in selected_ids:
                new_patches = [zoom_patch]
        if not new_patches:
            if strict_minimal_evidence and not _allow_add_roi(judgement):
                process.append({
                    "round": round_idx,
                    "mode": "minimal_evidence_stop",
                    "selected_patch_ids": [p.patch_id for p in selected],
                    "reason": "strict_minimal_evidence prevented automatic ROI expansion",
                    "judgement": judgement,
                })
                break
            if not remaining:
                break
            missing_info = str(judgement.get("missing_info") or question)
            new_patches = rank_patches_by_text(remaining, missing_info)[:add_per_round]
        for patch in new_patches:
            if len(selected) >= max_selected_patches:
                break
            selected.append(patch)
            selected_ids.add(patch.patch_id)

    final_prompt = _build_report_prompt(question, selected, reference)
    final_image_paths = [p.image_path for p in selected]
    if dry_run:
        report = """PATCH_ANALYSIS:
DRY_RUN: Qwen patch analysis was skipped.

LONG_REPORT:
Specimen/Site: Not specified from ROI evidence.
Diagnosis: DRY_RUN report generation skipped.
Tumor/Lesion Description: DRY_RUN.
Key Microscopic Findings: DRY_RUN.
Invasion/Necrosis/Margins: Not assessable from provided ROI patches.
Immunohistochemistry/Ancillary Studies: Not provided.
Stage/Grade/Prognostic Features: Not assessable from provided ROI patches.
Final Summary: DRY_RUN report generation skipped."""
        explanation = "Selected patch evidence and prompts were saved for inspection."
        final_raw = {"content": report, "reasoning": explanation}
    else:
        if qwen_client is None:
            raise ValueError("qwen_client is required unless dry_run=True")
        final_raw = qwen_client.chat(final_prompt, image_paths=final_image_paths)
        report = final_raw.get("content", "")
        explanation = final_raw.get("reasoning", "")
        if not _has_required_report_blocks(report):
            retry_prompt = _build_report_retry_prompt(question, selected, reference)
            retry_raw = qwen_client.chat(retry_prompt, image_paths=final_image_paths)
            report = retry_raw.get("content", "")
            explanation = retry_raw.get("reasoning", "") or explanation
            final_raw = {
                **retry_raw,
                "first_attempt": final_raw,
                "retry_reason": "missing_required_report_blocks",
            }
        if not _has_required_report_blocks(report):
            raise ValueError("Qwen returned incomplete final report content after retry")

    process.append({
        "round": "final",
        "mode": "slide_report_generation",
        "selected_patch_ids": [p.patch_id for p in selected],
        "input": {
            "prompt": final_prompt,
            "image_paths": final_image_paths,
        },
        "prompt": final_prompt,
        "raw_content": final_raw.get("content", ""),
        "reasoning": final_raw.get("reasoning", ""),
    })

    result = PathAgentBaselineResult(
        slide_id=slide_id,
        question=question,
        selected_patch_ids=[p.patch_id for p in selected],
        report=report,
        explanation=explanation,
        process=process,
        evidence=[asdict(p) for p in selected],
        reference=reference,
    )
    save_pathagent_baseline_result(result, out_dir)
    save_roi_selection_trace(result, patches, out_dir)
    return result


def save_pathagent_baseline_result(result: PathAgentBaselineResult, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    (out_dir / "pathagent_qwen_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "evidence.json").write_text(
        json.dumps({
            "slide_id": result.slide_id,
            "mode": result.mode,
            "selected_patch_ids": result.selected_patch_ids,
            "evidence": result.evidence,
            "future_extensions": {
                "segmentation": "reserved for MedSAM ROI masks and tissue stats",
                "cells": "reserved for cell counts, density, and morphology summaries",
                "knowledge": "reserved for retrieved pathology knowledge snippets",
            },
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "inference_trace.json").write_text(
        json.dumps({
            "slide_id": result.slide_id,
            "mode": result.mode,
            "question": result.question,
            "inputs": {
                "selected_patch_ids": result.selected_patch_ids,
                "evidence": result.evidence,
            },
            "outputs": {
                "report": result.report,
                "reasoning": result.explanation,
            },
            "thinking_process": result.process,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md = [
        f"# PathAgent-Qwen Baseline: {result.slide_id}",
        "",
        "## Question",
        "",
        result.question,
        "",
        "## Selected Evidence",
        "",
    ]
    for ev in result.evidence:
        md.append(f"- `{ev['patch_id']}`: {ev.get('image_path')}")
    report_payload = _split_report_blocks_for_markdown(result.report or "")
    if report_payload.get("patch_analysis"):
        md.extend(["", "## Patch Analysis", "", report_payload["patch_analysis"]])
    md.extend(["", "## Generated Report", "", report_payload.get("reports", result.report or "")])
    if result.explanation:
        md.extend(["", "## Reasoning", "", result.explanation])
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def save_roi_selection_trace(
    result: PathAgentBaselineResult,
    candidates: List[PatchEvidence],
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    selected_ids = set(result.selected_patch_ids)
    candidate_rows = [
        _trace_patch_row(patch, selected=patch.patch_id in selected_ids)
        for patch in sorted(candidates, key=lambda p: p.score, reverse=True)
    ]
    decisions = []
    for step in result.process:
        mode = step.get("mode")
        if mode not in {"evaluate_selected_patches", "minimal_evidence_stop"}:
            continue
        judgement = step.get("judgement") or {}
        decisions.append({
            "round": step.get("round"),
            "mode": mode,
            "selected_patch_ids": step.get("selected_patch_ids") or [],
            "can_diagnose_from_current_roi": judgement.get("can_diagnose_from_current_roi"),
            "diagnostic_confidence": judgement.get("diagnostic_confidence"),
            "sufficient": judgement.get("sufficient"),
            "next_action": judgement.get("next_action"),
            "target_patch_index": judgement.get("target_patch_index"),
            "zoom_region": judgement.get("zoom_region"),
            "zoom_reason": judgement.get("zoom_reason"),
            "must_add_roi": judgement.get("must_add_roi"),
            "add_roi_justification": judgement.get("add_roi_justification"),
            "missing_info": judgement.get("missing_info"),
            "answer": judgement.get("answer"),
            "stop_reason": step.get("reason"),
        })
    payload = {
        "slide_id": result.slide_id,
        "mode": result.mode,
        "roi_selection_summary": {
            "candidate_count": len(candidate_rows),
            "selected_count": len(result.selected_patch_ids),
            "selected_patch_ids": result.selected_patch_ids,
        },
        "foundation_or_candidate_rois": candidate_rows,
        "qwen_roi_decisions": decisions,
        "final_selected_evidence": result.evidence,
    }
    (out_dir / "roi_selection_trace.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "roi_selection_trace.md").write_text(
        render_roi_selection_trace_markdown(payload),
        encoding="utf-8",
    )


def _trace_patch_row(patch: PatchEvidence, selected: bool) -> dict:
    return {
        "patch_id": patch.patch_id,
        "selected": selected,
        "image_path": patch.image_path,
        "x": patch.x,
        "y": patch.y,
        "level": patch.level,
        "width": patch.width,
        "height": patch.height,
        "source": patch.source,
        "score": patch.score,
        "zoom_depth": patch.extras.get("zoom_depth"),
        "zoom_parent_patch_id": patch.extras.get("zoom_parent_patch_id"),
        "zoom_region": patch.extras.get("zoom_region"),
        "foundation_roi_ranking_path": patch.extras.get("foundation_roi_ranking_path"),
        "foundation_site_prediction": patch.extras.get("foundation_site_prediction"),
    }


def render_roi_selection_trace_markdown(payload: dict) -> str:
    lines = [
        f"# ROI Selection Trace: {payload.get('slide_id')}",
        "",
        "## Summary",
        "",
    ]
    summary = payload.get("roi_selection_summary") or {}
    lines.append(f"- Candidate ROIs: {summary.get('candidate_count')}")
    lines.append(f"- Final selected ROIs: {summary.get('selected_count')}")
    lines.append("")
    lines.extend(["## Qwen Decisions", ""])
    decisions = payload.get("qwen_roi_decisions") or []
    if not decisions:
        lines.append("No Qwen ROI decisions recorded.")
    for decision in decisions:
        lines.append(f"### Round {decision.get('round')}")
        lines.append(f"- Selected: {', '.join(decision.get('selected_patch_ids') or [])}")
        lines.append(f"- Can diagnose: {decision.get('can_diagnose_from_current_roi')}")
        lines.append(f"- Confidence: {decision.get('diagnostic_confidence')}")
        lines.append(f"- Sufficient: {decision.get('sufficient')}")
        lines.append(f"- Next action: {decision.get('next_action')}")
        if decision.get("zoom_region"):
            lines.append(f"- Zoom region: {decision.get('zoom_region')}")
        if decision.get("zoom_reason"):
            lines.append(f"- Zoom reason: {decision.get('zoom_reason')}")
        if decision.get("add_roi_justification"):
            lines.append(f"- Add-ROI justification: {decision.get('add_roi_justification')}")
        if decision.get("missing_info"):
            lines.append(f"- Missing info: {decision.get('missing_info')}")
        if decision.get("stop_reason"):
            lines.append(f"- Stop reason: {decision.get('stop_reason')}")
        lines.append("")
    lines.extend(["## Candidate ROIs", ""])
    for idx, roi in enumerate(payload.get("foundation_or_candidate_rois") or [], start=1):
        mark = "selected" if roi.get("selected") else "candidate"
        lines.append(
            f"- {idx}. {mark}; score={roi.get('score')}; "
            f"xy=({roi.get('x')},{roi.get('y')}); size={roi.get('width')}x{roi.get('height')}; "
            f"path={roi.get('image_path')}"
        )
    lines.append("")
    lines.extend(["## Final Evidence", ""])
    for idx, ev in enumerate(payload.get("final_selected_evidence") or [], start=1):
        lines.append(
            f"- {idx}. {ev.get('patch_id')}; score={ev.get('score')}; "
            f"source={ev.get('source')}; path={ev.get('image_path')}"
        )
    return "\n".join(lines) + "\n"


def rank_patches_by_text(patches: List[PatchEvidence], query: str) -> List[PatchEvidence]:
    query_terms = {tok for tok in _tokenize(query) if len(tok) > 2}

    def score(patch: PatchEvidence) -> float:
        hay = " ".join([patch.patch_id, patch.description, patch.source])
        terms = set(_tokenize(hay))
        overlap = len(query_terms & terms)
        return float(overlap) + float(patch.score or 0.0)

    return sorted(patches, key=score, reverse=True)


def _build_sufficiency_prompt(
    question: str,
    patches: List[PatchEvidence],
    reference: Optional[dict],
    enable_zoom: bool = False,
    max_selected_patches: int | None = None,
    strict_minimal_evidence: bool = False,
) -> str:
    budget_text = (
        f"You may use at most {max_selected_patches} final evidence images, but this is only a safety cap. "
        "Prefer one diagnostic ROI whenever possible."
        if max_selected_patches
        else "Prefer one diagnostic ROI whenever possible."
    )
    zoom_instruction = ""
    zoom_fields = ""
    if enable_zoom:
        zoom_instruction = """
Decision policy:
First ask: can you provide a certain or reasonably likely morphology-based diagnosis/differential from the current ROI evidence?
- If yes and the report can state limitations, set sufficient="Yes" and next_action="generate_report".
- If the current ROI looks diagnostically relevant but the field is too broad or cellular detail is insufficient, set next_action="zoom_in".
- If the current ROI is nondiagnostic, mostly artifact/background, or lacks the lesion pattern needed for diagnosis, set next_action="add_next_roi".
Do not choose add_next_roi just to increase confidence or because more candidates are available."""
        zoom_fields = """,
  "can_diagnose_from_current_roi": "Yes/No/Uncertain",
  "diagnostic_confidence": "high/medium/low",
  "next_action": "generate_report or zoom_in or add_next_roi",
  "target_patch_index": 1,
  "zoom_region": "center/top-left/top-right/bottom-left/bottom-right",
  "zoom_reason": "why this action is needed",
  "must_add_roi": false,
  "add_roi_justification": "only required when next_action is add_next_roi" """
    minimal_instruction = ""
    if strict_minimal_evidence:
        minimal_instruction = """
Strict minimal-evidence mode is enabled:
- The maximum ROI count is only a safety cap, not a target.
- Do not choose add_next_roi unless another independent ROI is essential.
- If choosing add_next_roi, set must_add_roi=true and state the specific missing visual pattern in add_roi_justification.
- If the issue is only lack of cellular detail inside the current ROI, choose zoom_in instead of add_next_roi.
- If the current ROI is diagnostic enough for a limitation-aware report, set sufficient="Yes" and next_action="generate_report"."""
    return f"""/no_think
You are reproducing the PathAgent patch-evaluation step for pathology WSI analysis.

Task objective:
{question}

Current selected patch evidence:
{evidence_text(patches)}

Inspect the attached patch images and the evidence text. Decide whether the current ROI evidence can support a diagnosis/differential and report, or whether you need a zoom-in or the next independent ROI.
{budget_text}
{zoom_instruction}
{minimal_instruction}

Important anti-leakage rules:
- Do not infer organ, cancer type, diagnosis, or specimen site from case IDs, slide IDs, file names, directory names, dataset-specific codes, coordinates, or patch labels.
- Patch labels and coordinates are technical crop identifiers only.
- Use the attached image content as the diagnostic evidence.
- Do not request more ROIs just because more are available.
- Stop early when the current ROI evidence is enough for a cautious, limitation-aware diagnosis/differential and report.

Return ONLY one JSON object:
{{
  "answer": "short provisional diagnosis/differential or why current ROI is nondiagnostic",
  "thinking_steps": "brief evidence-based reasoning",
  "sufficient": "Yes or No",
  "missing_info": "specific visual evidence needed only if current ROI cannot support diagnosis/report"{zoom_fields}
}}"""


def _wants_zoom(judgement: dict) -> bool:
    return str(judgement.get("next_action") or "").strip().lower() == "zoom_in"


def _wants_report(judgement: dict) -> bool:
    return str(judgement.get("next_action") or "").strip().lower() == "generate_report"


def _allow_add_roi(judgement: dict) -> bool:
    action = str(judgement.get("next_action") or "").strip().lower()
    if action != "add_next_roi":
        return False
    must_add = judgement.get("must_add_roi")
    if isinstance(must_add, bool):
        return must_add
    if isinstance(must_add, str):
        return must_add.strip().lower() in {"true", "yes", "1"}
    justification = str(judgement.get("add_roi_justification") or judgement.get("missing_info") or "")
    return len(justification.strip()) >= 20


def make_zoom_patch_from_judgement(
    selected: List[PatchEvidence],
    judgement: dict,
    zoom_dir: Path,
    max_zoom_depth: int,
) -> PatchEvidence | None:
    try:
        target_idx = int(judgement.get("target_patch_index") or len(selected)) - 1
    except Exception:
        target_idx = len(selected) - 1
    target_idx = max(0, min(target_idx, len(selected) - 1))
    patch = selected[target_idx]
    depth = int(patch.extras.get("zoom_depth") or 0)
    if depth >= max_zoom_depth:
        return None
    slide_path = str(patch.extras.get("slide_path") or "")
    if not slide_path:
        return None
    x = patch.x
    y = patch.y
    width = patch.width
    height = patch.height
    if x is None or y is None or not width or not height:
        return None
    region = str(judgement.get("zoom_region") or "center").strip().lower()
    rx, ry = {
        "center": (0.50, 0.50),
        "top-left": (0.25, 0.25),
        "top-right": (0.75, 0.25),
        "bottom-left": (0.25, 0.75),
        "bottom-right": (0.75, 0.75),
    }.get(region, (0.50, 0.50))
    new_w = max(256, int(width * 0.5))
    new_h = max(256, int(height * 0.5))
    cx = x + int(width * rx)
    cy = y + int(height * ry)
    new_x = max(0, cx - new_w // 2)
    new_y = max(0, cy - new_h // 2)
    zoom_dir.mkdir(parents=True, exist_ok=True)
    zoom_id = f"{patch.patch_id}_zoom{depth + 1}_{region.replace('-', '')}_x{new_x}_y{new_y}_w{new_w}_h{new_h}"
    out_path = zoom_dir / f"{zoom_id}.png"
    if not out_path.exists():
        from pathxai_agent.wsi.slide_reader import SlideReader

        reader = SlideReader(slide_path)
        try:
            image = reader.read_region(new_x, new_y, new_w, new_h, patch.level)
        finally:
            reader.close()
        image.save(out_path)
    extras = dict(patch.extras)
    extras.update({
        "zoom_depth": depth + 1,
        "zoom_parent_patch_id": patch.patch_id,
        "zoom_region": region,
        "zoom_reason": str(judgement.get("zoom_reason") or ""),
    })
    return PatchEvidence(
        patch_id=zoom_id,
        image_path=str(out_path),
        x=new_x,
        y=new_y,
        level=patch.level,
        width=new_w,
        height=new_h,
        source=f"{patch.source}:llm_zoom",
        description="LLM-requested zoom-in crop from a selected ROI.",
        score=patch.score + 0.001 * (depth + 1),
        extras=extras,
    )


def _build_report_prompt(question: str, patches: List[PatchEvidence], reference: Optional[dict]) -> str:
    return f"""/no_think
You are a senior pathology assistant generating a WSI-level pathology report from PathAgent-selected ROI patches.

Task objective:
{question}

Selected patch evidence:
{evidence_text(patches)}

Foundation model specimen/site hint:
{foundation_site_hint_text(patches)}

First produce a short internal patch-analysis summary, then generate one report text from the same ROI evidence:
- PATCH_ANALYSIS is saved separately for debugging and has no ground truth metric.
- LONG_REPORT must follow the unified cleaned pathology-report style used for evaluation.

Output exactly these two blocks:

PATCH_ANALYSIS:
Briefly summarize which visual findings in the selected ROI images support the diagnosis/differential. You may mention Patch 1, Patch 2, etc. here only.

LONG_REPORT:
Specimen/Site: Use the foundation site hint only if it is visually plausible from the ROI morphology. If not visually identifiable, write "Not specified from ROI evidence." If the hint and morphology conflict, state uncertainty rather than guessing.
Diagnosis: State the most likely diagnosis only if supported by morphology. If uncertain, write a differential diagnosis and explicitly mark uncertainty.
Tumor/Lesion Description: Describe lesion architecture, cellular morphology, cytoplasm, nuclear features, necrosis, and background tissue when visible.
Key Microscopic Findings: Give 2-5 concise image-grounded findings in report style. Do not mention patch labels.
Invasion/Necrosis/Margins: Report invasion, necrosis, capsule/margin status only if visible. Otherwise write "Not assessable from provided ROI patches."
Immunohistochemistry/Ancillary Studies: Usually "Not provided." Do not infer IHC or molecular results from H&E morphology.
Stage/Grade/Prognostic Features: Report grade/stage/prognostic features only if visually assessable or provided. Otherwise write "Not assessable from provided ROI patches."
Final Summary: One concise final interpretation, including uncertainty and limitations.

Rules:
- Base claims only on attached ROI images and provided patch descriptions.
- Treat the foundation site hint as auxiliary zero-shot evidence, not ground truth and not a dataset label.
- Do not mention Patch 1, Patch 2, ROI, crop, image filename, coordinates, selected evidence, or model/agent steps in LONG_REPORT.
- Patch labels are allowed only in PATCH_ANALYSIS.
- Do not infer organ, cancer type, diagnosis, or specimen site from case IDs, slide IDs, file names, directory names, dataset-specific codes, coordinates, or patch labels.
- Be explicit about uncertainty.
- Use "Not specified", "Not provided", or "Not assessable from provided ROI patches" instead of inventing missing information.
- Do not invent segmentation, cell-count, immunohistochemistry, grade, stage, organ/site, margin, or molecular details unless visible or provided.
- Keep LONG_REPORT concise but complete: 150-300 words.
- The final report should look like the cleaned long-report target, not patch-level notes.
- Do not include hidden chain-of-thought."""


def _build_report_retry_prompt(question: str, patches: List[PatchEvidence], reference: Optional[dict]) -> str:
    return f"""/no_think
Generate the required output blocks only. Your previous response was incomplete or did not follow the required format.

Task objective:
{question}

Selected anonymous ROI evidence:
{evidence_text(patches)}

Foundation model specimen/site hint:
{foundation_site_hint_text(patches)}

Anti-leakage rules:
- Ignore all case identifiers, slide identifiers, filenames, directory names, TCGA codes, and coordinates for diagnosis/site inference.
- Use only the visual appearance of the attached ROI images and any provided patch descriptions.
- Refer to evidence only as Patch 1, Patch 2, etc.

Output exactly:

PATCH_ANALYSIS:
Brief visible-evidence summary. Patch labels may be mentioned only here.

LONG_REPORT:
Specimen/Site: Use the foundation site hint only if visually plausible; otherwise "Not specified from ROI evidence" or uncertainty.
Diagnosis: State most likely diagnosis only if morphology supports it; otherwise give an uncertainty-marked differential.
Tumor/Lesion Description: Describe visible architecture, cellular morphology, cytoplasm, nuclear features, necrosis, and background.
Key Microscopic Findings: Give 2-5 image-grounded findings in report style. Do not mention patch labels.
Invasion/Necrosis/Margins: Visible findings only; otherwise "Not assessable from provided ROI patches."
Immunohistochemistry/Ancillary Studies: "Not provided" unless supplied.
Stage/Grade/Prognostic Features: Visible/provided findings only; otherwise "Not assessable from provided ROI patches."
Final Summary: Concise interpretation with uncertainty and limitations.

Do not mention Patch 1, Patch 2, ROI, crop, image filename, coordinates, selected evidence, or model/agent steps in LONG_REPORT.
Do not infer diagnosis/site from filenames, IDs, directory names, or coordinates."""


def _has_required_report_blocks(report: str) -> bool:
    return "LONG_REPORT:" in report


def _split_report_blocks_for_markdown(report: str) -> dict:
    patch_marker = "PATCH_ANALYSIS:"
    long_marker = "LONG_REPORT:"
    if patch_marker in report and long_marker in report and report.find(patch_marker) < report.find(long_marker):
        patch_start = report.find(patch_marker) + len(patch_marker)
        long_start = report.find(long_marker)
        return {
            "patch_analysis": report[patch_start:long_start].strip(),
            "reports": report[long_start:].strip(),
        }
    return {"patch_analysis": "", "reports": report.strip()}


def _reference_text(reference: Optional[dict]) -> str:
    if not reference:
        return "No reference text provided."
    summary = reference.get("summary_report_text") or reference.get("summary")
    full = reference.get("full_report_text") or reference.get("full")
    parts = []
    if summary:
        parts.append(f"Summary target: {summary[:1200]}")
    if full:
        parts.append(f"Full report target preview: {full[:1200]}")
    return "\n".join(parts) if parts else "No reference text provided."


def _description_keys(patch: PatchEvidence) -> List[str]:
    keys = [Path(patch.image_path).name, Path(patch.image_path).stem]
    if patch.x is not None and patch.y is not None:
        keys.extend([f"{patch.x}_{patch.y}.jpg", f"{patch.x}_{patch.y}.png", f"{patch.x}_{patch.y}"])
    return keys


def _tokenize(text: str) -> List[str]:
    import re

    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _int_or_none(value) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _format_optional_float(value) -> str:
    try:
        if value is None:
            return "n/a"
        return f"{float(value):.3f}"
    except Exception:
        return "n/a"
