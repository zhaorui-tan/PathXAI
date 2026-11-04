"""Evidence graph + claims — structured evidence representation and claim verification.

Builds a graph of case→slide→ROI→evidence nodes, then generates diagnosis slots,
extracts claims, and verifies them against evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathxai_agent_v2.config import Settings
from pathxai_agent_v2.llm import LLMClient, parse_llm_json
from pathxai_agent_v2.pipeline.base import (
    DiagnosisPrediction,
    EvidenceSufficiency,
    PipelineStage,
    RoiInfo,
    SitePrediction,
    StageResult,
    build_llm_image_paths,
    ok,
)


@dataclass
class EvidenceGraphOutput:
    """Structured evidence graph for a case."""
    graph: dict[str, Any] = field(default_factory=dict)
    diagnosis_slots: dict[str, Any] = field(default_factory=dict)
    claims: list[dict] = field(default_factory=list)
    verified_claims: list[dict] = field(default_factory=list)
    filtered_slots: dict[str, Any] = field(default_factory=dict)
    seg_verification_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph,
            "diagnosis_slots": self.diagnosis_slots,
            "claims": self.claims,
            "verified_claims": self.verified_claims,
            "filtered_slots": self.filtered_slots,
            "seg_verification_summary": self.seg_verification_summary,
        }


class EvidenceGraphStage(PipelineStage[dict, EvidenceGraphOutput]):
    """Stage: Build evidence graph, diagnosis slots, and verify claims."""

    stage_name = "evidence_graph"

    def run(
        self,
        stage_input: dict,
        dry_run: bool = False,
    ) -> StageResult[EvidenceGraphOutput]:
        """Build evidence graph and verify claims.

        stage_input expects:
            case_id, slides_used, candidates, selected_rois,
            site_prediction, diagnosis_prediction, evidence_sufficiency,
            segmentation_context, knowledge_context, morphology_context
        """
        case_id = stage_input.get("case_id", "unknown")
        selected: list[RoiInfo] = stage_input.get("selected_rois", [])
        site_pred: SitePrediction | None = stage_input.get("site_prediction")
        diag: DiagnosisPrediction | None = stage_input.get("diagnosis_prediction")
        suff: EvidenceSufficiency | None = stage_input.get("evidence_sufficiency")
        seg_ctx = stage_input.get("segmentation_context", {})
        morph_ctx = stage_input.get("morphology_context", {})
        know_ctx = stage_input.get("knowledge_context", {})
        broad = stage_input.get("broad_hypotheses")
        subtype = stage_input.get("subtype_sufficiency")
        reinspection = stage_input.get("reinspection_history") or []
        enable_claim_verification = bool(stage_input.get("enable_claim_verification", True))
        enable_llm_claim_verifier = bool(stage_input.get("enable_llm_claim_verifier", True))

        # Build evidence graph
        graph = _build_graph(case_id, selected, site_pred, diag, suff, seg_ctx, morph_ctx, know_ctx, broad, subtype, reinspection)

        # Build diagnosis slots from diagnosis prediction
        slots = _build_slots(diag, site_pred, suff)
        claims = _slots_to_claims(slots)

        # Heuristic + optional LLM verification
        if enable_claim_verification:
            verified, seg_summary = _verify_claims(
                claims,
                graph,
                seg_ctx,
                selected,
                suff,
                self.llm,
                dry_run,
                enable_llm=enable_llm_claim_verifier,
            )
            filtered = _filter_slots(slots, verified)
        else:
            verified = [{**c, "status": "unverified", "verification_method": "ablated"} for c in claims]
            filtered = dict(slots)
            seg_summary = {}

        if dry_run:
            verified = claims  # no verification in dry run

        return ok(
            EvidenceGraphOutput(
                graph=graph,
                diagnosis_slots=slots,
                claims=claims,
                verified_claims=verified,
                filtered_slots=filtered,
                seg_verification_summary=seg_summary,
            ),
            self.stage_name,
        )


def _build_graph(
    case_id: str,
    selected: list[RoiInfo],
    site_pred: SitePrediction | None,
    diag: DiagnosisPrediction | None,
    suff: EvidenceSufficiency | None,
    seg_ctx: dict,
    morph_ctx: dict,
    know_ctx: dict,
    broad: Any,
    subtype: Any,
    reinspection: list,
) -> dict:
    nodes = []
    edges = []

    # Case node
    nodes.append({"node_id": f"case::{case_id}", "type": "case", "case_id": case_id})

    # ROI nodes
    for roi in selected:
        roi_node = {
            "node_id": f"roi::{roi.roi_id}",
            "type": "roi",
            "roi_id": roi.roi_id,
            "image_path": roi.image_path,
            "score": roi.score,
            "segmentation": _find_seg(seg_ctx, roi.roi_id),
        }
        nodes.append(roi_node)
        edges.append({"source": f"case::{case_id}", "target": f"roi::{roi.roi_id}", "relation": "contains_roi"})

    # Evidence nodes
    evidence = [
        ("site", site_pred.to_dict() if site_pred else {}),
        ("diagnosis", diag.to_dict() if diag else {}),
        ("sufficiency", suff.to_dict() if suff else {}),
        ("morphology", _strip_heavy(morph_ctx)),
        ("knowledge", _strip_heavy(know_ctx)),
        ("broad_hypotheses", broad.to_dict() if hasattr(broad, "to_dict") else (broad or {})),
        ("subtype_sufficiency", subtype.to_dict() if hasattr(subtype, "to_dict") else (subtype or {})),
        ("reinspection", {"rounds": [r.to_dict() if hasattr(r, "to_dict") else r for r in reinspection]}),
    ]
    for etype, payload in evidence:
        nodes.append({"node_id": f"evidence::{etype}", "type": etype, "payload": payload})
        for roi in selected:
            edges.append({"source": f"roi::{roi.roi_id}", "target": f"evidence::{etype}", "relation": "supports"})

    return {"graph_version": "v2", "nodes": nodes, "edges": edges}


def _find_seg(seg_ctx: dict, roi_id: str) -> dict:
    for r in seg_ctx.get("roi_results", []):
        if r.get("roi_id") == roi_id:
            stats = r.get("stats", {}) or {}
            named_raw = stats.get("named_class_fractions") or r.get("named_fractions") or {}
            named = {
                name: info.get("fraction", info)
                for name, info in named_raw.items()
                if name != "Background"
            } if isinstance(named_raw, dict) else {}
            return {
                "tissue_types": named,
                "tumor_pct": stats.get("class_fractions", {}).get("1", 0),
            }
    return {}


def _build_slots(
    diag: DiagnosisPrediction | None,
    site_pred: SitePrediction | None,
    suff: EvidenceSufficiency | None,
) -> dict:
    if diag is None:
        return {}
    return {
        "site": site_pred.predicted_site if site_pred else "",
        "primary_diagnosis": diag.primary_diagnosis,
        "differential_diagnosis": diag.differential,
        "tumor_architecture": [diag.tumor_architecture] if diag.tumor_architecture else [],
        "cytology": [diag.cytology] if diag.cytology else [],
        "necrosis": diag.necrosis,
        "invasion": diag.invasion,
        "stroma_inflammation": [diag.stroma_inflammation] if diag.stroma_inflammation else [],
        "ihc_molecular": diag.ihc_molecular,
        "confidence": diag.confidence,
        "missing_evidence": diag.missing_evidence,
        "recommended_ancillary_studies": diag.recommended_ancillary_studies,
        "broad_type": diag.broad_type,
        "subtype": diag.subtype,
        "malignancy": diag.malignancy,
        "grade": diag.grade,
    }


def _slots_to_claims(slots: dict) -> list[dict]:
    claims = []
    cid = 0
    scalar_fields = ["site", "primary_diagnosis", "broad_type", "subtype", "malignancy", "grade", "necrosis", "invasion", "ihc_molecular"]
    list_fields = ["tumor_architecture", "cytology", "stroma_inflammation"]

    for field in scalar_fields:
        val = slots.get(field, "")
        if val:
            cid += 1
            claims.append({"claim_id": f"C{cid:03d}", "field": field, "claim": str(val), "status": "unverified"})

    for field in list_fields:
        vals = slots.get(field, [])
        for val in (vals if isinstance(vals, list) else [vals]):
            if val:
                cid += 1
                claims.append({"claim_id": f"C{cid:03d}", "field": field, "claim": str(val), "status": "unverified"})

    return claims


def _heuristic_verify(
    claims: list[dict],
    seg_ctx: dict,
    selected: list[RoiInfo],
    suff: EvidenceSufficiency | None = None,
) -> tuple[list[dict], dict]:
    """Rule-based claim verification — seg-grounded, with keyword fallback.

    Uses ``claim_seg_verifier.verify_all_claims`` as the primary structured
    verifier.  Fields that segmentation cannot assess (``ihc_molecular``,
    ``differential_diagnosis``, ``confidence``, ``site``) fall through to
    keyword-based heuristics.
    """
    from pathxai_agent_v2.pipeline.claim_seg_verifier import verify_all_claims as _seg_verify

    can_report = bool(suff.can_confidently_report) if suff else False
    seg_results = seg_ctx.get("roi_results", []) if seg_ctx else []

    # ── Primary: claim-seg alignment ─────────────────────────────────────
    if seg_results and seg_ctx.get("available"):
        claims_with_seg, seg_summary = _seg_verify(claims, seg_results)
    else:
        claims_with_seg = [
            {**c, "seg_verification": {"status": "not_assessable", "reason": "segmentation unavailable"}}
            for c in claims
        ]
        seg_summary = {"total": len(claims), "not_assessable": len(claims)}

    # ── Fallback: keyword heuristics for fields seg cannot verify ────────
    SEG_UNVERIFIABLE = {"ihc_molecular", "differential_diagnosis", "confidence", "site"}

    verified = []
    for claim in claims_with_seg:
        field = claim.get("field", "")
        claim_text = str(claim.get("claim", "")).lower()
        seg_v = claim.get("seg_verification", {})
        seg_status = seg_v.get("status", "not_assessable")

        # Explicitly unassessable claim text → always not_assessable
        if any(t in claim_text for t in (
            "not assessable", "not available", "not provided",
            "unavailable", "cannot be assessed", "cannot assess",
        )):
            verified.append({**claim, "status": "not_assessable",
                "verification_method": "text_parse", "reason": "claim states item is unavailable"})
            continue

        # IHC/molecular — never verifiable from H&E seg
        if field in {"ihc_molecular", "grade"}:
            verified.append({**claim, "status": "not_assessable",
                "verification_method": "heuristic",
                "reason": "requires ancillary/non-image evidence"})
            continue

        # Site — known input, always supported
        if field == "site":
            verified.append({**claim, "status": "supported",
                "verification_method": "heuristic", "reason": "known/predicted site context"})
            continue

        # Differential / confidence — reasoning metadata
        if field in {"differential_diagnosis", "confidence"}:
            verified.append({**claim, "status": "not_assessable",
                "verification_method": "heuristic", "reason": "reasoning metadata"})
            continue

        # ── Use seg verification result ────────────────────────────────
        if seg_status == "contradicted":
            # Seg explicitly contradicts the claim — strong signal
            verified.append({**claim, "status": "contradicted",
                "verification_method": "segmentation", "reason": seg_v.get("reason", "")})
        elif seg_status == "supported":
            verified.append({**claim, "status": "supported",
                "verification_method": "segmentation", "reason": seg_v.get("reason", "")})
        elif seg_status == "not_visible":
            # Seg shows the relevant tissue class is absent
            verified.append({**claim, "status": "weak",
                "verification_method": "segmentation",
                "reason": f"seg shows no relevant tissue: {seg_v.get('reason', '')}"})
        else:
            # Seg couldn't assess — fall back to tumor-presence heuristic
            tumor_pct = _max_class_pct(seg_ctx, "1") if seg_results else 0.0
            if tumor_pct > 0.1 and (can_report or field in {"tumor_architecture", "cytology", "broad_type"}):
                verified.append({**claim, "status": "supported",
                    "verification_method": "heuristic", "reason": "tumor-rich ROI evidence available"})
            else:
                verified.append({**claim, "status": "weak",
                    "verification_method": "heuristic",
                    "reason": "visual evidence exists but insufficient for definitive claim"})

    return verified, seg_summary


def _verify_claims(
    claims: list[dict],
    graph: dict,
    seg_ctx: dict,
    selected: list[RoiInfo],
    suff: EvidenceSufficiency | None,
    llm: LLMClient,
    dry_run: bool,
    *,
    enable_llm: bool = True,
) -> list[dict]:
    heuristic, seg_summary = _heuristic_verify(claims, seg_ctx, selected, suff)
    if dry_run or not claims or not enable_llm:
        return heuristic, seg_summary

    prompt = f"""/no_think
Verify pathology claims against the provided evidence graph and ROI images.

Statuses:
- supported: directly supported by visual ROI evidence, morphology text, or segmentation stats.
- weak: plausible but not directly proven.
- unsupported: no available evidence supports the claim.
- contradicted: available evidence suggests the claim is false.
- not_assessable: cannot be assessed from H&E ROI evidence.

Rules:
- Do not use filenames, case IDs, TCGA codes, directory names, or coordinates.
- KEEP is background knowledge, not patient-specific evidence.
- IHC/molecular/stage/gross/margins are not assessable unless explicitly supplied.
- Cite evidence using ROI ids, evidence::morphology, evidence::knowledge, evidence::subtype_sufficiency, or evidence::sufficiency.

Claims:
{json.dumps(claims, indent=2, ensure_ascii=False)}

Evidence graph:
{json.dumps(_strip_for_llm(graph), indent=2, ensure_ascii=False)}

Segmentation:
{json.dumps(_strip_for_llm(seg_ctx or {}), indent=2, ensure_ascii=False)}

Return ONLY JSON:
{{
  "claims": [
    {{"claim_id": "C001", "status": "supported/weak/unsupported/contradicted/not_assessable", "evidence": ["..."], "reason": "..."}}
  ],
  "unsupported_summary": ["..."],
  "missing_evidence": ["..."]
}}"""
    try:
        raw = llm.chat_json(prompt, image_paths=build_llm_image_paths(selected, seg_ctx))
        parsed = parse_llm_json(raw)
    except Exception:
        return heuristic, seg_summary
    llm_claims = parsed.get("claims") if isinstance(parsed.get("claims"), list) else []
    if not llm_claims:
        return heuristic, seg_summary
    return _merge_verification(heuristic, llm_claims), seg_summary


def _merge_verification(heuristic: list[dict], llm_claims: list[dict]) -> list[dict]:
    by_id = {str(row.get("claim_id")): dict(row) for row in heuristic}
    for row in llm_claims:
        cid = str(row.get("claim_id") or "")
        if cid not in by_id:
            continue
        merged = dict(by_id[cid])
        field = str(merged.get("field") or "").lower()
        claim_text = str(merged.get("claim") or "").lower()
        status = str(row.get("status") or "").lower()
        if status in {"supported", "weak", "unsupported", "contradicted", "not_assessable"}:
            # Do not let the LLM downgrade protected bookkeeping claims.
            # Known-site context is metadata, not a visual claim to re-litigate.
            # Explicit "not assessable" statements are the desired safe output,
            # not contradictions.
            if field == "site" and by_id[cid].get("status") == "supported":
                status = "supported"
            elif "not assessable" in claim_text and status in {"unsupported", "contradicted", "weak"}:
                status = by_id[cid].get("status") or "not_assessable"
            merged["status"] = status
        if row.get("evidence"):
            merged["evidence"] = row.get("evidence")
        if row.get("reason"):
            merged["reason"] = row.get("reason")
        merged["verification_method"] = "llm_plus_heuristic"
        by_id[cid] = merged
    return [by_id[str(row.get("claim_id"))] for row in heuristic]


def _max_class_pct(seg_ctx: dict, class_id: str) -> float:
    """Get max fraction of a tissue class across all ROIs."""
    max_pct = 0.0
    for r in seg_ctx.get("roi_results", []):
        pct = float(r.get("stats", {}).get("class_fractions", {}).get(class_id, 0))
        if pct > max_pct:
            max_pct = pct
    return max_pct


def _filter_slots(slots: dict, verified: list[dict]) -> dict:
    """Filter diagnosis slots: remove unsupported claims."""
    filtered = dict(slots)
    unsupported = {c["field"] for c in verified if c.get("status") in ("unsupported", "contradicted")}
    not_assessable = {
        c["field"] for c in verified
        if c.get("status") == "not_assessable" and c.get("field") in {"ihc_molecular", "grade"}
    }
    weak_specific = {
        c["field"] for c in verified
        if c.get("status") == "weak" and c.get("field") in {"grade"}
    }

    for field in unsupported:
        if field in filtered:
            filtered[field] = "not supported by available ROI evidence"
    for field in not_assessable:
        if field in filtered:
            filtered[field] = "not assessable from H&E images"
    for field in weak_specific:
        if field in filtered:
            filtered[field] = "not assessable from provided ROI evidence"

    return filtered


def _strip_heavy(obj: Any, max_str: int = 900, max_list: int = 8) -> Any:
    """Small JSON-safe context for LLM/HTML graph nodes."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k).endswith("_path") or str(k) in {"image_path", "mask_path", "overlay_path", "composite_path"}:
                out[k] = v
            else:
                out[k] = _strip_heavy(v, max_str=max_str, max_list=max_list)
        return out
    if isinstance(obj, list):
        return [_strip_heavy(x, max_str=max_str, max_list=max_list) for x in obj[:max_list]]
    if isinstance(obj, str) and len(obj) > max_str:
        return obj[:max_str] + "...[truncated]"
    return obj


def _strip_for_llm(obj: Any, max_str: int = 900, max_list: int = 8) -> Any:
    """JSON-safe evidence context with patient/file identifiers removed."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = str(k)
            if key in {
                "case_id",
                "slide_id",
                "image_path",
                "mask_path",
                "overlay_path",
                "composite_path",
                "path",
                "annotated_path",
            } or key.endswith("_path"):
                continue
            if key in {"node_id", "source", "target"} and isinstance(v, str):
                out[key] = _anonymize_node_id(v)
            else:
                out[key] = _strip_for_llm(v, max_str=max_str, max_list=max_list)
        return out
    if isinstance(obj, list):
        return [_strip_for_llm(x, max_str=max_str, max_list=max_list) for x in obj[:max_list]]
    if isinstance(obj, str):
        value = _anonymize_node_id(obj)
        if len(value) > max_str:
            return value[:max_str] + "...[truncated]"
        return value
    return obj


def _anonymize_node_id(value: str) -> str:
    if "::" not in value:
        return value
    prefix, suffix = value.split("::", 1)
    if prefix == "case":
        return "case::current_case"
    if prefix == "roi":
        return "roi::" + str(abs(hash(suffix)) % 100000)
    return value
