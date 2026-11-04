#!/usr/bin/env python3
"""Render a tabbed evidence-review workspace HTML for one PathXAI Agent case.

This is intentionally independent from the existing report.html renderer so the
old visualization remains reproducible while the review UI can evolve.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from html import escape
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROI_PALETTE_HEX = [
    "#3b82f6", "#ef4444", "#22c55e", "#f59e0b",
    "#a855f7", "#14b8a6", "#ec4899", "#fb923c",
]


def load_json(path: Path, default: Any = None) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def load_text(path: Path, default: str = "") -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return default
    return default


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def short(text: Any, n: int = 180) -> str:
    s = str(text or "").strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "..."


def rel_src(path_value: Any, html_path: Path) -> str:
    if not path_value:
        return ""
    raw = str(path_value)
    p = Path(raw)
    if not p.is_absolute():
        if raw.startswith("outputs/") or raw.startswith("shared_artifacts/"):
            p = PROJECT_ROOT / raw
        else:
            p = html_path.parent / raw
    try:
        return os.path.relpath(p, html_path.parent)
    except Exception:
        return raw


def slug(text: Any) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", str(text or "").strip()).strip("-")
    return s or "item"


def chip(ref: str, label: str | None = None, cls: str = "") -> str:
    return (
        f'<button class="xref {escape(cls)}" data-ref="{escape(ref)}" type="button">'
        f"{escape(label or ref)}</button>"
    )


def evidence_chip_class(ref: Any) -> str:
    ref_s = str(ref or "")
    if ref_s.startswith("FIG-"):
        return "evidence-tag visual"
    if ref_s.startswith("MORPH-R"):
        return "evidence-tag morph"
    if ref_s.startswith("KEEP-"):
        return "evidence-tag kg"
    if ref_s == "SEG":
        return "evidence-tag seg"
    if ref_s in {"DX", "SITE"}:
        return "evidence-tag model"
    if ref_s in {"SUBTYPE", "BROAD", "SUFF"}:
        return "evidence-tag reasoning"
    if ref_s.startswith("C"):
        return "evidence-tag verify"
    return "evidence-tag"


def evidence_chip(ref: Any, label: str | None = None) -> str:
    return chip(str(ref or ""), label, evidence_chip_class(ref))


def evidence_legend() -> str:
    items = [
        ("Visual ROI", "visual"),
        ("Segmentation", "seg"),
        ("Morphology", "morph"),
        ("KEEP", "kg"),
        ("Model reasoning", "model"),
        ("Verification", "verify"),
    ]
    return (
        '<div class="evidence-legend">'
        + "".join(f'<span class="legend-chip {escape(cls)}">{escape(label)}</span>' for label, cls in items)
        + "</div>"
    )


def badge(text: Any, cls: str = "neutral") -> str:
    if text in (None, ""):
        return ""
    return f'<span class="badge {escape(cls)}">{escape(str(text))}</span>'


def roi_color(fig_number: Any) -> str:
    try:
        idx = int(fig_number) - 1
    except Exception:
        idx = 0
    return ROI_PALETTE_HEX[idx % len(ROI_PALETTE_HEX)]


def figure_chip(fig_number: Any, label: str | None = None) -> str:
    try:
        num = int(fig_number)
    except Exception:
        return ""
    color = roi_color(num)
    return (
        f'<button class="figure-chip" data-ref="FIG-{num}" type="button" style="--fig-color:{escape(color)}">'
        f'<span>{num}</span>{escape(label or f"Figure {num}")}</button>'
    )


def kv(label: str, value: Any) -> str:
    if value in (None, "", []):
        value = "not available"
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v) for v in value[:8])
    return (
        '<div class="kv">'
        f'<div class="kv-label">{escape(label)}</div>'
        f'<div class="kv-value">{escape(str(value))}</div>'
        "</div>"
    )


def list_html(items: Any, limit: int = 8) -> str:
    vals = [str(x) for x in as_list(items) if str(x).strip()]
    if not vals:
        return '<div class="muted">None recorded.</div>'
    more = ""
    if len(vals) > limit:
        more = f'<li class="muted">+ {len(vals) - limit} more</li>'
    return "<ul>" + "".join(f"<li>{escape(v)}</li>" for v in vals[:limit]) + more + "</ul>"


def first_existing(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def checkpoint_data(stage: dict[str, Any]) -> dict[str, Any]:
    stage_path = stage.get("path")
    if not stage_path:
        return {}
    p = Path(stage_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / stage_path
    raw = load_json(p, {}) or {}
    data = raw.get("data", raw) if isinstance(raw, dict) else {}
    return data if isinstance(data, dict) else {}


def parse_medgemma_sections(text: Any) -> dict[str, str]:
    s = str(text or "")
    labels = [
        "FINDINGS",
        "IMPRESSION",
        "TOP 3 SUSPECTED DIAGNOSES",
        "LIMITATIONS",
        "FINAL DIAGNOSIS",
        "FINAL PATHOLOGY DIAGNOSIS",
    ]
    out: dict[str, str] = {}
    pattern = re.compile(rf"(?im)^({'|'.join(re.escape(x) for x in labels)})\s*:\s*")
    matches = list(pattern.finditer(s))
    if not matches:
        return {"Summary": short(s, 1200)}
    for idx, match in enumerate(matches):
        label = match.group(1).title()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(s)
        content = s[start:end].strip()
        # MedGemma sometimes repeats FINAL DIAGNOSIS many times. Keep it readable.
        if label in out:
            continue
        out[label] = short(content, 900)
    return out


NFC_PATTERNS = [
    r",?\s*not further classifiable from ROI evidence(?: without molecular markers)?",
    r",?\s*not further classifiable",
]


def clean_diagnosis_label(text: Any) -> tuple[str, str]:
    """Split display diagnosis from uncertainty/limitations.

    The model may store cautious limitations inside primary_diagnosis for
    machine-readable provenance. For the review UI, keep the diagnostic entity
    readable and show the limitation separately.
    """
    raw = re.sub(r"\s+", " ", str(text or "").strip())
    if not raw:
        return "Diagnosis not available", ""
    label = raw
    limitation = ""
    for pat in NFC_PATTERNS:
        if re.search(pat, label, flags=re.I):
            label = re.sub(pat, "", label, flags=re.I).strip(" ,;")
            limitation = "Subtype not further classifiable from current ROI evidence."
            break
    return label or raw, limitation


def clean_report_text(text: Any) -> str:
    s = str(text or "")
    for pat in NFC_PATTERNS:
        s = re.sub(pat, "", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


EVIDENCE_ALIASES = {
    "evidence::site": ("SITE", "Site context"),
    "evidence::diagnosis": ("DX", "Diagnosis model"),
    "evidence::subtype_sufficiency": ("SUBTYPE", "Subtype sufficiency"),
    "evidence::broad_hypotheses": ("BROAD", "Broad hypotheses"),
    "evidence::sufficiency": ("SUFF", "Evidence sufficiency"),
    "evidence::knowledge": ("KEEP-1", "KEEP knowledge"),
}


def evidence_ref_label(ev: Any, figures: list[dict[str, Any]]) -> tuple[str | None, str]:
    ev_s = str(ev or "")
    fref = figure_ref_for_evidence(ev_s, figures)
    if fref:
        return fref, fref.replace("FIG-", "Figure ")
    if ev_s.startswith("FIG-"):
        return ev_s, ev_s.replace("FIG-", "Figure ")
    if ev_s.startswith("MORPH-R"):
        return ev_s, ev_s.replace("MORPH-R", "Morphology R")
    if ev_s.startswith("KEEP-"):
        return ev_s, ev_s.replace("KEEP-", "KEEP ")
    if ev_s in {"SITE", "DX", "SUBTYPE", "BROAD", "SUFF", "SEG"}:
        labels = {
            "SITE": "Site context",
            "DX": "Diagnosis model",
            "SUBTYPE": "Subtype sufficiency",
            "BROAD": "Broad hypotheses",
            "SUFF": "Evidence sufficiency",
            "SEG": "Segmentation",
        }
        return ev_s, labels.get(ev_s, ev_s)
    if ev_s in EVIDENCE_ALIASES:
        return EVIDENCE_ALIASES[ev_s]
    if ev_s.startswith("evidence::morphology") or ev_s.startswith("evidence::visual_description"):
        return "MORPH-R1", "Morphology"
    if ev_s.startswith("evidence::segmentation"):
        return "SEG", "Segmentation"
    if ev_s.startswith("evidence::"):
        return None, ev_s.replace("evidence::", "").replace("_", " ").title()
    return None, short(ev_s, 40)


def sanitise_internal_refs(obj: Any) -> Any:
    if isinstance(obj, str):
        return re.sub(r"\broi::\d+\b", "linked ROI evidence", obj)
    if isinstance(obj, list):
        return [sanitise_internal_refs(x) for x in obj]
    if isinstance(obj, tuple):
        return [sanitise_internal_refs(x) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitise_internal_refs(v) for k, v in obj.items()}
    return obj


def read_case(case_dir: Path) -> dict[str, Any]:
    payload = load_json(case_dir / "staged_trace.json", {}) or {}
    payload.setdefault("case_id", case_dir.name)
    payload["_case_dir"] = str(case_dir)
    payload["_files"] = {
        "eval": load_json(case_dir / "eval.json", {}) or {},
        "diagnostic_hierarchy": load_json(case_dir / "diagnostic_hierarchy_eval.json", {}) or {},
        "diagnosis_prediction": load_json(case_dir / "diagnosis_prediction.json", {}) or payload.get("diagnosis_prediction") or {},
        "diagnosis_slots": load_json(case_dir / "diagnosis_slots.json", {}) or payload.get("diagnosis_slots") or {},
        "claim_verification": load_json(case_dir / "claim_verification.json", {}) or payload.get("claim_verification") or {},
        "evidence_sufficiency": load_json(case_dir / "evidence_sufficiency.json", {}) or payload.get("evidence_sufficiency") or {},
        "knowledge_context": load_json(case_dir / "knowledge_context.json", {}) or payload.get("knowledge_context") or {},
        "medgemma": load_json(case_dir / "medgemma_morphology_context.json", {}) or payload.get("medgemma_morphology_context") or {},
        "segmentation": load_json(case_dir / "segmentation_context.json", {}) or payload.get("segmentation_context") or {},
        "progress_trace": load_json(case_dir / "progress_trace.json", {}) or {},
        "artifact_index": load_json(case_dir / "case_artifact_index.json", {}) or {},
    }
    payload["_reports"] = {
        "long": load_text(case_dir / "generated_long_report.txt", payload.get("generated_long_report") or ""),
        "rich": load_text(case_dir / "generated_long_report_rich.txt", payload.get("generated_long_report_rich") or ""),
        "md": load_text(case_dir / "report.md", ""),
    }
    return payload


def _path_keys(value: Any) -> list[str]:
    if not value:
        return []
    raw = str(value)
    p = Path(raw)
    return [raw, p.name, p.stem]


def figure_records(payload: dict[str, Any], html_path: Path) -> list[dict[str, Any]]:
    figs = payload.get("figures") or []
    seg_by_roi: dict[str, dict[str, Any]] = {}
    for row in (payload.get("_files", {}).get("segmentation") or {}).get("roi_results", []) or []:
        rid = row.get("roi_id")
        if rid:
            seg_by_roi[rid] = row
        for key in _path_keys(row.get("image_path")):
            seg_by_roi.setdefault(key, row)
    selected_by_roi = {r.get("roi_id"): r for r in payload.get("selected_rois") or [] if r.get("roi_id")}

    out: list[dict[str, Any]] = []
    for i, fig in enumerate(figs, start=1):
        rid = fig.get("roi_id") or fig.get("id") or fig.get("image_path") or f"roi-{i}"
        selected = selected_by_roi.get(rid, {})
        seg = fig.get("segmentation") or seg_by_roi.get(rid) or {}
        if not seg:
            for key in _path_keys(fig.get("image_path")):
                if key in seg_by_roi:
                    seg = seg_by_roi[key]
                    break
        image_path = fig.get("image_path") or selected.get("image_path") or seg.get("image_path")
        overlay_path = (
            fig.get("overlay_path")
            or fig.get("segmentation_overlay_path")
            or selected.get("overlay_path")
            or selected.get("segmentation_overlay_path")
            or seg.get("overlay_path")
        )
        composite_path = (
            fig.get("composite_path")
            or selected.get("composite_path")
            or selected.get("segmentation_composite_path")
            or seg.get("composite_path")
        )
        mask_path = (
            fig.get("mask_path")
            or selected.get("mask_path")
            or selected.get("segmentation_mask_path")
            or seg.get("mask_path")
        )
        label = fig.get("figure") or f"Figure {fig.get('roi_index') or i}"
        ref = f"FIG-{i}"
        stats = seg.get("stats") or fig.get("stats") or {}
        class_fracs = stats.get("named_class_fractions") or {}
        out.append(
            {
                "ref": ref,
                "label": label,
                "roi_id": rid,
                "roi_index": fig.get("roi_index") or selected.get("roi_index") or i,
                "slide_id": fig.get("slide_id") or selected.get("slide_id") or "",
                "score": fig.get("score") or selected.get("score"),
                "role": fig.get("evidence_role") or selected.get("evidence_role") or selected.get("role") or "",
                "is_zoom": bool(fig.get("is_zoom") or selected.get("is_zoom") or "zoom" in str(rid).lower()),
                "zoom_parent": fig.get("zoom_parent_roi_id") or selected.get("zoom_parent_roi_id") or "",
                "image": rel_src(image_path, html_path),
                "overlay": rel_src(overlay_path, html_path),
                "composite": rel_src(composite_path, html_path),
                "mask": rel_src(mask_path, html_path),
                "stats": stats,
                "class_fracs": class_fracs,
                "qc": fig.get("seg_qc") or selected.get("seg_qc") or {},
            }
        )
    return out


def report_sections(text: str) -> list[tuple[str, str]]:
    text = text.strip()
    if not text:
        return []
    headings = [
        "Specimen/Site",
        "Diagnosis",
        "Tumor/Lesion Description",
        "Key Microscopic Findings",
        "Invasion/Necrosis/Margins",
        "Immunohistochemistry/Ancillary Studies",
        "Stage/Grade/Prognostic Features",
        "Final Summary",
    ]
    pattern = re.compile(rf"^({'|'.join(re.escape(h) for h in headings)})\s*:\s*", re.M)
    matches = list(pattern.finditer(text))
    if not matches:
        return [("Generated Report", text)]
    sections = []
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections.append((m.group(1), text[start:end].strip()))
    return sections


def inject_report_refs(text: str) -> str:
    safe = escape(text)

    def repl(m: re.Match[str]) -> str:
        n = m.group(1)
        return chip(f"FIG-{n}", f"Figure {n}", "fig")

    safe = re.sub(r"\[Figure\s+(\d+)\]", repl, safe)
    return safe.replace("\n", "<br>")


def figure_ref_for_evidence(evidence_id: Any, figures: list[dict[str, Any]]) -> str | None:
    ev = str(evidence_id or "")
    for fig in figures:
        if ev == fig["roi_id"] or ev.endswith(fig["roi_id"]) or fig["roi_id"].endswith(ev):
            return fig["ref"]
    return None


def build_inspector(payload: dict[str, Any], figures: list[dict[str, Any]], html_path: Path) -> dict[str, Any]:
    files = payload.get("_files", {})
    inspector: dict[str, Any] = {}
    overview_rows = payload.get("wsi_overviews") or []
    if not overview_rows:
        overview_rows = [
            {"annotated_path": str(p), "slide_id": f"WSI {i}", "slide_path": str(p)}
            for i, p in enumerate(sorted(Path(payload["_case_dir"]).glob("wsi_thumbnail_*_annotated.jpg")), start=1)
        ]
    for idx, row in enumerate(overview_rows, start=1):
        image = rel_src(row.get("annotated_path") or row.get("thumbnail_path") or row.get("path"), html_path)
        related_figures = [
            {"ref": f"FIG-{int(n)}", "label": f"Figure {int(n)}"}
            for n in (row.get("figure_numbers") or [])
            if str(n).isdigit()
        ]
        inspector[f"WSI-{idx}"] = {
            "type": "WSI Overview",
            "title": f"Slide overview {idx}",
            "meta": row.get("slide_id") or row.get("slide_path") or "",
            "image": image,
            "summary": "Whole-slide overview with selected ROI boxes when available.",
            "details": row,
            "related": related_figures,
        }
    for fig in figures:
        fracs = fig.get("class_fracs") or {}
        top_fracs = []
        for name, obj in fracs.items():
            if isinstance(obj, dict):
                top_fracs.append((name, float(obj.get("fraction") or 0)))
        top_fracs.sort(key=lambda x: x[1], reverse=True)
        wsi_related = []
        for wsi_idx, row in enumerate(overview_rows, start=1):
            fig_numbers = {str(n) for n in (row.get("figure_numbers") or [])}
            if str(fig.get("roi_index")) in fig_numbers:
                wsi_related.append({"ref": f"WSI-{wsi_idx}", "label": f"Slide {wsi_idx}"})
        inspector[fig["ref"]] = {
            "type": "Visual Evidence",
            "title": f'{fig["label"]}: ROI {fig.get("roi_index")}',
            "meta": f'{fig.get("role") or "roi"} | {fig.get("slide_id") or "slide unknown"}',
            "he_image": fig.get("image"),
            "ov_image": fig.get("overlay"),
            "image": fig.get("composite") or fig.get("overlay") or fig.get("image"),
            "summary": "Zoom ROI" if fig.get("is_zoom") else "Selected tumor evidence ROI",
            "details": {
                "roi_id": fig.get("roi_id"),
                "score": fig.get("score"),
                "zoom_parent": fig.get("zoom_parent"),
                "top_segmentation_classes": [f"{n}: {v*100:.1f}%" for n, v in top_fracs[:5]],
                "image_path": fig.get("image"),
                "overlay_path": fig.get("overlay"),
                "mask_path": fig.get("mask"),
            },
            "related": wsi_related,
        }
        # Sub-refs: separate click targets for H&E and overlay images
        he_ref = f"{fig['ref']}-he"
        ov_ref = f"{fig['ref']}-ov"
        inspector[he_ref] = {
            "type": "ROI — H&E",
            "title": f'{fig["label"]}: Original H&E',
            "meta": f'{fig.get("role") or "roi"} | {fig.get("slide_id") or "slide unknown"}',
            "image": fig.get("image"),
            "summary": "Original H&E-stained ROI patch",
            "details": {"roi_id": fig.get("roi_id"), "image_path": fig.get("image")},
            "related": wsi_related,
        }
        inspector[ov_ref] = {
            "type": "ROI — Segmentation Overlay",
            "title": f'{fig["label"]}: D1 Segmentation Overlay',
            "meta": f'{fig.get("role") or "roi"} | {fig.get("slide_id") or "slide unknown"}',
            "image": fig.get("overlay"),
            "summary": "Color-coded tissue class overlay for this ROI",
            "details": {
                "roi_id": fig.get("roi_id"),
                "overlay_path": fig.get("overlay"),
                "top_segmentation_classes": [f"{n}: {v*100:.1f}%" for n, v in top_fracs[:5]],
            },
            "related": wsi_related,
        }
        inspector[f'ROI-{fig.get("roi_index")}'] = inspector[fig["ref"]]

    claims = (files.get("claim_verification") or {}).get("claims") or []
    for c in claims:
        cid = c.get("claim_id") or f"C{len(inspector)}"
        related = []
        for ev in c.get("evidence") or []:
            fref, label = evidence_ref_label(ev, figures)
            if fref:
                related.append({"ref": fref, "label": label})
        inspector[cid] = {
            "type": "Verified Claim",
            "title": f'{cid}: {c.get("field") or "claim"}',
            "meta": f'status: {c.get("status") or "unknown"}',
            "summary": c.get("claim") or "",
            "details": {"reason": c.get("reason"), "evidence": c.get("evidence") or []},
            "related": related,
        }

    inspector["SITE"] = {
        "type": "Site Context",
        "title": (files.get("site_prediction") or {}).get("predicted_site")
        or (files.get("site_prediction") or {}).get("prediction")
        or "Site context",
        "meta": (files.get("site_prediction") or {}).get("source") or "known/input site metadata",
        "summary": "This site is provided as case context, not independently inferred from ROI evidence in known-site runs.",
        "details": files.get("site_prediction") or {},
    }
    inspector["DX"] = {
        "type": "Diagnosis Model",
        "title": clean_diagnosis_label((files.get("diagnosis_prediction") or {}).get("primary_diagnosis"))[0],
        "meta": f"confidence: {(files.get('diagnosis_prediction') or {}).get('confidence') or 'n/a'}",
        "summary": (files.get("diagnosis_prediction") or {}).get("not_further_classifiable_reason")
        or (files.get("diagnosis_prediction") or {}).get("reason")
        or "",
        "details": files.get("diagnosis_prediction") or {},
    }
    inspector["SUBTYPE"] = {
        "type": "Subtype Sufficiency",
        "title": (payload.get("subtype_sufficiency") or {}).get("recommended_primary_diagnosis")
        or "Subtype sufficiency",
        "meta": f"level: {(payload.get('subtype_sufficiency') or {}).get('best_supported_level') or 'n/a'}",
        "summary": (payload.get("subtype_sufficiency") or {}).get("reason") or "",
        "details": payload.get("subtype_sufficiency") or {},
    }
    inspector["BROAD"] = {
        "type": "Broad Hypotheses",
        "title": (payload.get("broad_hypotheses") or {}).get("best_broad_type") or "Broad hypotheses",
        "meta": f"confidence: {(payload.get('broad_hypotheses') or {}).get('confidence') or 'n/a'}",
        "summary": (payload.get("broad_hypotheses") or {}).get("reason") or "",
        "details": payload.get("broad_hypotheses") or {},
    }
    inspector["SUFF"] = {
        "type": "Evidence Sufficiency",
        "title": (files.get("evidence_sufficiency") or {}).get("recommended_next_step") or "Evidence sufficiency",
        "meta": f"sufficient: {(files.get('evidence_sufficiency') or {}).get('sufficient_for_diagnosis')}",
        "summary": (files.get("evidence_sufficiency") or {}).get("reason") or "",
        "details": files.get("evidence_sufficiency") or {},
    }
    inspector["SEG"] = {
        "type": "Segmentation Evidence",
        "title": "Segmentation summary",
        "meta": f"{len((files.get('segmentation') or {}).get('roi_results') or [])} ROI mask(s)",
        "summary": "Segmentation overlays and tissue-class fractions for selected ROI evidence.",
        "details": files.get("segmentation") or {},
    }

    med = files.get("medgemma") or {}
    for idx, rnd in enumerate(med.get("rounds") or [], start=1):
        ref = f"MORPH-R{idx}"
        text = rnd.get("description") or rnd.get("text") or rnd.get("content") or rnd
        inspector[ref] = {
            "type": "Morphology Evidence",
            "title": f"MedGemma morphology round {idx}",
            "meta": rnd.get("stage") or rnd.get("name") or "",
            "summary": "\n".join(f"{k}: {v}" for k, v in parse_medgemma_sections(text).items()),
            "details": rnd,
        }

    kg = files.get("knowledge_context") or {}
    # Phase 2 format: per-hypothesis anchored knowledge
    per_hyp = kg.get("per_hypothesis")
    if isinstance(per_hyp, list) and per_hyp:
        for idx, hyp in enumerate(per_hyp, start=1):
            if not isinstance(hyp, dict):
                continue
            dg = hyp.get("disease_graph") or {}
            caps = hyp.get("morphology_captions") or []
            cap_snippets = [c.get("snippet", "") for c in caps[:3] if c.get("snippet")]
            ref = f"KEEP-{idx}"
            inspector[ref] = {
                "type": "Knowledge Context (Phase 2)",
                "title": f'{hyp.get("diagnosis", "?")} ({hyp.get("confidence", "?")})',
                "meta": f'doid: {dg.get("doid", "n/a")} | captions: {len(caps)}',
                "summary": dg.get("definition") or "",
                "details": {
                    "diagnosis": hyp.get("diagnosis"),
                    "confidence": hyp.get("confidence"),
                    "disease_graph": dg,
                    "morphology_captions": caps,
                    "cross_site_captions": [c for c in caps if c.get("cross_site")],
                },
            }
            # Sub-refs for each caption snippet with morphology detail
            for ci, cap in enumerate(caps[:5], start=1):
                snippet = cap.get("snippet") or ""
                if not snippet:
                    continue
                tag = " [cross-site]" if cap.get("cross_site") else ""
                inspector[f"{ref}-cap{ci}"] = {
                    "type": "KEEP Caption",
                    "title": f'{hyp.get("diagnosis", "?")} — caption {ci}{tag}',
                    "meta": f'anatomy: {cap.get("anatomy") or "unspecified"}',
                    "summary": snippet,
                    "details": cap,
                }
    else:
        # Phase 1 format (legacy): site-driven broad search
        hits = (kg.get("site_subtype_candidates") or []) + (kg.get("disease_hits") or [])
        for idx, hit in enumerate(hits[:30], start=1):
            ref = f"KEEP-{idx}"
            inspector[ref] = {
                "type": "Knowledge Context (Phase 1)",
                "title": hit.get("name") or hit.get("label") or ref,
                "meta": f'score: {hit.get("score", "n/a")}',
                "summary": hit.get("definition") or hit.get("caption") or "",
                "details": hit,
            }

    for idx, stage in enumerate((files.get("progress_trace") or {}).get("stages") or [], start=1):
        ref = f"STEP-{idx}"
        stage_path = stage.get("path")
        data = {}
        if stage_path:
            p = Path(stage_path)
            if not p.is_absolute():
                p = PROJECT_ROOT / stage_path
            data = load_json(p, {}) or {}
        summary_bits = []
        if data:
            action = first_existing(data.get("action"), data.get("recommended_next_step"), data.get("reinspection_action"))
            primary = first_existing(data.get("primary_diagnosis"), data.get("diagnosis"), data.get("predicted_site"))
            sufficient = first_existing(data.get("sufficient_for_diagnosis"), data.get("sufficient_for_report"))
            if primary is not None:
                summary_bits.append(f"Result: {primary}")
            if action is not None:
                summary_bits.append(f"Action: {action}")
            if sufficient is not None:
                summary_bits.append(f"Sufficient: {sufficient}")
        inspector[ref] = {
            "type": "Decision Step",
            "title": stage.get("stage") or ref,
            "meta": stage_path or "",
            "summary": "\n".join(summary_bits) or short(json.dumps(data.get("data", data), ensure_ascii=False), 800),
            "details": data.get("data", data),
        }
    return inspector


def has_evaluation(payload: dict[str, Any]) -> bool:
    """Distinguish saved evaluation scores (including zero) from empty metadata."""
    files = payload.get("_files") or {}
    ev = files.get("eval") or {}
    hierarchy = files.get("diagnostic_hierarchy") or ev.get("diagnostic_hierarchy") or {}
    for record in (ev, hierarchy, ev.get("primary_metrics") or {}):
        if any(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for key, value in record.items()
            if key.endswith("_score") or key in {
                "image_only_core_score", "pathology_report_score", "full_report_quality_score",
            }
        ):
            return True
        for row in (record.get("field_scores") or {}).values():
            if not isinstance(row, dict) or row.get("status") == "not_evaluated":
                continue
            score = row.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                return True
    return False


def render_final_report(payload: dict[str, Any], figures: list[dict[str, Any]]) -> str:
    files = payload.get("_files", {})
    report = payload.get("_reports", {}).get("rich") or payload.get("_reports", {}).get("long") or ""
    diag = files.get("diagnosis_prediction") or {}
    suff = files.get("evidence_sufficiency") or {}
    site = files.get("site_prediction") or payload.get("site_prediction") or {}
    claims = (files.get("claim_verification") or {}).get("claims") or []
    stages = (files.get("progress_trace") or {}).get("stages") or []
    sections = report_sections(report)
    dx_label, dx_limit = clean_diagnosis_label(diag.get("primary_diagnosis"))
    body_rows = []
    claims_by_field = {
        str(c.get("field") or "").lower(): c
        for c in claims
        if c.get("field")
    }

    def claim_for_section(section_name: str) -> dict[str, Any] | None:
        name_l = section_name.lower()
        field_order: list[str]
        if "specimen" in name_l or "site" in name_l:
            field_order = ["site"]
        elif name_l == "diagnosis":
            field_order = ["primary_diagnosis", "subtype", "broad_type"]
        elif "tumor" in name_l or "lesion" in name_l:
            field_order = ["tumor_architecture", "primary_diagnosis", "subtype"]
        elif "microscopic" in name_l or "finding" in name_l:
            field_order = ["cytology", "tumor_architecture", "stroma_inflammation"]
        elif "invasion" in name_l or "necrosis" in name_l or "margin" in name_l:
            field_order = ["invasion", "necrosis"]
        elif "immun" in name_l or "ancillary" in name_l or "molecular" in name_l:
            field_order = ["ihc_molecular"]
        elif "stage" in name_l or "grade" in name_l or "prognostic" in name_l:
            field_order = ["grade", "malignancy"]
        elif "summary" in name_l:
            field_order = ["primary_diagnosis", "subtype", "malignancy"]
        else:
            field_order = []
        return next((claims_by_field[f] for f in field_order if f in claims_by_field), None)

    def report_evidence_tags(content: str, matching_claim: dict[str, Any] | None) -> list[str]:
        refs: list[str] = []
        for n in re.findall(r"\[Figure\s+(\d+)\]", content):
            refs.append(f"FIG-{n}")
        if matching_claim:
            refs.extend(str(ev) for ev in (matching_claim.get("evidence") or []))
            cid = matching_claim.get("claim_id")
            if cid:
                refs.append(str(cid))
        deduped: list[str] = []
        seen: set[str] = set()
        for ref in refs:
            ref_s, label = evidence_ref_label(ref, figures)
            if not ref_s and str(ref).startswith("C"):
                ref_s, label = str(ref), f"Claim {ref}"
            if not ref_s:
                continue
            if ref_s in seen:
                continue
            seen.add(ref_s)
            deduped.append(evidence_chip(ref_s, label))
        return deduped

    for name, content in sections:
        display_content = clean_report_text(content) if name.lower() == "diagnosis" else content
        matching_claim = claim_for_section(name)
        evidence_tags = report_evidence_tags(content, matching_claim)
        evidence_html = "".join(evidence_tags) or '<span class="muted">No linked evidence</span>'
        body_rows.append(
            '<section class="report-row">'
            f'<div class="report-row-title">{escape(name)}</div>'
            f'<div class="report-row-content">{inject_report_refs(display_content)}</div>'
            f'<div class="report-row-tags">{evidence_html}</div>'
            '</section>'
        )
    body = "".join(body_rows)

    status_counts: dict[str, int] = {}
    for c in claims:
        status = str(c.get("status") or "unknown").lower()
        status_counts[status] = status_counts.get(status, 0) + 1

    verification_summary = f"""
    <div class="verification-footer">
      <div class="side-title">Verification Summary <button class="link-btn" data-tab-target="claims" type="button">View details</button></div>
      <div class="verify-grid">
        <div class="verify supported"><b>{status_counts.get("supported", 0)}</b><span>Supported</span></div>
        <div class="verify weak"><b>{status_counts.get("weak", 0)}</b><span>Weak</span></div>
        <div class="verify unsupported"><b>{status_counts.get("unsupported", 0)}</b><span>Unsupported</span></div>
        <div class="verify neutral"><b>{status_counts.get("not assessable", 0) + status_counts.get("not_assessable", 0)}</b><span>Not assessable</span></div>
      </div>
    </div>
    """

    related_visuals = "".join(
        f'<button class="related-thumb" data-ref="{escape(fig["ref"])}" type="button">'
        f'<img src="{escape(fig.get("composite") or fig.get("overlay") or fig.get("image") or "")}" alt=""></button>'
        for fig in figures[:4]
        if fig.get("composite") or fig.get("overlay") or fig.get("image")
    )

    # Hierarchical diagnosis tiers (broad / subtype / malignancy / grade).
    # Rendered only when the prediction carries these structured fields.
    tier_items = []
    for tier_label, tier_key in (
        ("Broad type", "broad_type"),
        ("Subtype", "subtype"),
        ("Malignancy", "malignancy"),
        ("Grade", "grade"),
    ):
        tier_val = str(diag.get(tier_key) or "").strip()
        if tier_val:
            tier_items.append(
                f'<div class="tier-item"><span class="tier-key">{escape(tier_label)}</span>'
                f'<span class="tier-val">{escape(tier_val)}</span></div>'
            )
    tier_strip = (
        f'<div class="tier-strip">{"".join(tier_items)}</div>' if tier_items else ""
    )
    eval_fields = ((files.get("diagnostic_hierarchy") or {}).get("field_scores") or {})
    eval_order = [
        "cancer_type",
        "histologic_subtype",
        "malignancy_status",
        "invasive_status",
        "grade",
        "necrosis",
    ]
    eval_rows = []
    for field in eval_order:
        row = eval_fields.get(field) or {}
        if not isinstance(row, dict) or not row:
            continue
        status = str(row.get("status") or "").lower()
        cls = "supported" if status in {"match", "exact_match", "strong_match"} else "warn" if status in {"partial", "partial_match", "weak", "not_evaluated"} else "unsupported" if status in {"mismatch", "missing"} else "neutral"
        eval_rows.append(
            f'<tr><td>{escape(field.replace("_", " ").title())}</td>'
            f'<td>{badge(status or "n/a", cls)}</td>'
            f'<td>{escape(short(row.get("prediction"), 70))}</td>'
            f'<td>{escape(short(row.get("reference"), 70))}</td></tr>'
        )
    eval_block = ""
    if eval_rows and has_evaluation(payload):
        eval_block = f"""
        <div class="report-eval-strip">
          <div class="section-title">Evaluation Extraction</div>
          <div class="table-scroll"><table class="claim-table compact">
            <thead><tr><th>Field</th><th>Match</th><th>Prediction</th><th>Reference</th></tr></thead>
            <tbody>{''.join(eval_rows)}</tbody>
          </table></div>
        </div>
        """

    return f"""
<div class="result-layout">
  <div class="result-main">
    <div class="diagnosis-banner">
      <div class="diagnosis-check">✓</div>
      <div>
        <div class="card-kicker">Final Diagnosis</div>
        <h2>{escape(dx_label)}</h2>
        {f'<div class="diagnosis-limitation">{escape(dx_limit)}</div>' if dx_limit else ''}
      </div>
      <div class="confidence-box">
        <small>Confidence</small>
        <strong>{escape(str(diag.get("confidence") or "n/a"))}</strong>
      </div>
      <p>{escape(suff.get("reason") or diag.get("not_further_classifiable_reason") or "Report generated from visual evidence, morphology summaries, segmentation statistics, and knowledge context.")}</p>
      {tier_strip}
      <div class="banner-tags">
        {badge(site.get("predicted_site") or site.get("prediction") or "site not available", "site")}
        {badge("sufficient" if suff.get("sufficient_for_report") else "limited evidence", "warn")}
        {badge(f"{len(claims)} verified claims", "supported")}
      </div>
    </div>
    {eval_block}
    <div class="card">
      <div class="section-title with-icon">Final Report</div>
      {evidence_legend()}
      {body or '<p class="muted">No generated report found.</p>'}
      {verification_summary}
    </div>
  </div>
</div>
<div class="related-section">
  <div class="section-title">Related Evidence</div>
  <div class="related-strip">
    <article class="related-card" data-tab-jump="visual"><div class="related-icon visual">V</div><div><h3>Visual Evidence</h3><p>Review ROI thumbnails, WSI overviews, zooms, and segmentation overlays.</p><button class="open-tab" data-tab-target="visual">Open</button></div><div class="related-images">{related_visuals}</div></article>
    <article class="related-card" data-tab-jump="visual"><div class="related-icon morph">M</div><div><h3>Morphology + Knowledge Evidence</h3><p>Inspect MedGemma summaries and KEEP support in the unified evidence page.</p><button class="open-tab" data-tab-target="visual">Open</button></div></article>
    <article class="related-card" data-tab-jump="claims"><div class="related-icon verify">C</div><div><h3>Claim Support</h3><p>Trace each report claim to visual, morphology, and knowledge evidence.</p><button class="open-tab" data-tab-target="claims">Open</button></div></article>
  </div>
</div>
"""


def render_evidence(payload: dict[str, Any], figures: list[dict[str, Any]], html_path: Path) -> str:
    kg = payload.get("_files", {}).get("knowledge_context") or {}
    med = payload.get("_files", {}).get("medgemma") or {}
    overview_cards = []
    for idx, row in enumerate(payload.get("wsi_overviews") or [], start=1):
        img = rel_src(row.get("annotated_path") or row.get("thumbnail_path") or row.get("path"), html_path)
        img_html = f'<img src="{escape(img)}" alt="">' if img else ""
        fig_nums = row.get("figure_numbers") or []
        fig_legend = "".join(figure_chip(n) for n in fig_nums)
        fallback_legend = '<span class="muted">No linked ROI figures</span>'
        overview_cards.append(
            f'<div class="overview-card" data-ref="WSI-{idx}">{img_html}'
            f'<div class="overview-title">Slide {idx}</div><div class="muted">{escape(short(row.get("slide_id") or row.get("slide_path"), 90))}</div>'
            f'<div class="wsi-legend">{fig_legend or fallback_legend}</div></div>'
        )
    if not overview_cards:
        for idx, p in enumerate(sorted(Path(payload["_case_dir"]).glob("wsi_thumbnail_*_annotated.jpg")), start=1):
            overview_cards.append(
                f'<div class="overview-card" data-ref="WSI-{idx}"><img src="{escape(rel_src(p, html_path))}" alt="">'
                f'<div class="overview-title">Slide overview {idx}</div></div>'
            )

    fig_cards = []
    for fig in figures:
        raw_img = fig.get("image") or ""
        # Use overlay (not composite) for display — consistent across zoom/non-zoom
        seg_img = fig.get("overlay") or ""
        fracs = fig.get("class_fracs") or {}
        bars = []
        for name, obj in sorted(fracs.items(), key=lambda kv: (kv[1].get("fraction") if isinstance(kv[1], dict) else 0), reverse=True)[:5]:
            frac = float(obj.get("fraction") or 0) if isinstance(obj, dict) else 0
            bars.append(
                f'<div class="seg-row"><span>{escape(str(name))}</span><div class="bar"><i style="width:{max(0,min(100,frac*100)):.1f}%"></i></div><b>{frac*100:.1f}%</b></div>'
            )
        zoom = badge("zoom", "zoom") if fig.get("is_zoom") else ""
        ref = fig["ref"]
        he_ref = f"{ref}-he"
        ov_ref = f"{ref}-ov"
        raw_img_html = (
            f'<button class="img-btn" data-ref="{escape(he_ref)}" type="button">'
            f'<img src="{escape(raw_img)}" alt="H&E ROI"></button>'
        ) if raw_img else '<div class="empty-img">No original ROI</div>'
        seg_img_html = (
            f'<button class="img-btn" data-ref="{escape(ov_ref)}" type="button">'
            f'<img src="{escape(seg_img)}" alt="Segmentation overlay"></button>'
        ) if seg_img else '<div class="empty-img">No paired segmentation generated</div>'
        score_html = ""
        if isinstance(fig.get("score"), (int, float)):
            score_html = badge(f"score {fig.get('score'):.3f}", "score")
        fig_num = str(fig["ref"]).replace("FIG-", "")
        title_dot = f'<span class="fig-dot" style="background:{escape(roi_color(fig_num))}">{escape(fig_num)}</span>'
        fig_cards.append(
            f'<article class="roi-card" data-ref="{escape(fig["ref"])}">'
            f'<div class="roi-pair"><div><span>H&E ROI</span>{raw_img_html}</div><div><span>Overlay</span>{seg_img_html}</div></div>'
            f'<div class="roi-body"><div class="roi-title">{title_dot}{escape(fig["label"])} {zoom}</div>'
            f'<div class="muted">{escape(short(fig.get("roi_id"), 110))}</div>'
            f'<div class="roi-meta">{badge(fig.get("role") or "roi", "neutral")} {score_html}</div>'
            f'<div class="seg-bars">{"".join(bars)}</div></div></article>'
        )
    med_cards = []
    for i, rnd in enumerate(med.get("rounds") or [], start=1):
        text = rnd.get("description") or rnd.get("text") or rnd.get("content") or json.dumps(rnd, ensure_ascii=False)
        parsed = parse_medgemma_sections(text)
        section_bits = "".join(
            f'<div class="morph-field"><b>{escape(k)}</b><p>{escape(v)}</p></div>'
            for k, v in list(parsed.items())[:4]
        )
        roi_ids = rnd.get("input_roi_ids") or []
        med_cards.append(
            f'<article class="morph-card" data-ref="MORPH-R{i}">'
            f'<div class="card-kicker">Morphology Evidence</div><h3>MedGemma round {i}</h3>'
            f'<div class="muted">Input ROIs: {escape(str(len(roi_ids)))} · {escape(short(", ".join(roi_ids), 120))}</div>'
            f'{section_bits}</article>'
        )

    # KEEP: Phase 2 (per_hypothesis) or Phase 1 (legacy)
    per_hyp = kg.get("per_hypothesis")
    kg_cards = []
    if isinstance(per_hyp, list) and per_hyp:
        for i, hyp in enumerate(per_hyp, start=1):
            if not isinstance(hyp, dict):
                continue
            dg = hyp.get("disease_graph") or {}
            caps = hyp.get("morphology_captions") or []
            cap_text = " | ".join(c.get("snippet", "")[:120] for c in caps[:2] if c.get("snippet"))
            kg_cards.append(
                f'<article class="knowledge-card" data-ref="KEEP-{i}"><div class="card-kicker">Knowledge Evidence (Phase 2)</div>'
                f'<h3>{escape(hyp.get("diagnosis", "?"))} ({escape(hyp.get("confidence", "?"))})</h3>'
                f'<p>{escape(short(dg.get("definition") or cap_text or "No definition.", 300))}</p>'
                f'<div class="kg-meta">{badge("doid: " + str(dg.get("doid", "n/a")), "score")} {badge("captions: " + str(len(caps)), "site")}</div></article>'
            )
    else:
        hits = (kg.get("site_subtype_candidates") or []) + (kg.get("disease_hits") or [])
        for i, hit in enumerate(hits[:10], start=1):
            matched = ", ".join(str(x) for x in (hit.get("matched_terms") or hit.get("site_matched_terms") or [])[:8])
            score_badge = badge(f"score {hit.get('score', 'n/a')}", "score")
            matched_badge = badge("site match: " + matched, "site") if matched else ""
            kg_cards.append(
                f'<article class="knowledge-card" data-ref="KEEP-{i}"><div class="card-kicker">Knowledge Evidence</div>'
                f'<h3>{escape(hit.get("name") or hit.get("label") or f"KEEP hit {i}")}</h3>'
                f'<p>{escape(short(hit.get("definition") or hit.get("caption") or "No definition saved.", 300))}</p>'
                f'<div class="kg-meta">{score_badge} {matched_badge}</div></article>'
            )

    return f"""
<div class="card">
  <div class="section-title">WSI Overviews</div>
  <div class="overview-grid">{''.join(overview_cards) or '<p class="muted">No WSI thumbnails found.</p>'}</div>
</div>
<div class="card">
  <div class="section-title">ROI + Segmentation Evidence</div>
  <p class="section-note">Each card opens in the inspector with source path, segmentation fractions, zoom lineage, and related evidence.</p>
  <div class="roi-grid">{''.join(fig_cards) or '<p class="muted">No figures found.</p>'}</div>
</div>
<div class="card">
  <div class="section-title">Morphology Evidence</div>
  <p class="section-note">MedGemma rounds are shown as morphology evidence, separated from final report generation.</p>
  <div class="morph-grid">{''.join(med_cards) or '<p class="muted">No morphology rounds found.</p>'}</div>
</div>
<div class="card">
  <div class="section-title">Knowledge Evidence</div>
  <p class="section-note">KEEP hits are supporting or differential knowledge, not direct visual observations.</p>
  <div class="knowledge-grid">{''.join(kg_cards) or '<p class="muted">No KEEP hits found.</p>'}</div>
</div>
"""


def render_decisions(payload: dict[str, Any]) -> str:
    stages = (payload.get("_files", {}).get("progress_trace") or {}).get("stages") or []
    rows = []
    labels = {
        "01_roi_selection": "Seed ROI Evidence",
        "02_site_prediction": "Site Context",
        "04_segmentation_qc": "Segmentation QC",
        "05_medgemma_morphology": "Morphology Summary",
        "06_preliminary_diagnosis_sufficiency": "Preliminary Diagnosis + Sufficiency",
        "07_reinspection": "Reinspection",
        "08_final_diagnosis": "Final Diagnosis Planning",
        "09_slots_claim_verification": "Structured Slots + Claim Verification",
    }
    for idx, st in enumerate(stages, start=1):
        name = st.get("stage") or f"step-{idx}"
        display = next((v for k, v in labels.items() if name.startswith(k)), name)
        data = checkpoint_data(st)
        facts = []
        chips = []
        if name.startswith("01_roi_selection"):
            selected = data.get("selected_rois") or data.get("selected") or data.get("selected_roi_indices") or []
            facts.append(("Output", f"{len(selected)} seed ROI(s) selected"))
            facts.append(("Policy", data.get("selection_strategy") or data.get("reason") or "foundation ROI seeding"))
            chips.extend(chip(f"FIG-{i}", f"Figure {i}", "fig") for i in range(1, min(len(selected), 5) + 1))
        elif name.startswith("02_site_prediction"):
            facts.append(("Output", data.get("predicted_site") or data.get("prediction") or "known site context"))
            facts.append(("Source", data.get("source") or "metadata / site context"))
        elif name.startswith("04_segmentation_qc"):
            qc = data.get("seg_qc") or data.get("qc") or {}
            facts.append(("Output", f"{len((payload.get('_files', {}).get('segmentation') or {}).get('roi_results') or [])} ROI masks available"))
            if isinstance(qc, dict):
                facts.append(("QC", short(qc.get("summary") or qc.get("reason") or json.dumps(qc, ensure_ascii=False), 160)))
        elif name.startswith("05_medgemma_morphology"):
            med = payload.get("_files", {}).get("medgemma") or {}
            facts.append(("Output", f"{len(med.get('rounds') or [])} morphology round(s) saved"))
            chips.append(chip("MORPH-R1", "Morphology R1", "morph"))
        elif name.startswith("06_preliminary_diagnosis_sufficiency"):
            diag = data.get("diagnosis_prediction") or data.get("diagnosis") or {}
            suff = data.get("evidence_sufficiency") or data.get("sufficiency") or data
            prelim = clean_diagnosis_label(diag.get("primary_diagnosis"))[0] if isinstance(diag, dict) else short(diag, 120)
            facts.append(("Preliminary diagnosis", prelim))
            facts.append(("Sufficient", suff.get("sufficient_for_diagnosis") if isinstance(suff, dict) else "unknown"))
            facts.append(("Next action", suff.get("recommended_next_step") or suff.get("reinspection_action") if isinstance(suff, dict) else "unknown"))
        elif name.startswith("07_reinspection"):
            plan = data.get("reinspection_plan") or data.get("plan") or data
            facts.append(("Action", plan.get("action") or plan.get("reinspection_action") or "reinspection"))
            facts.append(("Reason", plan.get("reason") or plan.get("reinspection_reason") or data.get("reason") or "missing evidence"))
            added = data.get("added_rois") or data.get("new_rois") or []
            if added:
                facts.append(("Added", f"{len(added)} ROI(s)"))
        elif name.startswith("08_final_diagnosis"):
            diag = data.get("diagnosis_prediction") or data.get("diagnosis") or data
            final_dx = clean_diagnosis_label(diag.get("primary_diagnosis"))[0] if isinstance(diag, dict) else short(diag, 140)
            facts.append(("Final planning output", final_dx))
            if isinstance(diag, dict):
                facts.append(("Differential", ", ".join(str(x) for x in as_list(diag.get("differential") or diag.get("differential_diagnoses"))[:4])))
        elif name.startswith("09_slots_claim_verification"):
            cv = data.get("claim_verification") or payload.get("_files", {}).get("claim_verification") or {}
            claims = cv.get("claims") or []
            facts.append(("Output", f"{len(claims)} verified claim(s)"))
            chips.append(chip("C001", "Open claims", "supported"))

        fact_html = "".join(
            f'<div class="decision-fact"><span>{escape(str(k))}</span><b>{escape(short(v, 160))}</b></div>'
            for k, v in facts
            if v not in (None, "", [])
        )
        rows.append(
            f'<article class="decision-node" data-ref="STEP-{idx}">'
            f'<div class="node-index">{idx}</div><div><h3>{escape(display)}</h3>'
            f'<p>{escape(name)}</p>{fact_html}<div class="decision-links">{chip(f"STEP-{idx}", "inspect raw step", "step")} {"".join(chips)}</div></div></article>'
        )
    return f"""
<div class="card">
  <div class="section-title">Decision Process</div>
  <p class="section-note">A directory-style timeline. Click a step to inspect the exact checkpoint payload without leaving the page.</p>
  <div class="decision-list">{''.join(rows) or '<p class="muted">No progress trace found.</p>'}</div>
</div>
"""


def render_claims(payload: dict[str, Any], figures: list[dict[str, Any]]) -> str:
    claims = (payload.get("_files", {}).get("claim_verification") or {}).get("claims") or []
    rows = []
    for c in claims:
        cid = c.get("claim_id") or ""
        status = str(c.get("status") or "unknown").lower()
        ev_chips = []
        for ev in c.get("evidence") or []:
            ref, label = evidence_ref_label(ev, figures)
            if ref:
                ev_chips.append(evidence_chip(ref, label))
            else:
                ev_chips.append(f'<span class="tiny-chip">{escape(label)}</span>')
        rows.append(
            f'<tr class="claim-{escape(status)}"><td>{chip(cid, cid, status)}</td>'
            f'<td>{badge(status, status)}</td><td>{escape(c.get("field") or "")}</td>'
            f'<td>{escape(clean_report_text(c.get("claim")) if c.get("field") == "primary_diagnosis" else str(c.get("claim") or ""))}<div class="muted">{escape(short(clean_report_text(c.get("reason")) if c.get("field") == "primary_diagnosis" else c.get("reason"), 160))}</div></td>'
            f'<td><div class="ref-wrap">{"".join(ev_chips)}</div></td></tr>'
        )
    return f"""
<div class="card">
  <div class="section-title">Evidence Support Map</div>
  <p class="section-note">Claims should be supported, weak, not assessable, or removed. The evidence column links claims back to figures or non-image evidence.</p>
  {evidence_legend()}
  <div class="table-scroll"><table class="claim-table">
    <thead><tr><th>Claim</th><th>Status</th><th>Field</th><th>Statement</th><th>Evidence</th></tr></thead>
    <tbody>{''.join(rows) or '<tr><td colspan="5">No claims found.</td></tr>'}</tbody>
  </table></div>
</div>
"""


def render_knowledge(payload: dict[str, Any]) -> str:
    kg = payload.get("_files", {}).get("knowledge_context") or {}
    med = payload.get("_files", {}).get("medgemma") or {}
    med_cards = []
    for i, rnd in enumerate(med.get("rounds") or [], start=1):
        text = rnd.get("description") or rnd.get("text") or rnd.get("content") or json.dumps(rnd, ensure_ascii=False)
        med_cards.append(
            f'<article class="knowledge-card" data-ref="MORPH-R{i}"><div class="card-kicker">Morphology R{i}</div>'
            f'<h3>MedGemma round {i}</h3><p>{escape(short(text, 420))}</p></article>'
        )
    # KEEP: Phase 2 (per_hypothesis) or Phase 1 (legacy)
    per_hyp = kg.get("per_hypothesis")
    kg_cards = []
    if isinstance(per_hyp, list) and per_hyp:
        for i, hyp in enumerate(per_hyp, start=1):
            if not isinstance(hyp, dict):
                continue
            dg = hyp.get("disease_graph") or {}
            caps = hyp.get("morphology_captions") or []
            cap_text = " | ".join(c.get("snippet", "")[:120] for c in caps[:2] if c.get("snippet"))
            kg_cards.append(
                f'<article class="knowledge-card" data-ref="KEEP-{i}"><div class="card-kicker">KEEP Phase 2 — {i}</div>'
                f'<h3>{escape(hyp.get("diagnosis", "?"))} ({escape(hyp.get("confidence", "?"))})</h3>'
                f'<p>{escape(short(dg.get("definition") or cap_text or "", 320))}</p>'
                f'<div class="muted">doid: {escape(str(dg.get("doid", "n/a")))} | captions: {len(caps)}</div></article>'
            )
    else:
        hits = (kg.get("site_subtype_candidates") or []) + (kg.get("disease_hits") or [])
        for i, hit in enumerate(hits[:12], start=1):
            kg_cards.append(
                f'<article class="knowledge-card" data-ref="KEEP-{i}"><div class="card-kicker">KEEP {i}</div>'
                f'<h3>{escape(hit.get("name") or hit.get("label") or f"Hit {i}")}</h3>'
                f'<p>{escape(short(hit.get("definition") or hit.get("caption") or "", 320))}</p>'
                f'<div class="muted">score: {escape(str(hit.get("score", "n/a")))}</div></article>'
            )
    return f"""
<div class="page-grid">
  <div class="main-col">
    <div class="card">
      <div class="section-title">MedGemma Morphology Evidence</div>
      <div class="knowledge-grid">{''.join(med_cards) or '<p class="muted">No morphology rounds found.</p>'}</div>
    </div>
  </div>
  <aside class="side-col">
    <div class="card">
      <div class="section-title">KEEP Knowledge Context</div>
      <div class="knowledge-grid single">{''.join(kg_cards) or '<p class="muted">No KEEP hits found.</p>'}</div>
    </div>
  </aside>
</div>
"""


def render_audit(payload: dict[str, Any], figures: list[dict[str, Any]]) -> str:
    files = payload.get("_files", {})
    ev = files.get("eval") or {}
    hierarchy = files.get("diagnostic_hierarchy") or ev.get("diagnostic_hierarchy") or {}
    pm = ev.get("primary_metrics") or {}
    seg = files.get("segmentation") or {}
    metrics = "".join(
        kv(label, value)
        for label, value in [
            ("Primary score", pm.get("primary_score")),
            ("Image-only core", pm.get("image_only_core_score")),
            ("Pathology score", pm.get("pathology_report_score")),
            ("Full report score", pm.get("full_report_quality_score")),
            ("Diagnostic hierarchy", hierarchy.get("overall_score")),
            ("Selected figures", len(figures)),
            ("Seg max ROIs", seg.get("max_rois")),
        ]
    )
    field_rows = []
    for field, row in (hierarchy.get("field_scores") or {}).items():
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").lower()
        field_rows.append(
            f'<tr><td>{escape(str(field).replace("_", " ").title())}</td>'
            f'<td>{badge(status or "n/a", "supported" if status == "match" else "warn" if status in {"partial", "weak"} else "unsupported" if status == "mismatch" else "neutral")}</td>'
            f'<td>{escape(str(row.get("score", "n/a")))}</td>'
            f'<td>{escape(short(row.get("prediction"), 80))}</td>'
            f'<td>{escape(short(row.get("reference"), 80))}</td>'
            f'<td>{escape(short(row.get("reason"), 120))}</td></tr>'
        )
    return f"""
<div class="card">
  <div class="section-title">Diagnostic Hierarchy Scores</div>
  <p class="section-note">Field-level diagnosis evaluation. This is more informative than one aggregate score for debugging site/type/subtype errors.</p>
  <div class="table-scroll"><table class="claim-table">
    <thead><tr><th>Field</th><th>Status</th><th>Score</th><th>Prediction</th><th>Reference</th><th>Reason</th></tr></thead>
    <tbody>{''.join(field_rows) or '<tr><td colspan="6">No diagnostic hierarchy scores found.</td></tr>'}</tbody>
  </table></div>
</div>
<div class="page-grid">
  <div class="main-col"><div class="card"><div class="section-title">Run Metrics</div>{metrics}</div></div>
  <aside class="side-col"><div class="card"><div class="section-title">Artifacts</div>
    {kv("Case directory", payload.get("_case_dir"))}
    {kv("Reference report", ev.get("reference_report_path"))}
    {kv("KEEP root", (files.get("knowledge_context") or {}).get("keep_root"))}
  </div></aside>
</div>
"""


def metric_card(label: str, value: Any, sub: str = "", icon: str = "") -> str:
    sub_html = f'<div class="metric-sub">{escape(sub)}</div>' if sub else ""
    return (
        '<div class="metric-card">'
        f'<div class="metric-icon">{escape(icon or label[:1])}</div>'
        '<div>'
        f'<div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value">{escape(str(value if value not in (None, "") else "n/a"))}</div>'
        f'{sub_html}'
        '</div></div>'
    )


def render_html(payload: dict[str, Any], html_path: Path) -> str:
    files = payload.get("_files", {})
    show_evaluation = has_evaluation(payload)
    audit_tab = '<button class="tab" data-tab="audit">Audit</button>' if show_evaluation else ""
    audit_page = ""
    case_id = payload.get("case_id") or Path(payload.get("_case_dir", "")).name
    cancer_type = (files.get("artifact_index") or {}).get("cancer_type") or payload.get("cancer_type") or ""
    figures = figure_records(payload, html_path)
    if show_evaluation:
        audit_page = f'<div id="tab-audit" class="page">{render_audit(payload, figures)}</div>'
    inspector = build_inspector(payload, figures, html_path)
    diag = files.get("diagnosis_prediction") or {}
    site = files.get("site_prediction") or payload.get("site_prediction") or {}
    suff = files.get("evidence_sufficiency") or {}
    supported = sum(1 for c in (files.get("claim_verification") or {}).get("claims", []) if str(c.get("status")).lower() == "supported")
    all_claims = len((files.get("claim_verification") or {}).get("claims", []))
    slides = payload.get("slides_used") or (files.get("artifact_index") or {}).get("slides") or []
    seg = files.get("segmentation") or {}
    seg_results = seg.get("roi_results") or []
    kg = files.get("knowledge_context") or {}
    keep_count = len(kg.get("site_subtype_candidates") or []) + len(kg.get("disease_hits") or [])
    confidence_text = diag.get("confidence") or "n/a"
    dx_label, dx_limit = clean_diagnosis_label(diag.get("primary_diagnosis"))
    metric_strip = "".join(
        [
            metric_card("Case ID", case_id, "", "ID"),
            metric_card("Cancer Type", cancer_type or "n/a", "", "CT"),
            metric_card("Known Site", site.get("predicted_site") or "n/a", "metadata context", "S"),
            metric_card("Slides Used", len(slides), "", "SL"),
            metric_card("ROI Count", len(figures), "", "ROI"),
            metric_card("Segmentation", f"{len(seg_results)} / {seg.get('max_rois') or len(figures)}", "available masks", "SEG"),
            metric_card("KEEP", keep_count, "knowledge hits", "KG"),
            metric_card("Final Diagnosis", short(dx_label or "n/a", 42), dx_limit, "DX"),
            metric_card("Confidence", confidence_text, "", "CF"),
        ]
    )
    inspector_json = json.dumps(sanitise_internal_refs(inspector), ensure_ascii=False)
    wsi_pin_cards = []
    case_dir = Path(payload.get("_case_dir") or ".")
    for idx, p in enumerate(sorted(case_dir.glob("wsi_thumbnail_*_annotated.jpg"))[:4], start=1):
        wsi_pin_cards.append(
            f'<button class="pin-wsi" data-ref="WSI-{idx}" type="button">'
            f'<img src="{escape(rel_src(p, html_path))}" alt=""><span>WSI {idx}</span></button>'
        )
    pin_grid_html = "".join(wsi_pin_cards) or '<p class="muted">No WSI overview thumbnails found.</p>'
    default_inspector = (
        '<div class="inspector-title">Case evidence overview</div>'
        '<p class="muted">Pinned slide overviews are shown here by default. Click any figure, claim, KEEP hit, morphology card, or decision step to replace this panel with source details.</p>'
        f'<div class="pin-grid">{pin_grid_html}</div>'
        '<div class="inspector-section"><strong>Tip</strong><p class="muted">Use the Evidence tab for full ROI/segmentation review; use Verification for claim-to-evidence tracing.</p></div>'
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PathXAI Case Review - {escape(case_id)}</title>
<style>
:root {{
  --bg:#f7f9fc; --panel:#ffffff; --ink:#101a33; --muted:#68758a; --line:#dbe4f0;
  --blue:#0f4bd8; --blue2:#eef4ff; --green:#188544; --green2:#e9f8ef; --amber:#a05a00; --amber2:#fff4df;
  --red:#b42318; --red2:#fff0ee; --purple:#6544c9; --purple2:#f0ecff;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:radial-gradient(circle at top left,#ffffff 0,#f7f9fc 34%,#f4f7fb 100%); }}
button {{ font:inherit; }}
.app-header {{ position:sticky; top:0; z-index:20; background:rgba(255,255,255,.94); border-bottom:1px solid var(--line); backdrop-filter:blur(12px); }}
.header-inner {{ padding:16px 28px 12px; display:grid; grid-template-columns:1fr auto 1fr; gap:18px; align-items:center; }}
.brand {{ display:flex; gap:12px; align-items:center; }}
.brand-mark {{ width:40px;height:40px;border-radius:12px;background:#0f4bd8;color:white;display:grid;place-items:center;font-weight:900;box-shadow:0 8px 20px rgba(15,75,216,.18); }}
.case-title {{ font-size:1.45rem; font-weight:850; letter-spacing:.005em; }}
.case-subtitle {{ color:var(--muted); margin-top:2px; font-size:.84rem; }}
.summary-pills {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }}
.actions {{ display:flex; justify-content:flex-end; gap:9px; }}
.action-btn {{ border:1px solid #bfd0ee;background:white;color:#123d9d;border-radius:7px;padding:9px 13px;font-weight:750;cursor:pointer;text-decoration:none; }}
.action-btn.primary {{ background:#0f4bd8;color:white;border-color:#0f4bd8; }}
.tabs {{ justify-self:center; display:flex; gap:0; overflow:hidden; border:1px solid var(--line); border-radius:8px; background:white; box-shadow:0 6px 16px rgba(22,40,72,.05); }}
.tab {{ border:0; border-right:1px solid var(--line); background:white; padding:13px 32px; color:#26344f; cursor:pointer; white-space:nowrap; font-weight:760; }}
.tab:last-child {{ border-right:0; }}
.tab.active {{ background:linear-gradient(180deg,#ffffff,#edf4ff); color:#0f4bd8; box-shadow:inset 0 -3px 0 #0f4bd8; }}
.metric-strip {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; padding:14px 28px 0; }}
.metric-card {{ min-height:76px; min-width:0; background:white; border:1px solid var(--line); border-radius:8px; padding:12px; display:grid; grid-template-columns:36px minmax(0,1fr); gap:10px; align-items:center; box-shadow:0 6px 18px rgba(26,43,71,.04); }}
.metric-icon {{ width:34px;height:34px;border-radius:10px;background:#eef4ff;color:#0f4bd8;display:grid;place-items:center;font-size:.72rem;font-weight:900; }}
.metric-label {{ color:#506076;font-size:.7rem;font-weight:850;text-transform:uppercase;letter-spacing:.04em; }}
.metric-value {{ font-weight:850; font-size:.92rem; line-height:1.2; margin-top:2px; overflow-wrap:anywhere; }}
.metric-sub {{ color:var(--muted); font-size:.72rem; margin-top:2px; line-height:1.25; overflow-wrap:anywhere; }}
.workspace {{ display:grid; grid-template-columns:minmax(0,1fr) 390px; gap:16px; padding:16px 28px 32px; align-items:start; }}
.page {{ display:none; }}
.page.active {{ display:block; }}
.page-grid {{ display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:16px; align-items:start; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; box-shadow:0 8px 24px rgba(18,32,54,.05); margin-bottom:16px; }}
.hero-card {{ border-left:5px solid var(--blue); }}
.hero-card h2 {{ margin:4px 0 10px; font-size:1.5rem; line-height:1.25; }}
.hero-meta {{ display:flex; gap:8px; flex-wrap:wrap; }}
.card-kicker {{ text-transform:uppercase; color:var(--muted); font-size:.7rem; font-weight:750; letter-spacing:.08em; }}
.section-title {{ font-weight:800; font-size:1.05rem; margin-bottom:10px; }}
.section-title.with-icon {{ color:#0c348d; }}
.section-note {{ color:var(--muted); font-size:.86rem; margin-top:-4px; }}
.report-section {{ border-top:1px solid var(--line); padding-top:13px; margin-top:13px; }}
.report-section:first-child {{ border-top:0; padding-top:0; }}
.report-section h3 {{ margin:0 0 6px; font-size:.92rem; color:#263345; }}
.report-section p {{ margin:0; line-height:1.68; color:#2a3442; }}
.badge {{ display:inline-flex; align-items:center; border-radius:999px; padding:4px 8px; font-size:.74rem; font-weight:700; background:#eef2f7; color:#3f4a59; }}
.badge.site {{ background:var(--blue2); color:#1748bb; }}
.badge.confidence,.badge.supported {{ background:var(--green2); color:var(--green); }}
.badge.warn,.badge.weak {{ background:var(--amber2); color:var(--amber); }}
.badge.unsupported,.badge.error {{ background:var(--red2); color:var(--red); }}
.badge.zoom {{ background:var(--purple2); color:var(--purple); }}
.badge.score {{ background:#eef2f7; color:#465466; }}
.result-layout {{ display:block; }}
.diagnosis-banner {{ background:linear-gradient(90deg,#f6fff9,#ffffff); border:1px solid #bde7cd; border-radius:8px; padding:18px; display:grid; grid-template-columns:28px 1fr 118px; gap:14px; align-items:start; margin-bottom:14px; }}
.diagnosis-banner p {{ grid-column:2 / span 2; margin:0; color:#344255; line-height:1.45; }}
.diagnosis-banner h2 {{ margin:5px 0 0; font-size:1.55rem; line-height:1.28; }}
.diagnosis-limitation {{ margin-top:6px; color:#6b4b00; background:#fff7e6; border:1px solid #ffe1a6; border-radius:6px; padding:6px 8px; font-size:.82rem; line-height:1.35; }}
.diagnosis-check {{ width:24px;height:24px;border-radius:50%;background:#1c9b4a;color:white;display:grid;place-items:center;font-weight:900; }}
.confidence-box {{ border:1px solid #bde7cd;background:white;border-radius:8px;text-align:center;padding:10px; }}
.confidence-box small {{ display:block;color:var(--muted);font-size:.72rem; }}
.confidence-box strong {{ display:block;color:#16823d;font-size:1.25rem;margin-top:3px; }}
.banner-tags {{ grid-column:2 / span 2; display:flex; gap:7px; flex-wrap:wrap; margin-top:2px; }}
.tier-strip {{ grid-column:2 / span 2; display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:8px; margin-top:10px; }}
.tier-item {{ background:#ffffff; border:1px solid #bde7cd; border-radius:8px; padding:8px 10px; display:flex; flex-direction:column; gap:2px; }}
.tier-key {{ color:#506076; font-size:.68rem; font-weight:850; text-transform:uppercase; letter-spacing:.04em; }}
.tier-val {{ color:#16823d; font-size:.95rem; font-weight:800; line-height:1.2; overflow-wrap:anywhere; }}
.result-side .card {{ margin-bottom:14px; }}
.side-title {{ display:flex; justify-content:space-between; align-items:center; font-weight:850; color:#0c348d; margin-bottom:12px; }}
.side-title span {{ color:#0f4bd8; font-size:.78rem; font-weight:750; }}
.evidence-snapshot {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }}
.evidence-thumb {{ border:0; background:transparent; text-align:left; padding:0; cursor:pointer; }}
.evidence-thumb img {{ width:100%; aspect-ratio:1.25; object-fit:cover; border-radius:7px; border:1px solid var(--line); display:block; }}
.evidence-thumb span {{ display:block; margin-top:6px; }}
.evidence-thumb b {{ display:block; font-size:.75rem; }}
.evidence-thumb small {{ display:block; color:var(--muted); font-size:.68rem; }}
.mini-timeline {{ display:flex; flex-direction:column; gap:10px; }}
.mini-step {{ display:grid; grid-template-columns:28px 1fr; gap:10px; align-items:start; border:0; background:white; text-align:left; cursor:pointer; padding:0; }}
.mini-step span {{ width:24px;height:24px;border-radius:50%;border:1px solid #0f4bd8;color:#0f4bd8;display:grid;place-items:center;font-weight:850;font-size:.76rem; }}
.mini-step b {{ display:block; font-size:.82rem; }}
.mini-step small {{ display:block; color:var(--muted); font-size:.72rem; margin-top:1px; }}
.verify-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }}
.verification-footer {{ border-top:1px solid var(--line); margin-top:16px; padding-top:14px; }}
.verify {{ border:1px solid var(--line); border-radius:8px; padding:9px; text-align:center; }}
.verify b {{ display:block; font-size:1.05rem; }}
.verify span {{ font-size:.7rem; }}
.verify.supported {{ background:var(--green2); color:var(--green); border-color:#bde7cd; }}
.verify.weak {{ background:var(--amber2); color:var(--amber); border-color:#ffd99b; }}
.verify.unsupported {{ background:var(--red2); color:var(--red); border-color:#ffc9c2; }}
.verify.neutral {{ background:#f8fafc; color:#475569; }}
.report-row {{ display:grid; grid-template-columns:210px minmax(0,1fr) 260px; gap:14px; align-items:start; border-top:1px solid var(--line); padding:12px 0; }}
.report-row:first-of-type {{ border-top:0; }}
.report-row-title {{ color:#0c348d; font-weight:850; font-size:.86rem; }}
.report-row-content {{ color:#344255; font-size:.86rem; line-height:1.5; }}
.report-row-tags {{ display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end; }}
.evidence-tag {{ display:inline-flex; border:1px solid var(--line); border-radius:6px; padding:4px 8px; font-size:.72rem; font-weight:750; }}
.evidence-tag.visual {{ color:#0f4bd8; background:#f2f6ff; }}
.evidence-tag.morph {{ color:#6d35c7; background:#f5f0ff; }}
.evidence-tag.kg {{ color:#1f5eff; background:#eef4ff; }}
.evidence-tag.verify {{ color:#16823d; background:#eefaf2; }}
.evidence-tag.seg {{ color:#0f766e; background:#ecfdf5; }}
.evidence-tag.model {{ color:#9a3412; background:#fff7ed; }}
.evidence-tag.reasoning {{ color:#854d0e; background:#fefce8; }}
.evidence-legend {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:8px 0 12px; }}
.legend-chip {{ display:inline-flex; align-items:center; gap:6px; border:1px solid var(--line); border-radius:999px; padding:4px 9px; font-size:.72rem; font-weight:750; color:#334155; background:#fff; }}
.legend-chip::before {{ content:""; width:8px; height:8px; border-radius:50%; background:#94a3b8; }}
.legend-chip.visual::before {{ background:#0f4bd8; }}
.legend-chip.seg::before {{ background:#0f766e; }}
.legend-chip.morph::before {{ background:#6d35c7; }}
.legend-chip.kg::before {{ background:#1f5eff; }}
.legend-chip.model::before {{ background:#9a3412; }}
.legend-chip.verify::before {{ background:#16823d; }}
.claim-preview {{ border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
.claim-preview-row {{ width:100%; border:0; border-bottom:1px solid var(--line); background:white; display:grid; grid-template-columns:180px minmax(0,1fr) 110px; gap:12px; padding:10px; align-items:center; text-align:left; cursor:pointer; }}
.claim-preview-row:last-child {{ border-bottom:0; }}
.claim-preview-row b {{ color:#0c348d; font-size:.82rem; }}
.claim-preview-row span {{ color:#344255; font-size:.82rem; }}
.related-strip {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-top:2px; }}
.related-card {{ border:1px solid var(--line); border-radius:8px; background:white; padding:16px; display:grid; grid-template-columns:54px 1fr auto; gap:14px; align-items:center; cursor:pointer; box-shadow:0 8px 24px rgba(18,32,54,.05); }}
.related-card h3 {{ margin:0 0 5px; color:#0c348d; font-size:.96rem; }}
.related-card p {{ margin:0 0 10px; color:var(--muted); font-size:.78rem; line-height:1.35; }}
.related-icon {{ width:46px;height:46px;border-radius:50%;display:grid;place-items:center;font-weight:900; }}
.related-icon.visual {{ background:#eef4ff;color:#0f4bd8; }}
.related-icon.morph {{ background:#f3eaff;color:#7b3fc9; }}
.related-icon.verify {{ background:#eefaf2;color:#16823d; }}
.open-tab {{ border:1px solid #bfd0ee;background:white;color:#0f4bd8;border-radius:6px;padding:6px 12px;font-size:.78rem;font-weight:800;cursor:pointer; }}
.related-images {{ display:flex; gap:4px; }}
.related-thumb {{ border:0;background:transparent;padding:0;cursor:pointer; }}
.related-thumb img {{ width:54px;height:54px;object-fit:cover;border-radius:5px;border:1px solid var(--line); }}
.kv {{ border-top:1px solid var(--line); padding:10px 0; }}
.kv:first-child {{ border-top:0; }}
.kv-label {{ color:var(--muted); font-size:.73rem; font-weight:750; text-transform:uppercase; letter-spacing:.04em; }}
.kv-value {{ margin-top:3px; font-size:.9rem; line-height:1.35; }}
.xref,.thumb-chip {{ border:1px solid var(--line); background:white; border-radius:999px; padding:3px 8px; color:var(--blue); cursor:pointer; display:inline-flex; align-items:center; gap:6px; margin:1px; }}
.xref:hover,.thumb-chip:hover,.roi-card:hover,.knowledge-card:hover,.decision-node:hover,.overview-card:hover {{ border-color:var(--blue); box-shadow:0 0 0 3px rgba(31,94,255,.08); }}
.thumb-strip {{ display:flex; flex-direction:column; gap:8px; max-height:520px; overflow:auto; }}
.thumb-chip {{ border-radius:7px; padding:6px; color:var(--ink); justify-content:flex-start; }}
.thumb-chip img {{ width:58px; height:44px; object-fit:cover; border-radius:5px; border:1px solid var(--line); }}
.overview-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:12px; }}
.overview-card {{ cursor:pointer; border:1px solid transparent; border-radius:8px; padding:6px; }}
.overview-card img {{ width:100%; max-height:260px; object-fit:contain; background:#f8fafc; border:1px solid var(--line); border-radius:7px; cursor:zoom-in; }}
.overview-title,.roi-title {{ font-weight:800; margin-top:8px; }}
.wsi-legend {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }}
.figure-chip {{ border:1px solid color-mix(in srgb, var(--fig-color) 45%, white); background:color-mix(in srgb, var(--fig-color) 12%, white); color:#152033; border-radius:999px; padding:3px 8px 3px 4px; display:inline-flex; align-items:center; gap:5px; font-size:.74rem; font-weight:800; cursor:pointer; }}
.figure-chip span,.fig-dot {{ width:18px; height:18px; border-radius:50%; display:inline-grid; place-items:center; color:white; background:var(--fig-color); font-size:.68rem; font-weight:900; flex:0 0 auto; }}
.fig-dot {{ margin-right:7px; vertical-align:middle; }}
.roi-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:14px; }}
.roi-card,.knowledge-card,.morph-card,.decision-node {{ border:1px solid var(--line); border-radius:8px; background:white; cursor:pointer; overflow:hidden; }}
.roi-pair {{ display:grid; grid-template-rows:auto auto; gap:0; background:#f8fafc; border-bottom:1px solid var(--line); }}
.roi-pair > div {{ border-bottom:1px solid var(--line); }}
.roi-pair > div:last-child {{ border-bottom:0; }}
.roi-pair span {{ display:block; padding:6px 9px; color:#344255; font-size:.72rem; font-weight:850; background:#eef4ff; }}
.roi-pair img {{ width:100%; height:190px; object-fit:contain; display:block; background:#fbfdff; }}
.img-btn {{ border:none; padding:0; width:100%; cursor:pointer; background:transparent; }}
.img-btn:hover img {{ outline:2px solid var(--accent); outline-offset:-2px; }}
.empty-img {{ height:220px; display:grid; place-items:center; color:var(--muted); }}
.roi-body {{ padding:12px; }}
.roi-meta {{ display:flex; gap:6px; flex-wrap:wrap; margin:8px 0; }}
.seg-row {{ display:grid; grid-template-columns:86px 1fr 52px; gap:8px; align-items:center; font-size:.75rem; margin-top:6px; }}
.bar {{ height:7px; background:#edf1f6; border-radius:99px; overflow:hidden; }}
.bar i {{ height:100%; display:block; background:linear-gradient(90deg,#1f5eff,#38bdf8); }}
.decision-list {{ display:flex; flex-direction:column; gap:10px; }}
.decision-node {{ display:grid; grid-template-columns:44px 1fr; padding:13px; gap:12px; }}
.decision-node h3 {{ margin:0; font-size:.98rem; }}
.decision-node p {{ margin:4px 0 8px; color:var(--muted); font-size:.82rem; }}
.decision-fact {{ display:grid; grid-template-columns:150px minmax(0,1fr); gap:10px; padding:5px 0; border-top:1px solid #eef2f7; }}
.decision-fact span {{ color:var(--muted); font-size:.74rem; font-weight:800; text-transform:uppercase; letter-spacing:.03em; }}
.decision-fact b {{ font-size:.84rem; font-weight:650; color:#2f3d52; }}
.decision-links {{ display:flex; flex-wrap:wrap; gap:4px; margin-top:8px; }}
.node-index {{ width:34px; height:34px; border-radius:50%; display:grid; place-items:center; background:#172033; color:white; font-weight:800; }}
.table-scroll {{ overflow:auto; }}
.claim-table {{ width:100%; border-collapse:collapse; font-size:.85rem; }}
.claim-table th,.claim-table td {{ border-bottom:1px solid var(--line); padding:10px; vertical-align:top; text-align:left; }}
.claim-table th {{ color:#344255; background:#f8fafc; font-size:.76rem; text-transform:uppercase; letter-spacing:.04em; }}
.claim-table td {{ overflow-wrap:anywhere; word-break:break-word; }}
.claim-table.compact th,.claim-table.compact td {{ padding:8px; font-size:.78rem; }}
.report-eval-strip {{ background:white; border:1px solid var(--line); border-radius:8px; padding:14px; margin-bottom:14px; }}
.ref-wrap {{ display:flex; flex-wrap:wrap; gap:3px; }}
.tiny-chip {{ display:inline-flex; border:1px solid var(--line); border-radius:999px; padding:3px 7px; color:var(--muted); margin:1px; }}
.knowledge-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:12px; }}
.knowledge-grid.single {{ grid-template-columns:1fr; }}
.knowledge-card,.morph-card {{ padding:13px; }}
.knowledge-card h3,.morph-card h3 {{ margin:4px 0 6px; font-size:.96rem; }}
.knowledge-card p,.morph-card p {{ margin:0; color:#364254; font-size:.84rem; line-height:1.5; }}
.morph-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:12px; }}
.morph-field {{ border-top:1px solid #eef2f7; margin-top:8px; padding-top:8px; }}
.morph-field b {{ color:#0c348d; font-size:.78rem; }}
.kg-meta {{ display:flex; gap:6px; flex-wrap:wrap; margin-top:9px; }}
.inspector {{ position:sticky; top:142px; background:white; border:1px solid var(--line); border-radius:8px; max-height:calc(100vh - 162px); overflow:auto; box-shadow:0 10px 30px rgba(19,33,52,.08); }}
.inspector-head {{ padding:14px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:10px; align-items:center; }}
.inspector-head strong {{ font-size:1rem; }}
.inspector-body {{ padding:14px; }}
.inspector-type {{ display:inline-flex; font-size:.7rem; font-weight:800; color:var(--blue); text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px; }}
.inspector-title {{ font-size:1rem; font-weight:850; margin-bottom:4px; }}
.inspector-meta,.muted {{ color:var(--muted); font-size:.8rem; }}
.inspector-img {{ width:100%; max-height:420px; object-fit:contain; background:#f8fafc; border:1px solid var(--line); border-radius:7px; margin:12px 0; cursor:zoom-in; }}
.inspector-section {{ border-top:1px solid var(--line); padding-top:10px; margin-top:10px; }}
.inspector-section pre {{ white-space:pre-wrap; word-break:break-word; background:#f8fafc; border:1px solid var(--line); padding:10px; border-radius:7px; font-size:.75rem; max-height:320px; overflow:auto; }}
.raw-details summary {{ cursor:pointer; color:#0f4bd8; font-weight:800; }}
.pin-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:10px; margin-top:12px; }}
.pin-wsi {{ border:1px solid var(--line); border-radius:7px; background:white; padding:0; overflow:hidden; cursor:pointer; text-align:left; }}
.pin-wsi img {{ width:100%; height:120px; object-fit:contain; background:#f8fafc; display:block; }}
.pin-wsi span {{ display:block; padding:7px 9px; color:#0c348d; font-size:.78rem; font-weight:800; }}
.lightbox {{ position:fixed; inset:0; background:rgba(8,13,22,.88); display:none; place-items:center; z-index:99; padding:24px; }}
.lightbox img {{ max-width:96vw; max-height:94vh; object-fit:contain; background:white; }}
@media (max-width:1350px) {{ .metric-strip {{ grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); }} }}
@media (max-width:1100px) {{ .workspace,.page-grid {{ grid-template-columns:1fr; }} .inspector {{ position:relative; top:auto; max-height:none; }} .header-inner {{ grid-template-columns:1fr; }} .tabs {{ justify-self:start; }} .report-row {{ grid-template-columns:1fr; }} .report-row-tags {{ justify-content:flex-start; }} .related-strip {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header class="app-header">
  <div class="header-inner">
    <div class="brand">
      <div class="brand-mark">PX</div>
      <div>
        <div class="case-title">Pathology Case Review</div>
        <div class="case-subtitle">{escape(case_id)} · Explainable pathology report with evidence, reasoning, and claim verification</div>
      </div>
    </div>
    <nav class="tabs">
      <button class="tab active" data-tab="final">Result</button>
      <button class="tab" data-tab="visual">Evidence</button>
      <button class="tab" data-tab="decision">Reasoning</button>
      <button class="tab" data-tab="claims">Verification</button>
      {audit_tab}
    </nav>
    <div class="summary-pills"></div>
  </div>
</header>
<section class="metric-strip">{metric_strip}</section>
<main class="workspace">
  <section>
    <div id="tab-final" class="page active">{render_final_report(payload, figures)}</div>
    <div id="tab-visual" class="page">{render_evidence(payload, figures, html_path)}</div>
    <div id="tab-decision" class="page">{render_decisions(payload)}</div>
    <div id="tab-claims" class="page">{render_claims(payload, figures)}</div>
    {audit_page}
  </section>
  <aside class="inspector">
    <div class="inspector-head"><strong>Evidence Inspector</strong><span class="muted">click any chip/card</span></div>
    <div id="inspectorBody" class="inspector-body">
      {default_inspector}
    </div>
  </aside>
</main>
<div class="lightbox" id="lightbox" onclick="this.style.display='none'"><img id="lightboxImg" alt=""></div>
<script>
const INSPECTOR = {inspector_json};
function esc(s) {{
  return String(s ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
}}
function renderObj(obj) {{
  return '<pre>' + esc(JSON.stringify(obj, null, 2)) + '</pre>';
}}
function renderRelated(items) {{
  if (!items || !items.length) return '';
  return '<div class="inspector-section"><strong>Related</strong><div class="ref-wrap">' +
    items.map(r => '<button class="xref" data-ref="' + esc(r.ref) + '" type="button">' + esc(r.label || r.ref) + '</button>').join('') +
    '</div></div>';
}}
function inspect(ref) {{
  const item = INSPECTOR[ref];
  const body = document.getElementById('inspectorBody');
  if (!item) {{
    body.innerHTML = '<div class="inspector-title">' + esc(ref) + '</div><p class="muted">No detail was indexed for this reference.</p>';
    return;
  }}
  let html = '<span class="inspector-type">' + esc(item.type || 'Evidence') + '</span>';
  html += '<div class="inspector-title">' + esc(item.title || ref) + '</div>';
  if (item.meta) html += '<div class="inspector-meta">' + esc(item.meta) + '</div>';
  if (item.image) html += '<img class="inspector-img" src="' + esc(item.image) + '" onclick="openLightbox(this.src)" alt="">';
  if (item.summary) html += '<div class="inspector-section"><strong>Summary</strong><p style="white-space:pre-wrap;line-height:1.5;">' + esc(item.summary) + '</p></div>';
  if (item.details) html += '<div class="inspector-section"><details class="raw-details"><summary>Source details</summary>' + renderObj(item.details) + '</details></div>';
  html += renderRelated(item.related);
  body.innerHTML = html;
}}
function openLightbox(src) {{
  document.getElementById('lightboxImg').src = src;
  document.getElementById('lightbox').style.display = 'grid';
}}
document.addEventListener('click', e => {{
  const tab = e.target.closest('.tab');
  if (tab) {{
    activateTab(tab.dataset.tab);
    return;
  }}
  const openTabButton = e.target.closest('.open-tab, [data-tab-target]');
  if (openTabButton) {{
    activateTab(openTabButton.dataset.tabTarget || openTabButton.dataset.tabJump);
    return;
  }}
  const refEl = e.target.closest('[data-ref]');
  if (refEl) {{
    inspect(refEl.dataset.ref);
    document.querySelector('.inspector').scrollIntoView({{block:'nearest', behavior:'smooth'}});
    e.stopPropagation();
    return;
  }}
  const openTab = e.target.closest('[data-tab-jump]');
  if (openTab) {{
    activateTab(openTab.dataset.tabJump);
    return;
  }}
}});
function activateTab(name) {{
  if (!name) return;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  const page = document.getElementById('tab-' + name);
  if (page) page.classList.add('active');
  window.scrollTo({{top:0, behavior:'smooth'}});
}}
</script>
</body>
</html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case_dir", required=True, type=Path)
    ap.add_argument("--out_name", default="report_workspace.html")
    args = ap.parse_args()
    case_dir = args.case_dir.resolve()
    html_path = case_dir / args.out_name
    payload = read_case(case_dir)
    html = render_html(payload, html_path)
    html_path.write_text(html, encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(html_path), "bytes": len(html.encode("utf-8"))}, indent=2))


if __name__ == "__main__":
    main()
