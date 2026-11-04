"""MedGemma morphology description stage.

Calls MedGemma (short_report_client) for educational descriptive text of
pathology patches, cached per ROI.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathxai_agent_v2.config import Settings
from pathxai_agent_v2.llm import LLMClient
from pathxai_agent_v2.pipeline.base import (
    PipelineStage,
    RoiInfo,
    SitePrediction,
    StageResult,
    ok,
)
from pathxai_agent_v2.prompts import medgemma_short_description_prompt


@dataclass
class MorphologyOutput:
    """Output of the MedGemma morphology stage."""
    available: bool = False
    description: str = ""
    rounds: list[dict] = field(default_factory=list)
    round_count: int = 0
    source: str = "medgemma"

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "description": self.description,
            "rounds": self.rounds,
            "round_count": self.round_count,
            "source": self.source,
        }


class MorphologyStage(PipelineStage[list[RoiInfo], MorphologyOutput]):
    """Stage: Generate MedGemma morphology descriptions for ROIs.

    Uses a separate client (short_report_client / MedGemma).
    """

    stage_name = "morphology"

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        out_dir: str | Path,
        *,
        medgemma_client: LLMClient | None = None,
    ):
        super().__init__(settings, llm, out_dir)
        self.medgemma_client = medgemma_client or llm

    def run(
        self,
        stage_input: list[RoiInfo],
        dry_run: bool = False,
        *,
        site_prediction: SitePrediction | None = None,
        previous_rounds: list[dict] | None = None,
        round_index: int = 1,
        input_role: str = "initial",
    ) -> StageResult[MorphologyOutput]:
        if dry_run or not self.medgemma_client:
            return ok(
                MorphologyOutput(available=False),
                self.stage_name,
            )

        all_rounds = list(previous_rounds or [])
        new_round = self._describe_round(
            stage_input, site_prediction, round_index, input_role
        )
        all_rounds.append(new_round)

        # Accumulate descriptions
        descriptions = []
        for r in all_rounds:
            desc = r.get("description", "")
            if desc:
                descriptions.append(desc)

        return ok(
            MorphologyOutput(
                available=bool(descriptions),
                description="\n\n".join(descriptions),
                rounds=all_rounds,
                round_count=len(all_rounds),
                source="medgemma",
            ),
            self.stage_name,
        )

    def _describe_round(
        self,
        rois: list[RoiInfo],
        site_prediction: SitePrediction | None,
        round_index: int,
        input_role: str,
    ) -> dict:
        """Run MedGemma on one round of ROIs, with caching."""
        site_dict = site_prediction.to_dict() if site_prediction else {}
        site_dict.setdefault("predicted_site", "unknown")

        descriptions = []
        for roi in rois[:self.settings.roi.selected_rois]:
            # Cache key
            cache_key = _cache_key(roi.image_path, site_dict.get("predicted_site", ""), str(round_index))
            cache_path = self.out_dir / "llm_cache" / f"{cache_key}.json"

            if cache_path.exists():
                cached = json.loads(cache_path.read_text())
                descriptions.append(cached.get("description", ""))
                continue

            prompt = medgemma_short_description_prompt(site_dict)
            try:
                raw = self.medgemma_client.chat(
                    prompt, image_paths=[roi.image_path]
                )
                desc = str(raw.get("content", "")).strip()
            except Exception:
                desc = ""

            if desc:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({"description": desc}))

            descriptions.append(desc)

        combined = "\n\n".join(d for d in descriptions if d)
        return {
            "round": round_index,
            "input_role": input_role,
            "description": combined,
            "roi_count": len(rois),
            "roi_ids": [r.roi_id for r in rois],
        }


def _cache_key(image_path: str, site: str, round_str: str) -> str:
    h = hashlib.sha1(f"{image_path}|{site}|{round_str}".encode()).hexdigest()[:16]
    return h


class SegReaderMorphologyStage(PipelineStage[list[RoiInfo], MorphologyOutput]):
    """Use a fine-tuned paired ROI+seg reader as structured morphology evidence."""

    stage_name = "seg_reader_morphology"

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        out_dir: str | Path,
        *,
        seg_reader_client: LLMClient,
    ):
        super().__init__(settings, llm, out_dir)
        self.seg_reader_client = seg_reader_client

    def run(
        self,
        stage_input: list[RoiInfo],
        dry_run: bool = False,
        *,
        site_prediction: SitePrediction | None = None,
        segmentation_context: dict[str, Any] | None = None,
        previous_rounds: list[dict] | None = None,
        round_index: int = 1,
        input_role: str = "initial",
    ) -> StageResult[MorphologyOutput]:
        if dry_run or not self.seg_reader_client:
            return ok(MorphologyOutput(available=False), self.stage_name)

        all_rounds = list(previous_rounds or [])
        new_round = self._describe_round(
            stage_input,
            site_prediction,
            segmentation_context or {},
            round_index,
            input_role,
        )
        all_rounds.append(new_round)
        descriptions = [str(r.get("description", "")).strip() for r in all_rounds]
        descriptions = [d for d in descriptions if d]
        return ok(
            MorphologyOutput(
                available=bool(descriptions),
                description="\n\n".join(descriptions),
                rounds=all_rounds,
                round_count=len(all_rounds),
                source="fine_tuned_seg_reader",
            ),
            self.stage_name,
        )

    def _describe_round(
        self,
        rois: list[RoiInfo],
        site_prediction: SitePrediction | None,
        segmentation_context: dict[str, Any],
        round_index: int,
        input_role: str,
    ) -> dict:
        site = site_prediction.predicted_site if site_prediction else "unknown"
        seg_by_roi = {
            str(r.get("roi_id", "")): r
            for r in (segmentation_context.get("roi_results") or [])
            if isinstance(r, dict)
        }
        records: list[dict[str, Any]] = []
        descriptions: list[str] = []

        for local_idx, roi in enumerate(rois[: self.settings.roi.selected_rois], start=1):
            seg = seg_by_roi.get(roi.roi_id, {})
            prompt_version = _seg_reader_prompt_version(self.settings.llm.seg_reader_model)
            require_paired_seg = prompt_version.startswith("v3_")
            image_path, image_source = _seg_reader_image_path(
                roi,
                seg,
                require_pair=require_paired_seg,
            )
            prompt = _seg_reader_prompt(site, model_name=self.settings.llm.seg_reader_model)
            cache_key = _cache_key(
                str(image_path),
                site,
                f"seg_reader_{round_index}_{self.settings.llm.seg_reader_model}_{prompt_version}",
            )
            cache_path = self.out_dir / "seg_reader_cache" / f"{cache_key}.json"
            invalid_reason = ""
            raw = ""
            if not image_path:
                parsed = None
                invalid_reason = "missing_paired_seg_reader_image"
            elif cache_path.exists():
                cached = json.loads(cache_path.read_text())
                parsed = cached.get("prediction")
                raw = cached.get("raw", "")
                invalid_reason = str(cached.get("invalid_reason", "") or "")
                if not invalid_reason:
                    parsed, invalid_reason = _validate_seg_reader_prediction(parsed, raw)
            else:
                try:
                    raw_resp = self.seg_reader_client.chat_json(prompt, image_paths=[image_path])
                    parsed = raw_resp.get("json") if isinstance(raw_resp, dict) else None
                    raw = str((raw_resp or {}).get("content", ""))
                    if not isinstance(parsed, dict):
                        parsed = _parse_seg_reader_raw(raw)
                    elif isinstance(parsed, dict):
                        parsed = _normalise_seg_reader_prediction(parsed)
                except Exception as exc:
                    parsed = None
                    raw = f"ERROR: {exc}"
                parsed, invalid_reason = _validate_seg_reader_prediction(parsed, raw)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(
                        {
                            "prediction": parsed,
                            "raw": _bounded_reader_raw(raw),
                            "invalid_reason": invalid_reason,
                            "prompt_version": prompt_version,
                            "image_path": str(image_path),
                            "image_source": image_source,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )

            record = {
                # Keep prompts and reader-facing summaries anonymous.  The raw
                # ROI id may contain case/slide tokens; a short hash is enough
                # for checkpoint traceability without exposing dataset priors.
                "roi_trace_id": hashlib.sha1(str(roi.roi_id).encode()).hexdigest()[:10],
                "display_id": f"ROI {local_idx}",
                "image_path": image_path,
                "image_source": image_source,
                "prompt_version": prompt_version,
                "invalid_reason": invalid_reason,
                "prediction": parsed if isinstance(parsed, dict) else None,
                "raw": raw,
            }
            records.append(record)
            descriptions.append(_format_seg_reader_record(record))

        combined = "\n".join(d for d in descriptions if d)
        return {
            "round": round_index,
            "input_role": input_role,
            "source": "fine_tuned_seg_reader",
            "description": combined,
            "roi_count": len(rois),
            "roi_display_ids": [f"ROI {i}" for i, _ in enumerate(rois, start=1)],
            "roi_trace_ids": [hashlib.sha1(str(r.roi_id).encode()).hexdigest()[:10] for r in rois],
            "records": records,
        }


def apply_seg_reader_guard(morph: MorphologyOutput) -> MorphologyOutput:
    """Keep only positive, localized reader evidence for diagnosis/report prompts.

    The raw reader records remain in checkpoints for audit, but the accumulated
    description exposed to downstream LLM stages is conservative.  In particular,
    absent classes are not treated as contradiction/subtype-exclusion evidence.
    """
    if not morph or not morph.available or "seg_reader" not in str(morph.source).lower():
        return morph

    guarded_rounds: list[dict[str, Any]] = []
    guarded_descriptions: list[str] = []

    for round_data in deepcopy(morph.rounds or []):
        records = round_data.get("records") or []
        lines: list[str] = []
        for record in records:
            line = _format_guarded_seg_reader_record(record)
            if line:
                lines.append(line)
        if lines:
            guarded_desc = "\n".join(lines)
        else:
            guarded_desc = (
                "Seg-reader guard: no positive localized reader evidence passed "
                "filtering in this round; use raw ROI images and compact stats."
            )
        round_data["raw_reader_description"] = round_data.get("description", "")
        round_data["description"] = guarded_desc
        round_data["reader_guard"] = {
            "enabled": True,
            "policy": (
                "Only positive localized support is exposed downstream. Absent "
                "classes, weak/trace negative evidence, and stroma-only cues are "
                "not allowed to exclude subtype candidates or override image-only "
                "diagnosis."
            ),
        }
        guarded_rounds.append(round_data)
        guarded_descriptions.append(guarded_desc)

    return MorphologyOutput(
        available=bool(guarded_descriptions),
        description="\n\n".join(guarded_descriptions),
        rounds=guarded_rounds,
        round_count=len(guarded_rounds),
        source=f"{morph.source}_guarded",
    )


def _seg_reader_prompt_version(model_name: str = "") -> str:
    if "v3" in str(model_name or "").lower():
        return "v3_train_prompt_runtime_pair_v1"
    return "legacy_train_prompt_runtime_pair_v1"


def _seg_reader_prompt(site: str, *, model_name: str = "") -> str:
    site_text = site or "not provided"
    if "v3" in str(model_name or "").lower():
        # Keep online inference isomorphic to the v3 SFT training prompt.  The
        # v3 adapter is sensitive to looser schema prose and can otherwise
        # drift into long repeated lists that are useless for report writing.
        return (
            "You are a segmentation-grounded pathology morphology evidence reader. "
            "The image is a paired ROI: left panel is original H&E, right panel is "
            "segmentation overlay. "
            f"Known anatomical site: {site_text}.\n"
            "Return exactly one JSON object. Do not diagnose. Do not mention case IDs, "
            "filenames, coordinates, TCGA, IHC, molecular tests, stage, margins, gross "
            "size, or clinical outcomes. Separate high-confidence localized evidence "
            "from weak/non-localized morphology terms. Support labels must agree with "
            "fractions: if fraction is near zero, support must be absent or weak, never "
            "strong. Do not copy allowed-value lists or placeholder examples.\n"
            "Required keys: schema_version, site_context, roi_quality, "
            "segmentation_reliability, localized_evidence, visible_morphology, "
            "report_writer_guidance, consistency_policy."
        )
    return (
        "Read one paired pathology ROI image. Left: original H&E ROI. Right: "
        "segmentation overlay. Known site context: "
        f"{site_text}.\n"
        "Return ONE compact JSON object only, no markdown, no prose, no repeated keys. "
        "Do not diagnose. Do not mention case IDs, filenames, coordinates, TCGA, IHC, "
        "molecular, stage, margins, or outcome.\n"
        "Separate high-confidence localized evidence from weak or non-localized "
        "morphology terms. Support labels must agree with fractions: if a fraction "
        "is near zero, support must be absent or weak, never strong. Do NOT copy "
        "allowed-value lists or placeholder text.\n"
        "Required keys: schema_version, site_context, roi_quality, "
        "segmentation_reliability, localized_evidence, visible_morphology, "
        "report_writer_guidance. For visible_morphology, include one_sentence, "
        "high_confidence_features, low_confidence_features, and not_assessable."
    )


def _seg_reader_image_path(
    roi: RoiInfo,
    seg: dict[str, Any],
    *,
    require_pair: bool = False,
) -> tuple[str, str]:
    """Create the same left-H&E/right-overlay paired view used for SFT.

    Segmentation results may carry legacy composite paths from old experiments.
    For the reader ablation we want a stable v2 runtime pair with the same layout
    as the SFT images, so the adapter is not asked to parse a different visual
    format.
    """
    original = str(seg.get("image_path") or roi.image_path or "")
    overlay = str(seg.get("overlay_path") or "")
    if original and overlay and Path(original).exists() and Path(overlay).exists():
        key = hashlib.sha1(f"{original}|{overlay}|seg_reader_pair_v2".encode()).hexdigest()[:20]
        out_dir = Path("/local/PathXAI_Agent_v2/shared_artifacts/seg_reader_runtime_pairs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{key}.jpg"
        if out_path.exists():
            return str(out_path), "runtime_paired_he_seg"
        try:
            from PIL import Image, ImageDraw

            left = Image.open(original).convert("RGB")
            right = Image.open(overlay).convert("RGB")
            max_edge = 768
            for img in (left, right):
                img.thumbnail((max_edge, max_edge))
            h = max(left.height, right.height)
            canvas = Image.new("RGB", (left.width + right.width, h + 28), "white")
            canvas.paste(left, (0, 28))
            canvas.paste(right, (left.width, 28))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), "H&E ROI", fill=(20, 20, 20))
            draw.text((left.width + 8, 8), "Segmentation overlay", fill=(20, 20, 20))
            canvas.save(out_path, quality=88)
            return str(out_path), "runtime_paired_he_seg"
        except Exception:
            pass
    if require_pair:
        return "", "missing_paired_he_seg"
    fallback = str(
        seg.get("composite_path")
        or seg.get("paired_path")
        or seg.get("overlay_path")
        or roi.image_path
    )
    return fallback, "fallback_unpaired"


def _parse_seg_reader_raw(raw: str) -> dict[str, Any] | None:
    """Recover JSON-like seg-reader responses that the generic LLM parser missed."""
    text = str(raw or "").strip()
    if not text or text.startswith("ERROR:"):
        return None
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict):
            return _normalise_seg_reader_prediction(obj)
    return None


def _normalise_seg_reader_prediction(pred: dict[str, Any]) -> dict[str, Any]:
    """Map loosely formatted reader output onto the compact schema used downstream."""
    if str(pred.get("schema_version") or "").startswith("seg_reader_v3"):
        return _normalise_seg_reader_v3_prediction(pred)
    out = dict(pred)
    if "glandular_or_epithelial_evidence" not in out:
        for key in ("glandular_support", "epithelial_evidence", "glandular_evidence"):
            if key in out:
                out["glandular_or_epithelial_evidence"] = out[key]
                break
    if "tumor_evidence" not in out:
        for key in ("tumor", "tumour", "tumor_presence"):
            if key in out:
                out["tumor_evidence"] = out[key]
                break
    if "necrosis_evidence" not in out:
        for key in ("necrosis", "necrosis_support"):
            if key in out:
                out["necrosis_evidence"] = out[key]
                break
    fractions = out.get("tissue_fractions")
    if not isinstance(fractions, dict):
        fractions = {}
    cleaned_fractions: dict[str, Any] = {}
    for key, value in fractions.items():
        norm_key = _normalise_fraction_key(key)
        if isinstance(value, dict):
            value = value.get("fraction", value.get("value", value.get("extent", "")))
        cleaned_fractions[norm_key] = _coerce_fraction_value(value)
    out["tissue_fractions"] = cleaned_fractions
    out["roi_quality"] = _normalise_choice(
        out.get("roi_quality"),
        {"diagnostic", "artifact", "low_tumor", "background"},
        default="not specified",
    )
    for key in ("tumor_evidence", "necrosis_evidence", "glandular_or_epithelial_evidence"):
        value = out.get(key)
        if isinstance(value, dict):
            norm = dict(value)
            norm["support"] = _normalise_choice(
                norm.get("support"),
                {"strong", "moderate", "weak", "absent"},
                default=str(norm.get("support", "") or "not specified"),
            )
            if "fraction" in norm:
                norm["fraction"] = _coerce_fraction_value(norm.get("fraction"))
            out[key] = norm
    out["architecture"] = _normalise_choice_list(
        out.get("architecture"),
        {"solid", "nested", "glandular", "papillary", "diffuse", "trabecular", "not_assessable"},
    )
    out["cytology"] = _normalise_choice_list(
        out.get("cytology"),
        {"clear_cytoplasm", "eosinophilic_cytoplasm", "spindle", "pleomorphic", "small_blue_cells", "not_assessable"},
    )
    out["visual_task_note"] = _normalise_reader_notes(out.get("visual_task_note"))
    return out


def _normalise_seg_reader_v3_prediction(pred: dict[str, Any]) -> dict[str, Any]:
    out = dict(pred)
    loc = out.get("localized_evidence") if isinstance(out.get("localized_evidence"), dict) else {}
    fractions: dict[str, Any] = {}
    for src_key, dst_key in (
        ("tumor", "neoplastic"),
        ("necrosis", "dead_or_necrotic"),
        ("glandular_or_epithelial", "glandular_or_epithelial"),
        ("stroma_or_connective", "connective"),
        ("inflammation", "inflammation"),
        ("foreground", "foreground"),
    ):
        item = loc.get(src_key)
        if isinstance(item, dict) and "fraction" in item:
            fractions[dst_key] = _coerce_fraction_value(item.get("fraction"))
    out["tissue_fractions"] = fractions
    out["roi_quality"] = _normalise_choice(
        out.get("roi_quality"),
        {"diagnostic", "limited", "low_tumor", "background_or_artifact", "artifact", "background"},
        default="not specified",
    )
    vm = out.get("visible_morphology") if isinstance(out.get("visible_morphology"), dict) else {}
    out["high_confidence_features"] = _normalise_reader_notes(vm.get("high_confidence_features"))
    out["low_confidence_features"] = _normalise_reader_notes(vm.get("low_confidence_features"))
    out["visual_task_note"] = _normalise_reader_notes(vm.get("one_sentence"))
    out["not_assessable"] = _normalise_reader_notes(vm.get("not_assessable"))
    if "tumor_evidence" not in out:
        out["tumor_evidence"] = loc.get("tumor") or {}
    if "necrosis_evidence" not in out:
        out["necrosis_evidence"] = loc.get("necrosis") or {}
    if "glandular_or_epithelial_evidence" not in out:
        out["glandular_or_epithelial_evidence"] = loc.get("glandular_or_epithelial") or {}
    for key in ("tumor_evidence", "necrosis_evidence", "glandular_or_epithelial_evidence"):
        value = out.get(key)
        if isinstance(value, dict):
            norm = dict(value)
            norm["support"] = _normalise_choice(
                norm.get("support"),
                {"strong", "moderate", "weak", "absent"},
                default=str(norm.get("support", "") or "not specified"),
            )
            if "fraction" in norm:
                norm["fraction"] = _coerce_fraction_value(norm.get("fraction"))
            out[key] = norm
    return out


def _normalise_choice(value: Any, allowed: set[str], *, default: str = "") -> str:
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if "|" in text:
        return default
    return text if text in allowed else default


def _normalise_choice_list(value: Any, allowed: set[str]) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    cleaned: list[str] = []
    for item in values:
        text = str(item or "").strip().lower().replace(" ", "_").replace("-", "_")
        if not text or "|" in text:
            continue
        if text in allowed and text not in cleaned:
            cleaned.append(text)
    return cleaned[:3]


def _normalise_reader_notes(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    cleaned: list[str] = []
    placeholders = {
        "short visible finding",
        "brief visible morphology observation",
        "visible finding",
        "not_assessable",
    }
    for item in values:
        text = _clean_reader_field(item, max_chars=160)
        if not text:
            continue
        if text.lower().strip(" .") in placeholders:
            continue
        if text not in cleaned:
            cleaned.append(text)
    return cleaned[:3]


def _validate_seg_reader_prediction(
    parsed: Any,
    raw: str,
) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(parsed, dict):
        if _looks_like_repetitive_reader_failure(raw):
            return None, "repetitive_or_truncated_non_json"
        return None, "non_json"
    schema = str(parsed.get("schema_version") or "")
    if schema.startswith("seg_reader_v3"):
        loc = parsed.get("localized_evidence")
        vm = parsed.get("visible_morphology")
        if not isinstance(loc, dict) or not isinstance(vm, dict):
            return None, "v3_missing_localized_evidence_or_visible_morphology"
        required_loc = {"tumor", "necrosis", "glandular_or_epithelial"}
        if not any(k in loc for k in required_loc):
            return None, "v3_missing_core_localized_evidence"
        for key, value in loc.items():
            if isinstance(value, list) and len(value) > 8:
                return None, f"v3_repeated_list_in_localized_evidence:{key}"
        if _looks_like_repetitive_reader_failure(json.dumps(parsed, ensure_ascii=False)):
            return None, "v3_repetitive_output"
    return parsed, ""


def _looks_like_repetitive_reader_failure(raw: str) -> bool:
    text = str(raw or "")
    if len(text) > 12000:
        return True
    lowered = text.lower()
    for token in ("inflammation", "necrosis", "tumor architecture: necrosis", "necrotic/dead component"):
        if lowered.count(token) > 25:
            return True
    return False


def _bounded_reader_raw(raw: str, *, limit: int = 3000) -> str:
    text = str(raw or "")
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated raw seg-reader output, original_chars={len(text)}]"


def _coerce_fraction_value(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip().lower()
    if not text:
        return value
    aliases = {
        "absent": 0.0,
        "none": 0.0,
        "trace": 0.01,
        "focal": 0.05,
        "limited": 0.10,
        "weak": 0.10,
        "moderate": 0.35,
        "prominent": 0.70,
        "strong": 0.70,
    }
    if text in aliases:
        return aliases[text]
    if text.startswith("<"):
        try:
            return float(text[1:].replace("%", "")) / 100.0
        except Exception:
            return value
    if text.endswith("%"):
        try:
            return float(text[:-1]) / 100.0
        except Exception:
            return value
    try:
        return float(text)
    except Exception:
        return value


def _format_seg_reader_record(record: dict[str, Any]) -> str:
    pred = record.get("prediction")
    # Do not put raw ROI identifiers into prompts. They can contain TCGA case or
    # slide tokens; keep those only in audit/checkpoint records.
    roi_id = record.get("display_id") or "ROI"
    if not isinstance(pred, dict):
        return f"{roi_id}: seg-reader unavailable or non-JSON response; excluded from morphology evidence."
    if str(pred.get("schema_version") or "").startswith("seg_reader_v3"):
        return _format_seg_reader_v3_record(record)
    fractions = pred.get("tissue_fractions") if isinstance(pred.get("tissue_fractions"), dict) else {}
    frac_bits = []
    numeric_fractions: dict[str, float] = {}
    for k, v in fractions.items():
        try:
            fv = float(v)
            norm_key = _normalise_fraction_key(k)
            numeric_fractions[norm_key] = fv
            frac_bits.append(f"{norm_key}={fv:.3f}")
        except Exception:
            frac_bits.append(f"{_normalise_fraction_key(k)}={v}")
    necrosis_label = _seg_reader_necrosis_phrase(
        pred.get("necrosis_evidence", "not specified"),
        numeric_fractions,
    )
    tumor_label = _seg_reader_tumor_phrase(
        pred.get("tumor_evidence", "not specified"),
        numeric_fractions,
    )
    epithelial = _seg_reader_support_phrase(
        pred.get("glandular_or_epithelial_evidence", "not specified"),
        "glandular/epithelial",
    )
    architecture = _clean_reader_field(pred.get("architecture", ""))
    cytology = _clean_reader_field(pred.get("cytology", ""))
    note = _clean_reader_field(pred.get("visual_task_note", ""))
    detail_bits = []
    if architecture:
        detail_bits.append(f"architecture={architecture}")
    if cytology:
        detail_bits.append(f"cytology={cytology}")
    if note:
        detail_bits.append(f"visible note={note}")
    details = "; ".join(detail_bits)
    return (
        f"{roi_id}: ROI quality={_plain_reader_label(pred.get('roi_quality', 'not specified'))}; "
        f"tumor evidence={tumor_label}; "
        f"necrosis evidence={necrosis_label}; "
        f"glandular/epithelial evidence={epithelial or 'not specified'}; "
        f"fractions: {', '.join(frac_bits) if frac_bits else 'not provided'}"
        + (f"; {details}." if details else ".")
    )


def _format_seg_reader_v3_record(record: dict[str, Any]) -> str:
    pred = record.get("prediction") or {}
    roi_id = record.get("display_id") or "ROI"
    fractions = pred.get("tissue_fractions") if isinstance(pred.get("tissue_fractions"), dict) else {}
    frac_bits = []
    numeric_fractions: dict[str, float] = {}
    for k, v in fractions.items():
        try:
            fv = float(v)
            norm_key = _normalise_fraction_key(k)
            numeric_fractions[norm_key] = fv
            frac_bits.append(f"{norm_key}={fv:.3f}")
        except Exception:
            pass
    high = _normalise_reader_notes(pred.get("high_confidence_features"))
    low = _normalise_reader_notes(pred.get("low_confidence_features"))
    note = _normalise_reader_notes(pred.get("visual_task_note"))
    useful = []
    if high:
        useful.append("high-confidence localized features: " + "; ".join(high[:4]))
    if low:
        useful.append("low-confidence/case-level cues: " + "; ".join(low[:3]))
    if note:
        useful.append("summary: " + "; ".join(note[:1]))
    return (
        f"{roi_id}: ROI quality={_plain_reader_label(pred.get('roi_quality', 'not specified'))}; "
        f"tumor evidence={_seg_reader_tumor_phrase(pred.get('tumor_evidence', 'not specified'), numeric_fractions)}; "
        f"necrosis evidence={_seg_reader_necrosis_phrase(pred.get('necrosis_evidence', 'not specified'), numeric_fractions)}; "
        f"glandular/epithelial evidence={_seg_reader_support_phrase(pred.get('glandular_or_epithelial_evidence', 'not specified'), 'glandular/epithelial')}; "
        f"fractions: {', '.join(frac_bits) if frac_bits else 'not provided'}"
        + (f"; {'; '.join(useful)}." if useful else ".")
    )


def _format_guarded_seg_reader_record(record: dict[str, Any]) -> str:
    pred = record.get("prediction")
    roi_id = record.get("display_id") or "ROI"
    if not isinstance(pred, dict):
        reason = str(record.get("invalid_reason") or "unavailable")
        return f"{roi_id}: reader evidence excluded ({reason})."

    fractions = pred.get("tissue_fractions") if isinstance(pred.get("tissue_fractions"), dict) else {}
    numeric: dict[str, float] = {}
    for key, value in fractions.items():
        try:
            numeric[_normalise_fraction_key(key)] = float(value)
        except Exception:
            continue

    accepted: list[str] = []
    rejected: list[str] = []

    quality = _plain_reader_label(pred.get("roi_quality", "not specified"))
    if quality in {"diagnostic", "limited"}:
        accepted.append(f"ROI quality={quality}")
    elif quality and quality != "not specified":
        rejected.append(f"ROI quality={quality}")

    tumor_frac = max(numeric.get("neoplastic", 0.0), numeric.get("tumor", 0.0), numeric.get("tumour", 0.0))
    if tumor_frac >= 0.20:
        accepted.append(f"tumor-rich/diagnostic tissue supported (tumor fraction {tumor_frac:.3f})")
    elif tumor_frac > 0:
        rejected.append(f"limited tumor fraction {tumor_frac:.3f}")

    dead_frac = max(numeric.get("dead_or_necrotic", 0.0), numeric.get("necrosis", 0.0), numeric.get("dead", 0.0))
    if dead_frac >= 0.05:
        accepted.append(f"necrotic/dead component present but proportional to fraction {dead_frac:.3f}")
    elif dead_frac > 0:
        rejected.append(f"trace necrotic/dead component {dead_frac:.3f}")

    gland_frac = max(numeric.get("glandular_or_epithelial", 0.0), numeric.get("gland", 0.0), numeric.get("epithelial", 0.0))
    gland_support = _support_value(pred.get("glandular_or_epithelial_evidence"))
    if gland_frac >= 0.05 and gland_support not in {"absent", "none", "not specified"}:
        accepted.append(f"glandular/epithelial component positively supported (fraction {gland_frac:.3f})")
    else:
        rejected.append("absent/trace glandular-epithelial label not used as subtype-exclusion evidence")

    high = _normalise_reader_notes(pred.get("high_confidence_features"))
    for item in high:
        low = item.lower()
        if any(term in low for term in ("absent", "no ", "not identified", "not seen")):
            rejected.append(item)
            continue
        if any(term in low for term in ("tumor-rich", "tumour-rich", "necrotic", "gland", "epithelial")):
            accepted.append(item)

    record["accepted_reader_evidence"] = accepted[:6]
    record["rejected_reader_evidence"] = rejected[:6]
    if not accepted:
        return f"{roi_id}: reader guard found no positive localized evidence suitable for diagnosis."
    return (
        f"{roi_id}: guarded reader evidence: "
        + "; ".join(dict.fromkeys(accepted[:6]))
        + ". Do not use absent classes to exclude differential diagnoses."
    )


def _support_value(raw_label: Any) -> str:
    if isinstance(raw_label, dict):
        raw_label = raw_label.get("support", "")
    return str(raw_label or "").strip().lower().replace(" ", "_").replace("-", "_") or "not specified"


def _normalise_fraction_key(key: Any) -> str:
    text = str(key or "").strip().lower()
    text = text.replace(" ", "_").replace("-", "_")
    aliases = {
        "dead": "dead_or_necrotic",
        "necrosis": "dead_or_necrotic",
        "necrotic": "dead_or_necrotic",
        "tumor": "neoplastic",
        "tumour": "neoplastic",
        "neoplastic_cells": "neoplastic",
        "fibrous": "connective",
        "stroma": "connective",
        "stromal": "connective",
        "glands": "gland",
    }
    return aliases.get(text, text)


def _clean_reader_field(value: Any, *, max_chars: int = 220) -> str:
    """Keep reader text as morphology evidence, not diagnostic leakage."""
    text = str(value or "").strip()
    if not text:
        return ""
    text = " ".join(text.replace("\n", " ").split())
    # Remove sentences that try to become diagnosis / IHC / molecular claims.
    bad = re.compile(
        r"\b("
        r"diagnos(?:is|tic)|consistent with|compatible with|favor(?:s|ing)?|"
        r"carcinoma|adenocarcinoma|adenoma|sarcoma|lymphoma|melanoma|"
        r"pheochromocytoma|paraganglioma|metastatic|metastasis|"
        r"immunohistochem|ihc|molecular|marker|positive|negative|"
        r"stage|margin|lymph node|tcga|case id|filename|coordinate"
        r")\b",
        flags=re.IGNORECASE,
    )
    kept = []
    for sent in re.split(r"(?<=[.!?;])\s+", text):
        if not bad.search(sent):
            kept.append(sent)
    cleaned = " ".join(kept).strip(" ;.")
    return cleaned[:max_chars]


def _plain_reader_label(value: Any, *, max_chars: int = 80) -> str:
    text = str(value or "").strip()
    if not text:
        return "not specified"
    text = " ".join(text.replace("\n", " ").split())
    return text[:max_chars]


def _seg_reader_support_phrase(raw_label: Any, name: str) -> str:
    if isinstance(raw_label, dict):
        support = _plain_reader_label(raw_label.get("support", "not specified"))
        fraction = raw_label.get("fraction")
        try:
            fraction_text = f"{float(fraction):.3f}"
        except Exception:
            fraction_text = "not specified"
        return f"{support} support (fraction {fraction_text})"
    return _clean_reader_field(raw_label, max_chars=120) or "not specified"


def _seg_reader_tumor_phrase(raw_label: Any, fractions: dict[str, float]) -> str:
    support = ""
    if isinstance(raw_label, dict):
        support = _plain_reader_label(raw_label.get("support", ""))
    label = _seg_reader_support_phrase(raw_label, "tumor")
    tumor_fraction = max(
        fractions.get("neoplastic", 0.0),
        fractions.get("tumor", 0.0),
        fractions.get("tumour", 0.0),
    )
    if tumor_fraction >= 0.60:
        return f"tumor-rich ROI (neoplastic fraction {tumor_fraction:.3f})"
    if tumor_fraction >= 0.20:
        return f"moderate tumor component (neoplastic fraction {tumor_fraction:.3f})"
    if tumor_fraction > 0:
        return f"limited tumor component (neoplastic fraction {tumor_fraction:.3f})"
    if support in {"strong", "moderate"}:
        return f"{support} visual tumor support, but compact segmentation tumor fraction is 0.000; treat cautiously"
    return label


def _seg_reader_necrosis_phrase(raw_label: Any, fractions: dict[str, float]) -> str:
    """Keep reader necrosis evidence proportional to the segmented fraction.

    The fine-tuned reader is useful for translating paired ROI+seg images, but
    the final report writer tends to amplify a boolean "present" into
    "extensive/geographic necrosis".  This helper gives the writer calibrated
    language tied to the compact mask fractions.
    """
    label = str(raw_label.get("support", "") if isinstance(raw_label, dict) else raw_label or "not specified").strip().lower()
    dead_fraction = max(
        [
            fractions.get("dead_or_necrotic", 0.0),
            fractions.get("necrosis", 0.0),
            fractions.get("dead", 0.0),
        ]
    )
    if label in {"absent", "no", "none", "not present"} or "absent" in label:
        return "absent/not supported"
    if dead_fraction >= 0.20:
        return f"prominent necrotic/dead component supported by segmentation fraction {dead_fraction:.3f}"
    if dead_fraction >= 0.05:
        return f"limited necrotic/dead component supported by segmentation fraction {dead_fraction:.3f}"
    if dead_fraction >= 0.005:
        return f"focal/trace necrotic-dead component only (fraction {dead_fraction:.3f}); do not describe as extensive"
    if "present" in label or "yes" in label or "suggest" in label:
        return "necrosis is uncertain because the dead/necrotic fraction is minimal or unspecified; do not describe as extensive"
    return _clean_reader_field(raw_label, max_chars=120) or "not specified"
