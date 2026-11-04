from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image, ImageDraw, ImageFont

from pathxai_agent.llm.qwen_client import QwenVisionClient

from .lazyslide_backend import lazyslide_tissue_tile_candidates
from .patch_store import cache_roi_crops
from .roi_candidates import RoiCandidate
from .slide_reader import SlideReader


ZOOM_REGIONS = {
    "center": (0.50, 0.50),
    "top-left": (0.25, 0.25),
    "top-right": (0.75, 0.25),
    "bottom-left": (0.25, 0.75),
    "bottom-right": (0.75, 0.75),
}


def qwen_coarse_to_fine_rois(
    slide_path: str | Path,
    slide_id: str,
    out_dir: Path,
    qwen_client: Optional[QwenVisionClient],
    coarse_roi_size: int = 8192,
    coarse_pool_rois: int = 24,
    coarse_rois_per_round: int = 6,
    max_coarse_rounds: int = 3,
    fine_roi_size: int = 4096,
    max_final_rois: int = 4,
    coarse_qwen_max_edge: int = 1024,
    mpp: Optional[float] = 0.5,
    dry_run: bool = False,
) -> tuple[list[dict], dict]:
    """Bounded coarse-to-fine ROI selection using LazySlide candidates and Qwen triage.

    Qwen only sees anonymized contact sheets with C01/C02 style labels.
    Full slide/case identifiers are saved in metadata but are not included in
    triage prompts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    coarse_dir = out_dir / "coarse_rois"
    coarse_qwen_dir = coarse_dir / "qwen"
    fine_dir = out_dir / "fine_rois"
    contact_dir = out_dir / "contact_sheets"
    for path in [coarse_qwen_dir, fine_dir, contact_dir]:
        path.mkdir(parents=True, exist_ok=True)

    pool, lazy_meta = lazyslide_tissue_tile_candidates(
        slide_path,
        slide_id=slide_id,
        patch_size=coarse_roi_size,
        mpp=mpp,
        max_rois=max(coarse_pool_rois * 2, coarse_rois_per_round),
    )
    pool = spatially_diverse(pool, coarse_pool_rois)
    coarse_records = []
    for idx, roi in enumerate(pool, start=1):
        record = _coarse_record(idx, roi)
        record["qwen_image_path"] = str(coarse_qwen_dir / f"{record['anon_id']}.jpg")
        coarse_records.append(record)

    selected_fine: list[RoiCandidate] = []
    triage_rounds = []
    used_coarse_ids: set[str] = set()

    for round_idx in range(max_coarse_rounds):
        start = round_idx * coarse_rois_per_round
        batch = coarse_records[start:start + coarse_rois_per_round]
        if not batch:
            break
        for record in batch:
            if not Path(record["qwen_image_path"]).exists():
                _save_qwen_view(slide_path, record["roi"], Path(record["qwen_image_path"]), coarse_qwen_max_edge)

        contact_path = contact_dir / f"coarse_round_{round_idx + 1:02d}.jpg"
        make_contact_sheet(
            [(record["anon_id"], Path(record["qwen_image_path"])) for record in batch],
            contact_path,
        )
        triage = dry_triage(batch) if dry_run or qwen_client is None else qwen_triage_contact_sheet(qwen_client, contact_path, batch)
        triage_rounds.append({
            "round": round_idx + 1,
            "contact_sheet": str(contact_path),
            "shown_rois": [_public_record(record) for record in batch],
            "qwen_triage": triage,
        })

        selected_fine.extend(
            fine_rois_from_triage(
                triage,
                {record["anon_id"]: record for record in batch},
                slide_id=slide_id,
                fine_roi_size=fine_roi_size,
                max_new=max_final_rois - len(selected_fine),
                used_coarse_ids=used_coarse_ids,
            )
        )
        if len(selected_fine) >= max_final_rois:
            break
        if not triage.get("need_more_coarse_regions", False) and selected_fine:
            break

    if not selected_fine:
        selected_fine = fallback_fine_rois(pool, slide_id, fine_roi_size, max_final_rois)

    selected_fine = selected_fine[:max_final_rois]
    cached_fine = cache_roi_crops(str(slide_path), slide_id, selected_fine, fine_dir)

    metadata = {
        "mode": "qwen_coarse_to_fine",
        "slide_path": str(slide_path),
        "lazy_meta": {k: v for k, v in lazy_meta.items() if k != "rois"},
        "settings": {
            "coarse_roi_size": coarse_roi_size,
            "coarse_pool_rois": coarse_pool_rois,
            "coarse_rois_per_round": coarse_rois_per_round,
            "max_coarse_rounds": max_coarse_rounds,
            "fine_roi_size": fine_roi_size,
            "max_final_rois": max_final_rois,
            "coarse_qwen_max_edge": coarse_qwen_max_edge,
            "mpp": mpp,
            "dry_run": dry_run,
        },
        "coarse_pool": [_public_record(record) for record in coarse_records],
        "triage_rounds": triage_rounds,
        "selected_fine_rois": [asdict(roi) for roi in selected_fine],
        "cached_fine_rois": cached_fine,
    }
    (out_dir / "roi_triage.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return cached_fine, metadata


def spatially_diverse(candidates: list[RoiCandidate], max_count: int) -> list[RoiCandidate]:
    candidates = sorted(candidates, key=lambda roi: roi.score, reverse=True)
    if not candidates:
        return []
    selected: list[RoiCandidate] = []
    min_dist = max(candidates[0].width, candidates[0].height) * 0.75
    for roi in candidates:
        if all(_center_dist(roi, prev) >= min_dist for prev in selected):
            selected.append(roi)
        if len(selected) >= max_count:
            return selected
    for roi in candidates:
        if roi not in selected:
            selected.append(roi)
        if len(selected) >= max_count:
            break
    return selected


def qwen_triage_contact_sheet(qwen_client: QwenVisionClient, contact_path: Path, batch: list[dict]) -> dict:
    ids = ", ".join(record["anon_id"] for record in batch)
    prompt = f"""/no_think
You are triaging low-resolution pathology ROI thumbnails from one WSI.
The contact sheet contains anonymized coarse regions labeled: {ids}.

Task:
Select regions that are most useful for later high-resolution zoom and report generation.
Do not make a final diagnosis from this overview.
Ignore filenames, case IDs, slide IDs, and coordinates. Use visual content only.

Prefer regions with abnormal cellular density, lesion/tumor-like tissue, necrosis,
heterogeneity, viable tissue, or possible lesion-normal interface.

Return ONLY valid JSON:
{{
  "need_more_coarse_regions": false,
  "rois": [
    {{
      "roi_id": "C01",
      "diagnostic_score": 0,
      "abnormality_score": 0,
      "tumor_score": 0,
      "quality_score": 0,
      "zoom_needed": true,
      "zoom_regions": ["center"],
      "reason": "short visual reason"
    }}
  ],
  "selected_roi_ids": ["C01"]
}}
Scores are integers from 0 to 5. Use zoom_regions from:
center, top-left, top-right, bottom-left, bottom-right.
Select at most 3 ROI IDs.
"""
    raw = qwen_client.chat_json(prompt, image_paths=[contact_path])
    triage = raw.get("json") or {}
    triage["_raw_content"] = raw.get("content", "")
    triage["_reasoning"] = raw.get("reasoning", "")
    return normalize_triage(triage, [record["anon_id"] for record in batch])


def normalize_triage(triage: dict, valid_ids: list[str]) -> dict:
    valid = set(valid_ids)
    rows = []
    for row in triage.get("rois") or []:
        rid = str(row.get("roi_id") or "").strip().upper()
        if rid not in valid:
            continue
        zoom_regions = [str(z).strip().lower() for z in (row.get("zoom_regions") or ["center"])]
        zoom_regions = [z for z in zoom_regions if z in ZOOM_REGIONS] or ["center"]
        rows.append({
            "roi_id": rid,
            "diagnostic_score": _score(row.get("diagnostic_score")),
            "abnormality_score": _score(row.get("abnormality_score")),
            "tumor_score": _score(row.get("tumor_score")),
            "quality_score": _score(row.get("quality_score")),
            "zoom_needed": bool(row.get("zoom_needed", True)),
            "zoom_regions": zoom_regions[:2],
            "reason": str(row.get("reason") or "")[:500],
        })
    if not rows:
        rows = [{"roi_id": rid, "diagnostic_score": 3, "abnormality_score": 3, "tumor_score": 2, "quality_score": 3, "zoom_needed": True, "zoom_regions": ["center"], "reason": "fallback"} for rid in valid_ids[:2]]
    rows = sorted(rows, key=triage_score, reverse=True)
    selected = [rid for rid in (triage.get("selected_roi_ids") or []) if str(rid).upper() in valid]
    if not selected:
        selected = [row["roi_id"] for row in rows[:3]]
    return {
        "need_more_coarse_regions": bool(triage.get("need_more_coarse_regions", False)),
        "rois": rows,
        "selected_roi_ids": selected[:3],
        "_raw_content": triage.get("_raw_content", ""),
        "_reasoning": triage.get("_reasoning", ""),
    }


def dry_triage(batch: list[dict]) -> dict:
    return normalize_triage({
        "need_more_coarse_regions": False,
        "rois": [
            {
                "roi_id": record["anon_id"],
                "diagnostic_score": 3,
                "abnormality_score": 3,
                "tumor_score": 2,
                "quality_score": 3,
                "zoom_needed": True,
                "zoom_regions": ["center"],
                "reason": "dry-run fallback triage",
            }
            for record in batch[:3]
        ],
    }, [record["anon_id"] for record in batch])


def fine_rois_from_triage(
    triage: dict,
    records_by_id: dict[str, dict],
    slide_id: str,
    fine_roi_size: int,
    max_new: int,
    used_coarse_ids: set[str],
) -> list[RoiCandidate]:
    out: list[RoiCandidate] = []
    ranked = sorted(triage.get("rois") or [], key=triage_score, reverse=True)
    selected_ids = set(str(x).upper() for x in triage.get("selected_roi_ids") or [])
    ranked = sorted(ranked, key=lambda row: (row["roi_id"] in selected_ids, triage_score(row)), reverse=True)
    for row in ranked:
        if len(out) >= max_new:
            break
        record = records_by_id.get(row["roi_id"])
        if record is None:
            continue
        for region in row.get("zoom_regions") or ["center"]:
            if len(out) >= max_new:
                break
            key = f"{row['roi_id']}:{region}"
            if key in used_coarse_ids:
                continue
            used_coarse_ids.add(key)
            out.append(fine_roi_from_coarse(record["roi"], slide_id, fine_roi_size, row["roi_id"], region, row))
    return out


def fine_roi_from_coarse(coarse: RoiCandidate, slide_id: str, fine_size: int, coarse_id: str, region: str, triage_row: dict) -> RoiCandidate:
    rx, ry = ZOOM_REGIONS.get(region, ZOOM_REGIONS["center"])
    center_x = coarse.x + int(coarse.width * rx)
    center_y = coarse.y + int(coarse.height * ry)
    x = max(0, center_x - fine_size // 2)
    y = max(0, center_y - fine_size // 2)
    roi_id = f"{slide_id}_fine_{coarse_id}_{region.replace('-', '')}_x{x}_y{y}_l0_w{fine_size}_h{fine_size}"
    return RoiCandidate(
        slide_id=slide_id,
        roi_id=roi_id,
        x=x,
        y=y,
        width=fine_size,
        height=fine_size,
        level=0,
        source=f"qwen_coarse_to_fine:{coarse_id}:{region}",
        score=triage_score(triage_row),
    )


def fallback_fine_rois(pool: list[RoiCandidate], slide_id: str, fine_size: int, max_count: int) -> list[RoiCandidate]:
    return [
        fine_roi_from_coarse(roi, slide_id, fine_size, f"C{idx:02d}", "center", {"diagnostic_score": 1, "abnormality_score": 1, "tumor_score": 1, "quality_score": 1})
        for idx, roi in enumerate(pool[:max_count], start=1)
    ]


def make_contact_sheet(items: list[tuple[str, Path]], out_path: Path, thumb_size: int = 384, cols: int = 3) -> None:
    rows = int(math.ceil(len(items) / cols))
    label_h = 34
    sheet = Image.new("RGB", (cols * thumb_size, rows * (thumb_size + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    for idx, (label, path) in enumerate(items):
        col = idx % cols
        row = idx // cols
        x = col * thumb_size
        y = row * (thumb_size + label_h)
        img = Image.open(path).convert("RGB")
        img.thumbnail((thumb_size, thumb_size))
        tile = Image.new("RGB", (thumb_size, thumb_size), "white")
        tile.paste(img, ((thumb_size - img.width) // 2, (thumb_size - img.height) // 2))
        sheet.paste(tile, (x, y + label_h))
        draw.rectangle((x, y, x + thumb_size, y + label_h), fill=(245, 245, 245))
        draw.text((x + 10, y + 4), label, fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, "JPEG", quality=90)


def _save_qwen_view(slide_path: str | Path, roi: RoiCandidate, out_path: Path, max_edge: int) -> None:
    reader = SlideReader(slide_path)
    try:
        image = reader.read_region(roi.x, roi.y, roi.width, roi.height, roi.level)
    finally:
        reader.close()
    image.thumbnail((max_edge, max_edge))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, "JPEG", quality=82, optimize=True)


def _coarse_record(idx: int, roi: RoiCandidate) -> dict:
    anon_id = f"C{idx:02d}"
    return {
        "anon_id": anon_id,
        "roi": roi,
        "qwen_image_path": "",
        "score": roi.score,
    }


def _public_record(record: dict) -> dict:
    roi = record["roi"]
    return {
        "anon_id": record["anon_id"],
        "x": roi.x,
        "y": roi.y,
        "width": roi.width,
        "height": roi.height,
        "level": roi.level,
        "source": roi.source,
        "score": roi.score,
        "qwen_image_path": record.get("qwen_image_path"),
    }


def _center_dist(a: RoiCandidate, b: RoiCandidate) -> float:
    ax = a.x + a.width / 2.0
    ay = a.y + a.height / 2.0
    bx = b.x + b.width / 2.0
    by = b.y + b.height / 2.0
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def triage_score(row: dict) -> float:
    return (
        0.40 * float(row.get("diagnostic_score", 0))
        + 0.30 * float(row.get("abnormality_score", 0))
        + 0.20 * float(row.get("tumor_score", 0))
        + 0.10 * float(row.get("quality_score", 0))
    )


def _score(value: Any) -> int:
    try:
        return max(0, min(5, int(round(float(value)))))
    except Exception:
        return 0
