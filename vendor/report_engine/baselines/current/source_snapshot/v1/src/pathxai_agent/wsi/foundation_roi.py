from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .lazyslide_backend import (
    LazySlideUnavailable,
    _import_lazyslide,
    _iter_rows,
    _jsonable_properties,
    _shape_bounds,
)
from .patch_store import cache_roi_crops
from .roi_candidates import RoiCandidate


DEFAULT_POSITIVE_PROMPTS = [
    "histopathology image of malignant tumor",
    "histopathology image of invasive carcinoma",
    "pathology image showing cancer cells",
    "pathology image with cellular atypia and high tumor cellularity",
    "pathology image showing tumor necrosis",
    "pathology image showing tumor invasion at tissue interface",
]

DEFAULT_NEGATIVE_PROMPTS = [
    "normal benign histology tissue",
    "blank background or glass slide",
    "adipose tissue without tumor",
    "poor quality artifact out of focus tissue",
]

DEFAULT_SITE_PROMPTS = [
    {"site": "adrenal gland / adrenocortical tissue", "prompt": "H&E histopathology image of adrenal gland or adrenocortical tissue"},
    {"site": "paraganglia / pheochromocytoma tissue", "prompt": "H&E histopathology image of paraganglia or pheochromocytoma tissue"},
    {"site": "bladder / urothelial tissue", "prompt": "H&E histopathology image of bladder urothelium or urothelial carcinoma"},
    {"site": "breast tissue", "prompt": "H&E histopathology image of breast tissue or breast carcinoma"},
    {"site": "cervix / cervical tissue", "prompt": "H&E histopathology image of cervix or cervical squamous glandular tissue"},
    {"site": "colon or rectum / colorectal mucosa", "prompt": "H&E histopathology image of colon or rectal colorectal mucosa"},
    {"site": "esophagus / squamous or glandular mucosa", "prompt": "H&E histopathology image of esophageal squamous or glandular mucosa"},
    {"site": "stomach / gastric mucosa", "prompt": "H&E histopathology image of stomach or gastric mucosa"},
    {"site": "head and neck squamous mucosa", "prompt": "H&E histopathology image of head and neck squamous mucosa"},
    {"site": "kidney / renal tissue", "prompt": "H&E histopathology image of kidney renal parenchyma or renal cell carcinoma"},
    {"site": "liver / hepatic tissue", "prompt": "H&E histopathology image of liver hepatic tissue or hepatocellular carcinoma"},
    {"site": "bile duct / cholangiocarcinoma tissue", "prompt": "H&E histopathology image of bile duct or cholangiocarcinoma tissue"},
    {"site": "lung tissue", "prompt": "H&E histopathology image of lung parenchyma or lung carcinoma"},
    {"site": "pleura / mesothelium", "prompt": "H&E histopathology image of pleura or mesothelial tissue"},
    {"site": "brain / glial tissue", "prompt": "H&E histopathology image of brain glial tissue or glioma"},
    {"site": "ovary / ovarian tissue", "prompt": "H&E histopathology image of ovary or ovarian tumor tissue"},
    {"site": "uterus / endometrium", "prompt": "H&E histopathology image of uterus or endometrial tissue"},
    {"site": "prostate tissue", "prompt": "H&E histopathology image of prostate gland tissue or prostate adenocarcinoma"},
    {"site": "skin / melanocytic tissue", "prompt": "H&E histopathology image of skin or melanocytic melanoma tissue"},
    {"site": "soft tissue / sarcoma", "prompt": "H&E histopathology image of soft tissue spindle cell or sarcoma tissue"},
    {"site": "testis / germ cell tissue", "prompt": "H&E histopathology image of testis or germ cell tumor tissue"},
    {"site": "thyroid tissue", "prompt": "H&E histopathology image of thyroid follicular tissue or thyroid carcinoma"},
    {"site": "thymus tissue", "prompt": "H&E histopathology image of thymus or thymic epithelial tumor tissue"},
    {"site": "eye / uveal melanocytic tissue", "prompt": "H&E histopathology image of uveal melanoma or ocular melanocytic tissue"},
    {"site": "lymphoid tissue", "prompt": "H&E histopathology image of lymph node lymphoid tissue or lymphoma"},
]

FOUNDATION_RANKING_VERSION = "20260625_site_type_tumor_v2"


def foundation_text_similarity_rois(
    slide_path: str | Path,
    slide_id: str,
    out_dir: Path,
    model_name: str = "conch",
    model_path: str | None = None,
    token: str | None = None,
    tile_size: int = 1024,
    final_roi_size: int = 4096,
    max_final_rois: int = 4,
    mpp: Optional[float] = 0.5,
    positive_prompts: list[str] | None = None,
    negative_prompts: list[str] | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str | None = None,
    amp: bool | None = None,
    max_tiles: int | None = None,
    diversity_factor: float = 0.75,
    site_prompts: list[dict[str, str]] | None = None,
    site_top_k_tiles: int = 8,
    ranking_mode: str = "diagnostic",
    site_roi_fraction: float = 0.4,
    max_site_rois: int = 3,
    max_tumor_rois: int = 5,
    pbar: bool | None = True,
) -> tuple[list[dict], dict]:
    """Rank LazySlide tissue tiles with an image-text pathology foundation model.

    This is a cheap first-stage selector: it uses local image/text embeddings to
    retrieve tumor-like tiles, then crops larger final ROIs around the selected
    tile centers for report generation.
    """
    if model_name not in {"conch", "plip", "omiclip", "titan"}:
        raise ValueError("foundation text-image ROI selection supports: conch, plip, omiclip, titan")
    if ranking_mode not in {"diagnostic", "site_tumor", "separate_site_tumor", "site_type_tumor"}:
        raise ValueError("ranking_mode must be 'diagnostic', 'site_tumor', 'separate_site_tumor', or 'site_type_tumor'")

    zs, open_wsi = _import_lazyslide()
    out_dir.mkdir(parents=True, exist_ok=True)
    tile_key = "foundation_tiles"
    feature_key = f"{model_name}_{tile_key}"
    final_dir = out_dir / "foundation_rois"
    final_dir.mkdir(parents=True, exist_ok=True)

    slide_path = Path(slide_path)
    wsi = open_wsi(str(slide_path), store=None)
    metadata: dict[str, Any] = {
        "mode": "foundation_text_similarity",
        "slide_path": str(slide_path),
        "model_name": model_name,
        "model_path": model_path,
        "tile_key": tile_key,
        "feature_key": feature_key,
        "settings": {
            "tile_size": tile_size,
            "final_roi_size": final_roi_size,
            "max_final_rois": max_final_rois,
            "mpp": mpp,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "device": device,
            "amp": amp,
            "max_tiles": max_tiles,
            "diversity_factor": diversity_factor,
            "site_top_k_tiles": site_top_k_tiles,
            "ranking_mode": ranking_mode,
            "site_roi_fraction": site_roi_fraction,
            "max_site_rois": max_site_rois,
            "max_tumor_rois": max_tumor_rois,
            "ranking_version": FOUNDATION_RANKING_VERSION,
        },
    }
    props = getattr(wsi, "properties", None)
    if props is not None:
        metadata["properties"] = _jsonable_properties(props)

    try:
        zs.pp.find_tissues(wsi, key_added="foundation_tissues")
        tile_kwargs: dict[str, Any] = {"key_added": tile_key, "tissue_key": "foundation_tissues"}
        if mpp is not None:
            tile_kwargs["mpp"] = mpp
        try:
            zs.pp.tile_tissues(wsi, tile_size, **tile_kwargs)
        except TypeError:
            tile_kwargs.pop("tissue_key", None)
            zs.pp.tile_tissues(wsi, tile_size, **tile_kwargs)

        tiles = list(_iter_rows(wsi[tile_key]))
        if max_tiles is not None and max_tiles > 0:
            tiles = tiles[:max_tiles]
            try:
                wsi.shapes[tile_key] = wsi.shapes[tile_key].iloc[:max_tiles].copy()
            except Exception:
                pass
        metadata["tile_count"] = len(tiles)
        if not tiles:
            raise ValueError(f"LazySlide produced no tissue tiles for {slide_path}")

        model = _load_image_text_model(model_name, model_path=model_path, token=token)
        features = zs.tl.feature_extraction(
            wsi,
            model=model,
            model_name=model_name,
            tile_key=tile_key,
            key_added=feature_key,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            amp=amp,
            pbar=pbar,
            return_features=True,
        )
        prompts = list(positive_prompts or DEFAULT_POSITIVE_PROMPTS)
        neg_prompts = list(negative_prompts or DEFAULT_NEGATIVE_PROMPTS)

        # ── Text encoder selection ─────────────────────────────────────────────
        # TITAN's patch encoder (CONCHv1.5) produces 768-dim features.
        # Its text encoder also operates in 768-dim space.  Use TITAN's own
        # zero_shot_classifier() to get aligned text embeddings for patch scoring.
        # Other models (CONCH v1, PLIP, etc.) use the standard _encode_text path.
        is_titan = hasattr(model, "encode_slide") and callable(
            getattr(model, "encode_slide", None)
        )
        if is_titan:
            try:
                import torch as _torch
                # Rebuild attn_mask buffer — PyTorch 2.12+ bug with persistent=False
                _fix_titan_attn_mask(model, device)
                all_prompts = prompts + neg_prompts
                # zero_shot_classifier expects list-of-lists: [["prompt1"], ["prompt2"], ...]
                # Use "CLASSNAME" template so the prompt text is used verbatim.
                # Returns (embed_dim, n_prompts); transpose → (n_prompts, embed_dim)
                classifier = model.model.zero_shot_classifier([[p] for p in all_prompts], ["CLASSNAME"], device=device)
                text_embeddings = classifier.T.detach().cpu().numpy()
                scores = score_tiles(features, text_embeddings, positive_count=len(prompts))
            except Exception as exc:
                raise RuntimeError(
                    "TITAN tile scoring failed. Refusing to continue with all-zero "
                    f"ROI scores because that would make ROI selection arbitrary: {exc}"
                ) from exc
        else:
            text_embeddings = _encode_text(model, prompts + neg_prompts, device=device, amp=amp)
            scores = score_tiles(features, text_embeddings, positive_count=len(prompts))

        # ── Slide-level site prediction ────────────────────────────────────────
        tile_geoms = [row.get("geometry") for row in tiles]
        if is_titan:
            site_prediction = _titan_slide_site_prediction(
                model=model,
                features=features,
                tile_geoms=tile_geoms,
                tile_size=tile_size,
                site_prompts=site_prompts or DEFAULT_SITE_PROMPTS,
                device=device,
            )
        else:
            site_prediction = predict_site_from_top_tiles(
                features=features,
                tumor_scores=scores,
                model=model,
                site_prompts=site_prompts or DEFAULT_SITE_PROMPTS,
                top_k_tiles=site_top_k_tiles,
                device=device,
                amp=amp,
            )
        if ranking_mode in {"site_tumor", "separate_site_tumor"}:
            target_site = None
            if (site_prediction.get("confidence") or "").lower() in {"high", "medium"}:
                target_site = site_prediction.get("top_site")
            site_scores, site_score_metadata = score_site_discriminative_tiles(
                features=features,
                model=model,
                site_prompts=site_prompts or DEFAULT_SITE_PROMPTS,
                device=device,
                amp=amp,
                is_titan=is_titan,
                target_site=target_site,
            )
        else:
            site_scores = None
            site_score_metadata = None

        if ranking_mode == "site_type_tumor":
            site_type_scores, site_type_metadata = score_slide_site_types(
                features=features,
                model=model,
                site_prompts=site_prompts or DEFAULT_SITE_PROMPTS,
                tumor_scores=scores,
                device=device,
                amp=amp,
                is_titan=is_titan,
            )
            site_ranked, tumor_ranked = build_site_type_tumor_rois(
                slide_id=slide_id,
                tiles=tiles,
                tumor_scores=scores,
                site_support_scores=site_type_scores,
                tile_size=tile_size,
                final_roi_size=final_roi_size,
                max_site_rois=max_site_rois,
                max_tumor_rois=max_tumor_rois,
                diversity_factor=diversity_factor,
            )
            site_cached = cache_roi_crops(str(slide_path), slide_id, site_ranked, out_dir / "site_support_rois")
            tumor_cached = cache_roi_crops(str(slide_path), slide_id, tumor_ranked, out_dir / "tumor_rois")
            _attach_site_tumor_scores(site_cached, site_ranked, evidence_role="site_support")
            _attach_site_tumor_scores(tumor_cached, tumor_ranked, evidence_role="tumor")
            _attach_site_type_metadata(site_cached, site_ranked, site_type_metadata)
            _attach_site_type_metadata(tumor_cached, tumor_ranked, site_type_metadata)
            cached = site_cached + tumor_cached
            ranked = site_ranked + tumor_ranked
            metadata.update({
                "slide_site_candidates": site_type_metadata.get("slide_site_candidates") or [],
                "all_roi_site_scores_path": str(out_dir / "all_roi_site_scores.json"),
                "site_support_rois": [_roi_candidate_payload(roi) for roi in site_ranked],
                "tumor_rois": [_roi_candidate_payload(roi) for roi in tumor_ranked],
                "cached_site_support_rois": site_cached,
                "cached_tumor_rois": tumor_cached,
                "site_type_score_metadata": site_type_metadata,
            })
            (out_dir / "all_roi_site_scores.json").write_text(
                json.dumps(site_type_metadata.get("all_roi_site_scores") or [], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (out_dir / "slide_site_candidates.json").write_text(
                json.dumps(site_type_metadata.get("slide_site_candidates") or [], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (out_dir / "site_support_rois" / "rois.json").write_text(
                json.dumps(site_cached, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (out_dir / "tumor_rois" / "rois.json").write_text(
                json.dumps(tumor_cached, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            site_scores = site_type_scores
            site_score_metadata = {
                "available": True,
                "method": "slide_site_type_aggregation",
                "slide_site_candidates": site_type_metadata.get("slide_site_candidates") or [],
                "all_roi_site_scores_path": str(out_dir / "all_roi_site_scores.json"),
            }
            if site_type_metadata.get("slide_site_candidates"):
                top_site = site_type_metadata["slide_site_candidates"][0]
                site_prediction = {
                    "available": True,
                    "method": "slide_site_type_aggregation",
                    "top_site": top_site.get("site"),
                    "top_prompt": top_site.get("prompt"),
                    "top_score": top_site.get("aggregate_score"),
                    "margin_to_second": top_site.get("margin_to_second"),
                    "confidence": top_site.get("confidence"),
                    "scores": site_type_metadata.get("slide_site_candidates") or [],
                }
        elif ranking_mode == "separate_site_tumor":
            site_ranked, tumor_ranked = build_separate_site_tumor_rois(
                slide_id=slide_id,
                tiles=tiles,
                tumor_scores=scores,
                site_scores=site_scores,
                tile_size=tile_size,
                final_roi_size=final_roi_size,
                max_site_rois=max_site_rois,
                max_tumor_rois=max_tumor_rois,
                diversity_factor=diversity_factor,
                use_site_rois=bool(site_score_metadata.get("available")),
            )
            site_cached = cache_roi_crops(str(slide_path), slide_id, site_ranked, out_dir / "site_rois")
            tumor_cached = cache_roi_crops(str(slide_path), slide_id, tumor_ranked, out_dir / "tumor_rois")
            _attach_site_tumor_scores(site_cached, site_ranked, evidence_role="site")
            _attach_site_tumor_scores(tumor_cached, tumor_ranked, evidence_role="tumor")
            cached = site_cached + tumor_cached
            ranked = site_ranked + tumor_ranked
            metadata.update({
                "site_rois": [_roi_candidate_payload(roi) for roi in site_ranked],
                "tumor_rois": [_roi_candidate_payload(roi) for roi in tumor_ranked],
                "cached_site_rois": site_cached,
                "cached_tumor_rois": tumor_cached,
            })
            (out_dir / "site_rois" / "rois.json").write_text(
                json.dumps(site_cached, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (out_dir / "tumor_rois" / "rois.json").write_text(
                json.dumps(tumor_cached, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        elif ranking_mode == "site_tumor":
            effective_site_roi_fraction = site_roi_fraction
            if not site_score_metadata.get("available"):
                effective_site_roi_fraction = 0.0
            ranked = build_site_tumor_ranked_rois(
                slide_id=slide_id,
                tiles=tiles,
                tumor_scores=scores,
                site_scores=site_scores,
                tile_size=tile_size,
                final_roi_size=final_roi_size,
                max_final_rois=max_final_rois,
                diversity_factor=diversity_factor,
                site_roi_fraction=effective_site_roi_fraction,
            )
            cached = cache_roi_crops(str(slide_path), slide_id, ranked, final_dir)
            _attach_site_tumor_scores(cached, ranked)
            (final_dir / "rois.json").write_text(
                json.dumps(cached, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            ranked = build_ranked_rois(
                slide_id=slide_id,
                tiles=tiles,
                scores=scores,
                tile_size=tile_size,
                final_roi_size=final_roi_size,
                max_final_rois=max_final_rois,
                diversity_factor=diversity_factor,
            )
            cached = cache_roi_crops(str(slide_path), slide_id, ranked, final_dir)
        metadata.update({
            "positive_prompts": prompts,
            "negative_prompts": neg_prompts,
            "site_prediction": site_prediction,
            "selected_rois": [_roi_candidate_payload(roi) for roi in ranked],
            "cached_rois": cached,
            "top_scored_tiles": [row for row in scored_tile_records(slide_id, tiles, scores, tile_size)[:50]],
            "ranking_mode": ranking_mode,
        })
        if ranking_mode in {"site_tumor", "separate_site_tumor", "site_type_tumor"}:
            metadata["site_score_metadata"] = site_score_metadata
            metadata["top_site_discriminative_tiles"] = [
                row for row in scored_tile_records(slide_id, tiles, site_scores, tile_size)[:50]
            ]
        (out_dir / "foundation_roi_ranking.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return cached, metadata
    except LazySlideUnavailable:
        raise
    except Exception as exc:
        metadata["error"] = str(exc)
        (out_dir / "foundation_roi_error.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise


def _fix_titan_attn_mask(model: Any, device: str) -> None:
    """Rebuild the causal attention mask buffer on the correct device.

    In PyTorch 2.12+, persistent=False buffers are not properly initialized
    after .to(device) — they contain NaN/garbage instead of -inf/0.
    This rebuilds the causal mask so text encoding works correctly.
    """
    import torch
    te = getattr(getattr(model, "model", None), "text_encoder", None)
    if te is None:
        return
    if not hasattr(te, "num_pos") or not hasattr(te, "attn_mask"):
        return
    if te.attn_mask is None:
        return
    num_pos = te.num_pos
    mask = torch.empty(num_pos, num_pos, dtype=torch.float32, device=device)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # lower triangle = 0, upper triangle = -inf
    te.register_buffer("attn_mask", mask, persistent=False)


def _titan_slide_site_prediction(
    model: Any,
    features: np.ndarray,
    tile_geoms: list,
    tile_size: int,
    site_prompts: list[dict[str, str]],
    device: str | None = None,
) -> dict:
    """Slide-level zero-shot site prediction using TITAN's slide encoder + text scorer.

    TITAN aggregates ALL tile (CONCHv1.5) features through a transformer to produce a
    single slide embedding that captures tissue architecture. It then scores site
    class labels against the slide embedding using its language encoder.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [
        {"site": str(r.get("site") or "").strip(), "prompt": str(r.get("prompt") or "").strip()}
        for r in site_prompts
        if str(r.get("site") or "").strip()
    ]
    if not rows or len(features) == 0:
        return {"available": False, "reason": "no site prompts or features", "method": "titan_slide_level"}

    # Build coords tensor (top-left corners at level-0) from tile geometries
    coords_list = []
    patch_sizes = []
    for geom in tile_geoms:
        if geom is None or not hasattr(geom, "bounds"):
            continue
        b = geom.bounds  # (min_x, min_y, max_x, max_y) at level-0
        coords_list.append([int(b[0]), int(b[1])])
        patch_sizes.append(int(b[2] - b[0]))
    if not coords_list:
        return {"available": False, "reason": "could not extract tile coordinates from geometries", "method": "titan_slide_level"}

    patch_size_lv0 = int(np.median(patch_sizes))  # level-0 tile width (pixels)
    feat_arr = features[: len(coords_list)]        # align in case of mismatch
    feat_tensor = torch.from_numpy(feat_arr).float().to(device)
    coords_tensor = torch.tensor(coords_list, dtype=torch.long).to(device)

    try:
        model.to(device)
        # Rebuild attn_mask buffer — PyTorch 2.12+ bug with persistent=False buffers
        _fix_titan_attn_mask(model, device)
        # encode_slide_from_patch_features expects (1, N, C) and (1, N, 2)
        feat_batch = feat_tensor.unsqueeze(0)    # (1, N, C)
        coords_batch = coords_tensor.unsqueeze(0)  # (1, N, 2)
        # Use no_grad (not inference_mode) so Parameters can participate without error
        with torch.no_grad():
            slide_emb = model.model.encode_slide_from_patch_features(
                feat_batch, coords_batch, np.int64(patch_size_lv0)
            )
        # slide_emb shape from TITAN: (1, D)

        # zero_shot_classifier expects list-of-lists: [["class1"], ["class2"], ...]
        site_labels = [r["site"] for r in rows]
        with torch.no_grad():
            classifier = model.model.zero_shot_classifier(
                [[s] for s in site_labels], model.TEMPLATES, device=device
            )
            score_vec = model.model.zero_shot(slide_emb, classifier)
        # score_vec: (1, 1, N_classes) — squeeze all dims
        if hasattr(score_vec, "detach"):
            score_vec = score_vec.detach().cpu().numpy()
        scores_arr = np.asarray(score_vec).squeeze()  # (N_classes,)

        order = np.argsort(scores_arr)[::-1]
        ranked = [
            {
                "rank": int(rank + 1),
                "site": rows[idx]["site"],
                "prompt": rows[idx]["prompt"],
                "score": float(scores_arr[idx]),
            }
            for rank, idx in enumerate(order)
        ]
        margin = float(ranked[0]["score"] - ranked[1]["score"]) if len(ranked) > 1 else None
        if margin is None or margin >= 0.08:
            confidence = "high"
        elif margin >= 0.04:
            confidence = "medium"
        else:
            confidence = "low"
        return {
            "available": True,
            "method": "titan_slide_level_zero_shot",
            "top_site": ranked[0]["site"],
            "top_score": float(ranked[0]["score"]),
            "margin_to_second": margin,
            "confidence": confidence,
            "patch_size_lv0": patch_size_lv0,
            "n_tiles": len(coords_list),
            "scores": ranked,
        }
    except Exception as exc:
        return {
            "available": False,
            "method": "titan_slide_level_zero_shot",
            "reason": f"encode_slide/score failed: {exc}",
        }


def _load_image_text_model(model_name: str, model_path: str | None = None, token: str | None = None):
    from lazyslide.models import MODEL_REGISTRY

    if model_name == "plip":
        cls = _build_compatible_plip_class()
        return cls(model_path=model_path, token=token)
    cls = MODEL_REGISTRY[model_name]
    return cls(model_path=model_path, token=token)


def _build_compatible_plip_class():
    """Build _CompatiblePLIP lazily to avoid metaclass issues at import time.

    Newer lazyslide versions require ``key=`` and ``task=`` at class-definition
    time. We defer class creation until first use so the import always succeeds
    even when lazyslide is unavailable.
    """
    from lazyslide.models.base import ImageTextModel, ModelTask

    class _CompatiblePLIPImpl(ImageTextModel, key="_compat_plip"):
        """PLIP wrapper that uses the current ``token=`` Transformers API.

        Some LazySlide releases still call ``from_pretrained(..., use_auth_token=...)``
        for PLIP. Recent Transformers versions removed that keyword for CLIPModel,
        so this local shim keeps the same LazySlide image/text interface using the
        current ``token=`` argument.
        """

        task = ModelTask.multimodal

        def __init__(self, model_path=None, token=None):
            import torch
            from torchvision.transforms.functional import resize
            from transformers import CLIPModel, CLIPProcessor

            self._torch = torch
            self._resize = resize
            self.model_path = model_path or "vinid/plip"
            kwargs = {"token": token} if token else {}
            self.model = CLIPModel.from_pretrained(self.model_path, **kwargs)
            self.model.eval()
            self.processor = CLIPProcessor.from_pretrained(self.model_path, **kwargs)

        def get_transform(self):
            return None

        def to(self, device):
            self.model.to(device)
            return self

        @property
        def name(self):
            return "CompatiblePLIP"

        def encode_image(self, image):
            torch = self._torch
            with torch.inference_mode():
                if isinstance(image, torch.Tensor):
                    pixel_values = self._preprocess_tensor_batch(image.to(self.model.device))
                    output = self.model.get_image_features(pixel_values=pixel_values)
                else:
                    inputs = self.processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    output = self.model.get_image_features(**inputs)
                if hasattr(output, "pooler_output"):
                    output = output.pooler_output
                return output

        def _preprocess_tensor_batch(self, image):
            torch = self._torch
            image = image.float()
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if image.ndim == 4 and image.shape[1] not in {1, 3, 4} and image.shape[-1] in {1, 3, 4}:
                image = image.permute(0, 3, 1, 2).contiguous()
            if image.ndim != 4:
                raise ValueError(f"Expected PLIP image tensor with 3 or 4 dims, got shape {tuple(image.shape)}")
            if image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)
            elif image.shape[1] == 4:
                image = image[:, :3]
            elif image.shape[1] != 3:
                raise ValueError(f"Expected PLIP image tensor with 1/3/4 channels, got shape {tuple(image.shape)}")
            if image.max() > 2.0:
                image = image / 255.0
            if image.shape[-2:] != (224, 224):
                image = self._resize(image, [224, 224], antialias=True)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image.device).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image.device).view(1, 3, 1, 1)
            return (image - mean) / std

        def encode_text(self, text):
            torch = self._torch
            with torch.inference_mode():
                inputs = self.processor(
                    text=text,
                    return_tensors="pt",
                    max_length=77,
                    padding="max_length",
                    truncation=True,
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                text_features = self.model.get_text_features(**inputs)
                if hasattr(text_features, "pooler_output"):
                    text_features = text_features.pooler_output
                return torch.nn.functional.normalize(text_features, p=2, dim=-1)

    return _CompatiblePLIPImpl


def _encode_text(model: Any, texts: list[str], device: str | None = None, amp: bool | None = None) -> np.ndarray:
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model.to(device)
    except Exception:
        pass
    amp = bool(amp)
    autocast_dtype = torch.float16
    device_type = str(device).split(":", 1)[0]
    amp_ctx = torch.autocast(device_type, autocast_dtype) if amp else torch.no_grad()
    with amp_ctx, torch.inference_mode():
        try:
            embeddings = model.encode_text(texts)
        except AttributeError as exc:
            if "batch_encode_plus" not in str(exc) or not hasattr(model, "tokenizer"):
                raise
            tokens = _manual_conch_tokens(model.tokenizer, texts).to(device)
            embeddings = model.model.encode_text(tokens)
    if not isinstance(embeddings, np.ndarray):
        embeddings = embeddings.detach().cpu().numpy()
    return embeddings


def _manual_conch_tokens(tokenizer: Any, texts: list[str], context_length: int = 128):
    import torch

    rows = []
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    max_text_len = context_length - 1
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)
        ids = ids[:max_text_len]
        ids = ids + [pad_id] * (max_text_len - len(ids))
        ids.append(pad_id)
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


def score_tiles(features: np.ndarray, text_embeddings: np.ndarray, positive_count: int) -> np.ndarray:
    image = l2_normalize(features)
    text = l2_normalize(text_embeddings)
    sims = image @ text.T
    pos = sims[:, :positive_count]
    neg = sims[:, positive_count:]
    positive_score = 0.70 * pos.max(axis=1) + 0.30 * pos.mean(axis=1)
    negative_score = neg.max(axis=1) if neg.size else 0.0
    return positive_score - 0.35 * negative_score


def score_slide_site_types(
    features: np.ndarray,
    model: Any,
    site_prompts: list[dict[str, str]],
    tumor_scores: np.ndarray,
    device: str | None = None,
    amp: bool | None = None,
    is_titan: bool = False,
    top_k_sites_per_tile: int = 5,
    top_k_tiles_per_site: int = 20,
) -> tuple[np.ndarray, dict]:
    """Score all tissue tiles against all site prompts, then aggregate to slide site types.

    This is the v2 site-selection path. It does not require a prior target
    organ. Instead, every tile gets a zero-shot site ranking; slide-level site
    candidates are produced by aggregating the strongest supporting tiles for
    each site. The saved support ROIs are only explainability evidence; the
    actual site hypothesis is the aggregated ``slide_site_candidates`` list.
    """
    rows = [
        {"site": str(row.get("site") or "").strip(), "prompt": str(row.get("prompt") or "").strip()}
        for row in site_prompts
        if str(row.get("site") or "").strip() and str(row.get("prompt") or "").strip()
    ]
    if not rows or len(features) == 0:
        return np.zeros((len(features),), dtype=np.float32), {
            "available": False,
            "reason": "no site prompts or image features",
            "slide_site_candidates": [],
            "all_roi_site_scores": [],
        }

    if is_titan:
        text_embeddings = _encode_titan_text_embeddings(model, [row["site"] for row in rows], device=device)
    else:
        text_embeddings = _encode_text(model, [row["prompt"] for row in rows], device=device, amp=amp)

    image = l2_normalize(features)
    text = l2_normalize(text_embeddings)
    sims = image @ text.T
    order = np.argsort(sims, axis=1)[:, ::-1]
    top_idx = order[:, 0]
    top_score = sims[np.arange(sims.shape[0]), top_idx]
    second_score = sims[np.arange(sims.shape[0]), order[:, 1]] if sims.shape[1] > 1 else np.zeros_like(top_score)
    margin = top_score - second_score
    support_scores = 0.55 * _robust_unit(top_score) + 0.45 * _robust_unit(margin)

    top_k_sites_per_tile = max(1, min(int(top_k_sites_per_tile), len(rows)))
    all_roi_site_scores = []
    for tile_idx in range(sims.shape[0]):
        top_sites = []
        for rank, site_idx in enumerate(order[tile_idx, :top_k_sites_per_tile], start=1):
            top_sites.append({
                "rank": int(rank),
                "site": rows[int(site_idx)]["site"],
                "prompt": rows[int(site_idx)]["prompt"],
                "score": float(sims[tile_idx, int(site_idx)]),
            })
        all_roi_site_scores.append({
            "tile_index": int(tile_idx),
            "top_site": rows[int(top_idx[tile_idx])]["site"],
            "top_site_score": float(top_score[tile_idx]),
            "second_site_score": float(second_score[tile_idx]),
            "margin_to_second": float(margin[tile_idx]),
            "site_support_score": float(support_scores[tile_idx]),
            "top_k_sites": top_sites,
            "tumor_score": float(tumor_scores[tile_idx]) if tile_idx < len(tumor_scores) else None,
        })

    tumor_weight = _robust_unit(tumor_scores) if len(tumor_scores) == sims.shape[0] else np.ones((sims.shape[0],), dtype=np.float32)
    site_candidates = []
    top_k_tiles_per_site = max(1, min(int(top_k_tiles_per_site), sims.shape[0]))
    for site_idx, row in enumerate(rows):
        site_sim = sims[:, site_idx]
        site_order = np.argsort(site_sim)[::-1]
        top_tiles = site_order[:top_k_tiles_per_site]
        top_tile_scores = site_sim[top_tiles]
        top_tile_tumor = tumor_weight[top_tiles]
        aggregate = float(0.65 * top_tile_scores.mean() + 0.25 * top_tile_scores.max() + 0.10 * top_tile_tumor.mean())
        site_candidates.append({
            "site": row["site"],
            "prompt": row["prompt"],
            "aggregate_score": aggregate,
            "max_tile_similarity": float(top_tile_scores.max()),
            "mean_top_tile_similarity": float(top_tile_scores.mean()),
            "mean_top_tile_tumor_score": float(top_tile_tumor.mean()),
            "supporting_tile_indices": [int(idx) for idx in top_tiles[:8].tolist()],
        })
    site_candidates.sort(key=lambda row: row["aggregate_score"], reverse=True)
    for rank, row in enumerate(site_candidates, start=1):
        row["rank"] = int(rank)
        second = site_candidates[1]["aggregate_score"] if len(site_candidates) > 1 else None
        row["margin_to_second"] = float(row["aggregate_score"] - second) if rank == 1 and second is not None else None
        if rank == 1:
            margin_value = row["margin_to_second"]
            if margin_value is None or margin_value >= 0.04:
                row["confidence"] = "high"
            elif margin_value >= 0.02:
                row["confidence"] = "medium"
            else:
                row["confidence"] = "low"

    return support_scores.astype(np.float32), {
        "available": True,
        "method": "all_tiles_all_site_prompts_zero_shot_aggregation",
        "site_prompt_count": len(rows),
        "top_k_sites_per_tile": top_k_sites_per_tile,
        "top_k_tiles_per_site": top_k_tiles_per_site,
        "slide_site_candidates": site_candidates,
        "all_roi_site_scores": all_roi_site_scores,
    }


def score_site_discriminative_tiles(
    features: np.ndarray,
    model: Any,
    site_prompts: list[dict[str, str]],
    device: str | None = None,
    amp: bool | None = None,
    is_titan: bool = False,
    target_site: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Score tiles by how useful they are for organ/site discrimination.

    Tumor-rich regions are not always site-informative. When a slide-level/top
    tile site hint is confident, anchor ROI scoring to that site. Without a
    target site, selecting tiles that are merely discriminative for any organ can
    over-sample visually misleading mimics such as adrenal-vs-liver.
    """
    rows = [
        {"site": str(row.get("site") or "").strip(), "prompt": str(row.get("prompt") or "").strip()}
        for row in site_prompts
        if str(row.get("site") or "").strip() and str(row.get("prompt") or "").strip()
    ]
    if not rows or len(features) == 0:
        return np.zeros((len(features),), dtype=np.float32), {
            "available": False,
            "reason": "no site prompts or image features",
        }
    target_site = str(target_site or "").strip()
    target_idx = None
    if target_site:
        for idx, row in enumerate(rows):
            if row["site"] == target_site:
                target_idx = idx
                break
    if target_idx is None:
        return np.zeros((len(features),), dtype=np.float32), {
            "available": False,
            "reason": "no confident target site for site-anchored ROI ranking",
            "target_site": target_site or None,
        }

    if is_titan:
        text_embeddings = _encode_titan_text_embeddings(
            model,
            [row["site"] for row in rows],
            device=device,
        )
    else:
        text_embeddings = _encode_text(model, [row["prompt"] for row in rows], device=device, amp=amp)

    image = l2_normalize(features)
    text = l2_normalize(text_embeddings)
    sims = image @ text.T
    order = np.argsort(sims, axis=1)[:, ::-1]
    top_idx = order[:, 0]
    target_score = sims[:, target_idx]
    if sims.shape[1] > 1:
        masked = sims.copy()
        masked[:, target_idx] = -np.inf
        best_other = masked.max(axis=1)
        margin = target_score - best_other
    else:
        best_other = np.zeros_like(target_score)
        margin = target_score
    scores = 0.55 * _robust_unit(target_score) + 0.45 * _robust_unit(margin)
    top_records = []
    for idx in np.argsort(scores)[::-1][:50]:
        top_records.append({
            "tile_index": int(idx),
            "site": rows[target_idx]["site"],
            "prompt": rows[target_idx]["prompt"],
            "top_predicted_site_for_tile": rows[int(top_idx[idx])]["site"],
            "site_score": float(scores[idx]),
            "target_similarity": float(target_score[idx]),
            "best_other_similarity": float(best_other[idx]),
            "margin_to_second": float(margin[idx]),
        })
    return scores.astype(np.float32), {
        "available": True,
        "method": "site_anchored_tile_text_image_similarity",
        "target_site": rows[target_idx]["site"],
        "target_prompt": rows[target_idx]["prompt"],
        "site_prompt_count": len(rows),
        "top_site_tiles": top_records,
    }


def _encode_titan_text_embeddings(model: Any, texts: list[str], device: str | None = None) -> np.ndarray:
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    _fix_titan_attn_mask(model, device)
    with torch.no_grad():
        classifier = model.model.zero_shot_classifier([[text] for text in texts], ["CLASSNAME"], device=device)
    return classifier.T.detach().cpu().numpy()


def _robust_unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lo, hi = np.percentile(values, [5, 95])
    if float(hi - lo) < 1e-8:
        lo = float(values.min())
        hi = float(values.max())
    if float(hi - lo) < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def predict_site_from_top_tiles(
    features: np.ndarray,
    tumor_scores: np.ndarray,
    model: Any,
    site_prompts: list[dict[str, str]],
    top_k_tiles: int = 8,
    device: str | None = None,
    amp: bool | None = None,
) -> dict:
    """Zero-shot specimen/site hint from CONCH-like image-text similarity.

    This is an auxiliary prompt-scored hint, not ground truth and not a fixed
    organ classifier head. It scores site prompts on the tumor-ranked tiles so
    the hint follows diagnostically relevant tissue rather than blank/normal
    slide areas.
    """
    rows = [
        {"site": str(row.get("site") or "").strip(), "prompt": str(row.get("prompt") or "").strip()}
        for row in site_prompts
        if str(row.get("site") or "").strip() and str(row.get("prompt") or "").strip()
    ]
    if not rows or len(features) == 0:
        return {"available": False, "reason": "no site prompts or image features"}
    top_k = max(1, min(int(top_k_tiles or 1), len(features)))
    top_tile_indices = np.argsort(tumor_scores)[::-1][:top_k]
    text_embeddings = _encode_text(model, [row["prompt"] for row in rows], device=device, amp=amp)
    image = l2_normalize(features[top_tile_indices])
    text = l2_normalize(text_embeddings)
    sims = image @ text.T
    combined = 0.70 * sims.max(axis=0) + 0.30 * sims.mean(axis=0)
    order = np.argsort(combined)[::-1]
    ranked = [
        {
            "rank": int(rank + 1),
            "site": rows[idx]["site"],
            "prompt": rows[idx]["prompt"],
            "score": float(combined[idx]),
            "max_tile_similarity": float(sims[:, idx].max()),
            "mean_tile_similarity": float(sims[:, idx].mean()),
        }
        for rank, idx in enumerate(order)
    ]
    margin = float(ranked[0]["score"] - ranked[1]["score"]) if len(ranked) > 1 else None
    if margin is None or margin >= 0.04:
        confidence = "high"
    elif margin >= 0.02:
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "available": True,
        "method": "conch_style_zero_shot_text_image_similarity_on_top_tumor_tiles",
        "top_site": ranked[0]["site"],
        "top_prompt": ranked[0]["prompt"],
        "top_score": ranked[0]["score"],
        "margin_to_second": margin,
        "confidence": confidence,
        "top_tumor_tile_indices": [int(idx) for idx in top_tile_indices.tolist()],
        "top_k_tiles": top_k,
        "scores": ranked,
    }


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def build_ranked_rois(
    slide_id: str,
    tiles: list[dict],
    scores: np.ndarray,
    tile_size: int,
    final_roi_size: int,
    max_final_rois: int,
    diversity_factor: float,
) -> list[RoiCandidate]:
    records = scored_tile_records(slide_id, tiles, scores, tile_size)
    candidates = [
        roi_from_tile_record(slide_id, record, final_roi_size)
        for record in records
    ]
    return spatially_diverse(candidates, max_final_rois, min_dist=final_roi_size * diversity_factor)


def build_site_tumor_ranked_rois(
    slide_id: str,
    tiles: list[dict],
    tumor_scores: np.ndarray,
    site_scores: np.ndarray,
    tile_size: int,
    final_roi_size: int,
    max_final_rois: int,
    diversity_factor: float,
    site_roi_fraction: float,
) -> list[RoiCandidate]:
    """Build a mixed ROI list with separate site and tumor evidence.

    The output keeps old downstream behavior (one ordered ROI list) while
    avoiding the failure mode where all candidates are tumor-rich but
    site-ambiguous.
    """
    site_fraction = min(0.8, max(0.0, float(site_roi_fraction)))
    site_quota = int(round(max_final_rois * site_fraction))
    if max_final_rois >= 2 and site_fraction > 0:
        site_quota = max(1, site_quota)
    site_quota = min(site_quota, max_final_rois)
    tumor_quota = max_final_rois - site_quota

    site_records = scored_tile_records(slide_id, tiles, site_scores, tile_size)
    tumor_records = scored_tile_records(slide_id, tiles, tumor_scores, tile_size)
    # Final report/diagnosis should stay tumor-evidence first. Site scores are
    # auxiliary: they help keep organ-informative regions in the candidate pool,
    # but should not outrank diagnostically stronger tumor ROIs by themselves.
    combined_scores = 0.85 * _robust_unit(tumor_scores) + 0.15 * _robust_unit(site_scores)
    combined_records = scored_tile_records(slide_id, tiles, combined_scores, tile_size)

    site_candidates = [
        _roi_from_tile_record_with_scores(
            slide_id, record, final_roi_size,
            source_prefix="foundation_site_discriminative",
            site_scores=site_scores,
            tumor_scores=tumor_scores,
            combined_scores=combined_scores,
        )
        for record in site_records
    ]
    tumor_candidates = [
        _roi_from_tile_record_with_scores(
            slide_id, record, final_roi_size,
            source_prefix="foundation_tumor_discriminative",
            site_scores=site_scores,
            tumor_scores=tumor_scores,
            combined_scores=combined_scores,
        )
        for record in tumor_records
    ]
    combined_candidates = [
        _roi_from_tile_record_with_scores(
            slide_id, record, final_roi_size,
            source_prefix="foundation_site_tumor_combined",
            site_scores=site_scores,
            tumor_scores=tumor_scores,
            combined_scores=combined_scores,
        )
        for record in combined_records
    ]

    min_dist = final_roi_size * diversity_factor
    selected: list[RoiCandidate] = []
    _extend_spatially_diverse(selected, site_candidates, site_quota, min_dist)
    selected = _dedupe_rois(selected)
    _extend_spatially_diverse(selected, tumor_candidates, site_quota + tumor_quota, min_dist)
    selected = _dedupe_rois(selected)
    if len(selected) < max_final_rois:
        _extend_spatially_diverse(selected, combined_candidates, max_final_rois, min_dist)
    selected = _dedupe_rois(selected)
    if len(selected) < max_final_rois:
        selected.extend([roi for roi in combined_candidates if _roi_key(roi) not in {_roi_key(prev) for prev in selected}])
    return _dedupe_rois(selected)[:max_final_rois]


def build_separate_site_tumor_rois(
    slide_id: str,
    tiles: list[dict],
    tumor_scores: np.ndarray,
    site_scores: np.ndarray,
    tile_size: int,
    final_roi_size: int,
    max_site_rois: int,
    max_tumor_rois: int,
    diversity_factor: float,
    use_site_rois: bool,
) -> tuple[list[RoiCandidate], list[RoiCandidate]]:
    """Build separate ROI streams for site and tumor evidence.

    Site ROIs are intentionally few and organ-anchored. Tumor ROIs remain the
    main evidence for diagnosis and report generation.
    """
    combined_scores = 0.85 * _robust_unit(tumor_scores) + 0.15 * _robust_unit(site_scores)
    site_records = scored_tile_records(slide_id, tiles, site_scores, tile_size)
    tumor_records = scored_tile_records(slide_id, tiles, tumor_scores, tile_size)
    site_candidates = [
        _roi_from_tile_record_with_scores(
            slide_id, record, final_roi_size,
            source_prefix="foundation_site_roi",
            site_scores=site_scores,
            tumor_scores=tumor_scores,
            combined_scores=combined_scores,
            evidence_role="site",
            score_source="site",
        )
        for record in site_records
    ] if use_site_rois else []
    tumor_candidates = [
        _roi_from_tile_record_with_scores(
            slide_id, record, final_roi_size,
            source_prefix="foundation_tumor_roi",
            site_scores=site_scores,
            tumor_scores=tumor_scores,
            combined_scores=combined_scores,
            evidence_role="tumor",
            score_source="tumor",
        )
        for record in tumor_records
    ]
    min_dist = final_roi_size * diversity_factor
    site_ranked = spatially_diverse(site_candidates, max(0, int(max_site_rois)), min_dist=min_dist)
    tumor_ranked = spatially_diverse(tumor_candidates, max(0, int(max_tumor_rois)), min_dist=min_dist)
    return site_ranked, tumor_ranked


def build_site_type_tumor_rois(
    slide_id: str,
    tiles: list[dict],
    tumor_scores: np.ndarray,
    site_support_scores: np.ndarray,
    tile_size: int,
    final_roi_size: int,
    max_site_rois: int,
    max_tumor_rois: int,
    diversity_factor: float,
) -> tuple[list[RoiCandidate], list[RoiCandidate]]:
    """Build v2 evidence streams: site support ROIs plus tumor ROIs.

    Site type itself is predicted by aggregating every tissue tile. These ROIs
    only provide compact visual support for the top site hypotheses.
    """
    combined_scores = 0.85 * _robust_unit(tumor_scores) + 0.15 * _robust_unit(site_support_scores)
    site_records = scored_tile_records(slide_id, tiles, site_support_scores, tile_size)
    tumor_records = scored_tile_records(slide_id, tiles, tumor_scores, tile_size)
    site_candidates = [
        _roi_from_tile_record_with_scores(
            slide_id, record, final_roi_size,
            source_prefix="foundation_site_support_roi",
            site_scores=site_support_scores,
            tumor_scores=tumor_scores,
            combined_scores=combined_scores,
            evidence_role="site_support",
            score_source="site",
        )
        for record in site_records
    ]
    tumor_candidates = [
        _roi_from_tile_record_with_scores(
            slide_id, record, final_roi_size,
            source_prefix="foundation_tumor_roi",
            site_scores=site_support_scores,
            tumor_scores=tumor_scores,
            combined_scores=combined_scores,
            evidence_role="tumor",
            score_source="tumor",
        )
        for record in tumor_records
    ]
    min_dist = final_roi_size * diversity_factor
    site_ranked = spatially_diverse(site_candidates, max(0, int(max_site_rois)), min_dist=min_dist)
    tumor_ranked = spatially_diverse(tumor_candidates, max(0, int(max_tumor_rois)), min_dist=min_dist)
    return site_ranked, tumor_ranked


def _extend_spatially_diverse(
    selected: list[RoiCandidate],
    candidates: list[RoiCandidate],
    target_total: int,
    min_dist: float,
) -> None:
    for roi in candidates:
        if len(selected) >= target_total:
            break
        if _roi_key(roi) in {_roi_key(prev) for prev in selected}:
            continue
        if all(center_dist(roi, prev) >= min_dist for prev in selected):
            selected.append(roi)
    for roi in candidates:
        if len(selected) >= target_total:
            break
        if _roi_key(roi) not in {_roi_key(prev) for prev in selected}:
            selected.append(roi)


def _roi_from_tile_record_with_scores(
    slide_id: str,
    record: dict,
    final_roi_size: int,
    source_prefix: str,
    site_scores: np.ndarray,
    tumor_scores: np.ndarray,
    combined_scores: np.ndarray,
    evidence_role: str | None = None,
    score_source: str = "combined",
) -> RoiCandidate:
    roi = roi_from_tile_record(slide_id, record, final_roi_size)
    tile_index = int(record["tile_index"])
    roi.source = f"{source_prefix}:{record['tile_id']}"
    if score_source == "site":
        roi.score = float(site_scores[tile_index])
    elif score_source == "tumor":
        roi.score = float(tumor_scores[tile_index])
    else:
        roi.score = float(combined_scores[tile_index])
    setattr(roi, "site_rank_score", float(site_scores[tile_index]))
    setattr(roi, "tumor_rank_score", float(tumor_scores[tile_index]))
    setattr(roi, "combined_rank_score", float(combined_scores[tile_index]))
    setattr(roi, "tile_index", tile_index)
    if evidence_role:
        setattr(roi, "evidence_role", evidence_role)
    return roi


def _roi_key(roi: RoiCandidate) -> tuple[int, int, int, int, int]:
    return (int(roi.x), int(roi.y), int(roi.width), int(roi.height), int(roi.level))


def _dedupe_rois(candidates: list[RoiCandidate]) -> list[RoiCandidate]:
    seen = set()
    out = []
    for roi in candidates:
        key = _roi_key(roi)
        if key in seen:
            continue
        seen.add(key)
        out.append(roi)
    return out


def _attach_site_tumor_scores(cached: list[dict], ranked: list[RoiCandidate], evidence_role: str | None = None) -> None:
    by_id = {roi.roi_id: roi for roi in ranked}
    for row in cached:
        roi = by_id.get(str(row.get("roi_id") or ""))
        if roi is None:
            continue
        row["site_rank_score"] = getattr(roi, "site_rank_score", None)
        row["tumor_rank_score"] = getattr(roi, "tumor_rank_score", None)
        row["combined_rank_score"] = getattr(roi, "combined_rank_score", None)
        row["foundation_tile_index"] = getattr(roi, "tile_index", None)
        role = evidence_role or getattr(roi, "evidence_role", None)
        if role:
            row["evidence_role"] = role


def _attach_site_type_metadata(cached: list[dict], ranked: list[RoiCandidate], metadata: dict) -> None:
    scores = metadata.get("all_roi_site_scores") or []
    by_tile = {
        int(row.get("tile_index")): row
        for row in scores
        if isinstance(row, dict) and row.get("tile_index") is not None
    }
    by_id = {roi.roi_id: roi for roi in ranked}
    for row in cached:
        roi = by_id.get(str(row.get("roi_id") or ""))
        if roi is None:
            continue
        tile_index = getattr(roi, "tile_index", None)
        try:
            tile_index = int(tile_index)
        except Exception:
            tile_index = None
        site_row = by_tile.get(tile_index)
        if not site_row:
            continue
        row["top_site"] = site_row.get("top_site")
        row["top_site_score"] = site_row.get("top_site_score")
        row["site_margin_to_second"] = site_row.get("margin_to_second")
        row["top_k_sites"] = site_row.get("top_k_sites") or []


def _roi_candidate_payload(roi: RoiCandidate) -> dict:
    payload = asdict(roi)
    for key in [
        "site_rank_score",
        "tumor_rank_score",
        "combined_rank_score",
        "tile_index",
        "evidence_role",
    ]:
        if hasattr(roi, key):
            payload[key] = getattr(roi, key)
    return payload


def scored_tile_records(slide_id: str, tiles: list[dict], scores: np.ndarray, tile_size: int) -> list[dict]:
    records = []
    for idx, (tile, score) in enumerate(zip(tiles, scores)):
        geometry = tile.get("geometry")
        if geometry is None:
            continue
        minx, miny, maxx, maxy = _shape_bounds(geometry)
        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0
        records.append({
            "tile_id": f"{slide_id}_foundation_tile_{idx:05d}",
            "tile_index": idx,
            "x": max(0, int(round(cx - tile_size / 2))),
            "y": max(0, int(round(cy - tile_size / 2))),
            "center_x": int(round(cx)),
            "center_y": int(round(cy)),
            "width": tile_size,
            "height": tile_size,
            "score": float(score),
        })
    records.sort(key=lambda row: row["score"], reverse=True)
    return records


def roi_from_tile_record(slide_id: str, record: dict, final_roi_size: int) -> RoiCandidate:
    x = max(0, int(record["center_x"]) - final_roi_size // 2)
    y = max(0, int(record["center_y"]) - final_roi_size // 2)
    roi_id = f"{slide_id}_foundation_x{x}_y{y}_l0_w{final_roi_size}_h{final_roi_size}"
    return RoiCandidate(
        slide_id=slide_id,
        roi_id=roi_id,
        x=x,
        y=y,
        width=final_roi_size,
        height=final_roi_size,
        level=0,
        source=f"foundation_text_similarity:{record['tile_id']}",
        score=float(record["score"]),
    )


def spatially_diverse(candidates: list[RoiCandidate], max_count: int, min_dist: float) -> list[RoiCandidate]:
    selected: list[RoiCandidate] = []
    for roi in candidates:
        if all(center_dist(roi, prev) >= min_dist for prev in selected):
            selected.append(roi)
        if len(selected) >= max_count:
            break
    for roi in candidates:
        if len(selected) >= max_count:
            break
        if roi not in selected:
            selected.append(roi)
    return selected[:max_count]


def center_dist(a: RoiCandidate, b: RoiCandidate) -> float:
    ax = a.x + a.width / 2.0
    ay = a.y + a.height / 2.0
    bx = b.x + b.width / 2.0
    by = b.y + b.height / 2.0
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)
