#!/usr/bin/env python3
"""Export only the D1 inference adaptation after proving base-text identity.

The source checkpoint and pinned SAM3 base are memory-mapped on CPU. This
script refuses to export if any embedded D1 base_text_encoder tensor differs
from the pinned SAM3 language backbone. No model is instantiated and no GPU is
used. The original checkpoint is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


SOURCE_SHA256 = "41fd1867add7049bb6305e103b275c99ec513146f9d9e3f3f845b1dc55b0d031"
SOURCE_BYTES = 1484446541
SAM3_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
SAM3_BYTES = 3450062241


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_verified(path: Path, expected_size: int, expected_hash: str, label: str):
    path = path.expanduser().resolve(strict=True)
    if path.stat().st_size != expected_size:
        raise ValueError(f"{label} size mismatch")
    actual = sha256_file(path)
    if actual != expected_hash:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path, torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-d1", type=Path, required=True)
    p.add_argument("--sam3-base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    source_path, d1 = load_verified(args.source_d1, SOURCE_BYTES, SOURCE_SHA256, "source D1")
    sam_path, sam_raw = load_verified(args.sam3_base, SAM3_BYTES, SAM3_SHA256, "pinned SAM3 base")
    prompt = d1.get("prompt_enc")
    lora = d1.get("lora_state")
    adapter = d1.get("adapter_state")
    if not isinstance(prompt, dict) or not isinstance(lora, dict) or not isinstance(adapter, dict):
        raise ValueError("D1 checkpoint is missing required prompt, LoRA, or adapter state")
    if int(d1.get("stage", 0)) != 3:
        raise ValueError("expected the selected Stage 3 D1 checkpoint")

    sam_state = sam_raw.get("model", sam_raw)
    prefix = "base_text_encoder."
    base_keys = [key for key in prompt if key.startswith(prefix)]
    if not base_keys:
        raise ValueError("D1 prompt state has no base_text_encoder tensors to verify")
    equal_count = 0
    for key in base_keys:
        sam_key = "detector.backbone.language_backbone." + key.removeprefix(prefix)
        if sam_key not in sam_state:
            raise ValueError(f"pinned SAM3 tensor missing for {key}")
        left, right = prompt[key], sam_state[sam_key]
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            raise ValueError(f"D1 embedded base tensor differs from pinned SAM3: {key}")
        equal_count += 1

    learned_prompt = {key: value for key, value in prompt.items() if not key.startswith(prefix)}
    if not learned_prompt:
        raise ValueError("no learned prompt state remains after verified base removal")
    if not lora or not adapter:
        raise ValueError("expected non-empty D1 LoRA and adapter state")

    retained_training_state = (
        "tile_cls_state", "ssl_head_state", "pixel_ssl_state",
        "density_fusion_state", "density_gate_state", "contrast_proj_state",
        "morph_proj_state"
    )
    retained = {key: d1[key] for key in retained_training_state if d1.get(key) is not None}
    # Keep model-relevant settings only. Dataset paths, cache locations, and
    # training-run destinations are intentionally excluded from the artifact.
    cfg = d1.get("train_cfg") if isinstance(d1.get("train_cfg"), dict) else {}
    safe_cfg_keys = (
        "stage", "seg_mode", "use_organ_bias", "freeze_prompts_after_stage1",
        "freeze_lora_stage2", "no_freeze_lora_stage2", "native_crop_at_40x",
        "cell_cue_feature_fusion", "cell_density_source", "cell_density_train_mode"
    )
    safe_cfg = {key: cfg[key] for key in safe_cfg_keys if key in cfg}
    exported = {
        "format": "pathxai-d1-inference-adapter-v1",
        "stage": 3,
        "lora_state": lora,
        "prompt_enc": learned_prompt,
        "adapter_state": adapter,
        **retained,
        "train_cfg": safe_cfg,
        "provenance": {
            "source_checkpoint_sha256": SOURCE_SHA256,
            "source_checkpoint_bytes": SOURCE_BYTES,
            "source_reference_run": "tcga_expansion_20260924_1200_d1",
            "sam3_base_repository": "facebook/sam3",
            "sam3_base_revision": "3c879f39826c281e95690f02c7821c4de09afae7",
            "sam3_base_sha256": SAM3_SHA256,
            "verified_unchanged_base_text_tensors": equal_count,
            "removed_base_text_tensor_prefix": "prompt_enc.base_text_encoder.",
            "retained_auxiliary_states": sorted(retained),
            "excluded_training_state": ["opt_state", "scheduler_state", "scaler_state", "recent_losses"]
        }
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing export: {output}")
    temp = output.with_name(output.name + ".tmp")
    if temp.exists():
        raise FileExistsError(f"temporary export already exists: {temp}")
    torch.save(exported, temp)
    temp.replace(output)
    out_hash = sha256_file(output)
    print(f"Source D1 verified: {source_path}")
    print(f"SAM3 base verified: {sam_path}")
    print(f"Exact unchanged base text tensors: {equal_count}/{len(base_keys)}")
    print(f"Retained adaptation tensors: LoRA={len(lora)}, prompt={len(learned_prompt)}, adapter={len(adapter)}")
    print(f"Exported: {output}")
    print(f"Export bytes: {output.stat().st_size}")
    print(f"Export SHA-256: {out_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
