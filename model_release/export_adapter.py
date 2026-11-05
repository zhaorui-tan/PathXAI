#!/usr/bin/env python3
"""CPU-only exporter for the refined PathXAI D1 checkpoint.

Verifies the pinned SAM3 base-text tensors before removing them, preserves
all inference-relevant trained states, and writes a local manifest and receipt.
Never publishes anything and never modifies the source checkpoints.
"""
from __future__ import annotations
import hashlib, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import torch

SOURCE_SHA256 = "4d8ac978814f66ba5a716e26c8c44ada50f3f6c3f1bc987a97b24238babe9e5a"
SOURCE_BYTES = 1484446605
SAM3_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
SAM3_BYTES = 3450062241
SOURCE_RUN = "medsam3_d1_refined/d1_final/run_20260804_140335_stage3_tissue"
EXPECTED = {
    "prompt_enc": ("prompt_embeddings", "prompt_masks", "organ_bias"),
    "lora_state": None,
    "adapter_state": None,
    "tile_cls_state": None,
}
AUX_KEYS = ("tile_cls_state", "ssl_head_state", "pixel_ssl_state",
            "density_fusion_state", "density_gate_state", "contrast_proj_state",
            "morph_proj_state")
CFG_KEYS = ("stage", "seg_mode", "native_crop_at_40x", "use_organ_bias",
            "no_cell_info", "cell_density_source", "cell_density_train_mode",
            "cell_cue_feature_fusion", "freeze_prompts_after_stage1",
            "freeze_lora_stage2", "no_freeze_lora_stage2",
            "nuclei_density_logit_prior", "stage2_variant")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load_verified(path: Path, expected_size: int, expected_hash: str, label: str):
    path = path.expanduser().resolve(strict=True)
    if path.stat().st_size != expected_size:
        raise ValueError(f"{label} size mismatch: {path.stat().st_size} != {expected_size}")
    actual = sha256_file(path)
    if actual != expected_hash:
        raise ValueError(f"{label} SHA-256 mismatch: {actual}")
    return path, torch.load(path, map_location="cpu", weights_only=True, mmap=True)

def tensor_map(state):
    return {k: v for k, v in state.items() if torch.is_tensor(v)}

def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-d1", type=Path, required=True)
    p.add_argument("--sam3-base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args()
    src_path, d1 = load_verified(a.source_d1, SOURCE_BYTES, SOURCE_SHA256, "refined D1")
    sam_path, sam_raw = load_verified(a.sam3_base, SAM3_BYTES, SAM3_SHA256, "pinned SAM3 base")
    if int(d1.get("stage", 0)) != 3:
        raise ValueError("expected stage 3 source checkpoint")
    prompt, lora, adapter = d1.get("prompt_enc"), d1.get("lora_state"), d1.get("adapter_state")
    if not all(isinstance(x, dict) and x for x in (prompt, lora, adapter)):
        raise ValueError("required prompt_enc, lora_state, and adapter_state missing/empty")
    missing_prompt = sorted(set(EXPECTED["prompt_enc"]) - set(prompt))
    if missing_prompt:
        raise ValueError(f"required prompt keys missing: {missing_prompt}")
    sam_state = sam_raw.get("model", sam_raw)
    prefix = "base_text_encoder."
    base_keys = [k for k in prompt if k.startswith(prefix)]
    if not base_keys:
        raise ValueError("no embedded base_text_encoder tensors to verify")
    for key in base_keys:
        sam_key = "detector.backbone.language_backbone." + key.removeprefix(prefix)
        if sam_key not in sam_state:
            raise ValueError(f"pinned SAM3 tensor missing: {sam_key}")
        left, right = prompt[key], sam_state[sam_key]
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            raise ValueError(f"embedded base tensor differs from pinned SAM3: {key}")
    learned_prompt = {k: v for k, v in prompt.items() if not k.startswith(prefix)}
    if set(learned_prompt) != set(EXPECTED["prompt_enc"]):
        raise ValueError(f"unexpected refined learned prompt key set: {sorted(learned_prompt)}")
    aux = {k: d1[k] for k in AUX_KEYS if d1.get(k) is not None}
    if not aux.get("tile_cls_state"):
        raise ValueError("refined inference bundle requires trained tile_cls_state")
    cfg = d1.get("train_cfg") if isinstance(d1.get("train_cfg"), dict) else {}
    safe_cfg = {k: cfg[k] for k in CFG_KEYS if k in cfg}
    if safe_cfg.get("stage") != 3 or safe_cfg.get("seg_mode") != "tissue":
        raise ValueError(f"unexpected checkpoint runtime config: {safe_cfg}")
    exported = {
        "format": "pathxai-d1-inference-adapter-v1",
        "stage": 3,
        "prompt_enc": learned_prompt,
        "lora_state": lora,
        "adapter_state": adapter,
        **aux,
        "train_cfg": safe_cfg,
        "provenance": {
            "source_checkpoint_sha256": SOURCE_SHA256,
            "source_checkpoint_bytes": SOURCE_BYTES,
            "source_checkpoint_filename": "best_model.pth",
            "source_lineage": SOURCE_RUN,
            "lineage_note": "This is the refined 2026-08-04 D1 run; it is not the earlier tcga_expansion_20260924_1200_d1 run.",
            "sam3_base_repository": "facebook/sam3",
            "sam3_base_revision": "3c879f39826c281e95690f02c7821c4de09afae7",
            "sam3_base_sha256": SAM3_SHA256,
            "sam3_base_bytes": SAM3_BYTES,
            "verified_unchanged_base_text_tensors": len(base_keys),
            "removed_base_text_tensor_prefix": "prompt_enc.base_text_encoder.",
            "retained_auxiliary_states": sorted(aux),
            "excluded_training_state": ["opt_state", "scheduler_state", "scaler_state", "recent_losses", "epoch", "global_step", "best_dice"]
        }
    }
    output = a.output.expanduser().resolve()
    receipt_path, manifest_path = a.receipt.expanduser().resolve(), a.manifest.expanduser().resolve()
    for path in (output, receipt_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing file: {path}")
    tmp = output.with_name(output.name + ".tmp")
    torch.save(exported, tmp)
    tmp.replace(output)
    # Reload and prove every exported tensor matches its source object exactly.
    reloaded = torch.load(output, map_location="cpu", weights_only=True, mmap=True)
    checks = {}
    for section in ("prompt_enc", "lora_state", "adapter_state", *AUX_KEYS):
        if section not in exported:
            continue
        left, right = tensor_map(exported[section]), tensor_map(reloaded[section])
        if left.keys() != right.keys():
            raise ValueError(f"serialized key mismatch in {section}")
        for key in left:
            if left[key].shape != right[key].shape or left[key].dtype != right[key].dtype or not torch.equal(left[key], right[key]):
                raise ValueError(f"serialized tensor mismatch: {section}.{key}")
        checks[section] = len(left)
    if any(k.startswith(prefix) for k in reloaded["prompt_enc"]):
        raise ValueError("base text encoder tensors leaked into export")
    out_sha = sha256_file(output)
    manifest = {
        "schema": "pathxai-model-release-manifest-v1",
        "release_name": "PathXAI refined D1 inference adaptation",
        "release_kind": "adapter-only; pinned SAM3 base is a separately obtained dependency and is not included",
        "status": "local draft; not published",
        "asset": {"filename": output.name, "bytes": output.stat().st_size, "sha256": out_sha,
                  "required_for_inference": True, "source_lineage": SOURCE_RUN,
                  "source_checkpoint_bytes": SOURCE_BYTES, "source_checkpoint_sha256": SOURCE_SHA256},
        "sam3_base_dependency": {"repository": "facebook/sam3", "revision": "3c879f39826c281e95690f02c7821c4de09afae7",
                                  "filename": "sam3.pt", "bytes": SAM3_BYTES, "sha256": SAM3_SHA256,
                                  "distribution": "Not included; obtain separately from upstream."},
        "inference_configuration": safe_cfg,
        "verified_unchanged_base_text_tensors": len(base_keys),
        "removed_prefix": "prompt_enc.base_text_encoder.",
        "retained_states": {k: checks[k] for k in checks},
        "lineage_note": "This refined D1 is distinct from tcga_expansion_20260924_1200_d1. Do not attribute the earlier 1200 run metrics to this checkpoint without separate evidence.",
        "publication": "none"
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    receipt = {
        "verification": "passed",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "thread_limit": 4,
        "source": {"path": str(src_path), "bytes": SOURCE_BYTES, "sha256": SOURCE_SHA256, "stage": int(d1["stage"]),
                   "run": SOURCE_RUN},
        "sam3_base": {"path": str(sam_path), "bytes": SAM3_BYTES, "sha256": SAM3_SHA256,
                      "revision": "3c879f39826c281e95690f02c7821c4de09afae7"},
        "base_text_equality": {"verified_equal": len(base_keys), "embedded": len(base_keys), "mismatches": 0},
        "export": {"path": str(output), "bytes": output.stat().st_size, "sha256": out_sha,
                   "retained_tensor_counts": checks, "base_text_prefix_absent": True,
                   "required_runtime_states_present": ["prompt_enc", "lora_state", "adapter_state", "tile_cls_state"]},
        "runtime_config": safe_cfg,
        "lineage_guard": "refined D1 lineage explicitly recorded; not identified as the earlier 1200 run",
        "publication": "none"
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"export":str(output),"bytes":output.stat().st_size,"sha256":out_sha,
                      "equal_base_tensors":len(base_keys),"retained_tensors":checks,
                      "receipt":str(receipt_path),"manifest":str(manifest_path)},indent=2))
    return 0

if __name__ == "__main__":
    torch.set_num_threads(4)
    raise SystemExit(main())
