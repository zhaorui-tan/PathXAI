#!/usr/bin/env python3
"""Verify recorded delivery bytes and the scientific snapshot; no model loading."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "vendor/report_engine"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    errors = []
    manifest = json.loads((ROOT / "provenance/delivery_engine_manifest.json").read_text())
    upstream_prefix = "baselines/current/source_snapshot/archives/tcga_so1_so4_a5bridge_508_20260920/code/medsam3/"
    if not (ENGINE / upstream_prefix / "lora_layers.py").is_file():
        print("Provision the pinned upstream runtime first: python3 scripts/fetch_medsam_source.py", file=sys.stderr)
        return 1
    external = json.loads((ROOT / "provenance/external_medsam_source.json").read_text())
    registered = {row["path"]: row["sha256"] for row in manifest["files"]}
    for row in external["files"]:
        if registered.get(upstream_prefix + row["path"]) != row["final_sha256"]:
            errors.append("External source contract differs from frozen runtime: " + row["path"])
    for row in manifest["files"]:
        file = ENGINE / row["path"]
        if not file.is_file() or sha(file) != row["sha256"]:
            errors.append("delivery source mismatch: " + row["path"])
    sys.path.insert(0, str(ENGINE))
    from pathxai_demo.provenance import verify_manifest
    errors.extend(verify_manifest(ENGINE / "baselines/current/manifest.json"))
    nb = json.loads((ROOT / "PathXAI_report.ipynb").read_text())
    if any(c.get("outputs") or c.get("execution_count") is not None for c in nb["cells"] if c["cell_type"] == "code"):
        errors.append("Main runnable notebook must have cleared outputs and execution counts")
    a = json.loads((ROOT / "model_release/manifest.json").read_text())
    b = json.loads((ROOT / "model_release/upload_ready/manifest.json").read_text())
    if a != b:
        errors.append("Model manifest and upload manifest differ")
    if (ROOT / "examples/manifest.json").exists():
        from audit_examples import audit
        errors.extend(audit())
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Verified {len(manifest['files'])} delivery files, frozen scientific snapshot, model contracts, and clean notebook")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
