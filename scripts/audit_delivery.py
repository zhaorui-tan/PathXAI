#!/usr/bin/env python3
"""Verify executed notebook provenance, delivery bytes and source snapshot; no model loading."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "vendor/report_engine"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def notebook_source(value):
    """Normalize either Jupyter's list or string representation of source."""
    return "".join(value) if isinstance(value, list) else value

def builder_code_cells(path):
    """Read the literal code cells used by scripts/build_notebook.py."""
    tree = ast.parse(path.read_text())
    cells = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "code":
            continue
        if not node.args:
            continue
        try:
            source = ast.literal_eval(node.args[0])
        except (ValueError, TypeError):
            continue
        if isinstance(source, str):
            cells.append(source)
    return cells

def audit_main_notebook(root):
    errors = []
    notebook_path = root / "PathXAI_report.ipynb"
    nb = json.loads(notebook_path.read_text())
    code = [cell for cell in nb.get("cells", []) if cell.get("cell_type") == "code"]
    counts = [cell.get("execution_count") for cell in code]
    if not code or any(count is None for count in counts):
        errors.append("Main notebook must contain executed code cells with execution counts")
    if not any(cell.get("outputs") for cell in code):
        errors.append("Main notebook must retain outputs from its inference execution")
    if any(output.get("output_type") == "error" for cell in code for output in cell.get("outputs", [])):
        errors.append("Main notebook contains an execution error")
    example = nb.get("metadata", {}).get("pathxai_example", {})
    if example.get("notebook_kind") != "inference":
        errors.append("Main notebook metadata must identify an inference execution")
    if example.get("version") != "SO4":
        errors.append("Main notebook must record the SO4 PathXAI execution")
    if counts != example.get("source_execution_counts") or counts != list(range(1, len(code) + 1)):
        errors.append("Main notebook execution counts do not match the source run")
    if (root / "examples/manifest.json").exists():
        manifest = json.loads((root / "examples/manifest.json").read_text())
        entry = next((row for row in manifest["runs"] if row["version"] == "SO4"), {})
        for key in ("source_executed_notebook_sha256", "report_text_sha256"):
            if not entry.get(key) or example.get(key) != entry[key]:
                errors.append("Main notebook does not match the published SO4 run: " + key)
        output_text = json.dumps([cell.get("outputs", []) for cell in code])
        for key in ("source_report_html_sha256", "report_text_sha256"):
            if entry.get(key) and entry[key] not in output_text:
                errors.append("Main notebook output lacks its actual run checksum: " + key)
    expected = builder_code_cells(root / "scripts/build_notebook.py")
    actual = [notebook_source(cell.get("source", "")) for cell in code]
    if not expected or actual != expected:
        errors.append("Main notebook code cells differ from scripts/build_notebook.py")
    return errors

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
    errors.extend(audit_main_notebook(ROOT))
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
    print(f"Verified {len(manifest['files'])} delivery files, frozen scientific snapshot, model contracts, and executed inference notebook")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
