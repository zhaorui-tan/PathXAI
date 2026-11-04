#!/usr/bin/env python3
"""Create separate GitHub-source and HF-model ZIPs from reviewed allowlists."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = {".github", "examples", "configs", "docs", "environments", "evaluation", "model_release",
               "pathxai_tutorial", "provenance", "scoring", "scripts", "tests", "vendor"}
SOURCE_FILES = {"README.md", "PathXAI_report.ipynb", ".gitignore"}
EXCLUDED_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git"}
WEIGHT_SUFFIXES = {".pth", ".pt", ".safetensors", ".svs", ".ndpi", ".pem", ".key"}

def source_files():
    for name in sorted(SOURCE_FILES):
        yield ROOT / name
    for name in sorted(SOURCE_DIRS):
        for path in sorted((ROOT / name).rglob("*")):
            rel = path.relative_to(ROOT)
            if path.is_symlink():
                raise ValueError(f"Unexpected symlink in source release: {rel}")
            if not path.is_file() or any(part in EXCLUDED_DIRS for part in rel.parts):
                continue
            if str(rel).startswith("vendor/report_engine/baselines/current/source_snapshot/archives/tcga_so1_so4_a5bridge_508_20260920/code/medsam3/"):
                continue
            if path.suffix in WEIGHT_SUFFIXES or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.name.startswith(".env") or path.name == "ACADEMIC_LICENSE_DRAFT.md":
                continue
            yield path

def archive(path, files, prefix):
    if path.exists():
        raise FileExistsError(f"Choose a new output directory: {path}")
    rows = []
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for source, relative in files:
            data = source.read_bytes()
            z.writestr(str(Path(prefix) / relative), data)
            rows.append({"path":str(relative), "bytes":len(data), "sha256":hashlib.sha256(data).hexdigest()})
    return {"archive":path.name, "bytes":path.stat().st_size,
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest(), "files":rows}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()
    subprocess.run([sys.executable,str(ROOT / "scripts/audit_delivery.py")],check=True)
    subprocess.run([sys.executable,str(ROOT / "model_release/stage_upload.py")],check=True)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    source = archive(a.output_dir / "PathXAI_tutorial_source.zip",
                     [(f,f.relative_to(ROOT)) for f in source_files()], "PathXAI_tutorial")
    bundle = ROOT / "model_release/upload_ready"
    weights = json.loads((bundle / "manifest.json").read_text())["assets"]
    names = ["README.md", "LICENSE", "manifest.json"] + [row["repo_path"] for row in weights]
    model = archive(a.output_dir / "PathXAI-Seg_upload.zip",[(bundle / n,Path(n)) for n in names], "")
    manifest = {"source":source,"model":model,"note":"Local archives only; nothing uploaded."}
    (a.output_dir / "release_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    for row in (source,model):
        print(f"{row['archive']}: {row['bytes']} bytes, {len(row['files'])} files, sha256={row['sha256']}")

if __name__ == "__main__":
    main()
