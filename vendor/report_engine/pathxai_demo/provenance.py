"""Portable, file-level provenance snapshots for the PathXAI paper baseline.

This module intentionally uses an explicit source allowlist. It never reads
environment values and excludes result trees, model weights, and case data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "pathxai-baseline-v1"
SECRET_MARKERS = (b"-----BEGIN PRIVATE KEY-----", b"-----BEGIN OPENSSH PRIVATE KEY-----")
SECRET_NAME_PARTS = (".env", "credential", "secret", "token.txt", "access_token", "api_key", "apikey")
SECRET_VALUE_PATTERNS = (
    re.compile(rb"(?i)(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*['\"]([A-Za-z0-9_./+=:-]{20,})['\"]"),
    re.compile(rb"(?i)authorization\s*[:=]\s*['\"]Bearer\s+([A-Za-z0-9_./+=:-]{20,})['\"]"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{24,}\b"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_candidates(project: Path) -> list[tuple[Path, str, str]]:
    candidates: list[tuple[Path, str, str]] = []

    def add_tree(root: Path, dest_prefix: str, label: str, suffixes: set[str]) -> None:
        if not root.is_dir():
            return
        for item in sorted(root.rglob("*")):
            if item.is_file() and item.suffix.lower() in suffixes and "__pycache__" not in item.parts:
                candidates.append((item, f"{dest_prefix}/{item.relative_to(root).as_posix()}", label))

    add_tree(project / "src/pathxai_agent_v2", "v2/src/pathxai_agent_v2", "current_v2", {".py", ".yaml", ".yml"})
    # Runtime entrypoints and configuration which establish SO1/SO4 behavior.
    for rel in (
        "scripts/run_seg_reader_ablation.sh", "scripts/run_seg_supported_report_paper_ablation.sh",
        "scripts/run_seg_report_ablation.sh", "scripts/run_sf_ablation.sh",
        "scripts/export_report_samples_offline.py",
        "config/medgemma_llm.yaml",
    ):
        p = project / rel
        if p.is_file():
            candidates.append((p, f"v2/{rel}", "current_v2"))

    # v1 runtime modules referenced by the demo's source documentation.
    v1root = Path("/local/PathXAI_Agent/src/pathxai_agent")
    add_tree(v1root, "v1/src/pathxai_agent", "external_v1", {".py"})
    for rel in ("scripts/regenerate_report_html.py", "scripts/render_review_workspace_html.py"):
        p = Path("/local/PathXAI_Agent") / rel
        if p.is_file():
            candidates.append((p, f"v1/{rel}", "external_v1"))
    # External segmentation builder source used to establish checkpoint lineage.
    seg_builder = Path("/local/PathXAI/train_medsam_FINAL.py")
    if seg_builder.is_file():
        candidates.append((seg_builder, "external_seg_runtime/train_medsam_FINAL.py", "external_seg_runtime"))
    # Latest archived SO1/SO4 D1 and A5 bridge snapshots include executable
    # launcher code; include only code/config files, never patient inputs/results.
    for run_id in ("tcga_expansion_20260924_1200_d1", "tcga_so1_so4_a5bridge_508_20260920"):
        archive = project / "PathXAI_Seg2Report/results" / run_id / "provenance"
        if archive.is_dir():
            for item in sorted(archive.rglob("*")):
                if item.is_file() and item.suffix.lower() in {".py", ".yaml", ".yml", ".sh"} and "__pycache__" not in item.parts:
                    candidates.append((item, f"archives/{run_id}/{item.relative_to(archive).as_posix()}", f"archive:{run_id}"))
    # Keep first path for duplicate destinations and deterministic ordering.
    unique = {dest: (src, dest, label) for src, dest, label in candidates}
    return [unique[k] for k in sorted(unique)]


def _git_info(project: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=project, check=True, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"revision": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "status_porcelain_sha256": hashlib.sha256((git("status", "--porcelain") or "").encode()).hexdigest()}


def _runtime_inventory() -> dict[str, Any]:
    # Do not inspect environment variables: keys and values may contain secrets.
    return {"python": sys.version, "executable_basename": Path(sys.executable).name,
            "platform": platform.platform(), "machine": platform.machine(),
            "installed_distributions": _installed_distributions()}


def _installed_distributions() -> list[dict[str, str]]:
    try:
        from importlib.metadata import distributions
        return sorted(({"name": d.metadata.get("Name", "unknown"), "version": d.version}
                       for d in distributions()), key=lambda x: (x["name"].lower(), x["version"]))
    except Exception as exc:  # inventory failure is itself recorded
        return [{"error": type(exc).__name__}]


def freeze_baseline(project_root: Path, destination: Path, replace_existing: bool = False) -> Path:
    project = project_root.resolve()
    dest = destination.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    snapshot = dest / "source_snapshot"
    if snapshot.exists() and not replace_existing:
        raise FileExistsError(f"frozen source snapshot already exists: {snapshot}; choose a new output or explicitly replace")
    if snapshot.exists():
        shutil.rmtree(snapshot)
    snapshot.mkdir(parents=True)
    copied: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []
    for source, rel, origin in _source_candidates(project):
        lowered = source.name.lower()
        if any(mark in lowered for mark in SECRET_NAME_PARTS):
            skipped.append({"path": str(source), "reason": "secret_candidate_filename"})
            continue
        data = source.read_bytes()
        secret_reason = _secret_candidate_reason(data)
        if secret_reason:
            skipped.append({"path": str(source), "reason": secret_reason})
            continue
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        copied[rel] = {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
                       "origin": origin, "source_path": str(source)}

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_status": "current_tree_snapshot; historical paper baseline relationship incomplete",
        "project_root_name": project.name,
        "git": _git_info(project),
        "source_snapshot": {"directory": "source_snapshot", "files": copied,
                            "skipped_secret_candidates": skipped},
        "profiles": {
            "Core": {"variant": "SO1_ss21_seg_overlay_composite", "complete_historical_config": "unknown; see archived run references"},
            "Full": {"variant": "SO4_grade_aux_keep", "complete_historical_config": "unknown; see archived run references"}},
        "environment_inventory": _runtime_inventory(),
        "model_assets": _model_references(project),
        "archived_runs": _archive_evidence(project),
        "limitations": [
            "Current working tree is dirty; this snapshot records its current bytes and is not asserted to be the exact paper source.",
            "No weights or clinical/case artifacts are copied. Model hashes are retained only where already evidenced by scoped run metadata.",
            "Archived runs preserve configurations and launch metadata, but historical API responses are not complete enough for exact replay.",
        ],
    }
    manifest_path = dest / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (dest / "manifest.sha256").write_text(f"{sha256_file(manifest_path)}  manifest.json\n", encoding="ascii")
    return manifest_path


def _secret_candidate_reason(data: bytes) -> str | None:
    if any(marker in data for marker in SECRET_MARKERS):
        return "private_key_marker"
    for pattern in SECRET_VALUE_PATTERNS:
        match = pattern.search(data)
        if match:
            candidate = match.group(1) if match.lastindex else match.group(0)
            lowered = candidate.lower()
            if any(word in lowered for word in (b"placeholder", b"changeme", b"your_", b"example", b"dummy")):
                continue
            return "credential_value_candidate"
    return None


def _model_references(project: Path) -> dict[str, Any]:
    # Metadata only: never hash or copy multi-gigabyte weights in the freeze CLI.
    refs = []
    for run_id, name in (("tcga_expansion_20260924_1200_d1", "D1 checkpoint"),
                         ("tcga_so1_so4_a5bridge_508_20260920", "A5 bridge checkpoint")):
        plan_path = project / "PathXAI_Seg2Report/results" / run_id / "plan.json"
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            refs.append({"name": name, "path": plan.get("checkpoint", "unknown"),
                         "sha256_reference": plan.get("checkpoint_sha256"), "evidence": str(plan_path)})
        except (OSError, json.JSONDecodeError):
            refs.append({"name": name, "path": "unknown", "sha256_reference": None, "evidence": str(plan_path)})
    refs.extend([
        {"name": "CONCH / foundation model", "path": "not resolved by this freeze", "sha256_reference": None},
        {"name": "KEEP dataset", "path": "/local/datasets/public/KEEP_dataset (archived config reference)", "sha256_reference": None},
        {"name": "Qwen service model", "path": "E:/tanja1/qwen3.6/Qwen3.6-27B-Q6_K.gguf (archived config reference)", "sha256_reference": None},
    ])
    # Add stat metadata only for exact files available on this machine.
    for ref in refs:
        p = Path(ref["path"])
        if p.is_file():
            stat = p.stat()
            ref["available_here"] = True
            ref["size_bytes"] = stat.st_size
            ref["mtime_ns"] = stat.st_mtime_ns
        else:
            ref["available_here"] = False
    return {"references": refs, "weights_copied": False}


def _archive_evidence(project: Path) -> list[dict[str, Any]]:
    runs = []
    for run_id in ("tcga_expansion_20260924_1200_d1", "tcga_so1_so4_a5bridge_508_20260920"):
        base = project / "PathXAI_Seg2Report/results" / run_id
        evidence = []
        for rel in ("README.md", "plan.json", "launch_commands.json", "provenance/pipeline.yaml"):
            p = base / rel
            if p.is_file():
                evidence.append({"path": str(p), "sha256": sha256_file(p), "size_bytes": p.stat().st_size})
        plan_path = base / "plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else {}
        code_hashes = plan.get("code_sha256", {})
        exact_match = 0
        missing, mismatch = [], []
        for rel, expected in code_hashes.items():
            file_path = base / rel
            if not file_path.is_file():
                missing.append(rel)
            elif sha256_file(file_path) != expected:
                mismatch.append(rel)
            else:
                exact_match += 1
        config_path = base / "provenance/pipeline.yaml"
        config_entry = next((x for x in evidence if x["path"].endswith("provenance/pipeline.yaml")), None)
        runs.append({"run_id": run_id, "evidence": evidence,
                     "cohort_size": plan.get("n"), "methods": plan.get("methods", {}),
                     "checkpoint": {"path": plan.get("checkpoint"), "sha256": plan.get("checkpoint_sha256")},
                     "python_executables": plan.get("python"),
                     "pipeline_config": {"snapshot_path": f"source_snapshot/archives/{run_id}/pipeline.yaml",
                                         "sha256": config_entry.get("sha256") if config_entry else None},
                     "code_hash_audit": {"expected_files": len(code_hashes), "matched": exact_match,
                                         "missing": missing, "mismatched": mismatch},
                     "output_roots": [str(base / "SO1_D1"), str(base / "SO4_D1"), str(base / "SO1_A5bridge"), str(base / "SO4_A5bridge")],
                     "replay_assessment": "partial metadata; no complete historical request/response corpus established"})
    # Older direct SO1/SO4 result directories are useful result provenance,
    # but are not treated as source snapshots and are never traversed/copied.
    for run_id, variant in (("run_20260818_so1", "SO1_ss21_seg_overlay_composite"),
                            ("run_20260819_so4", "SO4_grade_aux_keep")):
        root = project / "outputs/seg_reader_ablation" / run_id / variant
        runs.append({"run_id": run_id, "variant": variant, "output_root": str(root),
                     "present": root.is_dir(),
                     "inspection": "directory presence only; case outputs and clinical content not traversed",
                     "replay_assessment": "historical result directory; exact source/config linkage not established by this inspection"})
    return runs


def verify_manifest(manifest_path: Path, live: bool = False) -> list[str]:
    manifest_path = manifest_path.resolve()
    base = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    sidecar = base / "manifest.sha256"
    if sidecar.exists():
        expected = sidecar.read_text(encoding="ascii").split()[0]
        if sha256_file(manifest_path) != expected:
            errors.append("manifest.json checksum mismatch")
    snapshot = base / manifest["source_snapshot"]["directory"]
    for rel, info in manifest["source_snapshot"]["files"].items():
        target = snapshot / rel
        if not target.is_file():
            errors.append(f"missing snapshot file: {rel}")
            continue
        if sha256_file(target) != info["sha256"]:
            errors.append(f"snapshot checksum mismatch: {rel}")
        if live:
            source = Path(info["source_path"])
            if not source.is_file():
                errors.append(f"missing live source: {source}")
            elif sha256_file(source) != info["sha256"]:
                errors.append(f"live source drift: {source}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="copy allowlisted source bytes and write manifest")
    freeze.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    freeze.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "baselines/current")
    freeze.add_argument("--replace-existing", action="store_true", help="replace an existing frozen snapshot explicitly")
    verify = sub.add_parser("verify", help="verify snapshot integrity; optionally compare live source")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        path = freeze_baseline(args.project_root, args.output, replace_existing=args.replace_existing)
        print(path)
        return 0
    errors = verify_manifest(args.manifest, args.live)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("manifest and frozen sources verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
