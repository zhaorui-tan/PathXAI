#!/usr/bin/env python3
"""Fetch the exact pinned MedSAM3 source files and reconstruct the frozen local tree.

The upstream archive is used as a source of the recorded files. Retrieval does not
imply any license grant; consult the upstream project and the notices in this repo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance/external_medsam_source.json"
DEFAULT_DESTINATION = (
    ROOT / "vendor/report_engine/baselines/current/source_snapshot/archives/"
    "tcga_so1_so4_a5bridge_508_20260920/code/medsam3"
)
CHUNK_SIZE = 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member_name(name: str) -> tuple[str, ...]:
    if "\\" in name or name.startswith("/"):
        raise ValueError(f"Unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe archive path: {name!r}")
    return parts


def apply_local_modifications(files: dict[str, bytes]) -> None:
    """Apply the two recorded, exact local-file modifications to upstream bytes."""
    key = "lora_layers.py"
    data = files[key]
    replacements = [
        (
            b"        # Handle batch_first\n        if self.batch_first:\n",
            b"        # Handle batch_first\n"
            b"        target_dtype = self.q_proj.weight.dtype\n"
            b"        if query.dtype != target_dtype:\n"
            b"            query = query.to(dtype=target_dtype)\n"
            b"            key = key.to(dtype=target_dtype)\n"
            b"            value = value.to(dtype=target_dtype)\n"
            b"            \n"
            b"        if self.batch_first:\n",
        ),
        (
            b"        # x shape: (..., in_features)\n"
            b"        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B\n",
            b"        # x shape: (..., in_features)\n"
            b"        if x.dtype != self.lora_A.dtype:\n"
            b"            x = x.to(dtype=self.lora_A.dtype)\n"
            b"        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B\n",
        ),
        (
            b"    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
            b"        \"\"\"\n"
            b"        Forward pass: original output + LoRA output\n"
            b"        \"\"\"\n"
            b"        return self.original_layer(x) + self.lora(x)\n",
            b"    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
            b"        \"\"\"\n"
            b"        Forward pass: original output + LoRA output.\n"
            b"        Cast x to match weight dtype so it is robust to gradient\n"
            b"        checkpointing dropping the outer autocast context.\n"
            b"        \"\"\"\n"
            b"        target_dtype = self.original_layer.weight.dtype\n"
            b"        if x.dtype != target_dtype:\n"
            b"            x = x.to(dtype=target_dtype)\n"
            b"        return self.original_layer(x) + self.lora(x)\n",
        ),
    ]
    for old, new in replacements:
        if data.count(old) != 1:
            raise ValueError("Pinned lora_layers.py no longer matches the recorded local patch")
        data = data.replace(old, new, 1)
    files[key] = data

    key = "sam3/model_builder.py"
    data = files[key]
    old = b"        return_interm_layers=False,\n"
    if data.count(old) != 1:
        raise ValueError("Pinned model_builder.py no longer matches the recorded local patch")
    data = data.replace(old, b"        return_interm_layers=True,\n", 1)
    anchor = b"\ndef build_sam3_video_model(\n"
    note = (
        b"\n# def download_ckpt_from_hf():\n"
        b"#     SAM3_MODEL_ID = \"facebook/sam3.1\"\n"
        b"#     SAM3_CFG_NAME = \"config.json\"\n"
        b"#     # sam3.1 uses sam3.1_multiplex.pt (sam3.pt may not exist there)\n"
        b"#     CKPT_CANDIDATES = [\"sam3.1_multiplex.pt\", \"sam3.pt\"]\n"
        b"\n"
        b"#     _ = hf_hub_download(repo_id=SAM3_MODEL_ID, filename=SAM3_CFG_NAME)\n"
        b"\n"
        b"#     last_err = None\n"
        b"#     for ckpt_name in CKPT_CANDIDATES:\n"
        b"#         try:\n"
        b"#             return hf_hub_download(repo_id=SAM3_MODEL_ID, filename=ckpt_name)\n"
        b"#         except Exception as e:\n"
        b"#             last_err = e\n"
        b"\n"
        b"#     raise RuntimeError(\n"
        b"#         f\"Could not download any checkpoint from {SAM3_MODEL_ID}. \"\n"
        b"#         f\"Tried: {CKPT_CANDIDATES}. Last error: {last_err}\"\n"
        b"#     )\n"
    )
    if data.count(anchor) != 1:
        raise ValueError("Pinned model_builder.py no longer matches the recorded local patch")
    data = data.replace(anchor, note + anchor, 1)
    files[key] = data


def verify_files(files: dict[str, bytes], manifest: dict) -> None:
    expected = {row["path"]: row for row in manifest["files"]}
    if set(files) != set(expected):
        missing = sorted(set(expected) - set(files))
        extra = sorted(set(files) - set(expected))
        raise ValueError(f"File inventory mismatch; missing={missing[:5]}, extra={extra[:5]}")
    for rel, data in files.items():
        row = expected[rel]
        if len(data) != row["bytes"] or sha256(data) != row["final_sha256"]:
            raise ValueError(f"Reconstructed file hash mismatch: {rel}")


def verify_existing(destination: Path, manifest: dict) -> bool:
    if not destination.exists():
        return False
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("Destination exists and is not a regular directory")
    expected = {row["path"]: row for row in manifest["files"]}
    actual: dict[str, Path] = {}
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise ValueError("Destination contains a symlink")
        # Importing the provisioned modules may create interpreter caches.
        # They are not source and must not make a second verification fail.
        if "__pycache__" in path.relative_to(destination).parts:
            continue
        if path.is_file():
            actual[path.relative_to(destination).as_posix()] = path
    if set(actual) != set(expected):
        raise ValueError("Existing destination inventory differs from the pinned source tree")
    for rel, path in actual.items():
        row = expected[rel]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["final_sha256"]:
            raise ValueError(f"Existing destination hash mismatch: {rel}")
    return True


def download_archive(manifest: dict, path: Path) -> None:
    upstream = manifest["upstream"]
    request = urllib.request.Request(upstream["archive_url"], headers={"User-Agent": "PathXAI-tutorial-source-bootstrap/1"})
    digest = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(request, timeout=60) as response, path.open("xb") as output:
        final_url = response.geturl()
        if not final_url.startswith("https://codeload.github.com/"):
            raise ValueError("Archive response did not come from official codeload.github.com")
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > manifest["limits"]["max_archive_bytes"]:
                raise ValueError("Upstream archive exceeds the configured size limit")
            digest.update(chunk)
            output.write(chunk)
    if total != upstream["archive_bytes"] or digest.hexdigest() != upstream["archive_sha256"]:
        raise ValueError("Downloaded upstream archive size or SHA-256 differs from the pinned archive")


def read_pinned_files(archive_path: Path, manifest: dict) -> dict[str, bytes]:
    upstream = manifest["upstream"]
    expected = {row["path"]: row for row in manifest["files"]}
    files: dict[str, bytes] = {}
    total_uncompressed = 0
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len(infos) > manifest["limits"]["max_members"]:
            raise ValueError("Upstream archive contains too many members")
        seen: set[str] = set()
        for info in infos:
            parts = safe_member_name(info.filename.rstrip("/"))
            if info.filename in seen:
                raise ValueError(f"Duplicate archive entry: {info.filename}")
            seen.add(info.filename)
            if info.file_size < 0:
                raise ValueError("Invalid negative archive member size")
            total_uncompressed += info.file_size
            if total_uncompressed > manifest["limits"]["max_uncompressed_bytes"]:
                raise ValueError("Upstream archive exceeds uncompressed size limit")
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ValueError(f"Archive contains a symlink: {info.filename}")
            prefix = tuple(PurePosixPath(upstream["archive_root"].rstrip("/")).parts)
            if parts[:len(prefix)] != prefix:
                raise ValueError(f"Unexpected archive root: {info.filename}")
            rel_parts = parts[len(prefix):]
            if not rel_parts:
                continue
            rel = "/".join(rel_parts)
            if rel not in expected:
                continue
            if info.is_dir():
                continue
            data = archive.read(info)
            if len(data) != info.file_size or sha256(data) != expected[rel]["upstream_sha256"]:
                raise ValueError(f"Upstream file hash mismatch: {rel}")
            files[rel] = data
    return files


def install(destination: Path, files: dict[str, bytes], manifest: dict) -> None:
    verify_files(files, manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Destination already exists; refusing to replace source files")
    stage = Path(tempfile.mkdtemp(prefix=".medsam3-source-", dir=destination.parent))
    try:
        for rel, data in files.items():
            out = stage.joinpath(*PurePosixPath(rel).parts)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("xb") as stream:
                stream.write(data)
        os.replace(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION,
                        help="Source tree destination (defaults to the frozen tutorial runtime path)")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    destination = args.destination.resolve()
    if verify_existing(destination, manifest):
        print(f"Verified existing pinned MedSAM3 source: {manifest['limits']['expected_file_count']} files, {manifest['limits']['expected_source_bytes']} bytes")
        return 0
    with tempfile.TemporaryDirectory(prefix="pathxai-medsam3-fetch-") as temp_dir:
        archive_path = Path(temp_dir) / "upstream.zip"
        download_archive(manifest, archive_path)
        files = read_pinned_files(archive_path, manifest)
        apply_local_modifications(files)
        verify_files(files, manifest)
        install(destination, files, manifest)
    print(f"Installed and verified pinned MedSAM3 source: {manifest['limits']['expected_file_count']} files, {manifest['limits']['expected_source_bytes']} bytes")
    print("Upstream source retrieval is provenance, not a license grant; follow upstream terms and bundled notices.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"MedSAM3 source bootstrap failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
