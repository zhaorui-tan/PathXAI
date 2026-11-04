#!/usr/bin/env python3
"""Verify the upload_ready bundle; upload only when --upload and --repo-id are set.

Uses HF_TOKEN when present, otherwise an existing `hf auth login` credential.
Token values are never printed. The target repository must already exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BUNDLE = HERE / "upload_ready"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--upload", action="store_true", help="Upload the verified bundle to Hugging Face")
    parser.add_argument("--repo-id", help="Existing HF model repository, OWNER/REPO")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Token environment variable name")
    args = parser.parse_args()
    if args.upload and not args.repo_id:
        parser.error("--upload requires --repo-id OWNER/REPO")
    if args.repo_id and not args.upload:
        parser.error("--repo-id requires --upload")

    try:
        bundle = args.bundle.expanduser().resolve(strict=True)
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != "pathxai-model-release-manifest-v1":
            raise ValueError("unsupported manifest schema")
        assets = manifest.get("assets", [])
        if len(assets) != 1 or assets[0].get("id") != "pathxai_d1_adapter":
            raise ValueError("expected exactly one pathxai_d1_adapter asset")
        asset = assets[0]
        weight = bundle / asset["repo_path"]
        if not weight.is_file():
            raise FileNotFoundError(f"missing manifest asset: {asset['repo_path']}")
        if weight.stat().st_size != int(asset["bytes"]):
            raise ValueError("adapter size does not match manifest")
        actual = sha256_file(weight)
        if actual != asset["sha256"]:
            raise ValueError("adapter SHA-256 does not match manifest")
        for name in ("README.md", "manifest.json", "LICENSE"):
            if not (bundle / name).is_file():
                raise FileNotFoundError(f"upload bundle missing {name}")
        print(f"Verified bundle: {bundle}")
        print(f"Adapter: {asset['repo_path']} ({asset['bytes']} bytes; sha256 {actual})")
        print("NOTICE: SAM License is included; review derivative-distribution compatibility and PathXAI rights before publication.")
        if not args.upload:
            print("Dry run only; no network upload attempted.")
            print(f"Credential status: {args.token_env} is {'set (value hidden)' if os.environ.get(args.token_env) else 'not set; saved `hf auth login` may still be used'}.")
            return 0

        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError("install huggingface_hub before uploading") from exc
        api = HfApi(token=os.environ.get(args.token_env) or True)
        try:
            api.repo_info(repo_id=args.repo_id, repo_type="model")
        except Exception as exc:
            raise RuntimeError("target repository is not accessible; create/configure it first") from exc
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(bundle),
            commit_message="Upload verified PathXAI D1 inference adaptation bundle",
        )
        print(f"Uploaded verified bundle to {args.repo_id}")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
