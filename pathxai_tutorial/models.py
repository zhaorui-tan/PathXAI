"""Download pinned adapters and separately authorized third-party resources."""
import hashlib
import json
from pathlib import Path
import re
from .paths import ROOT, ENGINE

def verify_asset(path, spec):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size != spec["bytes"]:
        raise ValueError(f"Missing/wrong-size model asset: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != spec["sha256"]:
        raise ValueError(f"Model checksum mismatch: {path.name}")
    return path

def download_models(*, d1_repo_id, d1_revision, version="SO1", token=None,
                    cache_dir="models/hf_cache", local_d1=None, local_sam3=None,
                    local_conch=None, local_keep=None):
    """Each remote resource needs the user's own access permission.

    SAM3/CONCH/KEEP are fetched from their original repositories, never our D1 repo.
    The custom repository's revision is mandatory and immutable. Local overrides
    still require exact size/hash matching against the shipped manifests.
    """
    from huggingface_hub import hf_hub_download
    if version.upper() not in {"SO1", "SO4", "CORE", "FULL"}:
        raise ValueError("version must be SO1 or SO4")
    release = json.loads((ROOT / "model_release" / "manifest.json").read_text())
    d1 = next(a for a in release["assets"] if a["id"] == "pathxai_d1_adapter")
    originals = {a["id"]: a for a in json.loads((ENGINE / "assets/model_references.json").read_text())["assets"]}
    cache = str(Path(cache_dir).expanduser().resolve())
    def fetch(spec, *, local=None, repo=None, revision=None, filename=None, repo_type="model"):
        if local:
            return verify_asset(local, spec)
        repo = repo or spec.get("repository")
        revision = revision or spec.get("revision")
        if not repo or not re.fullmatch(r"[0-9a-fA-F]{40}", revision or ""):
            raise ValueError("A repository and its exact 40-character commit revision are required.")
        path = hf_hub_download(repo_id=repo, repo_type=repo_type,
                               revision=revision, filename=filename or spec["filename"],
                               token=token, cache_dir=cache)
        return verify_asset(path, spec)
    result = {
        "d1_path": fetch(d1, local=local_d1, repo=d1_repo_id, revision=d1_revision, filename=d1["repo_path"]),
        "sam3_path": fetch(originals["sam3"], local=local_sam3),
        "conch_path": fetch(originals["conch"], local=local_conch),
    }
    if version.upper() in {"SO4", "FULL"}:
        keep = json.loads((ENGINE / "assets/keep_references.json").read_text())
        paths = [fetch(s, local=Path(local_keep) / s["filename"] if local_keep else None,
                       repo=keep["dataset"]["repository"], revision=keep["dataset"]["revision"],
                       repo_type="dataset") for s in keep["assets"]]
        # hf_hub_download returns snapshot symlinks; verification resolves them to
        # blobs, so use the snapshot directory with canonical filenames for KEEP.
        if local_keep:
            result["keep_dir"] = Path(local_keep).expanduser().resolve()
        else:
            reference = hf_hub_download(repo_id=keep["dataset"]["repository"], repo_type="dataset",
                revision=keep["dataset"]["revision"], filename=keep["assets"][0]["filename"], token=token, cache_dir=cache)
            result["keep_dir"] = Path(reference).parent
    return result

