from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_PATHAGENT_ROOT = Path("/local/PathAgent-main")
DEFAULT_CANDIDATE_ROOTS = [
    Path("/local"),
    Path("/local/models"),
    Path("/local/checkpoints"),
    Path("/local/MMF/.cache/huggingface"),
    Path("/local/datasets/public/Wsi-Caption-master/models"),
    Path("/local/WSI-VQA/WSI-VQA-master/models"),
]


@dataclass
class PathAgentAssetStatus:
    pathagent_root: str
    pathagent_exists: bool
    coordinate_generation_script: Optional[str]
    patch_generation_script: Optional[str]
    clam_available: bool
    clam_root: Optional[str]
    plip_candidates: List[str]
    qwen_candidates: List[str]
    patho_r1_candidates: List[str]
    notes: List[str]


def check_pathagent_assets(
    pathagent_root: Path = DEFAULT_PATHAGENT_ROOT,
    clam_root: Optional[Path] = None,
    candidate_roots: Optional[Iterable[Path]] = None,
) -> Dict[str, object]:
    pathagent_root = pathagent_root.resolve()
    coordinate_script = pathagent_root / "data_preparation_script" / "coordinate_generation.py"
    patch_script = pathagent_root / "data_preparation_script" / "patch_generation.py"
    notes: List[str] = []

    detected_clam_root = _detect_clam_root(clam_root)
    clam_available = detected_clam_root is not None
    if not clam_available:
        notes.append(
            "CLAM/wsi_core was not detected. PathAgent coordinate generation needs "
            "a CLAM-compatible checkout on PYTHONPATH or --clam_root."
        )

    roots = list(candidate_roots or DEFAULT_CANDIDATE_ROOTS)
    plip = _find_model_dirs(roots, ("plip",))
    qwen = _find_model_dirs(roots, ("qwen", "qwen2", "qwen2.5"))
    patho_r1 = _find_model_dirs(roots, ("patho", "pathor", "patho-r1", "patho_r1"))
    if not plip:
        notes.append("No local PLIP checkpoint directory was found in default candidate roots.")
    if not qwen:
        notes.append("No local Qwen checkpoint directory was found in default candidate roots.")
    if not patho_r1:
        notes.append("No local Patho-R1 checkpoint directory was found in default candidate roots.")

    status = PathAgentAssetStatus(
        pathagent_root=str(pathagent_root),
        pathagent_exists=pathagent_root.exists(),
        coordinate_generation_script=str(coordinate_script) if coordinate_script.exists() else None,
        patch_generation_script=str(patch_script) if patch_script.exists() else None,
        clam_available=clam_available,
        clam_root=str(detected_clam_root) if detected_clam_root else None,
        plip_candidates=plip,
        qwen_candidates=qwen,
        patho_r1_candidates=patho_r1,
        notes=notes,
    )
    return asdict(status)


def _detect_clam_root(clam_root: Optional[Path]) -> Optional[Path]:
    candidates = []
    if clam_root:
        candidates.append(clam_root)
    env = os.environ.get("CLAM_ROOT")
    if env:
        candidates.append(Path(env))
    candidates.extend([Path("/local/CLAM"), Path("/local/clam"), Path("/local/CLAM-master")])
    for root in candidates:
        if (root / "wsi_core" / "WholeSlideImage.py").exists():
            return root.resolve()
    return None


def _find_model_dirs(roots: Iterable[Path], needles: tuple[str, ...], max_hits: int = 20) -> List[str]:
    hits: List[str] = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if len(hits) >= max_hits:
                return hits
            if not path.is_dir():
                continue
            text = str(path).lower()
            if not any(n in text for n in needles):
                continue
            has_modelish_file = any(
                (path / name).exists()
                for name in ("config.json", "pytorch_model.bin", "model.safetensors", "tokenizer.json")
            )
            if has_modelish_file or any(part in text for part in needles):
                resolved = str(path.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    hits.append(resolved)
    return hits
