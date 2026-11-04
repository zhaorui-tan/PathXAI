from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py

from pathxai_agent.wsi.roi_candidates import RoiCandidate


def read_pathagent_h5_rois(
    h5_path: str | Path,
    slide_id: Optional[str] = None,
    patch_size: int = 4096,
    level: int = 0,
    max_rois: Optional[int] = None,
    source: str = "pathagent_h5",
) -> List[RoiCandidate]:
    """Read CLAM/PathAgent H5 coordinates into the project ROI contract."""
    h5_path = Path(h5_path)
    if slide_id is None:
        slide_id = h5_path.stem
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"{h5_path} does not contain a 'coords' dataset")
        coords = f["coords"][:]
        if "patch_size" in f.attrs:
            try:
                patch_size = int(f.attrs["patch_size"])
            except Exception:
                pass
        elif "patch_size" in f:
            try:
                patch_size = int(f["patch_size"][()])
            except Exception:
                pass

    rois = []
    for idx, (x, y) in enumerate(coords):
        if max_rois is not None and len(rois) >= max_rois:
            break
        x_i = int(x)
        y_i = int(y)
        roi_id = f"{slide_id}_pa{idx:05d}_x{x_i}_y{y_i}_l{level}_w{patch_size}_h{patch_size}"
        rois.append(
            RoiCandidate(
                slide_id=slide_id,
                roi_id=roi_id,
                x=x_i,
                y=y_i,
                width=patch_size,
                height=patch_size,
                level=level,
                source=source,
                score=0.0,
            )
        )
    return rois


def write_rois_json(rois: Iterable[RoiCandidate], out_path: str | Path, metadata: Optional[dict] = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "rois": [asdict(roi) for roi in rois],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pathagent_coordinate_generation(
    source_dir: str | Path,
    save_dir: str | Path,
    pathagent_root: str | Path = "/local/PathAgent-main",
    clam_root: Optional[str | Path] = None,
    patch_size: int = 4096,
    step_size: Optional[int] = None,
    patch_level: int = 0,
    preset: Optional[str | Path] = None,
    python_executable: str = sys.executable,
    extra_args: Optional[List[str]] = None,
) -> subprocess.CompletedProcess:
    """Call PathAgent's official CLAM-style coordinate generation script.

    This requires CLAM's `wsi_core` package. Provide `clam_root` or set
    `CLAM_ROOT` if it is not already importable.
    """
    source_dir = Path(source_dir)
    save_dir = Path(save_dir)
    pathagent_root = Path(pathagent_root)
    script = pathagent_root / "data_preparation_script" / "coordinate_generation.py"
    if not script.exists():
        raise FileNotFoundError(f"PathAgent coordinate_generation.py not found: {script}")
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_executable,
        str(script),
        "--source",
        str(source_dir),
        "--save_dir",
        str(save_dir),
        "--step_size",
        str(step_size or patch_size),
        "--patch_size",
        str(patch_size),
        "--patch_level",
        str(patch_level),
        "--seg",
        "--patch",
    ]
    if preset:
        cmd.extend(["--preset", str(preset)])
    if extra_args:
        cmd.extend(extra_args)

    env = None
    if clam_root:
        import os

        env = dict(os.environ)
        env["PYTHONPATH"] = f"{Path(clam_root)}:{env.get('PYTHONPATH', '')}"

    return subprocess.run(cmd, check=True, env=env)


def prepare_single_slide_source(slide_path: str | Path, source_dir: str | Path, symlink: bool = True) -> Path:
    """Create a one-slide source directory compatible with PathAgent script."""
    slide_path = Path(slide_path).resolve()
    source_dir = Path(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    target = source_dir / slide_path.name
    if target.exists():
        return target
    if symlink:
        target.symlink_to(slide_path)
    else:
        shutil.copy2(slide_path, target)
    return target


def expected_h5_path(save_dir: str | Path, slide_path: str | Path) -> Path:
    return Path(save_dir) / "patches" / f"{Path(slide_path).stem}.h5"


def pathagent_h5_summary(h5_path: str | Path) -> Dict[str, object]:
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        attrs = {k: _jsonable(v) for k, v in f.attrs.items()}
        coord_count = int(f["coords"].shape[0]) if "coords" in f else 0
        first_coords = f["coords"][:5].tolist() if "coords" in f else []
    return {
        "h5_path": str(h5_path),
        "keys": keys,
        "attrs": attrs,
        "coord_count": coord_count,
        "first_coords": first_coords,
    }


def _jsonable(value):
    try:
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return value
    except Exception:
        return str(value)
