"""Read-only import/version check for the two PathXAI interpreter roles.

This checks imports and framework versions only. It does not construct a model,
open a slide, access model weights, contact an API, or alter the environment.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_CODE = (APP_ROOT / "vendor" / "report_engine" / "baselines" / "current"
                / "source_snapshot" / "archives" / "tcga_so1_so4_a5bridge_508_20260920" / "code")
PIPELINE_V2 = APP_ROOT / "vendor" / "report_engine" / "baselines" / "current" / "source_snapshot" / "v2" / "src"

EXPECTED = {
    "notebook": {"python": (3, 10), "torch": "2.12.0", "torchvision": "0.27.0"},
    "segmentation": {"python": (3, 12), "torch": "2.11.0", "torchvision": "0.26.0"},
}


def _module_checks(role: str) -> list[str]:
    common = ["torch", "torchvision", "PIL", "yaml", "requests"]
    if role == "notebook":
        return common + ["lxml.html", "ipykernel", "notebook", "conch", "pathxai_agent_v2.config.settings",
                         "pathxai_agent.segmentation.medsam_roi", "pathxai_agent.segmentation.seg_visual"]
    return common + ["numpy", "matplotlib", "scipy.ndimage", "skimage.transform", "tqdm",
                     "sam3.model_builder", "sam3.train.data.collator",
                     "sam3.train.data.sam3_image_dataset", "sam3.train.transforms.basic_for_api",
                     "lora_layers", "all_datasets", "combine_dataset"]


def _configure_source_paths(role: str) -> dict[str, str]:
    paths: dict[str, Path]
    if role == "notebook":
        paths = {
            "pipeline_v2_src": PIPELINE_V2,
            "archived_v1_src": ARCHIVE_CODE / "v1",
        }
    else:
        paths = {
            "medsam3_source": ARCHIVE_CODE / "medsam3",
            "segmentation_runtime": ARCHIVE_CODE / "seg_runtime",
            "segmentation_datasets": ARCHIVE_CODE / "seg_runtime" / "datasets",
            "archived_v1_src": ARCHIVE_CODE / "v1",
        }
    for path in paths.values():
        if path.is_dir():
            sys.path.insert(0, str(path))
    return {name: str(path) for name, path in paths.items()}


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check(role: str) -> dict[str, Any]:
    expected = EXPECTED[role]
    source_paths = _configure_source_paths(role)
    results = []
    for module_name in _module_checks(role):
        try:
            module = importlib.import_module(module_name)
            results.append({"module": module_name, "ok": True,
                            "origin": getattr(module, "__file__", None)})
        except Exception as exc:  # record all missing/broken imports in one report
            results.append({"module": module_name, "ok": False,
                            "error": f"{type(exc).__name__}: {exc}"})

    package_versions = {name: _distribution_version(name) for name in
                        ("torch", "torchvision", "conch", "lxml", "Pillow", "PyYAML", "ipykernel", "notebook",
                         "hydra-core", "iopath", "open-clip-torch")}
    torch_module = sys.modules.get("torch")
    actual_torch = getattr(torch_module, "__version__", None)
    actual_vision = _distribution_version("torchvision")
    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    try:
        cuda_available = bool(torch_module.cuda.is_available()) if torch_module else None
    except Exception:
        cuda_available = None

    version_checks = {
        "python_major_minor": list(sys.version_info[:2]) == list(expected["python"]),
        "torch": bool(actual_torch and actual_torch.split("+", 1)[0] == expected["torch"]),
        "torchvision": bool(actual_vision and actual_vision.split("+", 1)[0] == expected["torchvision"]),
    }
    imports_ok = all(item["ok"] for item in results)
    return {
        "role": role,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "expected": {"python": ".".join(map(str, expected["python"])),
                     "torch": expected["torch"], "torchvision": expected["torchvision"]},
        "version_checks": version_checks,
        "package_versions": package_versions,
        "cuda_runtime_version": cuda_version,
        "cuda_available": cuda_available,
        "source_paths": source_paths,
        "imports": results,
        "ok": all(version_checks.values()) and imports_ok,
        "note": "No model, weights, slide, or API were accessed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=tuple(EXPECTED),
                        help="notebook (Python 3.10) or segmentation (Python 3.12)")
    args = parser.parse_args()
    report = check(args.role)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
