"""Relocation bootstrap for the archived MedSAM ROI subprocess.

It preloads the registered frozen `train_medsam_FINAL.py` and MedSAM3 source
before executing the unchanged archived `pathxai_agent.segmentation.medsam_roi`
CLI. The wrapper's own 0.55 GPU memory cap remains in effect.
"""

from __future__ import annotations

import importlib.util
import os
import runpy
import sys
from pathlib import Path


def _is_unconfigured_research_path(value: str) -> bool:
    normalized = os.path.normcase(os.path.normpath(value))
    forbidden_fragments = (
        os.path.normcase(os.path.normpath("/local/MedSAM/MedSAM3")),
        os.path.normcase(os.path.normpath("/local/PathXAI_Agent/src")),
        os.path.normcase(os.path.normpath("/local/PathXAI_Agent_v2/PathXAI_Seg2Report/results")),
    )
    return any(normalized == fragment or normalized.startswith(fragment + os.sep) for fragment in forbidden_fragments)


def preload_frozen_training_runtime(
    *,
    v1_source: Path,
    segmentation_runtime: Path,
    medsam3_source: Path,
) -> None:
    for root in (v1_source, segmentation_runtime, medsam3_source):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

    class FrozenPath(list):
        """Ignore the archived live-root insertion; keep configured roots first."""

        def insert(self, index, value):
            if _is_unconfigured_research_path(str(value)):
                return
            super().insert(index, value)

    sys.path = FrozenPath(sys.path)
    # The archived wrapper applies this same cap before loading the checkpoint.
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.55, 0)

    # The frozen train runtime requests HF's floating checkpoint by default.
    # Keep its model construction and flags intact while routing that one
    # asset lookup to the explicitly configured, revision-pinned local file.
    import sam3.model_builder as sam3_builder
    sam3_checkpoint = Path(os.environ["PATHXAI_SAM3_CHECKPOINT"]).resolve()
    if not sam3_checkpoint.is_file():
        raise FileNotFoundError(f"Configured SAM3 checkpoint missing: {sam3_checkpoint}")
    sam3_builder.download_ckpt_from_hf = lambda: str(sam3_checkpoint)

    entry = segmentation_runtime / "train_medsam_FINAL.py"
    if not entry.is_file():
        raise FileNotFoundError(f"Frozen segmentation runtime entry missing: {entry}")
    spec = importlib.util.spec_from_file_location("train_medsam_FINAL", entry)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load frozen segmentation runtime: {entry}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_medsam_FINAL"] = module
    spec.loader.exec_module(module)
    # The unchanged runtime script imports packages from its hard-coded
    # historical path, but this adapter has already placed the configured
    # archived MedSAM3 source ahead of that path. Point its tokenizer lookup
    # at the same configured source root.
    module.MEDSAM3_ROOT = str(medsam3_source)
    bpe_path = Path(os.environ["PATHXAI_SAM3_BPE"]).resolve()
    if not bpe_path.is_file():
        raise FileNotFoundError(f"Configured MedSAM3 BPE asset missing: {bpe_path}")
    module.BPE_PATH = str(bpe_path)


def main() -> None:
    v1_source = Path(os.environ["PATHXAI_V1_SOURCE"]).resolve()
    runtime = Path(os.environ["PATHXAI_SEG_RUNTIME"]).resolve()
    medsam3 = Path(os.environ["PATHXAI_MEDSAM3_SOURCE"]).resolve()
    preload_frozen_training_runtime(
        v1_source=v1_source,
        segmentation_runtime=runtime,
        medsam3_source=medsam3,
    )
    runpy.run_module("pathxai_agent.segmentation.medsam_roi", run_name="__main__")


if __name__ == "__main__":
    main()

