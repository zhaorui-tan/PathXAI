from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pathxai_agent.storage import SEG_CACHE_DIR, TMP_DIR


DEFAULT_CKPT = Path(
    "/local/PathXAI/medsam3_combined_runs_4096_fusion_fixed_data_fixed2/"
    "stage3_ablation_group_d/stage3_ablation_D1/"
    "run_20260619_144518_stage3_tissue/best_model.pth"
)


@dataclass
class MedSAMRoiSegmenter:
    ckpt_path: str | Path = DEFAULT_CKPT
    device: str = "cuda:0"
    max_rois: int = 5
    patch_size: int = 4096
    stride: int = 2048
    cache_dir: str | Path = SEG_CACHE_DIR / "medsam3_d1"
    python_path: str | Path = Path("/local/conda_staffs/conda_envs/medsam/bin/python")
    use_subprocess: bool = True

    def __post_init__(self) -> None:
        self.ckpt_path = Path(self.ckpt_path)
        self.cache_dir = Path(self.cache_dir)
        self.python_path = Path(self.python_path)
        self._loaded = False
        self._model = None
        self._transform = None
        self._tissue_prompts = None
        self._classes = None

    def load(self) -> None:
        if self._loaded:
            return
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"MedSAM checkpoint not found: {self.ckpt_path}")
        pathxai_root = Path("/local/PathXAI")
        if str(pathxai_root) not in sys.path:
            sys.path.insert(0, str(pathxai_root))
        import torch
        import train_medsam_FINAL as medsam

        device = torch.device(self.device if torch.cuda.is_available() or not self.device.startswith("cuda") else "cpu")
        model, prompt_enc, adapter, _lora_cfg, _tile_classifier, _ssl_head, _pixel_ssl_head = medsam.build_stage3_model(
            str(self.ckpt_path), device
        )
        ckpt = torch.load(self.ckpt_path, map_location=device, weights_only=False)
        raw_model = model.module if hasattr(model, "module") else model
        # build_stage3_model already loads most states through nested builders,
        # but re-load the saved components defensively for exact checkpoint use.
        if ckpt.get("lora_state"):
            sd = raw_model.state_dict()
            sd.update({k: v.to(device) for k, v in ckpt["lora_state"].items()})
            raw_model.load_state_dict(sd)
        if ckpt.get("prompt_enc"):
            prompt_enc.load_state_dict({
                k: v.to(device) if hasattr(v, "to") else v
                for k, v in ckpt["prompt_enc"].items()
            }, strict=False)
        if adapter is not None and ckpt.get("adapter_state"):
            adapter.load_state_dict({
                k: v.to(device) if hasattr(v, "to") else v
                for k, v in ckpt["adapter_state"].items()
            }, strict=False)

        model.eval()
        prompt_enc.eval()
        if adapter is not None:
            adapter.eval()
        self._model = model
        self._device = device
        self._medsam = medsam
        self._transform = medsam._build_transform(medsam.SAM3_RESOLUTION)
        self._tissue_prompts = dict(medsam.TISSUE_PROMPT_TEXTS)
        self._classes = dict(medsam.TISSUE_CLASSES)
        self._loaded = True

    def segment_rois(self, rois: list[dict], out_dir: str | Path) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.use_subprocess and os.environ.get("MEDSAM_ROI_WORKER") != "1":
            return self._segment_rois_subprocess(rois, out_dir)
        self.load()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for idx, row in enumerate(rois[: self.max_rois], start=1):
            try:
                results.append(self._segment_one(idx, row, out_dir))
            except Exception as exc:
                results.append({
                    "roi_index": row.get("roi_index", idx),
                    "roi_id": row.get("roi_id"),
                    "image_path": row.get("image_path"),
                    "error": str(exc),
                })
        summary = _aggregate_results(results, self._classes or {})
        payload = {
            "available": True,
            "checkpoint": str(self.ckpt_path),
            "device": str(getattr(self, "_device", self.device)),
            "max_rois": self.max_rois,
            "class_names": self._classes,
            "roi_results": results,
            "summary": summary,
        }
        (out_dir / "segmentation_results.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return payload

    def _segment_rois_subprocess(self, rois: list[dict], out_dir: Path) -> dict:
        if not self.python_path.exists():
            raise FileNotFoundError(f"MedSAM python not found: {self.python_path}")
        work_dir = out_dir / "_worker"
        work_dir.mkdir(parents=True, exist_ok=True)
        input_path = work_dir / "rois.json"
        output_path = work_dir / "segmentation_results.json"
        input_path.write_text(json.dumps(rois[: self.max_rois], indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_dir = TMP_DIR / "medsam"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "MEDSAM_ROI_WORKER": "1",
            "TMPDIR": str(tmp_dir),
            "MPLCONFIGDIR": str(tmp_dir),
            "NUMBA_CACHE_DIR": str(tmp_dir),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": "/local/PathXAI_Agent/src:/local/PathXAI:/local/MedSAM/MedSAM3",
        })
        cmd = [
            str(self.python_path),
            "-m",
            "pathxai_agent.segmentation.medsam_roi",
            "--input_rois", str(input_path),
            "--out_dir", str(out_dir),
            "--output_json", str(output_path),
            "--ckpt", str(self.ckpt_path),
            "--device", str(self.device),
            "--max_rois", str(self.max_rois),
            "--patch_size", str(self.patch_size),
            "--stride", str(self.stride),
            "--cache_dir", str(self.cache_dir),
        ]
        proc = subprocess.run(
            cmd,
            cwd="/local/PathXAI_Agent",
            env=env,
            text=True,
            capture_output=True,
            timeout=None,
            check=False,
        )
        if proc.returncode != 0:
            payload = {
                "available": False,
                "error": f"MedSAM subprocess failed with code {proc.returncode}",
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
                "command": cmd,
            }
            (out_dir / "segmentation_error.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return payload
        if not output_path.exists():
            payload = {
                "available": False,
                "error": "MedSAM subprocess completed but did not write output JSON",
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
                "command": cmd,
            }
            (out_dir / "segmentation_error.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return payload
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        payload["subprocess"] = {
            "python": str(self.python_path),
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
        (out_dir / "segmentation_results.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return payload

    def _segment_one(self, idx: int, row: dict, out_dir: Path) -> dict:
        image_path = Path(str(row.get("image_path") or ""))
        if not image_path.exists():
            raise FileNotFoundError(str(image_path))
        cache_key = _cache_key(
            image_path=image_path,
            roi_id=str(row.get("roi_id") or ""),
            ckpt_path=self.ckpt_path,
            patch_size=self.patch_size,
            stride=self.stride,
        )
        roi_cache = self.cache_dir / cache_key
        result_path = roi_cache / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["cache_hit"] = True
            return result
        roi_cache.mkdir(parents=True, exist_ok=True)
        pil = Image.open(image_path).convert("RGB")
        w, h = pil.size
        scores = np.zeros((len(self._classes), h, w), dtype=np.float32)
        # Include background score as zeros; foreground prompts compete by argmax.
        for cid, prompt in sorted(self._tissue_prompts.items()):
            if int(cid) <= 0:
                continue
            scores[int(cid)] = self._medsam.sliding_window_predict_class(
                self._model,
                self._transform,
                pil,
                prompt,
                self._device,
                patch_size=max(int(self.patch_size), int(self._medsam.SAM3_RESOLUTION)),
                stride=max(1, int(self.stride)),
                is_nuclei_prompt=False,
            )
        pred = scores.argmax(axis=0).astype(np.uint8)
        roi_id = str(row.get("roi_id") or f"roi_{idx}")
        mask_path = roi_cache / "mask.png"
        overlay_path = roi_cache / "overlay.jpg"
        Image.fromarray(pred).save(mask_path)
        _save_overlay(pil, pred, overlay_path)
        stats = _mask_stats(pred, self._classes)
        result = {
            "roi_index": row.get("roi_index", idx),
            "roi_id": roi_id,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "width": w,
            "height": h,
            "stats": stats,
            "cache_key": cache_key,
            "cache_hit": False,
        }
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result


def _mask_stats(mask: np.ndarray, classes: dict[int, str]) -> dict:
    total = int(mask.size)
    counts = {int(cid): int((mask == int(cid)).sum()) for cid in sorted(classes)}
    fractions = {str(cid): (counts[int(cid)] / total if total else 0.0) for cid in counts}
    named = {
        str(classes.get(int(cid), cid)): {
            "class_id": int(cid),
            "pixels": counts[int(cid)],
            "fraction": fractions[str(cid)],
        }
        for cid in counts
    }
    foreground = total - counts.get(0, 0)
    return {
        "total_pixels": total,
        "foreground_fraction": foreground / total if total else 0.0,
        "class_pixel_counts": {str(k): v for k, v in counts.items()},
        "class_fractions": fractions,
        "named_class_fractions": named,
    }


def _aggregate_results(results: list[dict], classes: dict[int, str]) -> dict:
    valid = [row for row in results if row.get("stats")]
    if not valid:
        return {"valid_roi_count": 0}
    avg = {}
    for cid, name in sorted(classes.items()):
        vals = [
            row["stats"]["class_fractions"].get(str(cid), 0.0)
            for row in valid
        ]
        avg[str(name)] = {
            "class_id": int(cid),
            "mean_fraction": float(np.mean(vals)) if vals else 0.0,
            "max_fraction": float(np.max(vals)) if vals else 0.0,
        }
    return {
        "valid_roi_count": len(valid),
        "mean_foreground_fraction": float(np.mean([row["stats"]["foreground_fraction"] for row in valid])),
        "mean_class_fractions": avg,
    }


def _save_overlay(image: Image.Image, mask: np.ndarray, path: Path) -> None:
    base = np.asarray(image.convert("RGB")).astype(np.float32)
    palette = np.array([
        [0, 0, 0],
        [220, 38, 38],
        [37, 99, 235],
        [22, 163, 74],
        [234, 179, 8],
        [147, 51, 234],
        [14, 165, 233],
    ], dtype=np.float32)
    colors = palette[np.clip(mask, 0, len(palette) - 1)]
    alpha = (mask > 0).astype(np.float32)[..., None] * 0.45
    out = (base * (1 - alpha) + colors * alpha).clip(0, 255).astype(np.uint8)
    Image.fromarray(out).save(path, quality=90)


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)[:120]


def _cache_key(
    image_path: Path,
    roi_id: str,
    ckpt_path: Path,
    patch_size: int,
    stride: int,
) -> str:
    payload = {
        "image_path": str(image_path.resolve()),
        "image_mtime": image_path.stat().st_mtime,
        "roi_id": roi_id,
        "ckpt_path": str(ckpt_path.resolve()),
        "ckpt_mtime": ckpt_path.stat().st_mtime if ckpt_path.exists() else None,
        "patch_size": int(patch_size),
        "stride": int(stride),
        "version": "medsam_roi_cache_v2",
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MedSAM ROI segmentation worker.")
    parser.add_argument("--input_rois", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_rois", type=int, default=5)
    parser.add_argument("--patch_size", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--cache_dir", type=Path, default=SEG_CACHE_DIR / "medsam3_d1")
    args = parser.parse_args()

    rois = json.loads(args.input_rois.read_text(encoding="utf-8"))
    segmenter = MedSAMRoiSegmenter(
        ckpt_path=args.ckpt,
        device=args.device,
        max_rois=args.max_rois,
        patch_size=args.patch_size,
        stride=args.stride,
        cache_dir=args.cache_dir,
        use_subprocess=False,
    )
    payload = segmenter.segment_rois(rois, args.out_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    _main()
