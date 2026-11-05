"""Reconstruct frozen sample candidate crops from source WSIs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image


_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "sample_candidates.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_rgb(image: Image.Image) -> str:
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def _open_slide(path: Path):
    """Use the same OpenSlide-first reader convention as the archived pipeline."""
    try:
        import openslide  # type: ignore

        return openslide.OpenSlide(str(path))
    except Exception as openslide_error:
        try:
            import tiffslide  # type: ignore

            return tiffslide.TiffSlide(str(path))
        except Exception as tiffslide_error:
            raise RuntimeError(
                f"Cannot open source WSI {path}: OpenSlide={openslide_error}; "
                f"TiffSlide={tiffslide_error}"
            ) from tiffslide_error


def prepare_sample_candidates(
    slide_paths: Iterable[str | Path],
    output_dir: str | Path,
    progress: Callable[[str], object] = print,
) -> list[dict]:
    """Recreate the frozen input ROIs from the original source slides.

    The incoming paths must be exactly the source WSI basenames recorded in
    the manifest. Each output crop is read at its recorded level-0 coordinates,
    decoded RGB pixels are checked against the frozen reference, and the PNG
    byte hash is also checked. Any discrepancy raises and prevents returning a
    partially valid candidate list.
    """
    config = json.loads(_CONFIG.read_text(encoding="utf-8"))
    case_id = config.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"Invalid candidate manifest case: {case_id!r}")
    source_slides = config.get("source_slides")
    candidate_rois = config.get("candidate_rois")
    if not isinstance(source_slides, list) or not source_slides:
        raise ValueError("Candidate manifest must contain at least one source slide")
    if not isinstance(candidate_rois, list) or not candidate_rois:
        raise ValueError("Candidate manifest must contain at least one candidate ROI")
    expected_count = config.get("candidate_count")
    if expected_count != len(candidate_rois):
        raise ValueError(
            f"Candidate manifest count mismatch: candidate_count={expected_count!r}, "
            f"candidate_rois={len(candidate_rois)}"
        )

    supplied: dict[str, Path] = {}
    for raw in slide_paths:
        path = Path(raw).expanduser().resolve()
        if path.name in supplied:
            raise ValueError(f"Duplicate source slide basename: {path.name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        supplied[path.name] = path
    expected_slides = {item["basename"]: item for item in source_slides}
    if set(supplied) != set(expected_slides):
        raise ValueError(
            f"Source slide basename set does not match frozen {case_id} input; "
            f"missing={sorted(set(expected_slides)-set(supplied))}, "
            f"extra={sorted(set(supplied)-set(expected_slides))}"
        )
    for basename, record in expected_slides.items():
        if not basename.startswith(f"{case_id}-"):
            raise ValueError(f"Non-sample slide in candidate manifest: {basename}")
        path = supplied[basename]
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Source WSI size mismatch for {basename}")
        actual_wsi_hash = _sha256_file(path)
        if actual_wsi_hash != record["sha256"]:
            raise ValueError(f"Source WSI SHA256 mismatch for {basename}: {actual_wsi_hash}")

    readers: dict[str, object] = {}
    dimensions: dict[str, tuple[int, int]] = {}
    try:
        for basename, path in supplied.items():
            reader = _open_slide(path)
            readers[basename] = reader
            dimensions[basename] = tuple(int(value) for value in reader.dimensions)
    except Exception:
        for reader in readers.values():
            reader.close()
        raise

    dest = Path(output_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    receipts = []
    try:
        total = len(candidate_rois)
        for index, frozen in enumerate(candidate_rois, start=1):
            basename = frozen["slide_basename"]
            if basename not in readers or frozen["slide_id"] not in frozen["roi_id"]:
                raise ValueError(f"Candidate {index} has inconsistent slide/case identifiers")
            roi_id = str(frozen["roi_id"])
            if Path(roi_id).name != roi_id or "/" in roi_id or "\\" in roi_id or ".." in Path(roi_id).parts:
                raise ValueError(f"Unsafe ROI filename in frozen candidate manifest: {roi_id!r}")
            x, y = int(frozen["x"]), int(frozen["y"])
            width, height = int(frozen["width"]), int(frozen["height"])
            level = int(frozen["level"])
            slide = readers[basename]
            slide_width, slide_height = dimensions[basename]
            level_count = int(getattr(slide, "level_count", 1))
            if level != 0 or level >= level_count:
                raise ValueError(f"Unsupported crop level {level} for {basename}")
            if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > slide_width or y + height > slide_height:
                raise ValueError(f"Candidate {index} crop is out of WSI bounds: {frozen['roi_id']}")
            image = slide.read_region((x, y), level, (width, height)).convert("RGB")
            if image.size != (width, height):
                raise ValueError(f"Reader returned wrong image size for {frozen['roi_id']}: {image.size}")
            rgb_hash = _sha256_rgb(image)
            if rgb_hash != frozen["decoded_rgb_sha256"]:
                raise ValueError(
                    f"Decoded RGB mismatch at candidate {index} ({frozen['roi_id']}): "
                    f"expected {frozen['decoded_rgb_sha256']}, got {rgb_hash}"
                )
            image_path = dest / f"{roi_id}.png"
            image.save(image_path, format="PNG")
            png_hash = _sha256_file(image_path)
            if png_hash != frozen["png_sha256"]:
                raise ValueError(
                    f"PNG SHA256 mismatch at candidate {index} ({frozen['roi_id']}): "
                    f"expected {frozen['png_sha256']}, got {png_hash}; decoded pixels matched"
                )
            row = {key: value for key, value in frozen.items() if key not in {"candidate_index", "slide_basename", "png_sha256", "decoded_rgb_sha256"}}
            row["image_path"] = str(image_path)
            row["metadata"] = {**row.get("metadata", {}), "slide_path": str(supplied[basename])}
            rows.append(row)
            receipts.append({
                "index": index,
                "roi_id": frozen["roi_id"],
                "source_wsi_basename": basename,
                "source_wsi_sha256": expected_slides[basename]["sha256"],
                "x": x, "y": y, "width": width, "height": height, "level": level,
                "decoded_rgb_sha256": rgb_hash,
                "expected_decoded_rgb_sha256": frozen["decoded_rgb_sha256"],
                "png_sha256": png_hash,
                "expected_png_sha256": frozen["png_sha256"],
                "verified": True,
                "image_path": str(image_path),
            })
            (dest / "reconstruction_receipt.json").write_text(
                json.dumps({"case_id": config["case_id"], "candidate_manifest_sha256": config["source_candidates_sha256"], "verified_count": len(receipts), "crops": receipts}, indent=2) + "\n",
                encoding="utf-8",
            )
            progress(f"verified candidate {index}/{total}: {frozen['roi_id']}")
    finally:
        for reader in readers.values():
            reader.close()
    if len(rows) != expected_count or len(receipts) != expected_count:
        raise RuntimeError(
            f"Strict candidate reconstruction incomplete: {len(rows)}/{expected_count}"
        )
    return rows
