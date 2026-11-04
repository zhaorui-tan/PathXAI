from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_ROOT = Path(os.environ.get("PATHXAI_SHARED_ARTIFACTS", PROJECT_ROOT / "shared_artifacts"))

MANIFESTS_DIR = SHARED_ROOT / "manifests"
REFERENCE_REPORTS_DIR = SHARED_ROOT / "reference_reports"
CASE_LISTS_DIR = SHARED_ROOT / "case_lists"
ROI_CACHE_DIR = SHARED_ROOT / "roi_cache"
SEG_CACHE_DIR = SHARED_ROOT / "seg_cache"
WSI_THUMBNAIL_CACHE_DIR = SHARED_ROOT / "wsi_thumbnails"
TMP_DIR = SHARED_ROOT / "tmp"

