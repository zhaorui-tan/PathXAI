"""Notebook API retaining the frozen scientific profiles and report exporter."""
import json
import os
from pathlib import Path
import sys
import threading
from datetime import datetime, timezone
from .paths import ROOT, activate_engine

def make_config(*, segmentation_python, model_dir="models", conch_path=None,
                d1_path=None, sam3_path=None,
                keep_dir=None, workspace="workspace", qwen_api=None, qwen_model=None,
                api_key=None, hf_token=None, device="cuda:0", pipeline_python=None):
    activate_engine()
    from pathxai_demo.profiles import config_defaults
    root = Path(workspace).expanduser().resolve()
    weights = Path(model_dir).expanduser().resolve()
    config = config_defaults(workspace=root)
    config.update({
        # Preserve venv executable symlinks: resolving them selects the base
        # interpreter and silently discards dependencies installed in the venv.
        "pipeline_python": os.path.abspath(os.path.expanduser(str(pipeline_python or sys.executable))),
        "seg_python": os.path.abspath(os.path.expanduser(str(segmentation_python))),
        "seg_ckpt": str(Path(d1_path).expanduser().resolve()) if d1_path else str(weights / "weights" / "pathxai_d1_adapter.pth"),
        "sam3_checkpoint": str(Path(sam3_path).expanduser().resolve()) if sam3_path else str(weights / "base" / "sam3.pt"),
        "foundation_model_path": str(Path(conch_path).expanduser().resolve()) if conch_path else str(weights / "conch" / "pytorch_model.bin"),
        "keep_root": str(Path(keep_dir).expanduser().resolve()) if keep_dir else str(weights / "keep"),
        "qwen_api": qwen_api or os.environ.get("QWEN_API_BASE", ""),
        "qwen_model": qwen_model or os.environ.get("QWEN_MODEL", ""),
        "api_key": api_key if api_key is not None else os.environ.get("QWEN_API_KEY", ""),
        "hf_token": hf_token if hf_token is not None else os.environ.get("HF_TOKEN", ""),
        "device": device,
        "trace_mode": "record",
    })
    if not config["qwen_api"] or not config["qwen_model"]:
        raise ValueError("Set QWEN_API_BASE and QWEN_MODEL, or pass qwen_api/qwen_model explicitly.")
    return config

def check_api(config, *, timeout_seconds=60):
    """One small synthetic-image request; no patient/case image is transmitted."""
    activate_engine()
    from pathxai_demo.api_probe import probe_api
    return probe_api(config, timeout_seconds=timeout_seconds)

def run_report(config, *, slide_paths, case_id, version="SO1", output_dir=None,
               known_site="", known_site_source="user", candidate_rois=None,
               max_seconds=3600, progress=print):
    """Run real WSI→CONCH→D1→Qwen→HTML, one case and one variant at a time.

    output_dir must be new. References/diagnosis labels are not accepted here.
    Model/API failures are retained as failures, not substituted with sample text.
    """
    activate_engine()
    from pathxai_demo.engine import run_engine
    from pathxai_demo.exporting import prepare_report
    mode = {"SO1": "core", "SO4": "full", "CORE": "core", "FULL": "full"}.get(version.upper())
    if mode is None:
        raise ValueError("version must be SO1 or SO4")
    if not case_id or Path(case_id).name != case_id or case_id in {".", ".."} or any(c in case_id for c in "/\\:"):
        raise ValueError("case_id must be a non-empty path-safe identifier")
    slides = [str(Path(p).expanduser().resolve()) for p in slide_paths]
    if not slides or any(not Path(p).is_file() for p in slides):
        raise FileNotFoundError("Supply one or more existing WSI files")
    if not isinstance(max_seconds, int) or isinstance(max_seconds, bool) or max_seconds <= 0:
        raise ValueError("max_seconds must be a positive integer")
    run = Path(output_dir or ROOT / "outputs" / (case_id + "_" + version.upper() + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))).expanduser().resolve()
    if run.exists():
        raise FileExistsError("Choose a new output directory; existing runs are never overwritten")
    run.mkdir(parents=True)
    secrets = [str(config.get(k) or "") for k in ("api_key", "hf_token")]
    def clean(value):
        text = str(value)
        for secret in secrets:
            if secret:
                text = text.replace(secret, "[redacted]")
        return text
    def save(name, value):
        (run / name).write_text(clean(json.dumps(value, indent=2, ensure_ascii=False)) + "\n")
    def event(item):
        line = clean(json.dumps(item, ensure_ascii=False))
        with (run / "progress.jsonl").open("a") as stream:
            stream.write(line + "\n")
        if progress:
            progress(clean(f"[{item.get('stage', 'run')}] {item.get('message', '')}"))
    if known_site_source not in {"user", "tcga", "tcga_site_only"}:
        raise ValueError("Unsupported known_site_source")
    if candidate_rois is not None and (not isinstance(candidate_rois, list) or not candidate_rois):
        raise ValueError("candidate_rois must be a non-empty list when supplied")
    case = {"case_id": case_id, "mode": mode, "slide_paths": slides, "known_site": known_site,
            "known_site_source": known_site_source}
    if candidate_rois is not None:
        case["candidate_rois"] = candidate_rois
    save("tutorial_run.json", {"case": case, "version": version, "started_utc": datetime.now(timezone.utc).isoformat(),
         "max_seconds": max_seconds, "workers": 1, "qwen_api": config.get("qwen_api"), "qwen_model": config.get("qwen_model"),
         "pipeline_python": config.get("pipeline_python"), "seg_python": config.get("seg_python"),
         "note": "New inference, no supplied reference diagnosis. API generation is not guaranteed byte-identical."})
    cancellation = threading.Event()
    timer = threading.Timer(max_seconds, cancellation.set)
    timer.daemon = True
    timer.start()
    try:
        outcome = run_engine(config, case, run, event, cancellation)
        save("scientific_outcome.json", outcome)
        if outcome.get("status") == "succeeded":
            outcome = prepare_report(config, case, run, outcome, event, cancellation)
        save("outcome.json", outcome)
        if outcome.get("status") != "succeeded":
            raise RuntimeError(f"Report generation {outcome.get('status')}; see {run / 'outcome.json'}")
        return {"run_dir": str(run), "report_html": outcome["report_path"],
                "report_text": str(run / "cases" / case_id / "generated_long_report.txt"),
                "pipeline_result": str(run / "cases" / case_id / "pipeline_result.json")}
    except BaseException as exc:
        cancellation.set()
        save("tutorial_error.json", {"type": type(exc).__name__, "message": clean(exc)})
        raise
    finally:
        timer.cancel()
