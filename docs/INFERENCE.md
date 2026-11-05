# Local inference guide

The notebook runs a new WSI inference and exports a portable HTML report. This guide covers its model/API configuration, variant selection and outputs. Environment setup is in [INSTALLATION.md](INSTALLATION.md).

## Access and prerequisites

You need the local notebook and segmentation environments, a CUDA-capable host for model inference, and an API endpoint serving a vision-capable Qwen model. See the [installation guide](INSTALLATION.md) for the observed Python and Torch pairs. The included environment checks verify imports and versions; they are not a model-inference test.

The pipeline uses separate model resources:

| Resource | Where to obtain it | Required for |
| --- | --- | --- |
| PathXAI D1 inference adapter | Hugging Face model repo [`RayTan/PathXAI-Seg`](https://huggingface.co/RayTan/PathXAI-Seg); manual access approval required | SO1 and SO4 segmentation |
| SAM3 base checkpoint | Directly from the gated [`facebook/sam3`](https://huggingface.co/facebook/sam3) repository, at the pinned revision in `vendor/report_engine/assets/model_references.json` | SO1 and SO4 segmentation |
| CONCH checkpoint | Directly from the [`MahmoodLab/conch`](https://huggingface.co/MahmoodLab/conch) repository, at the pinned revision in `vendor/report_engine/assets/model_references.json` | WSI ROI selection in SO1 and SO4 |
| KEEP knowledge files | Directly from the gated [`Loie/KEEP_dataset`](https://huggingface.co/datasets/Loie/KEEP_dataset) dataset, at the pinned revision in `vendor/report_engine/assets/keep_references.json` | SO4 only |

The SAM3 and CONCH checkpoints are not hosted in the PathXAI repository. Users must request and receive access from each upstream provider and download them with their own Hugging Face account. KEEP access is required only for SO4. The notebook downloads the separate assets directly and verifies their pinned sizes and SHA-256 values; it does not substitute other weights.

**D1 is published at revision `abfe61cb8e6e916b7226fd9dba077ef95ee9ee4f`.** Its release files and adapter identity are recorded in [release verification](../provenance/hf_model_release.json). Request access on the model page and wait for approval before downloading. The notebook selects this immutable revision. The staged adapter file is Git-ignored and is not a GitHub artifact. The model bundle includes the SAM License without an additional academic-only restriction. See [model release notes](MODEL_RELEASE.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

## Configure the notebook

Start the Jupyter server from `PathXAI_tutorial/` and select the `pathxai` kernel registered by the installation instructions. In the notebook's setup cell, choose a version and set the segmentation interpreter and D1 release identifiers:

```python
VERSION = "SO1"  # or "SO4"
SEGMENTATION_PYTHON = "/absolute/path/to/pathxai-segmentation-py312/bin/python"
D1_REPO_ID = "RayTan/PathXAI-Seg"
D1_REVISION = "abfe61cb8e6e916b7226fd9dba077ef95ee9ee4f"
QWEN_API_BASE = "https://your-vision-endpoint/v1"
QWEN_MODEL = "your-server-model-id"
```

The notebook can read these values from environment variables (`PATHXAI_VERSION`, `PATHXAI_SEG_PYTHON`, `PATHXAI_D1_REPO`, `PATHXAI_D1_REVISION`, `QWEN_API_BASE`, and `QWEN_MODEL`). Set `PATHXAI_D1_REPO=RayTan/PathXAI-Seg`; the default `PATHXAI_D1_REVISION` is already pinned to the verified release. Set `QWEN_API_KEY` and `HF_TOKEN` in your shell or enter them at the notebook's secure prompts. They are passed to the runtime but must not be added to notebook cells, committed configuration, or outputs.

Alternatively, `PATHXAI_LOCAL_MODEL_CONFIG` can point to a local JSON file with path/config values accepted by `make_config`, such as `segmentation_python`, `d1_path`, `sam3_path`, `conch_path`, `keep_dir`, `workspace`, `qwen_api`, `qwen_model`, and `device`. Keep `api_key` and `hf_token` out of that file. Local paths still have to match the bundled asset manifests' size and checksum. Use this option for files you already obtained with authorization; the tutorial does not bundle model weights.

`device` defaults to `cuda:0`. Make sure both interpreters can see the intended GPU before starting a long inference.

## Run the sample

The sample helper downloads both pinned source WSIs listed in `configs/sample_tcga.json` for TCGA-FD-A5BS, whose known site is bladder / urothelial tissue: one DX1 slide and one TSB slide, totaling 2,623,173,488 bytes (about 2.44 GiB). The GDC inventory matches these two slides, with none excluded. Each slide is checked against pinned GDC metadata, file size, MD5 and SHA-256; interrupted downloads can resume. `download_patient(data_dir, existing_dir=None)` returns the verified slide paths. When `PATHXAI_SAMPLE_DIR` is set, the helper searches that directory for both pinned filenames and verifies the slides.

`sample_metadata()` exposes the patient information at the top level, including `case_id`, `known_site` and the pinned `slides` list. The notebook passes both pinned slides and the known anatomical site to one `run_report(...)` call, producing one patient-level report. An incomplete or mismatched slide set must stop before inference; it must not silently fall back to fewer slides.

The sample uses `prepare_sample_candidates(slides, output_dir)` to reconstruct the 20 pinned candidate crops in `configs/sample_candidates.json`. Their coordinates, order and pixel hashes are checked before inference. Pass the returned list as `candidate_rois` and preserve the sample's `known_site_source` marker from its manifest.

For your own data, provide all WSI paths for one patient and the anatomical sampling site that you knew independently before inference. For example:

```python
from pathlib import Path

case_id = "case_001"  # use a local pseudonymous identifier
slides = [
    Path("/data/case_001/slide_a.svs"),
    Path("/data/case_001/slide_b.svs"),
]
known_site = "adrenal gland"  # illustrative; replace with the independently known anatomical site

result = run_report(
    config,
    slide_paths=slides,
    case_id=case_id,
    version=VERSION,
    known_site=known_site,
    max_seconds=MAX_SECONDS,
)
```

Use one list for the slides belonging to a single patient case. Do not combine different patients in one call or include a tumor diagnosis in `known_site`.

The notebook then:

1. Downloads and verifies the D1, SAM3 and CONCH assets. For SO4 it also downloads and verifies KEEP.
2. Sends a small synthetic image to the configured Qwen API as a connectivity probe. This probe does not send the TCGA WSI.
3. Runs the selected SO1/Core or SO4/Full profile on the WSI.
4. Writes a new run directory containing the HTML report, its assets, generated report text, pipeline result and provenance records.

The report stage sends selected WSI-derived ROI images and report prompts to the configured Qwen endpoint. Confirm that this transfer is allowed for your data before proceeding. For the TCGA sample, the notebook supplies `known_site` from the pinned sample metadata. For your own data, provide it only when you knew the anatomical site independently before inference; the notebook does not infer it from case identifiers or filenames. It does not pass the TCGA project label or a reference diagnosis into report generation.

Each run requires a new output directory. To preserve the portable viewer, keep `report.html` together with its `assets/` directory and other generated artifacts. To change where a new run is written, set `PATHXAI_OUTPUT_DIR` to a path that does not already exist.

## Optional command-line execution

After obtaining the models, you can run the same report helper without Jupyter. Copy `configs/inference.example.json` to your private `local_config.json`, fill in the interpreter, model paths and endpoint, and supply credentials through environment variables. Use absolute model paths when possible. Relative configuration paths resolve from the shell's current working directory, so run the following from the tutorial root:

```bash
python scripts/run_report.py \
  --config local_config.json \
  --slide /absolute/path/to/slide.svs \
  --case-id YOUR_CASE_ID --version SO1 \
  --output outputs/YOUR_NEW_RUN --check-api
```

This command expects existing model and WSI files; automatic model/sample downloads are notebook steps. Its default maximum run time is 3600 seconds, configurable with `--max-seconds`. Preserve the terminal/session or use your usual job scheduler for a long CLI run.

## Evaluation and physician scoring

The downloaded sample has no verified reference report. Automatic evaluation is optional: provide an independently curated reference for the same case after inference. Follow [EVALUATION.md](EVALUATION.md); do not create a reference from the generated report or infer patient histology from a TCGA cohort label.

The [local scoring tools](../scoring/README.md) are separate workflows. One supports physician reference report authoring and optional assessment scores; the other supports blinded ranking, acceptability and safety review of five reports. Their outputs are human annotations, not the automatic evaluator's metrics.

## Current validation status

See [VALIDATION.md](VALIDATION.md) for the current validation scope and status. An environment import check or synthetic API probe alone does not establish full WSI-to-report inference.
