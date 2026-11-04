# PathXAI: TCGA patient to pathology report

This repository contains a local Jupyter workflow for running PathXAI on a patient-level set of whole-slide images (WSIs) and exporting one portable `report.html` for the case. It supports **SO1/Core** and **SO4/Full**. Both use the D1 segmentation adaptation; SO4 also requires the KEEP knowledge resources.

The workflow uses a local notebook and two local Python environments. It does not use Colab or Docker. The report pipeline calls a vision-capable Qwen API that you configure; selected ROI images and report prompts are sent to that endpoint. Check your data-handling requirements before running it on private slides.

## Review the committed example artifacts

The repository includes two recorded notebooks and report artifacts, which can be read without installing anything or obtaining model access:

| Version | Completed notebook, viewable on GitHub | Generated report |
| --- | --- | --- |
| SO1/Core | [SO1 notebook with actual outputs](examples/notebooks/SO1_completed.ipynb) | [SO1 HTML](examples/SO1/report.html) |
| SO4/Full | [SO4 notebook with actual outputs](examples/notebooks/SO4_completed.ipynb) | [SO4 HTML](examples/SO4/report.html) |

These are completed outputs from the five-slide patient-level rerun. ROI selection and the main diagnosis match the historical D1 demo; grade/necrosis discrepancies remain. Read the [result comparison](docs/RESULT_COMPARISON.md). The known Pages URL is [https://zhaorui-tan.github.io/PathXAI/](https://zhaorui-tan.github.io/PathXAI/), but **Pages has not been deployed yet**; GitHub's file browser displays HTML source. For offline viewing, download the full repository and open [`examples/index.html`](examples/index.html). [Example details](examples/README.md) · [Enable the Pages report viewer](docs/PAGES.md).

## Quick start

1. From the checkout root, run `python3 scripts/fetch_medsam_source.py` to prepare the external MedSAM3 runtime source required by segmentation. This uses the pinned upstream source and tutorial patches; it needs network access but does not download model weights or require a GPU. Run it before environment import checks or opening the notebook. See [installation](docs/INSTALLATION.md).
2. Create the Python 3.10 notebook/pipeline and Python 3.12 segmentation environments as described in [installation](docs/INSTALLATION.md).
3. Obtain your own authorized access to the separate SAM3 and CONCH models. The D1 adapter is published at [RayTan/PathXAI-Seg](https://huggingface.co/RayTan/PathXAI-Seg/tree/62894f38c67e28389f43a9f7efe89dd4a40e5608); the notebook pins its verified immutable revision. SO4 additionally requires access to the gated KEEP dataset. See the [inference guide](docs/INFERENCE.md) for the current release state and configuration.
4. Set the Qwen API base URL, model ID and credential in environment variables or provide the API key at the notebook's secure prompt. Do not store credentials in notebook cells or configuration files.
5. Start Jupyter from this directory with the Python 3.10 environment and open [PathXAI_report.ipynb](PathXAI_report.ipynb). The notebook downloads and verifies the five pinned diagnostic slides DX1–DX5 for one open TCGA patient from the NCI Genomic Data Commons (8,055,455,029 bytes total, about 7.50 GiB), runs one selected variant across the complete slide set, and exports one patient-level `report.html` plus its assets. The additional TS1 slide is excluded to match the historical five-diagnostic-slide scope.

The patient-level sample is for a software walkthrough. The notebook supplies the known anatomical site from pinned metadata and does not pass the TCGA project label or a reference diagnosis to the report generator. The sample does not contain a verified reference report. The included completed notebooks and HTML reports come from the five-slide rerun; see [validation](docs/VALIDATION.md) for its exact scope.

## Repository contents

- `PathXAI_report.ipynb` — local Jupyter workflow for sample download, API check, inference, HTML export and optional report evaluation.
- `examples/` — two completed notebooks with saved outputs, actual HTML reports and their unchanged shared image assets; also the static GitHub Pages site.
- `pathxai_tutorial/` — helpers for checksum-verified patient-slide download, model access, inference configuration and report generation.
- `vendor/report_engine/` — frozen pipeline snapshot, HTML renderer, input assets and source manifests. The MedSAM3 source subtree under `vendor/report_engine/baselines/current/source_snapshot/archives/tcga_so1_so4_a5bridge_508_20260920/code/medsam3/` is omitted from the public source archive; setup prepares the runtime copy separately from pinned upstream source and tutorial patches. Model weights are separate files.
- `environments/` and [installation](docs/INSTALLATION.md) — separate Python 3.10 and 3.12 specs plus import/version checks.
- `evaluation/` and [evaluation guide](docs/EVALUATION.md) — automatic, deterministic report-to-reference agreement metrics.
- `scoring/` and [scoring guide](scoring/README.md) — local tools for physician reference scoring and blinded comparison of five reports.
- `model_release/` — D1 adapter manifest and Hugging Face staging materials. The adapter file is excluded from Git and published at `RayTan/PathXAI-Seg`; its verified immutable revision is recorded in `provenance/hf_model_release.json`. SAM3 and CONCH base weights are not included.

## Evaluation and human scoring

Automatic evaluation requires an independently curated reference for the same case; it is separate from inference and is skipped by default in the notebook. See [EVALUATION.md](docs/EVALUATION.md) for metrics and limitations. The two local scoring apps support clinician review and blinded report comparison. Their annotations are physician/user scores; they must not be confused with automatic evaluator metrics.

## Validation status

See [VALIDATION.md](docs/VALIDATION.md) for the current checks, their scope and live inference status. Environment/import checks alone do not establish full WSI-to-report inference.

Source identities and runtime records are described in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). See [third-party notices](docs/THIRD_PARTY_NOTICES.md) for upstream code and model notices; using the MedSAM3 source does not grant additional rights to that upstream code.

PathXAI is research software, not a medical device or a clinical diagnostic service.
