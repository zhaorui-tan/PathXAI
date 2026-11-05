# PathXAI: TCGA patient to pathology report
PREPRINT PAPER WILL BE ONLINE SOON :)

This repository contains a local Jupyter workflow for running PathXAI on a patient-level set of whole-slide images (WSIs) and exporting one portable `report.html` for the case. It supports **SO1/Core** and **SO4/Full**. Both use the refined D1 segmentation adaptation; SO4 also requires the KEEP knowledge resources.

The workflow uses a local notebook and two local Python environments. It does not use Colab or Docker. The report pipeline calls a vision-capable Qwen API that you configure; selected ROI images and report prompts are sent to that endpoint. Check your data-handling requirements before running it on private slides.

## Review the committed example artifacts

The repository includes two notebooks showing captured progress and final reports from runs of the tutorial Python API. They can be read without installing anything or obtaining model access; use `PathXAI_report.ipynb` for fresh inference.

| Version | Recorded-run notebook, viewable on GitHub | Generated report |
| --- | --- | --- |
| SO1/Core | [Recorded SO1 run notebook with captured outputs](examples/notebooks/SO1_completed.ipynb) | [View SO1 report](https://zhaorui-tan.github.io/PathXAI/SO1/report.html) |
| SO4/Full | [Recorded SO4 run notebook with captured outputs](examples/notebooks/SO4_completed.ipynb) | [View SO4 report](https://zhaorui-tan.github.io/PathXAI/SO4/report.html) |

The linked reports and notebooks with recorded outputs are from SO1 and SO4 runs for TCGA-OR-A5LL, using refined D1 segmentation and six pinned source slides. Read the [execution summary](docs/RESULT_COMPARISON.md). **[Open the live report viewer](https://zhaorui-tan.github.io/PathXAI/)** to browse the currently deployed reports without running inference. For offline viewing, download the full repository and open [`examples/index.html`](examples/index.html). [Example details](examples/README.md) · [Pages publishing guide](docs/PAGES.md).

**Qwen output variability:** Report wording and details may vary with the Qwen model version, deployment configuration and sampling parameters. Repeated runs may produce different outputs even with the same input slides.

## Quick start

1. From the checkout root, run `python3 scripts/fetch_medsam_source.py` to prepare the external MedSAM3 runtime source required by segmentation. This uses the pinned upstream source and tutorial patches; it needs network access but does not download model weights or require a GPU. Run it before environment import checks or opening the notebook. See [installation](docs/INSTALLATION.md).
2. Create the Python 3.10 notebook/pipeline and Python 3.12 segmentation environments as described in [installation](docs/INSTALLATION.md).
3. Request approval for [PathXAI D1](https://huggingface.co/RayTan/PathXAI-Seg) and obtain your own authorized access to the separate SAM3 and CONCH models. The D1 adapter is published at [RayTan/PathXAI-Seg](https://huggingface.co/RayTan/PathXAI-Seg/tree/abfe61cb8e6e916b7226fd9dba077ef95ee9ee4f); the notebook pins this immutable revision. SO4 additionally requires access to the gated KEEP dataset. See the [inference guide](docs/INFERENCE.md) for the current release state and configuration.
4. Set the Qwen API base URL, model ID and credential in environment variables or provide the API key at the notebook's secure prompt. Do not store credentials in notebook cells or configuration files.
5. Start Jupyter from this directory with the Python 3.10 environment and open [PathXAI_report.ipynb](PathXAI_report.ipynb). The notebook downloads and verifies all six pinned source WSIs for TCGA-OR-A5LL (adrenal gland / adrenocortical tissue) from the NCI Genomic Data Commons: DX1–DX5 and TS1, totaling 11,156,338,264 bytes (about 10.39 GiB). The GDC inventory matches these six slides, with none excluded. It runs the selected variant and exports one patient-level `report.html` plus its assets.

The notebook supplies the known anatomical site from the pinned sample metadata. The notebooks with recorded outputs and reports are available in [examples](examples/README.md); see [validation](docs/VALIDATION.md) for the recorded run details.

## Repository contents

- `PathXAI_report.ipynb` — local Jupyter workflow for sample download, API check, inference, HTML export and optional report evaluation.
- `examples/` — two recorded-run notebooks with captured progress and reports, actual HTML reports and their unchanged shared image assets; also the static GitHub Pages site. Use `PathXAI_report.ipynb` for fresh inference.
- `pathxai_tutorial/` — helpers for checksum-verified patient-slide download, model access, inference configuration and report generation.
- `vendor/report_engine/` — frozen pipeline snapshot, HTML renderer, input assets and source manifests. The MedSAM3 source subtree under `vendor/report_engine/baselines/current/source_snapshot/archives/tcga_so1_so4_a5bridge_508_20260920/code/medsam3/` is omitted from the public source archive; setup prepares the runtime copy separately from pinned upstream source and tutorial patches. Model weights are separate files.
- `environments/` and [installation](docs/INSTALLATION.md) — separate Python 3.10 and 3.12 specs plus import/version checks.
- `evaluation/` and [evaluation guide](docs/EVALUATION.md) — automatic, deterministic report-to-reference agreement metrics.
- `data_lists/` — [public report and segmentation data lists](data_lists/README.md), including the final 456-case development subset, 1,200-case public evaluation cohort, segmentation splits and evaluation windows.
- `scoring/` and [scoring guide](scoring/README.md) — local tools for physician reference scoring and blinded comparison of five reports.
- `model_release/` — D1 adapter manifest and Hugging Face staging materials. The adapter file is excluded from Git and published at `RayTan/PathXAI-Seg`; its verified immutable revision is recorded in `provenance/hf_model_release.json`. SAM3 and CONCH base weights are not included.

## Evaluation and human scoring

The **[report data lists](data_lists/report/README.md)** contain the final **456 development cases and 1,200 public evaluation cases**, with 4,055 listed source WSI associations. The **[segmentation lists](data_lists/seg/README.md)** include all 24 recorded dataset split pools, fixed validation subsets, internal/external evaluation and supplemental window IDs. Dataset sources, counts, provenance limits and checksums accompany the lists. Raw datasets are obtained from their original providers.

The evaluator matches the current manuscript version; its source hashes are pinned in [evaluation/version.json](evaluation/version.json). Automatic evaluation requires an independent reference for the same case; it is separate from inference and is skipped by default in the notebook. See [EVALUATION.md](docs/EVALUATION.md) for metrics and limitations. The two local scoring apps support clinician review and blinded report comparison. Their annotations are physician/user scores; they must not be confused with automatic evaluator metrics.

## Validation status

See [VALIDATION.md](docs/VALIDATION.md) for the current checks, their scope and live inference status. Environment/import checks alone do not establish full WSI-to-report inference.

Source identities and runtime records are described in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). See [third-party notices](docs/THIRD_PARTY_NOTICES.md) for upstream code and model notices; using the MedSAM3 source does not grant additional rights to that upstream code.

PathXAI is research software, not a medical device or a clinical diagnostic service.
