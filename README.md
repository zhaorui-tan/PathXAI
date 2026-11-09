# PathXAI: TCGA patient to pathology report
Preprint: [arXiv:2611.09117](https://arxiv.org/abs/2611.09117).

This repository contains a local Jupyter workflow for running PathXAI on a patient-level set of whole-slide images (WSIs) and exporting one portable `report.html` for the case. It supports **PathXAI(core)** and **PathXAI**. Both use the refined D1 segmentation adaptation; PathXAI also requires the KEEP knowledge resources.

The workflow uses a local notebook and two local Python environments. It does not use Colab or Docker. The report pipeline calls a vision-capable Qwen API that you configure; selected ROI images and report prompts are sent to that endpoint. Check your data-handling requirements before running it on private slides.

## Review the committed example artifacts

Open the executed notebooks, generated reports and step-by-step execution trajectories directly from the table below. **TCGA-OR-A5LL** is the notebook/report example; **TCGA-OR-A5JI** provides trajectories for an existing development case, and **TCGA-08-0516** provides trajectories for a randomly selected GDC case. No installation or model access is needed to read them.

| Case | Version | Direct links |
| --- | --- | --- |
| TCGA-OR-A5LL | PathXAI(core), SO1 | [Executed notebook](examples/notebooks/SO1_completed.ipynb) · [Open report](https://zhaorui-tan.github.io/PathXAI/SO1/report.html) |
| TCGA-OR-A5LL | PathXAI, SO4/Full | [Executed notebook](PathXAI_report.ipynb) · [Open report](https://zhaorui-tan.github.io/PathXAI/SO4/report.html) |
| TCGA-OR-A5JI | PathXAI(core), SO1 | **[Open Core trajectory — 22 calls](https://zhaorui-tan.github.io/PathXAI/traces/TCGA-OR-A5JI_SO1.html)** |
| TCGA-OR-A5JI | PathXAI, SO4/Full | **[Open Full trajectory — 22 calls](https://zhaorui-tan.github.io/PathXAI/traces/TCGA-OR-A5JI_SO4.html)** |
| TCGA-08-0516 | PathXAI(core), SO1 | **[Open Core trajectory — 22 calls](https://zhaorui-tan.github.io/PathXAI/traces/TCGA-08-0516_SO1.html)** |
| TCGA-08-0516 | PathXAI, SO4/Full | **[Open Full trajectory — 33 calls](https://zhaorui-tan.github.io/PathXAI/traces/TCGA-08-0516_SO4.html)** |

### What a trajectory shows

A trajectory records the sequence of model calls made while generating a report for one case. It lets you inspect how the workflow progresses from tissue assessment through diagnostic candidates and reinspection to the final report:

- **Inputs to each call:** the images sent to the model and the full prompt, including the site and any tissue statistics, prior outputs or retrieved knowledge supplied at that step.
- **Intermediate and final outputs:** broad-lineage and subtype assessments, diagnostic candidates, evidence-sufficiency judgments, claim checks, and report drafting or revision.
- **Progression through the workflow:** the initial pass, any reinspection rounds and the final pass. Stage, pass and branch labels are inferred from prompts, call order and saved checkpoint presence.
- **Resource use:** token usage and elapsed time for each model call.

Use the **Requests** sidebar to jump to a step, expand its prompt and output, or click an input image to enlarge it. Thinking was disabled in these runs, so the pages contain the recorded inputs, outputs and explicit explanations returned by the model; no internal reasoning text was returned. [Trajectory overview](https://zhaorui-tan.github.io/PathXAI/traces/) · [Run settings and case-selection notes](examples/traces/README.md).

**Notebook/report example:** The TCGA-OR-A5LL notebooks use refined D1 segmentation and six pinned source slides. The main [PathXAI_report.ipynb](PathXAI_report.ipynb) contains the executed SO4 workflow and its outputs; the completed notebooks preserve the SO1 and SO4 executions. Each HTML report is linked to its run by the published provenance and checksums. [Execution summary](docs/RESULT_COMPARISON.md) · [All examples](https://zhaorui-tan.github.io/PathXAI/) · [Example details](examples/README.md) · [Pages publishing guide](docs/PAGES.md). For offline viewing, download the repository and open [`examples/index.html`](examples/index.html).

**Qwen output variability:** Report wording and details may vary with the Qwen model version, deployment configuration and sampling parameters. Repeated runs may produce different outputs even with the same input slides.

## Quick start

1. From the checkout root, run `python3 scripts/fetch_medsam_source.py` to prepare the external MedSAM3 runtime source required by segmentation. This uses the pinned upstream source and tutorial patches; it needs network access but does not download model weights or require a GPU. Run it before environment import checks or opening the notebook. See [installation](docs/INSTALLATION.md).
2. Create the Python 3.10 notebook/pipeline and Python 3.12 segmentation environments as described in [installation](docs/INSTALLATION.md).
3. Request approval for [PathXAI D1](https://huggingface.co/RayTan/PathXAI-Seg) and obtain your own authorized access to the separate SAM3 and CONCH models. The D1 adapter is published at [RayTan/PathXAI-Seg](https://huggingface.co/RayTan/PathXAI-Seg/tree/abfe61cb8e6e916b7226fd9dba077ef95ee9ee4f); the notebook pins this immutable revision. PathXAI additionally requires access to the gated KEEP dataset. See the [inference guide](docs/INFERENCE.md) for the current release state and configuration.
4. Set the Qwen API base URL, model ID and credential in environment variables or provide the API key at the notebook's secure prompt. Do not store credentials in notebook cells or configuration files.
5. Start Jupyter from this directory with the Python 3.10 environment and open [PathXAI_report.ipynb](PathXAI_report.ipynb). The notebook downloads and verifies all six pinned source WSIs for TCGA-OR-A5LL (adrenal gland / adrenocortical tissue) from the NCI Genomic Data Commons: DX1–DX5 and TS1, totaling 11,156,338,264 bytes (about 10.39 GiB). The GDC inventory matches these six slides, with none excluded. It runs the selected variant and exports one patient-level `report.html` plus its assets.

The notebooks supply the known anatomical site from the pinned sample metadata. Executed notebooks and reports are available in [examples](examples/README.md); see [validation](docs/VALIDATION.md) for execution details.

## Repository contents

- `PathXAI_report.ipynb` — executed SO4 PathXAI notebook for sample download, API check, inference, HTML export and optional report evaluation; clear outputs before running it for a fresh case.
- `examples/` — executed SO1 and SO4 inference notebooks, reports and shared image assets; also the static GitHub Pages site.
- `pathxai_tutorial/` — helpers for checksum-verified patient-slide download, model access, inference configuration and report generation.
- `vendor/report_engine/` — frozen pipeline snapshot, HTML renderer, input assets and source manifests. The MedSAM3 source subtree under `vendor/report_engine/baselines/current/source_snapshot/archives/tcga_so1_so4_a5bridge_508_20260920/code/medsam3/` is omitted from the public source archive; setup prepares the runtime copy separately from pinned upstream source and tutorial patches. Model weights are separate files.
- `environments/` and [installation](docs/INSTALLATION.md) — separate Python 3.10 and 3.12 specs plus import/version checks.
- `evaluation/` and [evaluation guide](docs/EVALUATION.md) — automatic, deterministic report-to-reference agreement metrics.
- `source_data/` — [manuscript numerical results](source_data/README.md), with figure/table indices, confidence intervals, metric-specific counts and checksums for the 9 November 2026 revision.
- `data_lists/` — [public report and segmentation data lists](data_lists/README.md), including the final 456-case development subset, 1,200-case public evaluation cohort, segmentation splits and evaluation windows.
- `scoring/` and [scoring guide](scoring/README.md) — local tools for physician reference scoring and blinded comparison of five reports.
- `model_release/` — D1 adapter manifest and Hugging Face staging materials. The adapter file is excluded from Git and published at `RayTan/PathXAI-Seg`; its verified immutable revision is recorded in `provenance/hf_model_release.json`. SAM3 and CONCH base weights are not included.

## Evaluation and human scoring

The **[report data lists](data_lists/report/README.md)** contain the final **456 development cases and 1,200 public evaluation cases**, with 4,058 listed source WSI associations. The **[segmentation lists](data_lists/seg/README.md)** include all 24 recorded dataset split pools, fixed validation subsets, internal/external evaluation and supplemental window IDs. Dataset sources, counts, provenance limits and checksums accompany the lists. Raw datasets are obtained from their original providers.

The evaluator matches the current manuscript version; its source hashes are pinned in [evaluation/version.json](evaluation/version.json). Automatic evaluation requires an independent reference for the same case; it is separate from inference and is skipped by default in the notebook. See [EVALUATION.md](docs/EVALUATION.md) for metrics and limitations. The two local scoring apps support clinician review and blinded report comparison. Their annotations are physician/user scores; they must not be confused with automatic evaluator metrics.

## Validation status

See [VALIDATION.md](docs/VALIDATION.md) for the current checks, their scope and live inference status. Environment/import checks alone do not establish full WSI-to-report inference.

Source identities and runtime records are described in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). See [third-party notices](docs/THIRD_PARTY_NOTICES.md) for upstream code and model notices; using the MedSAM3 source does not grant additional rights to that upstream code.

## License

Original PathXAI code is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC 4.0)](LICENSE). It permits non-commercial use, modification and redistribution with attribution and an indication of changes; commercial use requires separate permission from the relevant rights holders.

This license applies to PathXAI's original code. Third-party code and adaptations subject to upstream terms, model checkpoints, datasets and externally sourced assets retain their respective licenses and access conditions. In particular, the published D1 adapter bundle retains its [SAM License](model_release/upload_ready/LICENSE); the code license does not relicense those weights. See [third-party notices](docs/THIRD_PARTY_NOTICES.md).

PathXAI is research software, not a medical device or a clinical diagnostic service.





