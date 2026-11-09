# PathXAI execution trajectories

These four saved execution trajectories show SO1/Core and SO4/Full runs for two public TCGA cases: **TCGA-OR-A5JI**, an existing case in the 456-case PathXAI-Report pipeline development subset, and **TCGA-08-0516**, randomly selected from an online GDC candidate pool outside the recorded case inventory. They supplement the existing TCGA-OR-A5LL report examples.

[Open the trajectory viewer](https://zhaorui-tan.github.io/PathXAI/traces/) or download an HTML file and open it locally. Each page includes input images, prompts, model outputs, call order and token usage. Thinking was disabled and no internal reasoning text was returned; explicit diagnostic explanations remain in the model outputs.

## Contents

| File | Case | Version | Requests | Final diagnosis in report |
|---|---|---|---:|---|
| [View SO1](https://zhaorui-tan.github.io/PathXAI/traces/TCGA-08-0516_SO1.html) | TCGA-08-0516 (randomly selected) | SO1 | 22 | Glioblastoma, IDH-wildtype |
| [View SO4](https://zhaorui-tan.github.io/PathXAI/traces/TCGA-08-0516_SO4.html) | TCGA-08-0516 (randomly selected) | SO4 | 33 | Glioblastoma |
| [View SO1](https://zhaorui-tan.github.io/PathXAI/traces/TCGA-OR-A5JI_SO1.html) | TCGA-OR-A5JI (existing development case) | SO1 | 22 | Adrenocortical carcinoma |
| [View SO4](https://zhaorui-tan.github.io/PathXAI/traces/TCGA-OR-A5JI_SO4.html) | TCGA-OR-A5JI (existing development case) | SO4 | 22 | Adrenocortical carcinoma |

- `selection/tcga_never_included_primary_dx.csv` lists the 742 candidate cases the external case was drawn from (see below).
- [trace_viewer.py](../../pathxai_tutorial/trace_viewer.py) generates the pages from saved run records and image files. From the repository root, run `python -m pathxai_tutorial.trace_viewer outputs/<run_dir> --context-window 262144`. This reads existing outputs and does not run inference.

Each page is a single self-contained HTML file of 3–4.5 MB with the images embedded, so it works on GitHub Pages without extra assets. Open it in a browser.

## Why we added an external case

The additional case illustrates pipeline execution outside the recorded development and evaluation case lists. The existing case has a different role:

- **TCGA-OR-A5JI is in the development subset.** It appears in the final [456-case list](../../data_lists/report/public_development456.txt). TCGA-08-0516 appears in neither that list nor the final [1,200-case evaluation list](../../data_lists/report/public_evaluation1200.txt).
- **Its site text names the cell lineage.** This run's `known_site` is `adrenal gland / adrenocortical tissue`, taken from the TCGA project site map for TCGA-ACC. "Adrenocortical" already points toward the expected diagnosis.

For TCGA-08-0516, the supplied site is `brain`. These are illustrative trajectories, not additional cohort-level performance estimates.

## How TCGA-08-0516 was chosen

1. **All TCGA cases with slides.** We queried the NCI GDC API on 5 November 2026 for every open-access TCGA slide image: 11,120 cases.
2. **Removed cases in the recorded repository snapshot.** The selection used a 5 November 2026 snapshot, including the then-recorded 1,708 report-study inventory, the 9,009-case source inventory and segmentation split lists. 2,109 cases were absent. These are historical screening inventories, not the final 456/1,200 manuscript inclusion lists linked above.
3. **Kept cases with a primary-tumour diagnostic slide.** 742 of those have at least one DX slide from a primary tumour (sample type `01`). This is the list in `selection/`.
4. **Picked one at random.** We drew one case from the 742 using Python's `random.SystemRandom`, which was not seeded, so the draw cannot be reproduced. It returned TCGA-08-0516.

The 742 candidates by project:

| Project | Cases |
|---|---:|
| TCGA-GBM | 389 |
| TCGA-PAAD | 183 |
| TCGA-TGCT | 117 |
| TCGA-LGG | 25 |
| TCGA-KICH | 15 |
| TCGA-SARC | 7 |
| Six other projects | 1 each |

TCGA-GBM and TCGA-PAAD were absent from the 9,009-case source inventory used for selection. TCGA-GBM had no entry in the site map used for these runs; `brain` was supplied explicitly.

## Inputs used for TCGA-08-0516

- **Slide:** one diagnostic slide, `TCGA-08-0516-01Z-00-DX1.B384463F-9206-424F-B829-600FB602722C.svs` (1.75 GB, primary tumour). The case's two other slides are tissue slides (TS1, BS1) and were left out. The slide was verified against its GDC size and MD5.
- **Known site:** `brain`. A pathologist normally receives the specimen site, so we supplied one, but kept it to plain anatomy so it would not hint at the tumour type. It matches GDC's record for the case (primary site "Brain", tissue of origin "Brain, NOS").
- **Reference:** GDC records the diagnosis as glioblastoma (ICD-O-3 `9440/3`). The case's GDC "Pathology Report" file is a "TCGA Missing Pathology Report Form", so that file does not provide a narrative reference for scoring this generated report.

The TCGA-OR-A5JI runs retain their original site text, `adrenal gland / adrenocortical tissue`.

## Model and settings

| Setting | Value |
|---|---|
| Model | Qwen3.6-27B, Q6_K GGUF quantization |
| Server | llama.cpp (build b9856), run locally |
| Temperature | 0.2 |
| Thinking | Off (`enable_thinking: false` plus the `/no_think` prompt prefix); no reasoning was returned |
| Output cap | 4,096 tokens per request |
| Context limit | 262,144 tokens |
| System prompt | None; every request is a single user message |

These settings describe the four additional runs shown here; they do not replace the existing A5LL examples or manuscript evaluation results.

## Reading the pages

- **Stage, pass and branch names are inferred.** The trace does not record them. The viewer derives them from each prompt's first line, the call order and whether the final-pass checkpoint exists. For all four runs, we checked the labels against the pipeline's checkpoint files (`ckpt_06*`, `ckpt_08a–c_final_*` and `ckpt_dual_path_diagnosis_*`), and every checkable label matched. In each run, one diagnosis pair (the last reinspection round) could not be checked, because the final pass overwrites its dual-path checkpoint. Those pairs follow the orchestrator's call order, as all checkable pairs did.
- **Images are verified.** Each image shown was re-encoded with the recorded settings and matched against the payload hash in the trace, confirming it is the exact image that was sent.
- **Image and text tokens are combined.** llama.cpp does not report image tokens separately, so the context chart shows prompt text and images as one bar.
- **The username was redacted.** In these copies, the local username in the model file path and in local file paths was replaced with `<user>`. Nothing else in the pages was changed.

## Observations you may want to look at

- **SO4 sends local file paths to the model.** In both SO4 runs, the report-writing prompt includes absolute local paths (`cache_path` and `image_path`) inside the auxiliary grade-region block (request 32 of the TCGA-08-0516 SO4 run, request 21 of the TCGA-OR-A5JI SO4 run). They are shown redacted here, but the endpoint received the full paths.
- **The subtype check and final diagnosis disagree in the TCGA-08-0516 SO4 run.** The final subtype sufficiency step rated the evidence as `broad_type` and recommended "Glioma", yet the final report says "Glioblastoma". We have not looked into why.

## Limits

- **"Never included" in the selection filename refers to case-ID absence from the recorded selection snapshot.** Some segmentation datasets are listed without patient IDs (PanNuke and CoNIC2022 by array index, CPM17 by generic file names), and some of these contain TCGA-derived images, so they cannot be ruled out. We also did not check the pretraining data behind CONCH or SAM3.
- **This is one randomly selected additional case, with one run per case/configuration.** Differences between runs may partly reflect run-to-run variation, even at temperature 0.2.
- **The candidate list reflects GDC on 5 November 2026.** It was checked against the repository at that date.
