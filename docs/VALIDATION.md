# Validation record

Both SO1/Core and SO4/Full completed real local Jupyter notebook execution for **TCGA-OR-A5JI, all five diagnostic slides DX1–DX5**, with the known anatomical site supplied explicitly. The 20-candidate pool, initial selected ROI IDs, final six ROI IDs and main diagnosis match the historical D1 demo. The aligned reference still disagrees with generated grade and necrosis statements, and internal report inconsistencies remain. Read [the result comparison](RESULT_COMPARISON.md); this is not full clinical agreement or word-for-word reproduction.

| Check | SO1/Core | SO4/Full |
| --- | --- | --- |
| Notebook execution | Passed, 1681.6 seconds | Passed, 621.2 seconds |
| Patient input slides / candidate ROIs | 5 / 20 | 5 / 20 |
| Final diagnostic ROIs / reinspection rounds | 6 / 2 | 6 / 2 |
| Main diagnosis | Adrenocortical carcinoma | Adrenocortical carcinoma |
| D1 segmentation | Available | Available |
| KEEP | Not used by this profile | Available; adrenal-site retrieval |
| Auxiliary grade evidence cards | 0 | 2 |
| Pipeline errors | None | None |
| Portable HTML bytes | 190,918 | 226,396 |
| Displayed WSI count | 2 | 2 |

All five slides enter CONCH candidate generation. The selected diagnostic ROIs come from two slides, so those two are displayed in the report. SO1 includes first-time feature extraction for four added slides; SO4 reuses identity-checked caches. Runtime is not a benchmark across versions.

[Machine-readable validation](../provenance/validation_results.json) records checksums, timings, input scope and limits. [Completed notebooks and HTML reports](../examples/README.md) retain real outputs; old single-slide attempts remain in the local workspace and are not the published examples. No reference diagnosis was supplied to inference. Actual saved Qwen prompts were checked for the TCGA project label; none contained it. References were used only after generation.

## Checks performed

- 14 focused software tests passed: 9 delivery/patient-download checks, 3 evaluator checks and 2 scoring isolation checks.
- The registered frozen scientific/runtime files passed their hash checks. The compact D1 adapter and separate pinned SAM3 assets were used in both runs.
- The three matching initial ROI inputs and mask PNGs match the historical full-D1 saved artifacts byte-for-byte; class pixel counts and fractions also match. This checks these three inputs, not universal model parity.
- A fresh official upstream source download and the bootstrap script reconstructed all 169 external MedSAM3 files exactly. That source subtree is excluded from the public code archive and fetched during setup.
- All five local WSI files passed size, GDC MD5 and pinned SHA-256 checks. Live GDC inventory and HTTP Range-prefix checks matched each file. A fresh complete 8.06 GB transfer was not repeated.
- Report/export audits confirmed unchanged scientific report and evidence text. Original exported image bytes were preserved when building the examples.
- An isolated headless Chrome session opened the example index and both reports successfully. Required resource requests had no failures and no JavaScript errors were reported. Only optional favicon requests returned 404. The local test server was stopped.
- Both owned inference process groups were confirmed inactive after completion. The temporarily paused 8002 queue was restored and started its next case.

Run local checks from the tutorial root after provisioning the upstream source:

```bash
python3 scripts/fetch_medsam_source.py
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s evaluation/tests -v
python3 -m unittest discover -s scoring/tests -v
python3 scripts/audit_delivery.py
```

The publisher's raw execution record is `workspace/patient_validation_20261104/`: `status.json`, `operator.log`, `SO1_executed.ipynb`, `SO4_executed.ipynb`, and separate `SO1/` / `SO4/` run folders. For local inspection, use `cat workspace/patient_validation_20261104/status.json` and `tail -n 30 workspace/patient_validation_20261104/operator.log`. These workspace files are excluded from the release; sanitized completed notebooks and artifact hashes are included instead.

## Scope and remaining limits

Validation used an NVIDIA RTX 5090 (32 GB), the existing Python 3.12 segmentation environment, and a Python 3.10 notebook environment inheriting installed pipeline packages. The published dependency specifications were not installed from scratch. Notebook cells ran through `nbclient` with a real Jupyter kernel; interactive Jupyter and scoring browser flows were not tested. The source-archive relocation check is recorded separately with the release packaging results.

Qwen generation used the same model identifier and temperature as the historical run, but no fixed seed; API runtime and exact report wording are not guaranteed identical. The software completed despite limited-evidence flags, and grade/necrosis inconsistencies remain documented. The report-only evaluator also has an observed invasion-parser limitation described in [EVALUATION.md](EVALUATION.md).

The SO1/SO4 notebook runs used existing checksum-verified local model files. Subsequent upload/download verification is recorded in [the HF release record](../provenance/hf_model_release.json).

## Fresh GitHub and Hugging Face inference test

On 2026-11-04, a fresh clone of public GitHub commit `2eb529101c9e3ef2beb252d8cfede05fd2b4ca74` fetched the official upstream MedSAM3 source and passed the frozen-source audit. The published download helper downloaded D1 revision `62894f38c67e28389f43a9f7efe89dd4a40e5608` into an empty HF cache, verified its SHA-256, and passed that downloaded file to the actual segmentation model builder.

Model construction and inference on three 4096 × 4096 TCGA-OR-A5JI ROIs succeeded in 48.6 seconds. All three predictions used a fresh segmentation cache (`cache_hit=false`). Mask PNGs and overlay PNGs matched the prior SO1 run byte-for-byte, and class statistics were identical. See [input/output hashes and the test receipt](../provenance/hf_inference_validation.json).

This test reused authorized, checksum-verified SAM3 and CONCH files and the installed environment. It did not repeat a clean dependency installation, full WSI download, or Qwen report generation. The recorded SO1/SO4 reports above remain the full report validation.

The access check also found that the HF repository was initially private. It is now publicly discoverable with **manual approval required for downloads**; anonymous metadata access, blocked anonymous weight access, and authorized weight access were verified. The weight revision and bytes did not change. Readers must request D1 access and obtain separate upstream model permissions.
