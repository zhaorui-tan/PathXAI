# Reproducibility and source identity

The main tutorial notebook contains the executed PathXAI (SO4) inference for TCGA-OR-A5LL using the frozen PathXAI profile and refined D1 adaptation. The committed PathXAI(core) and PathXAI example notebooks contain the executed SO1 and SO4 inference workflows, with outputs preserved. Notebook metadata and the publication manifest link the execution to the run record, generated report text and published HTML checksums. Qwen can produce different results across model versions, quantizations, hardware and calls; record its model ID and configuration with each result. This delivery is not a claim of byte-identical reproduction of historical paper results.

## Fixed inputs

- `configs/sample_tcga.json` pins all six open source WSIs for TCGA-OR-A5LL (adrenal gland / adrenocortical tissue): DX1–DX5 and TS1, totaling 11,156,338,264 bytes (about 10.39 GiB). The live GDC Slide Image inventory matches these six records, with none excluded. Their GDC identities, filenames, sizes, MD5 and SHA-256 values are pinned. `download_patient(...)` verifies all six slides before inference, including when a local directory is supplied. A missing or mismatched slide is an error.
- `configs/sample_candidates.json` pins the ordered 20 sample ROI coordinates, scores and image hashes. `prepare_sample_candidates(...)` reconstructs and verifies these crops from the source WSIs. For other patients, omit `candidate_rois` so CONCH selects candidates from their slides.
- `model_release/manifest.json` pins the D1 adaptation checksum and its August 4 refined source-checkpoint identity (SHA-256 `4d8ac978814f66ba5a716e26c8c44ada50f3f6c3f1bc987a97b24238babe9e5a`). The separate SAM3 base, CONCH model and PathXAI KEEP resources have pinned identities under `vendor/report_engine/assets/`.
- The D1 adaptation removes unchanged base text tensors only after exact comparison against the pinned SAM3 checkpoint. LoRA, prompt, adapter and retained auxiliary learned states are preserved. The base checkpoint must still be obtained separately.
- `vendor/report_engine/baselines/current/manifest.json` records the frozen scientific source/configuration files. The MedSAM3 source subtree at `vendor/report_engine/baselines/current/source_snapshot/archives/tcga_so1_so4_a5bridge_508_20260920/code/medsam3/` is omitted from the public source archive and prepared at setup time from pinned upstream source plus two tutorial patches by `python3 scripts/fetch_medsam_source.py`. `provenance/external_medsam_source.json` records that runtime dependency separately; this bootstrap does not alter the frozen scientific snapshot. `provenance/report_engine_sources.json` records the original imported delivery files, while `provenance/delivery_changes.json` documents the tutorial-specific changes outside the frozen scientific snapshot.
- `evaluation/vendor/` contains the report-only evaluator and terminology sources. Each evaluation result includes their SHA-256 hashes.

Verify the frozen scientific snapshot from the tutorial root:

```bash
python3 vendor/report_engine/pathxai_demo/provenance.py verify \
  vendor/report_engine/baselines/current/manifest.json
python3 scripts/audit_delivery.py
```

Before importing the segmentation worker or opening the notebook, prepare the external MedSAM3 source from the tutorial root:

```bash
python3 scripts/fetch_medsam_source.py
```

This requires network access. It fetches source code only, not model weights. The upstream source's own terms apply; the tutorial does not add a license grant for it.

## Runtime records

Every run uses a new output directory. For patient-level inference, `tutorial_run.json` records the selected case, all WSI paths, the explicitly supplied `known_site`, variant, endpoint/model ID, interpreter paths and time limit. `progress.jsonl`, `scientific_outcome.json` and `outcome.json` preserve progress and final status. The engine also records its resolved profile, source identity, model integrity checks and case artifacts. Credentials are supplied at runtime and redacted from the wrapper's records.

The notebook supplies the sample's known adrenal gland / adrenocortical site from pinned metadata; it does not infer the site from a TCGA identifier. For your own data, provide `known_site` only when the anatomical sampling site was independently known before inference. The complete set of one patient's slides contributes to one report. Automatic evaluation happens after report generation, against an independent reference, and never feeds that reference back into inference.

The executed PathXAI(core)/PathXAI notebooks and outputs use TCGA-OR-A5LL with all six pinned source slides, refined D1 segmentation and the documented adrenal gland / adrenocortical site. The main `PathXAI_report.ipynb` records the SO4 PathXAI execution. See [validation](VALIDATION.md) and [the execution summary](RESULT_COMPARISON.md) for execution scope and artifact provenance.

The environment specifications record observed direct dependency versions and separate the Python 3.10 pipeline from Python 3.12 segmentation. They are not a complete transitive lockfile or proof of a clean installation. See [validation](VALIDATION.md) for checks actually performed.

## Output handling

Keep the full exported report folder: HTML may refer to adjacent image assets. Reports, traces and cached images can contain information from the input slide. Store them with your data, outside the public source repository. The scoring tools likewise write only to user-configured local output paths.
