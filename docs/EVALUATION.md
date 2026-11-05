# Report evaluation

The tutorial evaluator scores a generated report against a user-supplied reference. It runs locally, makes no model/API calls, and does not require model weights. The packaged code under `evaluation/vendor/` is a snapshot of the current deterministic report-only evaluator in `src/pathxai_agent_v2/eval/` (`eval_v4_report_only_20261017_r5`). Prefer `generated_long_report.txt` for evaluation. For PathXAI `report.html`, the wrapper extracts only final-report rows, excluding differential diagnoses, evidence panels and reference panels. Other HTML inputs must contain only the report being assessed. The versioned source modules are included so this evaluation package can run independently of the rest of the pipeline.

## Inputs

Provide a generated HTML or plain-text report and its matching reference report. The reference must come from an independently curated/physician reference source; this repository includes generated example reports but does not redistribute the original clinical reference report. If the reference is structured JSON, use an object with a `text` string and optional `structured` object. The `structured` keys follow the evaluator's existing convention: `primary_diagnosis`, `histologic_subtype`, `broad_type`, `malignancy_status`, `grade`, `invasive_status`, and `necrosis`.

An optional `--cancer-type TCGA-...` sets anatomy context for interpreting report language and computes a separate TCGA cohort compatibility field. It does not fill in patient histology or replace the independently supplied reference. Without this argument, strict report/reference subtype agreement is still computed, but anatomy context is absent.

## Run

From the tutorial repository root (no third-party dependencies are needed), run:

```bash
python3 evaluation/run_evaluation.py \
  --report path/to/report.html \
  --reference path/to/reference.txt \
  --cancer-type TCGA-BRCA \
  --output evaluation/result.json
```

For structured JSON reference input:

```bash
python3 evaluation/run_evaluation.py \
  --report path/to/report.html \
  --reference path/to/reference.json \
  --reference-format json \
  --output evaluation/result.json
```

Use `--diagnosis-only` when the reference or assessment covers diagnosis alone. This suppresses narrative and morphology/finding metrics that would otherwise be misleading. The evaluator uses only the Python standard library.

The detailed JSON includes per-field applicability, coverage, exact agreement, abstention/QA flags, metric scope, report/reference SHA-256 hashes, and the evaluator/terminology versions. A `null` score means the reference field is not applicable or was not documented; it is not a zero. Preserve the JSON alongside the input versions when reporting results.

## What the scores mean

This is automatic **report-to-reference agreement**, not a clinical quality determination. The v4 schema preserves v3 metrics except the subtype field, which uses method-blind synonym normalization and anatomy context; strict subtype agreement and TCGA cohort compatibility are distinct fields. TCGA compatibility measures cohort membership, not exact patient subtype. The diagnosis composite is weighted across reference-applicable broad diagnosis, subtype, and malignancy fields. The parser attempts to distinguish assertions, uncertainty, contradictions and missing findings, but its classifications require review. Narrative ROUGE overlap is descriptive and should not stand in for clinical correctness.

**Physician scoring is separate.** Clinician review requires a prespecified rubric, blinded review where appropriate, qualified reviewers, and adjudication rules. Do not present automatic evaluator metrics as physician scores, diagnostic accuracy in clinical practice, or validation for patient care.

## Dependencies and tests

The evaluation wrapper has no third-party dependency. From the repository root, run its focused tests with:

```bash
python3 -m unittest evaluation.tests.test_run_evaluation
```

