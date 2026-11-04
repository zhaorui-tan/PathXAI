# Patient-level reproduction comparison

## Summary

The five-slide SO1 and SO4 reruns reproduce the prior D1 runs' candidate pool, selected ROIs, and main diagnosis for TCGA-OR-A5JI. This is a meaningful pipeline reproduction result. It does **not** mean the reports reproduce the reference in every clinical feature: both reruns retain a high-grade classification and report necrosis, while the aligned reference describes a low-grade adrenal cortical carcinoma without coagulative necrosis.

The comparison baseline is the 2026-10-15 `PathXAI_demo` core/full validation for this case. It is not a case report from `tcga_expansion_20260924_1200_d1`. The old D1 run and tutorial reruns use the same five WSI slides and the same D1 segmentation checkpoint source. The previous tutorial attempt used DX1 alone; that run selected a DX1-only candidate pool and reported hepatocellular carcinoma. The five-slide rerun resolves that input mismatch.

## ROI and diagnosis reproduction

Both new runs have 20 ranked candidates distributed across the five slides (DX1 3, DX2 4, DX3 4, DX4 6, DX5 3). Their normalized candidate-pool hash, candidate ID order, initial three selected ROI IDs, and final six selected ROI IDs match the corresponding D1 baseline exactly. Both use SO1/SO4 profiles, an explicit `known_site` of `adrenal gland / adrenocortical tissue`, the same Qwen model identifier and temperature, and no reference diagnosis in the inference input.

For the three matching initial ROI images, the new compact D1 adapter's saved mask PNGs, class pixel counts and fractions also match the historical full-D1 saved outputs exactly. This is a check of these stored artifacts, not evidence of parity on all possible images; no probability arrays were saved. The [path-free candidate list](../provenance/sample_candidate_rois.json) records the observed candidate coordinates and scores.

The evaluator normalizes both SO1 and SO4 output to the same main diagnosis entity as the D1 baseline: adrenocortical carcinoma. SO1 uses the same capitalization as the baseline; SO4 differs only in capitalization. This is report-to-reference agreement, not a physician score or clinical validation.

| Variant | Diagnosis agreement | Exact grade | Necrosis | Invasion | Contradiction rate |
|---|---:|---:|---:|---:|---:|
| D1 SO1 baseline | 1.00 | 0.00 | 0.00 | 0.00 | 0.33 |
| Five-slide SO1 rerun | 1.00 | 0.00 | 0.00 | 1.00* | 0.33 |
| D1 SO4 baseline | 1.00 | 0.00 | 0.00 | 0.00 | 0.67 |
| Five-slide SO4 rerun | 1.00 | 0.00 | 0.00 | 0.00 | 0.67 |

Scores are from `eval_v4_report_only_20261017_r5`, comparing each generated report with the same aligned image-only reference. The composite diagnosis score is built from applicable report/reference diagnosis fields; it is not clinical accuracy. A score of 1.00 for diagnosis does not imply the grade, necrosis, or invasion fields are correct.

\* The SO1 evaluator assigns an invasion score of 1.00 by interpreting report language about unconfirmed invasive criteria as absence. The report itself says capsular/vascular invasion is not assessable from the supplied H&E ROIs. Treat this as a parser-derived result requiring review, not evidence that the case has no vascular invasion.

## Remaining report differences

The aligned reference describes a right adrenal cortical carcinoma as low grade, with no coagulative necrosis and no vascular or sinusoidal invasion. The TCGA project label was not used as a substitute for this patient-level reference.

- **Grade:** Both new structured outputs say high grade; both receive exact-grade agreement 0.00. SO1's report first says the exact grade is not assessable, then states “Grade: High” ([SO1 report, lines 25–29](../examples/SO1/generated_long_report.txt)). SO4 reports high grade ([SO4 report, lines 21–24](../examples/SO4/generated_long_report.txt)). The D1 baselines also called the case high grade, so this disagreement predates the tutorial rerun.
- **Necrosis:** Both new reports receive necrosis agreement 0.00. SO1 says extensive necrosis is present ([lines 10–19](../examples/SO1/generated_long_report.txt)); SO4 says focal necrosis is present, then later says necrosis is not supported by the available ROIs ([lines 13–27](../examples/SO4/generated_long_report.txt)). SO4's pipeline field says necrosis is absent in the provided fields of view. These statements conflict internally and disagree with the reference.
- **Invasion:** SO4 says invasion is not definitively assessable ([lines 13–16](../examples/SO4/generated_long_report.txt)); the reference reports no vascular or sinusoidal invasion. SO1's evaluator score should be read with the caveat above.

These remaining differences mean that only the main diagnosis and ROI-selection result are reproduced here. The report text is not reproduced word-for-word, and the field-level report agreement is incomplete.

## Artifacts and method

- [SO1 HTML report](../examples/SO1/report.html) · [SO1 text report](../examples/SO1/generated_long_report.txt) · [SO1 completed notebook](../examples/notebooks/SO1_completed.ipynb)
- [SO4 HTML report](../examples/SO4/report.html) · [SO4 text report](../examples/SO4/generated_long_report.txt) · [SO4 completed notebook](../examples/notebooks/SO4_completed.ipynb)
- [Path-free comparison provenance](../provenance/patient_reproduction.json)
- [Evaluation method and score meanings](EVALUATION.md) · [Inference instructions](INFERENCE.md) · [Validation scope](VALIDATION.md)
- [Inference example configuration](../configs/inference.example.json) · [Pinned TCGA sample metadata](../configs/sample_tcga.json)

The evaluator runs locally and makes no model/API calls. Its detailed per-run JSON is generated under the local workspace when the comparison script runs; the committed provenance summary omits local paths and the full reference report. Physician review/scoring was not performed. These repository links describe checked-in artifacts; this document makes no claim that a website is currently published.
