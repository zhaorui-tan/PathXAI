# PathXAI manuscript Source Data

Numerical results for the manuscript revision dated **9 November 2026**.
The [preprint record](https://arxiv.org/abs/2611.09117) may display an earlier version until its replacement is announced.

These files contain the current manuscript analysis outputs using evaluator eval_v4_report_only_20261109_r7. Existing reports were rescored after correcting grade extraction; grade-related values and dependent completeness endpoints were regenerated. No reports were generated for this update. Diagnosis and morphology scores are unchanged. Estimates, confidence intervals, counts and statistical definitions match the current manuscript. Non-data provenance fields (internal source paths and case-list hashes) are omitted.

## Find the results

| Manuscript item | Starting files |
| --- | --- |
| Fig. 1(c), reporting comparisons | `report_primary_evaluation/public_summary.csv`, `public_contrasts.csv`, and the Fig. 2/3 sources below |
| Fig. 2, public evaluation | `report_primary_evaluation/public_summary.csv`, `public_contrasts.csv`, `public_organ_counts.csv` |
| Fig. 2, private evaluation | `aggregate_results/table4_fig9_private_reference.csv`, `private_reference_cancer_compatibility.csv`, `table8_private_hospital_sources.csv` |
| Fig. 3, reader assessment | `aggregate_results/table5_fig10_physician_ratings.csv`, `fig3_joint_criterion.csv` |
| Fig. 3, tissue inputs and writing | `aggregate_results/table2_fig7_four_policy.csv`, `fig3_workflow_comparisons.csv`, `table7_suppfig3_report_ablation.csv` |
| Fig. 5, segmentation | `aggregate_results/table1_fig4_segmentation.csv`, `table6_suppfig2_seg_ablation.csv`, `seg_shared_external_bootstrap.csv` |
| Supplementary results | [`table_index.json`](table_index.json), covering all 44 supplementary table groups; definition/protocol tables remain in the manuscript |

[`figure_index.json`](figure_index.json) lists the numerical source files by main figure. Fig. 4 presents report examples and Fig. 6 is a methods schematic; public example artifacts are in [`../examples/`](../examples/).

## Cohorts and units

- The primary public evaluation and the current tissue-input/writing comparisons use **1,200 cases**, separate from the **456 cases used for PathXAI-Report pipeline development**. Older evidence-organization/acquisition analyses retain their actual development-case subsets. The reader assessment uses 132 development cases and 264 ratings per method.
- The primary private automated analysis uses 151 shared cases, evaluated separately against the two pathologists' references. Available-case sensitivities and reference-agreement analyses retain their stated denominators. Private files contain aggregate statistics only, without private case IDs, images, full reports or individual ratings.
- The public cohort metadata tables describe their recorded metadata snapshot. Their recoverable-slide totals are not a replacement for the current complete WSI associations in `data_lists/report/slides.csv`.
- Use `n_cases`, `n_applicable` or the corresponding count field for each metric. Do not sum overlapping subsets as independent cases.
- Values are not uniformly percentages: respect the `unit`, `metric`, `estimand` and column labels. A 0–1 fraction is multiplied by 100 when plotted as a percentage. A `_pp` field is already in percentage points.
- Paired contrasts follow their recorded left/right methods. A negative contradiction-rate difference favours the left method because lower contradiction is better.
- Stable filenames and internal scope IDs retain historical labels for compatibility. In particular, `input_policy_shared507` and the current `agent_d508` writing arrays now refer to the 1,200-case evaluation, as shown by their actual counts. Human-readable legacy `Public curated references` and `Public OCR references` in sensitivity arrays denote the 456-case development subset and the 1,200-case evaluation cohort, respectively; they do not imply different reference-preparation pipelines.

## Sample lists and provenance

The final public case/WSI lists are in [`../data_lists/report/`](../data_lists/report/); segmentation sample lists are in [`../data_lists/seg/`](../data_lists/seg/). Raw datasets remain with their original providers under their own access terms. This directory adds numerical results and does not replace those lists.

[`release_manifest.json`](release_manifest.json) records source/export hashes and file-level row counts. [`SHA256SUMS.txt`](SHA256SUMS.txt) covers all files in this release. The scoring implementation is documented in [`../evaluation/version.json`](../evaluation/version.json).
