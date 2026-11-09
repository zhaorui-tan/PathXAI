# Report evaluation

This package runs the current manuscript report evaluator locally using only the Python standard library. It makes no model/API calls and generates no reports. The six implementation modules in `evaluation/vendor/` match the manuscript revision `eval_v4_report_only_20261109_r7`. `evaluation/version.json` records their source hashes. This revision corrects grade extraction and rescores existing reports; diagnosis and morphology rules are unchanged.

## Inputs and run

Supply a generated report and its matching independent reference. Prefer `generated_long_report.txt`. For PathXAI `report.html`, the wrapper extracts final-report rows and excludes differential diagnoses, evidence and reference panels. Other HTML inputs must contain only the report being assessed. Plain text is read directly. Original clinical references are not redistributed here.

From the repository root:

```bash
python3 evaluation/run_evaluation.py \
  --report path/to/generated_long_report.txt \
  --reference path/to/reference.txt \
  --cancer-type TCGA-BRCA \
  --output evaluation/result.json
```

For JSON references, add `--reference-format json`; the file must be an object with a `text` string and optional `structured` object. Supported structured fields are `primary_diagnosis`, `histologic_subtype`, `broad_type`, `malignancy_status`, `grade`, `invasive_status`, and `necrosis`. Keep the supplied reference independent of the generated report.

`--cancer-type TCGA-...` supplies anatomy and a separate TCGA cohort compatibility target. It does not supply the patient's exact histologic subtype. `--site breast` (or another supported anatomy, such as `kidney`) supplies anatomy without a TCGA cohort target. Use `--diagnosis-only` when findings/grade/narrative were not assessed; their manuscript metrics are then `null`.

## Manuscript metric mapping

The detailed JSON preserves the frozen evaluator's original `metrics` and adds `manuscript_metrics`. The five main public evaluation measures map to these per-case outputs (all nested keys below start at the JSON root):

| Paper measure | Exact output key | Meaning |
| --- | --- | --- |
| Cancer family match rate | `metrics.cohort_compatibility` (also `manuscript_metrics.diagnostic_mapping.cohort_compatibility`) | Compatibility with the supplied TCGA cancer-family target. |
| Exact diagnosis match rate | `metrics.subtype` (also `manuscript_metrics.reference_specific_diagnostic_mapping.strict_subtype`) | Exact normalized histologic-entity agreement with the independent case reference. |
| Finding F1 | `manuscript_metrics.findings.finding_f1` | 2C/(D+A). |
| Finding recall | `manuscript_metrics.findings.correct_fact_coverage` | C/D. |
| Finding contradiction rate | `manuscript_metrics.findings.contradiction_asserted` | E/A. |

Public cohort percentages are 100 times the equally weighted mean of applicable per-case scores, without cancer-type weighting. A definite, specific reference is required for Exact diagnosis match rate. Inapplicable references are excluded; applicable cases with missing, uncertain, conflicting or incompatible diagnostic predictions score zero. Finding applicability follows the denominators below. Cohort compatibility does not establish the exact patient subtype. Anatomy-conditioned synonyms are accepted for exact matching, while distinct sibling diagnoses remain distinct. The released scorer does not implement the separate private cohort family-matching analysis.

Additional diagnostic composites are retained: `metrics.diagnosis` (also `manuscript_metrics.reference_specific_diagnostic_mapping.diagnosis`) weights broad diagnosis, strict subtype and malignancy over applicable reference fields. `metrics.diagnosis_cohort` (also `manuscript_metrics.diagnostic_mapping.diagnosis_cohort`) substitutes cohort compatibility for the subtype component. The latter is used in the tissue-input workflow policy analysis; it is not one of the five main public measures above.

`manuscript_metrics.findings` uses the manuscript aggregate's exact finding audit. Let **D** be reference-documented binary findings (asserted present/absent), **A** those with an asserted prediction, **C** matching assertions and **E** opposite assertions:

| Finding output | Formula | Undefined when |
| --- | --- | --- |
| `finding_f1` | 2C/(D+A) | D+A=0 |
| `assertion_coverage` | A/D | D=0 |
| `correct_fact_coverage` | C/D | D=0 |
| `contradiction_asserted` | E/A | A=0 |

Missing, uncertain and conflicting predictions are counted separately. Assertions on undocumented reference findings are unverified; they are not included in A/C/E. **The manuscript Finding contradiction rate is `manuscript_metrics.findings.contradiction_asserted` (E/A).** The preserved legacy `metrics.contradiction_rate` uses D as its denominator; it is not the manuscript Finding contradiction rate.

`manuscript_metrics.grade` retains per-case exact, ordinal and compatible-coverage outputs. **Balanced grade accuracy** (`class_balanced_grade`) requires a cohort: within each grading system, compute exact correctness within each represented reference grade, then average those grade-specific accuracies equally. Missing/incompatible predictions count as incorrect; grading systems are reported separately. A single report/reference pair cannot determine that cohort score, so this wrapper returns `null` for it. Per-case exact and ordinal values are sensitivity inputs. This package does not add cohort aggregation or confidence interval estimation. Individual output cannot provide a cohort percentage or its confidence interval; those require the applicable case set and the manuscript's metric-specific interval procedure.

## Grade extraction

The same rules apply to reference and generated reports. Anatomy distinguishes renal WHO/ISUP grades (1–4) from prostate ISUP Grade Groups (1–5), scopes explicit primary-lesion sections and excludes grades of another organ. Four-tier and three-tier nuclear grades remain separate. Generic grades are not converted into named systems based on the reference.

Numeric and descriptive ranges remain uncertain; their endpoints are not exact grades. Unavailable, conflicting and hypothetical grades do not provide a definite answer. A definite grade is retained when a hedged repetition gives the same value, but a different possible value or range remains ambiguous. Explicit Gleason and prostate Grade Group values are retained separately without numerical conversion; the reference Grade Group is used when both are recorded.

Per-case outputs retain extraction evidence and per-system assertions. Grade-dependent completeness uses the corrected fields. Unassessable reference grades are inapplicable; missing or incompatible predictions receive zero on eligible cases. Thus zero agreement can reflect missing answers or a different grading system as well as an incorrect value.

## Interpretation and provenance

Scores measure automatic report-to-reference agreement. A `null` score means the metric is not applicable, not documented or not assessed; it is not zero. The parser distinguishes assertions, uncertainty, contradictions and missing findings, but classifications require review. Narrative ROUGE overlap is descriptive. Physician scoring requires a separate prespecified rubric and qualified review; automatic metrics are not physician scores or clinical validation.

The result records evaluator/terminology versions, input SHA-256 hashes, hashes of all five scorer modules, the manuscript metric helper hash, context and applicability/QA fields. Preserve the detailed JSON with the input versions. The included tests and fixtures contain synthetic examples only; private reports and case-level clinical metadata are excluded.

## Tests

```bash
python3 -m unittest discover -s evaluation/tests -t .
```

The suite includes the frozen scientific contracts (only package imports adapted), synthetic parser counterexamples, synonym/sibling checks, plain-text and PathXAI HTML CLI runs, exact wrapper/scorer parity, and the distinct manuscript/legacy finding denominators.
