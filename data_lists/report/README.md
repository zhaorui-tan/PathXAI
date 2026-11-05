# Report data lists

These are the **final public case lists used in the manuscript**.

| Analysis group | Cases | TCGA projects | Listed source WSIs |
| --- | --- | --- | --- |
| [Public development subset](public_development456.txt) | 456 | 27 | 934 |
| [Public evaluation cohort](public_evaluation1200.txt) | 1,200 | 22 | 3,121 |
| Combined, disjoint cases | 1,656 | Projects overlap across groups | 4,055 |

The 456-case subset was used for method development and component comparisons. The 1,200-case cohort supplies the primary public evaluation. Group membership is fixed independently of an individual method's output availability; metric-specific reference applicability can give smaller denominators. SO1/Core and SO4/Full use these same case definitions.

[patients.csv](patients.csv) has one row per included case; [slides.csv](slides.csv) lists the corresponding source WSI associations; [project_counts.csv](project_counts.csv) gives group-by-project counts. The technically unusable case TCGA-AA-3939 is excluded from every final list. The older development and screening inventories are not part of this release.

For the development subset, case-associated slides were recovered from the public WsiCaption source manifest; `in_frozen_roi_candidates` identifies which slides were represented among the saved ROI candidates. For the evaluation cohort, `recorded_primary_slide` identifies the preferred WSI in the patient manifest. `source_slide_count` counts the associations actually listed here. A listed slide need not have contributed an ROI to every report. DX, TS and other source slide types are retained.

## Obtaining the data

Use the [NCI GDC portal](https://portal.gdc.cancer.gov/) or each row's `download_url`. GDC filenames and UUIDs identify the source WSIs; the list export does not re-download them. For the runnable tutorial case, use [sample_tcga.json](../../configs/sample_tcga.json) and the notebook downloader.

These files contain public identifiers and source associations, not images, clinical report text or private cases. TCGA project labels support cohort auditing and evaluation; they are not supplied as a known diagnosis to inference. Source hashes and the inclusion-list version are recorded in [provenance.json](../provenance.json).
