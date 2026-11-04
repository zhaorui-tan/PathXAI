# Report data lists

## Main study cohorts

| Cohort | Cases | TCGA projects | Listed source WSIs |
| --- | --- | --- | --- |
| [original508_development](original508_development.txt) | 508 | 30 | 1,039 |
| [new1200_confirmation](new1200_confirmation.txt) | 1,200 | 22 | 3,121 |
| Total, disjoint cases | 1,708 | Projects overlap across cohorts | 4,160 |

[patients.csv](patients.csv) has one row per study case; [slides.csv](slides.csv) has one row per source WSI association. [project_counts.csv](project_counts.csv) gives cohort-by-project counts. The SO1/SO4 and A5/D1 runs share these cohort definitions; they are not additional independent patient sets.

For **original508**, the frozen report manifest contains precomputed candidate ROIs, not a complete WSI inventory. The full case-associated WSI list was recovered from the public Wsi-Caption source manifest. Of its 1,039 WSIs, **822** are represented among the frozen candidate ROIs; `in_frozen_roi_candidates` distinguishes them. The remaining 217 are source associations, not proven model inputs. `list_scope=case_association_inventory` preserves this distinction.

For **new1200**, all 3,121 WSIs come from the frozen patient manifest's `slide_paths`. `recorded_primary_slide` identifies the preferred WSI recorded in that manifest. This is the frozen source inventory, not a claim that every source slide yielded an ROI in every generated report. The original, A5-bridge and D1 expansion manifests have identical bytes. These lists retain DX, TS and other source slide types rather than applying the tutorial's five-DX sample policy to the entire study.

## Complete public source inventory

[source_inventory_patients.csv](source_inventory_patients.csv) contains all **9,009** public TCGA cases in the source manifest. [source_inventory_slides.csv](source_inventory_slides.csv) contains its **10,949** listed WSIs. Some source cases have no listed WSI; `listed_slide_count=0` preserves those entries.

The 1,708 study cases are included and marked `in_report_study=1`. The other **7,301** cases belong to the source/screening inventory; they must not be counted as cases evaluated in the released report study. This flag does not describe participation in unrelated historical experiments.

## Obtaining the data

Use the [NCI GDC portal](https://portal.gdc.cancer.gov/) or the `download_url` derived from each recorded GDC file ID. The filenames and UUIDs are taken from the source manifests; all 10,949 files were not re-downloaded or revalidated against the live GDC API during this list export. For the runnable, checksum-verified tutorial sample, use [sample_tcga.json](../../configs/sample_tcga.json) and the notebook downloader.

`slide_type` classifies the filename's DX/TS token, including letter suffixes such as TSA; it is not a pathology label. Local machine paths, report text and clinical reference text are omitted. Project names are metadata for cohort auditing: do not feed a TCGA project label into the diagnostic prompt as a known diagnosis.
