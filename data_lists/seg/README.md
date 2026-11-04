# Segmentation data lists

## Recorded source pools and fixed validation

[train_pool/](train_pool/) and [heldout_pool/](heldout_pool/) provide one TXT file per dataset, preserving the recorded split-file order. There are **24 datasets**, **360,122 eligible training entries** and **90,040 held-out entries**. [counts.csv](counts.csv) lists dataset-level counts and supervision roles. This includes SICAPv2, which was absent from the earlier A5-only list export.

IDs are paths relative to the local public-data collection root, with image extensions retained. `CoNIC2022` and `PanNuke` instead use the recorded zero-based array indices; these require the matching source array version/order and are not patient IDs. File-like IDs should be resolved against the provider's matching dataset layout; local collection directory names can differ from upstream archive names.

The source split files use a seeded, per-dataset 80:20 split. They define eligible pools; finite-step training does not establish that every entry was sampled. This export records the available source split files and their hashes, not a historical log of sampled minibatches or proof that current source files are byte-identical to every training-day copy. Train and held-out IDs are disjoint within each dataset. This is not a claim of patient-level independence across tiles or across different public datasets.

[fixed_validation.csv](fixed_validation.csv) contains **56 quick model-selection entries drawn from the training pool** and **34 monitoring entries from the held-out pools**. The model-selection subset is not independent validation. The monitoring subset overlaps the larger held-out pool; it is not an extra test cohort. The original and refined D1 setup caches are byte-identical. ID mapping comes from the archived dataset-index export, and membership in the respective full pools was verified.

## Evaluation lists

| List | Dataset counts | Identity source |
| --- | --- | --- |
| [test_internal.csv](test_internal.csv) | BCSS 31; DigestPath2019 50; GlaS 33; ICIAR2018_WSI 2; WSSS4LUAD 8 | Shared aligned internal manifest, 124 source images |
| [test_external.csv](test_external.csv) | ACDC_LungHP 5,000; PAIP2019 5,000; RINGS_Gland 500; RINGS_Tumor 500; METU_CCTGS 16 | Shared aligned external image/ROI manifest |
| [aggc_windows.csv](aggc_windows.csv) | 2,539 windows, 128 slide IDs, 73 specimens | Frozen AGGC window manifest; keys match refined D1 per-window output |
| [aggc_specimens.csv](aggc_specimens.csv) | 73 specimen IDs | Deduplicated from the window manifest; not an additional cohort |
| [supplemental_camelyon16_windows.csv](supplemental_camelyon16_windows.csv) | 9,800 windows, 49 slides | Recovered and deduplicated from refined D1 class/window metric output |

AGGC and CAMELYON window coordinates are retained so readers can identify the exact regions. Their overlapping class-level metric rows were not counted as additional windows. CAMELYON16 is supplemental. TIGER_WSI is recorded in [the source index](../sources.csv) as excluded because its ROI/WSI annotation and evaluation protocols were not aligned; there is no canonical TIGER test list in this release.

The internal and non-AGGC external tables are the shared aligned identity manifests. D1's corresponding aggregate/class metric CSVs omit sample identifiers, so those outputs cannot independently prove exact case-by-case identity. We do not relabel this as a newly rerun D1 identity validation. The AGGC output does retain sufficient keys to verify all 2,539 windows; CAMELYON identities are explicitly output-derived rather than a separately frozen input manifest.

## D1 lineage

The HF adaptation used by the report tutorial derives from the **original June 19 D1 run** (`stage3_ablation_D1/run_20260619_144518_stage3_tissue`), with source checkpoint SHA-256 `41fd1867add7049bb6305e103b275c99ec513146f9d9e3f3f845b1dc55b0d031`. The **August 4 refined D1** (`medsam3_d1_refined/d1_final/run_20260804_140335_stage3_tissue`) is a separate segmentation benchmark lineage. They must not be treated as one checkpoint. Both config identities and their common validation-cache hashes are recorded in [provenance.json](../provenance.json); the external/supplemental benchmark lists above refer to the archived segmentation benchmark scope.

All image/mask/annotation paths in the released lists are portable identifiers. No source images, masks, private cohort IDs or model weights are included. Dataset-version mappings that remain unresolved are stated in [sources.csv](../sources.csv).
