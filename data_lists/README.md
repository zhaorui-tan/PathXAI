# Public data lists

The lists are separated into **[report](report/README.md)** and **[seg](seg/README.md)**. They contain public case IDs, source filenames or array indices, split membership and evaluation window coordinates. Images, masks, model weights and pathology report text are obtained separately from their original providers.

| Collection | Coverage | Start here |
| --- | --- | --- |
| Report study cohorts | 508 development + 1,200 confirmation cases; 4,160 associated/source WSIs | [Patients](report/patients.csv), [slides](report/slides.csv) |
| Complete report source inventory | 9,009 public TCGA cases; 10,949 listed WSIs; includes the study cohorts | [Source cases](report/source_inventory_patients.csv), [source slides](report/source_inventory_slides.csv) |
| Segmentation source pools | 24 datasets; 360,122 training-pool and 90,040 held-out entries | [Counts](seg/counts.csv), [train](seg/train_pool/), [held-out](seg/heldout_pool/) |
| Segmentation evaluation | 124 internal images; 11,016 external image/ROI entries; 2,539 AGGC windows from 73 specimens | [Internal](seg/test_internal.csv), [external](seg/test_external.csv), [AGGC](seg/aggc_windows.csv) |
| Supplemental segmentation evaluation | 9,800 CAMELYON16 windows from 49 slides | [Window list](seg/supplemental_camelyon16_windows.csv) |

The source inventory is larger than the evaluated report cohort. Training-pool membership does not establish that every item was sampled during a finite-step run. Images, windows, specimens and patients are different counting units; do not sum them as independent patients. The detailed guides document selection, overlap and provenance limits.

[sources.csv](sources.csv) records source links and unresolved dataset-version mappings. A blank link means the original download location was not verified, not that the sample list is absent. Original providers determine dataset access and terms. These lists do not grant access to controlled datasets or redistribute their contents.

Private clinical cohorts are outside these public lists. The local tutorial example remains pinned separately in [sample_tcga.json](../configs/sample_tcga.json).

## Integrity and reconstruction

[manifest.json](manifest.json) records the size, row count and SHA-256 of each released list and guide. [provenance.json](provenance.json) identifies the original input manifests/configs and their hashes, using source-relative names. Run from the repository root:

```bash
python3 scripts/audit_data_lists.py
```

The publisher's [export script](../scripts/export_public_data_lists.py) reconstructs the identifier lists from the archived project and segmentation trees. Its inputs are supplied explicitly; it does not load models or modify source data. Readers do not need those internal source trees to use the released CSV/TXT lists. Report labels in the lists are cohort metadata, not diagnostic inputs to inference.
