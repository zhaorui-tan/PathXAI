# Validation record for TCGA-OR-A5LL

SO1/Core and SO4/Full completed notebook execution and report export for the 5 diagnostic slides listed in `configs/sample_tcga.json` (10,988,208,661 bytes; 10.23 GiB). The notebook receives the sample's configured anatomical site.

## Recorded runs

| Version | Notebook runtime | Pipeline runtime | Candidate ROIs | Final ROIs | Displayed WSIs | D1 segmentation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SO1 | 1429.2 s | 515.3 s | 20 | 6 | 2 | true |
| SO4 | 1031.4 s | 647.2 s | 20 | 6 | 2 | true |

Environment specifications were not installed from scratch for these runs (clean install tested: false).

## Historical standalone model-load check

A separate test on 2026-11-04 loaded the published D1 adapter and ran segmentation on three TCGA-OR-A5JI ROIs. That historical A5JI test reported its own mask-output comparison; it is not an A5LL result. See [the archived test receipt](../provenance/hf_inference_validation.json).
