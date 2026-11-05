# Execution summary for TCGA-OR-A5LL

The SO1/Core and SO4/Full tutorial runs processed the 5 diagnostic slides in the pinned sample manifest (10,988,208,661 bytes; 10.23 GiB). Both notebook executions completed and their reports passed the recorded export audits.

| Run | Notebook runtime | Pipeline runtime | Candidate ROIs | Final ROIs | Displayed WSIs | D1 segmentation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SO1 | 1429.2 s | 515.3 s | 20 | 6 | 2 | true |
| SO4 | 1031.4 s | 647.2 s | 20 | 6 | 2 | true |

Environment specifications were not installed from scratch for these runs (clean install tested: false).

The separate 2026-11-04 D1 standalone segmentation test used three TCGA-OR-A5JI ROIs and is retained as a historical A5JI record in [the test receipt](../provenance/hf_inference_validation.json).
