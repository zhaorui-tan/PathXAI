# Validated sample execution

TCGA-OR-A5LL: six source slides (DX1–DX5 and TS1), 20 pinned candidate ROIs, and the documented anatomical site.

| Version | Notebook runtime | Pipeline runtime | Final ROIs | D1 segmentation |
| --- | ---: | ---: | ---: | --- |
| SO1 | 908.4 s | 522.9 s | 6 | passed |
| SO4 | 740.2 s | 517.1 s | 6 | passed |

The runs loaded the verified HF D1 adapter with the separately pinned SAM3 base. Source WSIs and all 20 candidate crops passed checksum verification. The frozen SO1/SO4 inference code generated the reports; exported clinical text and image bytes were checked against those outputs.

[Completed notebooks and HTML reports](../examples/README.md) · [Machine-readable validation](../provenance/validation_results.json)

Environment specifications describe the runtime used. A fresh installation of all specifications was not tested.
