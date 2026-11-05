# Validated sample execution

TCGA-FD-A5BS: two source slides (DX1 and TSB), 20 pinned candidate ROIs, and the documented anatomical site.

| Version | Notebook runtime | Pipeline runtime | Final ROIs | Refined D1 segmentation |
| --- | ---: | ---: | ---: | --- |
| SO1 | 690.1 s | 614.2 s | 6 | passed |
| SO4 | 682.5 s | 548.4 s | 6 | passed |

The runs loaded the verified HF refined D1 adapter with the separately pinned SAM3 base. Source WSIs and all 20 candidate crops passed checksum verification. The frozen SO1/SO4 inference code generated the reports; exported clinical text and image bytes were checked against those outputs.

[Completed notebooks and HTML reports](../examples/README.md) · [Machine-readable validation](../provenance/validation_results.json)

Environment specifications describe the runtime used. A fresh installation of all specifications was not tested.
