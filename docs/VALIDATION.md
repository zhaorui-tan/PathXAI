# Validated sample execution

TCGA-OR-A5LL: six source slides (DX1–DX5 and TS1), 20 pinned candidate ROIs, and the documented adrenal gland / adrenocortical site.

| Version | Notebook runtime | Pipeline runtime | Final ROIs | Input slides |
| --- | ---: | ---: | ---: | ---: |
| PathXAI(core) | 794.3 s | 515.7 s | 6 | 6 |
| PathXAI | 939.9 s | 574.6 s | 6 | 6 |

Both runs completed through executed tutorial notebooks using a checksum-verified HF refined D1 adapter from the previous fresh HF download and the separately pinned SAM3 base. The main notebook records the SO4 PathXAI execution; the completed example notebooks record SO1 and SO4. All six WSI hashes and all 20 reconstructed candidate crops matched the pinned inputs. Exported report text and image bytes were verified against the original run outputs.

The notebook execution outputs, run records and HTML report checksums are published together to show which run produced each report. Qwen outputs can vary across model versions, quantizations, hardware and calls.

[Recorded outputs and HTML reports](../examples/README.md) · [Machine-readable validation](../provenance/validation_results.json)

Environment specifications describe the runtime used. A fresh installation of all specifications was not tested.
