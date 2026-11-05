# Validated sample execution

TCGA-OR-A5LL: six source slides (DX1–DX5 and TS1), 20 pinned candidate ROIs, and the documented adrenal gland / adrenocortical site.

| Version | Report runtime | Pipeline runtime | Final ROIs | Input slides |
| --- | ---: | ---: | ---: | ---: |
| SO1 | 545.5 s | 528.2 s | 6 | 6 |
| SO4 | 513.3 s | 510.6 s | 6 | 6 |

Both runs completed through the tutorial Python API using a freshly downloaded, checksum-verified HF refined D1 adapter and the separately pinned SAM3 base. All six WSI hashes and all 20 reconstructed candidate crops matched the pinned inputs. Exported report text and image bytes were verified against the original run outputs.

The example notebooks display these captured runs; their display cells were separately executed successfully. The main `PathXAI_report.ipynb` invokes the same API for fresh inference.

[Recorded outputs and HTML reports](../examples/README.md) · [Machine-readable validation](../provenance/validation_results.json)

Environment specifications describe the runtime used. A fresh installation of all specifications was not tested.
