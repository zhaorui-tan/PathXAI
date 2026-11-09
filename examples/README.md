# Executed PathXAI examples

Final PathXAI(core) and PathXAI outputs for **TCGA-OR-A5LL**, using six pinned source slides and the refined D1 adapter.

| Version | Executed inference notebook | HTML report | Report text |
| --- | --- | --- | --- |
| PathXAI(core) | [Notebook](notebooks/SO1_completed.ipynb) | [View report](https://zhaorui-tan.github.io/PathXAI/SO1/report.html) | [Text](SO1/generated_long_report.txt) |
| PathXAI | [Notebook](notebooks/SO4_completed.ipynb) | [View report](https://zhaorui-tan.github.io/PathXAI/SO4/report.html) | [Text](SO4/generated_long_report.txt) |

The notebooks contain executed PathXAI inference cells for these TCGA-OR-A5LL reports. The main [PathXAI notebook](../PathXAI_report.ipynb) records the SO4 execution. The publication manifest links notebook provenance and run records to generated report text and HTML checksums. Both variants use the six pinned source slides and refined D1 segmentation.

The six slides total 11,156,338,264 bytes (10.39 GiB). Reports retain their original text and image bytes. Machine-specific output paths and repeated progress messages are normalized for display.

GitHub previews the notebooks without execution. For offline HTML viewing, download the full `examples/` folder and open [index.html](index.html). Keep the report folders, shared `assets/` and notebook records together. WSI files and model weights are obtained separately.

## Execution trajectories

Four additional request/response trajectories are available for **TCGA-OR-A5JI** (existing development case) and **TCGA-08-0516** (randomly selected GDC case), each with SO1/Core and SO4/Full runs. They show input images, prompts, model outputs and token usage in call order.

[Open the trajectory viewer](https://zhaorui-tan.github.io/PathXAI/traces/) · [Run settings and selection notes](traces/README.md)

Thinking was disabled in these runs; no internal reasoning text was returned. The existing A5LL reports and notebooks above are retained.
