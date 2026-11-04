# Recorded patient-level examples

These are actual completed local Jupyter executions for TCGA-OR-A5JI using **all five diagnostic slides DX1–DX5** and the explicitly supplied anatomical site. SO1 and SO4 both use the D1 adaptation with separately obtained SAM3/CONCH weights; SO4 also uses KEEP.

| Version | Notebook with actual outputs | HTML report | Saved text |
| --- | --- | --- | --- |
| SO1/Core | [Recorded notebook](notebooks/SO1_completed.ipynb) | [Report](SO1/report.html) | [Text](SO1/generated_long_report.txt) |
| SO4/Full | [Recorded notebook](notebooks/SO4_completed.ipynb) | [Report](SO4/report.html) | [Text](SO4/generated_long_report.txt) |

The candidate pool, initial/final selected ROI identities and main diagnosis match the historical D1 demo. **Grade/necrosis discrepancies and internal report inconsistencies remain.** Read [the comparison](comparison.html) or [the detailed audit](../docs/RESULT_COMPARISON.md). This single case is not clinical validation. All five slides enter the candidate pool; the selected diagnostic ROIs come from two slides, which are displayed in the reports.

GitHub previews notebook outputs without execution. To view HTML as a page, use the deployed GitHub Pages site, or download/clone the full `examples/` folder and open [index.html](index.html). Keep `assets/`, `SO1/`, and `SO4/` together. These saved reports need no model, API or Python service. Original image bytes are preserved and identical assets are deduplicated. No original WSI or model checkpoint is included.

Recorded code cells and execution counts are retained. Presentation cleanup removes ANSI formatting and heartbeat refreshes, condenses feature progress bars to recorded completion lines, merges adjacent stdout records, and replaces machine paths with relative links or labeled placeholders. The final Markdown report is copied from the actual saved text. No model response was fabricated or corrected. Original/published hashes and cleanup counts are in [manifest.json](manifest.json).

These executions reused existing checksum-verified local model files and GDC slides. Live GDC inventory and Range-prefix checks passed for all five slides; a fresh complete 8.06 GB transfer was not repeated. Download of our D1 from HF remains separate from this local validation. Use the clean [main notebook](../PathXAI_report.ipynb) to run inference; the archived notebooks are reading records.
