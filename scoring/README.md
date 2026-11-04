# PathXAI local scoring package

Two independent apps are included:

- [`single_report/`](single_report/): reference report authoring, optional
  correctness/unsupported-claim scoring, and diagnostic ROIs.
- [`compare_reports/`](compare_reports/): blinded five-report ranking,
  acceptability, and safety scoring.

Install system OpenSlide libraries and the Python dependencies in
[`requirements.txt`](requirements.txt), configure private manifests and output
paths, then follow the run commands in [SCORING.md](../docs/SCORING.md).
Both apps bind to localhost and keep data on the configured machine.
