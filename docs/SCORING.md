# Local scoring tools

This tutorial package contains two standalone local applications adapted from
the repository's Human_eval and Human_Compare_SO4_D1 interfaces:

1. **Single-report authoring and scoring** in `scoring/single_report/`.
   A reviewer inspects one slide, records structured findings and report prose,
   marks diagnostic regions, and can score diagnostic correctness and
   unsupported claims.
2. **Five-report comparison** in `scoring/compare_reports/`.
   Five reports are assigned blinded labels A–E per local session and case.
   The reviewer ranks each report overall and by section, rates clinical
   acceptability per report, and marks major unsupported claims and omissions.

Both services bind to `127.0.0.1` by default. They do not upload slides,
reports, or scores, and do not call model APIs. Start them on the machine that
can read the image and report files. The browser communicates with that local
server. Do not configure a public host. Score files are written only to the
output directory you pass at startup; export them from the app's CSV and JSONL
links.

## Install native slide dependencies

The applications use OpenSlide for whole-slide images and Pillow for tile
encoding. Install the system OpenSlide library first, then Python packages.

```bash
# Debian/Ubuntu
sudo apt-get install openslide-tools libopenslide-dev
python3 -m pip install openslide-python Pillow
```

On macOS, install `openslide` with Homebrew before `pip install openslide-python
Pillow`. On Windows, install an OpenSlide binary distribution and ensure its
DLLs are on `PATH` before installing the Python packages. The server reports a
clear startup error if the Python binding is missing.

## Configure your data paths

Use anonymized case IDs in manifests. Keep manifests, input reports, WSI files,
score output, and generated DZI tile caches in directories you control. The
sample report text in `scoring/examples/` is explicitly
synthetic demonstration text and is not a real patient record. There are no
slide images in this package.

### Single-report authoring

Copy `scoring/single_report/manifests/cases.example.json` to a
private location. For each case, set `case_id`, `anon_id`, and `slide_path` to
your anonymized case key and absolute local slide path. Keep `anon_id` free of
patient identifiers. Keep slide paths within your local data directory.

```bash
cd scoring/single_report
python3 app.py \
  --manifest /your/private/config/cases.json \
  --data-root /your/private/slides \
  --annotation-root /your/private/output/single_report_scores \
  --dzi-cache-root /your/private/output/single_report_dzi \
  --host 127.0.0.1 --port 8787
```

Open `http://127.0.0.1:8787`. The UI creates a random local session code and
does not request a name, specialty, or other identity. The code is a storage
key, not authentication. Use separate browser profiles if separate local
sessions are needed. `Save Draft` writes a draft; `Submit / Lock` marks it
submitted. The prior version is retained when an edit is saved. CSV and JSONL
exports are available in the page header.

The form preserves the source authoring workflow: structured diagnosis facts,
narrative sections, optional diagnostic ROIs, diagnostic correctness (1–5),
unsupported-claim severity (none/minor/major), and notes. See
`scoring/single_report/config/scoring_schema.json`.

### Five-report comparison

Copy `scoring/compare_reports/manifests/cases.example.json` to
a private location. Each entry needs an anonymized `case_id`, absolute
`slide_path`, and exactly five report paths under `reports`. Report-map keys
are internal identifiers and are not shown in the browser; labels A–E are
generated deterministically from the local session code and case. Point paths
to your own reports. Relative slide and report paths are resolved from the
manifest file's directory. The example report files are synthetic text and
have no clinical images.

```bash
cd scoring/compare_reports
python3 app.py \
  --manifest /your/private/config/comparison_cases.json \
  --annotation-root /your/private/output/comparison_scores \
  --dzi-cache-root /your/private/output/comparison_dzi \
  --secret-file /your/private/config/assignment_secret.txt \
  --host 127.0.0.1 --port 8792
```

Open `http://127.0.0.1:8792`. A random local session code separates each
reviewer's score files; no identity fields are collected. Keep the assignment
secret private and stable for the duration of a scoring run so each session
sees a consistent order. The service creates a random secret if the file does
not exist. Drafts and submitted score records are written under the configured
annotation root. Export CSV or JSONL from the app header.

The five-report rubric is preserved: required overall ordering on a 1–5 scale
(1 is best; ties are allowed), section rankings with required core domains when
present (Diagnosis, Key Microscopic Findings, Stage / Grade / Prognostic
Features), and per-report acceptability (`no_correction`, `minor_correction`,
`major_correction`, `not_usable`), unsupported-major-claim, and
major-visible-omission judgments. See
`scoring/compare_reports/config/scoring_schema.json`.

## Export and cleanup

CSV and JSONL exports contain score records and random session codes. Treat
exports as potentially sensitive research data and store them accordingly.
The applications do not store names or physician profiles. Clear browser local
storage and remove the configured output/cache directories when you no longer
need those local records, according to your own retention policy.

Run the synthetic isolation checks with:

```bash
python3 -m unittest discover -s scoring/tests -v
```

The tests mock OpenSlide and use temporary synthetic reports and slide paths;
they verify per-session score separation and identity-field filtering. They do
not validate a system OpenSlide installation or open a real WSI.

## Adaptation notes

- The Human_eval-derived app keeps its authoring fields, save/lock history,
  OpenSlide-backed DZI viewer, ROI handling, and CSV/JSONL exports. Its default
  case path is now an example manifest; deployment-specific paths are CLI
  arguments. Name, specialty, role, and experience collection were removed.
- The SO4_D1-derived app keeps the five-report UI and scoring rubric,
  per-session/case blinding, multi-slide viewer, draft synchronization,
  submission locking, and score export. Fixed study method IDs are replaced by
  the five report keys supplied in each user's manifest. Original study
  annotation imports, randomized re-review rounds, score prefill, and
  rank-scale migration are not activated by the standalone launcher.
- The shared OpenSlide implementation is included locally under
  `compare_reports/vendor/`; it does not import or read from the original study
  directory. The vendored OpenSeadragon library and its license are included
  in each app's static assets.

No existing annotations, credentials, study manifests, source reports, WSI
files, or physician identities were copied into this package.
