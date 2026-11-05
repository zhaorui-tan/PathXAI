"""Generate the distributable notebook without outputs or credentials."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
def code(text, tag):
    cells.append({"cell_type": "code", "metadata": {"tags": [tag]}, "execution_count": None,
                  "outputs": [], "source": text.splitlines(True)})

md("""# PathXAI: a TCGA patient to an HTML report

Run this notebook on your own NVIDIA GPU computer/server after following [installation](docs/INSTALLATION.md).
Choose SO1/Core or SO4/Full; both use refined D1 segmentation internally. This is real inference, not a stored-report viewer.
The public sample is the six-slide patient input for TCGA-OR-A5LL, with a documented
anatomical site supplied before inference. The source slides and pinned candidate regions produce one patient report. The exact slide inventory, sizes and checksums are pinned in `configs/sample_tcga.json`.

Obtain your own access to SAM3, CONCH, and PathXAI D1 on Hugging Face. SO4 additionally needs KEEP access.
Supply a vision-capable Qwen API endpoint/model and credentials. Selected ROI images are sent to that API.
Never save credentials in notebook cells or outputs. References are used only in the optional evaluation cell after generation.
""")
code("""from pathlib import Path
import os, sys, json
from getpass import getpass

ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'pathxai_tutorial').is_dir()), None)
if ROOT is None:
    raise RuntimeError('Start Jupyter from the PathXAI_tutorial repository directory.')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from pathxai_tutorial import download_patient, sample_metadata, prepare_sample_candidates, make_config, check_api, run_report
from pathxai_tutorial.models import download_models
print('Tutorial:', ROOT.name)
""", "setup")
md("""## 1. Choose a version and configure paths

Set the segmentation interpreter to the `pathxai-segmentation-py312` environment from the installation guide.
The current notebook kernel runs the report/CONCH pipeline. The D1 repository is `RayTan/PathXAI-Seg`;
the published immutable revision is filled in below. Obtain your own access to the separate upstream models before downloading.
The optional local model configuration is documented in [the inference guide](docs/INFERENCE.md).
""")
code("""VERSION = os.environ.get('PATHXAI_VERSION', 'SO1')  # SO1 or SO4
SEGMENTATION_PYTHON = os.environ.get('PATHXAI_SEG_PYTHON', '/path/to/pathxai-segmentation-py312/bin/python')
D1_REPO_ID = os.environ.get('PATHXAI_D1_REPO', 'RayTan/PathXAI-Seg')
D1_REVISION = os.environ.get('PATHXAI_D1_REVISION', 'abfe61cb8e6e916b7226fd9dba077ef95ee9ee4f')  # immutable 40-character HF commit
QWEN_API_BASE = os.environ.get('QWEN_API_BASE', '')
QWEN_MODEL = os.environ.get('QWEN_MODEL', '')
MAX_SECONDS = int(os.environ.get('PATHXAI_MAX_SECONDS', '3600'))
LOCAL_MODEL_CONFIG = os.environ.get('PATHXAI_LOCAL_MODEL_CONFIG', '')

# This optional JSON file contains paths and endpoint/model settings, never keys.
local = json.loads(Path(LOCAL_MODEL_CONFIG).read_text()) if LOCAL_MODEL_CONFIG else {}
if 'api_key' in local or 'hf_token' in local:
    raise ValueError('Remove credentials from the local configuration; use environment variables.')
""", "configuration")
md("""## 2. Credentials and model downloads

The adapter download contains our LoRA, learned prompts and necessary D1 auxiliary parameters.
SAM3 base weights and CONCH come from their original repositories under your own permissions.
No automatic model substitution is performed when access is unavailable.
""")
code("""api_key = os.environ.get('QWEN_API_KEY', '')
if not api_key and os.environ.get('PATHXAI_API_NO_KEY', '') != '1':
    api_key = getpass('Qwen API key (leave blank only if your endpoint needs none): ')

if local:
    config = make_config(**local, api_key=api_key)
else:
    if not D1_REPO_ID or not D1_REVISION:
        raise ValueError('Set D1_REPO_ID and D1_REVISION from the model release before downloading.')
    hf_token = os.environ.get('HF_TOKEN') or getpass('HF read token for your approved model access: ')
    assets = download_models(d1_repo_id=D1_REPO_ID, d1_revision=D1_REVISION,
                             version=VERSION, token=hf_token)
    config = make_config(segmentation_python=SEGMENTATION_PYTHON,
                         qwen_api=QWEN_API_BASE, qwen_model=QWEN_MODEL,
                         api_key=api_key, hf_token=hf_token, **assets)
print('Selected report version:', VERSION)
""", "models")
md("""## 3. Download and validate the public TCGA patient

The downloader checks every pinned GDC record, file size, MD5 and SHA-256. Interrupted transfers can be resumed.
If you already have the slide set, set `PATHXAI_SAMPLE_DIR` to its parent directory (nested GDC UUID folders
are supported). Every expected slide must pass verification; an incomplete patient is never run silently.
For your own data, replace `slides`, `case_id` and `known_site` with your patient's slide list, identifier
and anatomical sampling site, and set `candidate_rois = None` and `known_site_source = "user"`
to generate candidate ROIs from your own slides. Do not pass a tumor diagnosis as `known_site`.
The included sample reconstructs its pinned candidate ROIs directly from the downloaded slides.
""")
code("""sample = sample_metadata()
slides = download_patient(ROOT / 'data', existing_dir=os.environ.get('PATHXAI_SAMPLE_DIR') or None)
case_id = sample['case_id']
known_site = sample['known_site']
known_site_source = sample['known_site_source']['pipeline_marker']
candidate_rois = prepare_sample_candidates(slides, ROOT / 'data' / 'sample_rois')
print('Patient:', case_id)
print('Known anatomical site:', known_site)
print('Verified slides:', len(slides))
for slide in slides:
    print(' -', slide.name)
""", "sample")
md("""## 4. Verify the configured API

This makes one small synthetic-image request. A successful probe checks connectivity only; it does not
guarantee the full scientific workflow will succeed. No sample WSI is sent in this step.
""")
code("""probe = check_api(config)
print(probe['message'])
if not probe['ok']:
    raise RuntimeError('API probe failed. Correct endpoint, model and key before running inference.')
""", "api_probe")
md("""## 5. Run report generation

One case and one version run at a time. Keep the kernel/server alive; progress is also written to disk.
The time budget cancels the owned worker and preserves partial artifacts. Re-running uses a new output directory.
The sample supplies its documented anatomical site explicitly, as the historical patient run did.
Only slide images, candidate regions, patient identifier and anatomical site are passed to inference; cancer-project labels,
reference diagnosis, grade and evaluation results are not supplied to the report generator.
""")
code("""result = run_report(config, slide_paths=slides, case_id=case_id, version=VERSION,
                    known_site=known_site, known_site_source=known_site_source,
                    candidate_rois=candidate_rois, max_seconds=MAX_SECONDS,
                    output_dir=os.environ.get('PATHXAI_OUTPUT_DIR') or None)
print('Report:', result['report_html'])
print('Report text:', result['report_text'])
""", "inference")
md("""## 6. Open/export the report

Keep the HTML together with its `assets/` directory. Download/copy the whole portable folder to share the report.
The raw scientific outputs and exact configuration/provenance are retained beside it.
""")
code("""from IPython.display import FileLink, display
report_html = Path(result['report_html'])
try:
    display(FileLink(str(report_html.relative_to(ROOT))))
except ValueError:
    print('Report outside the Jupyter root; open it locally:', report_html)
print('Portable report folder:', report_html.parent)
""", "output")
md("""## 7. Optional automatic report evaluation

Provide an independently curated reference report for the same case. This sample download does not contain
a verified reference report. Do not generate a reference from the model's output or infer it from a TCGA project label.
Use `generated_long_report.txt`, not the entire HTML viewer (which also contains evidence and differential diagnoses).
See [evaluation](docs/EVALUATION.md) for field applicability, metrics, structured references and batch input.
The two physician scoring tools are described separately in [scoring](docs/SCORING.md).
""")
code("""REFERENCE_REPORT = None  # Path('/your/reference_report.txt')
if REFERENCE_REPORT is None:
    print('Evaluation skipped: supply an independent reference report first.')
else:
    from evaluation.run_evaluation import evaluate_paths
    metrics = evaluate_paths(Path(result['report_text']), Path(REFERENCE_REPORT))
    evaluation_path = Path(result['run_dir']) / 'evaluation_v4.json'
    evaluation_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + '\\n')
    print('Evaluation saved:', evaluation_path)
""", "evaluation")

for i, cell in enumerate(cells):
    cell['id'] = f'pathxai-{i:02d}'
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python (PathXAI)",
    "language": "python", "name": "pathxai"}, "language_info": {"name": "python", "version": "3.10"}},
    "nbformat": 4, "nbformat_minor": 5}
if __name__ == '__main__':
    (ROOT / 'PathXAI_report.ipynb').write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n')
