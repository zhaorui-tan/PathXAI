# Installation

PathXAI uses two Python interpreters. The tested inference setup is Linux with an NVIDIA GPU. The Jupyter notebook runs in the Python 3.10 pipeline environment; the MedSAM segmentation subprocess runs in a separate Python 3.12 environment. Keep these environments separate because their tested Torch and TorchVision versions differ.

The commands below describe the versions observed in the project’s installed `conch` and `medsam` environments on the Linux host, with added notebook/HTML dependencies. The import checks used those existing environments and the frozen source shipped under `vendor/report_engine/baselines/current/source_snapshot/`, together with the separately prepared MedSAM3 runtime source described below. They did not recreate either environment from scratch. Both Torch interpreters reported CUDA 13.0 and `torch.cuda.is_available() == true`. The import checker itself runs no model inference. The TCGA-OR-A5LL sample subsequently completed real SO1/SO4 notebook execution; environment import checks alone are not inference evidence. See [VALIDATION.md](VALIDATION.md) for current status.

## Prepare the external MedSAM3 source

From the tutorial checkout root, run this once before running environment import checks or opening the notebook:

```bash
python3 scripts/fetch_medsam_source.py
```

The script obtains the pinned upstream MedSAM3 source and applies the two tutorial patches needed by this runtime. It uses Python's standard library and network access; it does not install packages, fetch model weights, require a GPU, or change the frozen scientific snapshot. The fetched files and patch provenance are recorded in `provenance/external_medsam_source.json`. See [third-party notices](THIRD_PARTY_NOTICES.md) for licensing context.

Allow at least 10,988,208,661 bytes (10.23 GiB) for the 5 pinned diagnostic WSI files, plus additional space for resumable partial downloads, derived tiles/caches, models and inference outputs. Keep temporary and output storage on a volume with sufficient free space before starting a full run. The sample input is the diagnostic-slide set listed in the pinned manifest.

## Python 3.10 notebook and pipeline environment

Create the base environment from the tutorial root:

```bash
conda env create -f environments/notebook.yml
conda activate pathxai-notebook-py310
```

Install the observed CUDA 13.0 Torch pair and the exact editable CONCH source revision:

```bash
python -m pip install torch==2.12.0 torchvision==0.27.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install --no-deps -e \
  'git+https://github.com/mahmoodlab/CONCH.git@141cc09c7d4ff33d8eda562bd75169b457f71a62#egg=conch'
```

The CONCH revision is the Git commit recorded by the existing editable distribution (`conch` 0.1.0). The environment file pins the observed CONCH dependencies, notebook kernel, report HTML dependencies (`lxml`, Pillow), and WSI imports. The bundled pipeline and renderer sources are loaded from this checkout; they are not installed from a second Git checkout.

The notebook server is installed in this same environment (`notebook==7.6.4`, Python 3.10 compatible; see the [Jupyter Notebook release](https://pypi.org/project/notebook/7.6.4/)). From the tutorial root, start it with:

```bash
jupyter notebook PathXAI_report.ipynb
```

The notebook declares the `pathxai` kernel. Register that kernel name so the server selects this interpreter:

```bash
python -m ipykernel install --user --name pathxai \
  --display-name 'Python (PathXAI)'
```

## Python 3.12 segmentation environment

Create a second environment:

```bash
conda env create -f environments/segmentation.yml
conda activate pathxai-segmentation-py312
```

Install the observed CUDA 13.0 Torch pair:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu130
```

The segmentation worker uses the separately prepared MedSAM3 source after the setup command above. The worker adds its source paths to `sys.path`; do not install a different `sam3` release. Model checkpoint files are a separate asset and are not part of these environments.

## Check both interpreters

From the tutorial root, run each check with its corresponding interpreter:

```bash
conda run -n pathxai-notebook-py310 python environments/check_environment.py --role notebook
conda run -n pathxai-segmentation-py312 python environments/check_environment.py --role segmentation
```

The checker imports the modules used by the worker paths, reports Python/Torch/TorchVision versions, source locations, and CUDA availability as JSON, and exits nonzero on missing imports or unexpected core versions. It does not initialize a model or contact an API. A `cuda_available: false` result means the installed driver/runtime cannot expose a GPU to that interpreter; the import check can still pass.

## Observed environment inventory

| Role | Python | Torch / TorchVision | Other relevant observed versions |
| --- | --- | --- | --- |
| Notebook/pipeline | 3.10.20 | 2.12.0+cu130 / 0.27.0 | `conch` 0.1.0 at commit `141cc09c7d4ff33d8eda562bd75169b457f71a62`; Pillow 12.2.0; PyYAML 6.0.3; requests 2.34.1 |
| Segmentation | 3.12.13 | 2.11.0+cu130 / 0.26.0 | Hydra 1.3.2; iopath 0.1.10; OpenCLIP 3.3.0; Pillow 12.2.0; PyYAML 6.0.3 |

The tutorial validation environment adds Notebook server `7.6.4`, `ipykernel` `7.3.0`, and `lxml` `6.1.3` to the existing pipeline environment. The recorded inference runs used a real Jupyter kernel; the Notebook server was installed afterward for the delivery environment check. The environment files have not been independently recreated or clean-install tested, so treat them as pinned setup specifications rather than a claim of a clean-room installation test.

The CUDA wheel commands follow PyTorch’s official installation matrix for the observed version pairs: [PyTorch previous-version commands](https://pytorch.org/get-started/previous-versions/). The selected CUDA 13.0 wheel does not install or guarantee a compatible NVIDIA driver; use a host with a driver that supports that runtime for GPU inference.
