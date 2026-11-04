# D1 model release and Hugging Face upload

## Bundle contents

The model repository content is assembled in [`model_release/upload_ready/`](../model_release/upload_ready/). Upload the files in that directory, preserving the relative path `weights/pathxai_d1_adapter.pth`. The bundle includes the adapter, its model card, a machine-readable manifest, and the SAM License. It contains no SAM3 base weights, CONCH weights, patient data, or case data.

The adapter is 25,256,190 bytes with SHA-256 `9e68ca4da8c990b164ce0ec0ea8d63b32f1bded2edddeaec9c228b936db443e0`. It was derived from the selected D1 checkpoint (1,484,446,541 bytes; SHA-256 `41fd1867add7049bb6305e103b275c99ec513146f9d9e3f3f845b1dc55b0d031`). A CPU, memory-mapped check compared all 295 embedded `prompt_enc.base_text_encoder.*` tensors with their mapped values in the pinned SAM3 checkpoint; all 295 were exactly equal. The exported file retains D1 LoRA, learned prompts, organ bias, prompt mask, adapter, and auxiliary trained states. It removes only those unchanged text-base tensors and optimizer/scheduler/scaler/resume state. The original checkpoint was not modified.

Inference still needs the separate pinned SAM3 dependency: `facebook/sam3`, revision `3c879f39826c281e95690f02c7821c4de09afae7`, `sam3.pt`, expected size 3,450,062,241 bytes and SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`. Each user obtains access and downloads SAM3 directly from the [official model page](https://huggingface.co/facebook/sam3). The model card links to the [PathXAI GitHub tutorial](https://github.com/zhaorui-tan/PathXAI), whose installation guide provisions the pinned upstream source and whose notebook runs patient-level SO1/SO4 inference.

## License setting

The model card uses `license: other`, `license_name: sam-license`, and `license_link: LICENSE`. The bundle's `LICENSE` is the SAM License text from the official SAM3 source, last updated November 19, 2025. Do not replace it with Apache-2.0, MIT, or CC-BY-NC-ND. The SAM License covers SAM materials including trained weights and derivatives and requires a copy to accompany distributed materials. Confirm the terms applicable to the pinned model revision and the D1 adaptation's distribution authority before publication; the bundle does not assert blanket rights clearance.

No additional academic-only or noncommercial restriction is applied by this bundle. The SAM License is not an academic-only license.

## Manual upload (recommended)

1. Confirm that the relevant rights holders and institution approve distribution of the D1 adaptation under the applicable SAM terms.
2. Create a Hugging Face **model** repository under the intended owner. Start private while checking the files and settings.
3. In the repository's Files page, upload every file from `model_release/upload_ready/`, preserving the `weights/` folder. `README.md` contains the model-card metadata and `manifest.json` records the adapter hash and external SAM3 dependency. The selected repository is [`RayTan/PathXAI-Seg`](https://huggingface.co/RayTan/PathXAI-Seg).
4. Confirm the card displays license `other`, name `SAM License`, and a link to `LICENSE`. Do not add a blanket project-license claim.
5. If using access requests, enable gated access and choose automatic or manual approval. Manual approval adds a waiting step. HF collects usernames/emails by default, but its gating settings allow disabling **Collect username** and **Collect email**; new requests can then be reviewed by request ID. For a reviewer-access route intended to preserve anonymity, also avoid identity-revealing custom fields and coordinate the access arrangement with the editor. These settings affect new requests only. This gate does not grant upstream SAM3 access.
6. Verify the uploaded `weights/pathxai_d1_adapter.pth` size and SHA-256 against `manifest.json`.

Hugging Face's current guides cover [gated access and manual approvals](https://huggingface.co/docs/hub/en/models-gated) and [uploading files](https://huggingface.co/docs/huggingface_hub/main/guides/upload).

For a single command that uploads the prepared folder, run this from the tutorial repository root:

```bash
cd model_release/upload_ready
hf upload RayTan/PathXAI-Seg .
```

## Optional CLI

From the `PathXAI_tutorial` directory, verify the prepared bundle without contacting the Hub:

```bash
python model_release/stage_upload.py
```

The optional Python uploader can also target the selected repository:

```bash
python model_release/stage_upload.py --upload --repo-id RayTan/PathXAI-Seg
```

The uploader verifies the checkpoint first, uses `HF_TOKEN` or an existing `hf auth login` credential, and never prints token contents. No upload has been made.
