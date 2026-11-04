---
license: other
license_name: sam-license
license_link: LICENSE
library_name: pytorch
tags:
  - pathology
  - segmentation
  - sam3
---

# PathXAI D1 inference adaptation

This model repository contains one compact D1 adaptation checkpoint. It does not contain SAM3 base weights. [`manifest.json`](manifest.json) records the adapter file's size and SHA-256, source checkpoint identity, and the pinned external SAM3 dependency.

## Files and inference dependencies

| File | Role | Size | SHA-256 |
|---|---|---:|---|
| `weights/pathxai_d1_adapter.pth` | D1 fine-tuned LoRA, prompt embeddings, organ bias, prompt mask, adapter, and retained auxiliary trained states | 25,256,190 bytes | `9e68ca4da8c990b164ce0ec0ea8d63b32f1bded2edddeaec9c228b936db443e0` |

Inference also requires the separate pinned SAM3 base model `facebook/sam3`, revision `3c879f39826c281e95690f02c7821c4de09afae7`, file `sam3.pt` (3,450,062,241 bytes; SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`). It is not hosted here. Users must request and receive access from the [official SAM3 model page](https://huggingface.co/facebook/sam3), then download it directly from Meta's repository.

The adapter was derived from D1 checkpoint `best_model.pth` (1,484,446,541 bytes; SHA-256 `41fd1867add7049bb6305e103b275c99ec513146f9d9e3f3f845b1dc55b0d031`) for run `tcga_expansion_20260924_1200_d1`. Before export, all 295 `prompt_enc.base_text_encoder.*` tensors were compared against the mapped tensors in the pinned SAM3 checkpoint and were exactly equal. The derived file omits those unchanged tensors, retains the D1 LoRA, prompt, adapter, and auxiliary trained states, and excludes optimizer/scheduler/scaler/resume-only state. The original checkpoint was not modified.

Use the [PathXAI tutorial on GitHub](https://github.com/zhaorui-tan/PathXAI) for the local Jupyter workflow, model configuration, SO1/SO4 patient-level inference, evaluation, and recorded HTML examples. Follow its installation guide to fetch the pinned upstream MedSAM3 source. CONCH weights and patient/case data are not included in this model repository.

## Access

Access to this repository does not grant access to the separate upstream SAM3 repository. If the repository owner enables gated access, users request access with their own Hugging Face account. The owner configures automatic or manual approval and whether profile information is collected; see the request form for the settings in effect.

This research software is not a medical device and is not intended to diagnose, treat, cure, or prevent disease. Users are responsible for validating it for their own data and setting.

## License and notices

HF metadata is `license: other` with the SAM License linked as `LICENSE`. This is the SAM License text published in the official SAM3 source and dated November 19, 2025. It covers SAM Materials including trained weights and derivatives and requires a copy to accompany distributed SAM materials. The licensing status of the D1 adaptation and applicability of this agreement to the pinned upstream revision must be confirmed by the publisher before distribution. This card does not claim blanket rights clearance.

## Integrity

Verify the adapter against [`manifest.json`](manifest.json) before loading. The manifest also identifies the separate SAM3 dependency and upstream checksum. Do not substitute a newer SAM3 base without checking compatibility.
