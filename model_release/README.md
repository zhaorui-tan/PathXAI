# D1 model release workspace

`manifest.json` is the public-safe, path-free adapter identity contract. The ready-to-review Hub folder is [`upload_ready/`](upload_ready/); its `.pth` payload is gitignored. It contains the derived D1 adaptation, its card, manifest, and SAM License. It does not contain SAM3 or CONCH base weights. No additional academic-only restriction is applied by the model bundle.

The adapter was exported from the pinned full D1 checkpoint only after an exact CPU comparison proved all 295 embedded `base_text_encoder` tensors equal the pinned SAM3 language-backbone tensors. See [`export_adapter.py`](export_adapter.py) and [`upload_ready/manifest.json`](upload_ready/manifest.json). The original D1 file was not modified.

From the `PathXAI_tutorial` directory, verify the prepared bundle without uploading:

```bash
python model_release/stage_upload.py
```

For the selected repository, upload the prepared folder with Hugging Face CLI:

```bash
cd model_release/upload_ready
hf upload RayTan/PathXAI-Seg .
```

The optional Python uploader uses `HF_TOKEN` if set, otherwise the account's existing `hf auth login` token. It never prints token contents. No Hub upload has been performed.
