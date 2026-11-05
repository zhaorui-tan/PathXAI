# Third-party materials

This package contains materials from multiple sources. A model-card license does not automatically license every file in the GitHub code package.

| Component | Source and notice |
| --- | --- |
| SAM3 code and separate SAM3 base weights | The SAM License is retained for SAM3 materials to which it applies, at `vendor/report_engine/assets/licenses/SAM-LICENSE.txt` and in the D1 model bundle. Original copyright notices are preserved. Users obtain base weights directly from `facebook/sam3`. |
| MedSAM3 adaptation source | The source subtree is omitted from the prepared source archive and fetched at setup from the pinned `Joey-S-Liu/MedSAM3` revision with two tutorial patches. Upstream copyright, license, and other notices apply to the upstream files. This package does not grant additional rights to that source or treat a model-card license as a blanket grant for code. |
| CLIP tokenizer vocabulary | OpenAI CLIP, MIT; original license and exact source identity are in `vendor/report_engine/assets/licenses/CLIP-LICENSE.txt` and `CLIP-ORIGIN.json`. |
| OpenSeadragon viewer | Bundled with the local scoring interfaces; its license is retained beside each viewer under `static/vendor/openseadragon/LICENSE.txt`. |
| CONCH | Installed from a pinned original source commit. Model access is separately authorized through `MahmoodLab/conch`; no CONCH weights are redistributed here. |
| KEEP resources | Downloaded from the original gated `Loie/KEEP_dataset` repository for PathXAI. No KEEP dataset files are redistributed here. |
| TCGA sample WSI | Obtained by the reader directly from NCI GDC using the recorded open-access file UUID. The image is not in the code package or D1 model bundle. |

PathXAI's original code and adaptations remain subject to their owners' rights. This preparation does not assign a new blanket license to all code. The D1 model bundle includes the SAM License; it does not implement an additional academic-only/noncommercial restriction. A proposed restrictive-license draft is kept outside the upload bundle and is not effective licensing text.

Official sources: [SAM3 license](https://huggingface.co/facebook/sam3/blob/main/LICENSE), [MedSAM3 source](https://github.com/Joey-S-Liu/MedSAM3), [CONCH](https://huggingface.co/MahmoodLab/conch), [KEEP](https://huggingface.co/datasets/Loie/KEEP_dataset), [GDC downloads](https://docs.gdc.cancer.gov/API/Users_Guide/Downloading_Files/).
