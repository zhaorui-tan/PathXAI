#!/usr/bin/env python3
"""Publish copies of verified successful runs without rerunning inference.

Keep all image bytes unchanged, deduplicate identical assets across the reports,
and preserve real notebook outputs with documented presentation-only cleanup.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("so1_run", "so1_notebook", "so4_run", "so4_notebook"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "examples")
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose an empty output directory; published examples are not overwritten")
    (out / "assets").mkdir(parents=True)
    (out / "notebooks").mkdir()
    (out / ".nojekyll").write_text("")
    entries = []
    from lxml import html
    import nbformat
    for version, mode in (("SO1", "core"), ("SO4", "full")):
        run = getattr(args, version.lower() + "_run").resolve()
        notebook = getattr(args, version.lower() + "_notebook").resolve()
        outcome = json.loads((run / "outcome.json").read_text())
        if outcome.get("status") != "succeeded" or outcome.get("mode") != mode:
            raise ValueError(f"{version} is not a successful matching run")
        export = json.loads((run / "export_audit.json").read_text())[0]
        if not export.get("clinical_and_evidence_text_unchanged"):
            raise ValueError("The source run lacks a successful export audit")
        case = outcome["case_id"]
        run_record = json.loads((run / "tutorial_run.json").read_text())
        case_input = run_record["case"]
        slide_names = [Path(name).name for name in case_input["slide_paths"]]
        known_site = case_input.get("known_site", "")
        run_date = run_record["started_utc"][:10]
        source = run / "portable/report"
        page = source / "report.html"
        if sha(page) != export["report_sha256"]:
            raise ValueError("The source HTML changed after its export audit")
        dest = out / version
        dest.mkdir()
        for asset in (source / "assets").iterdir():
            if not asset.is_file() or asset.is_symlink():
                raise ValueError("Unexpected source asset type")
            target = out / "assets" / asset.name
            if target.exists():
                if sha(target) != sha(asset):
                    raise ValueError("Asset filename collision with unequal bytes")
            else:
                shutil.copyfile(asset, target)
        original_html = page.read_text()
        published_html = original_html.replace("assets/", "../assets/")
        def visible_text(markup):
            document = html.fromstring(markup)
            # Embedded scripts contain the asset URLs being relocated; they
            # are not rendered report text.
            for node in document.xpath("//script|//style"):
                node.drop_tree()
            return document.text_content()
        if visible_text(original_html) != visible_text(published_html):
            raise ValueError("Published HTML must preserve all rendered text")
        (dest / "report.html").write_text(published_html)
        report_text = run / "cases" / case / "generated_long_report.txt"
        shutil.copyfile(report_text, dest / "generated_long_report.txt")
        nb = json.loads(notebook.read_text())
        if any(o.get("output_type") == "error" for c in nb["cells"] for o in c.get("outputs", [])):
            raise ValueError("Cannot publish a failed notebook as a successful example")
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        if not code_cells or any(c.get("execution_count") is None for c in code_cells):
            raise ValueError("Every code cell in the source notebook must have executed")
        original_counts = [c["execution_count"] for c in code_cells]
        root_prefix = str(ROOT.resolve()) + "/"
        replacements = [
            (str(report_text), f"../{version}/generated_long_report.txt"),
            (str(source), f"../{version}"),
            (str(source).removeprefix(root_prefix), f"../{version}"),
            (root_prefix, "<tutorial>/"),
            (str(run) + "/", "<run>/"),
            (str(ROOT.parent.resolve()) + "/", "<local-workspace>/"),
            ("/local/conda_staffs/conda_envs/", "<conda-environment>/"),
            ("/local/datasets/public/Wsi-Caption-master/datasets/", "<local-slide-data>/"),
            ("/home/tanzr3/.cache/", "<local-model-cache>/"),
            ("/local/PathXAI_Agent_v2/PathXAI_tutorial/workspace/", "<local-environment>/"),
            ("E:/tanja1/qwen3.6/", ""),
        ]
        removed_heartbeats = 0
        removed_progress_updates = 0
        last_feature_progress = None
        def clean(text):
            nonlocal removed_heartbeats, removed_progress_updates, last_feature_progress
            text = ANSI.sub("", text)
            for before, after in replacements:
                text = text.replace(before, after)
            kept = []
            for line in text.splitlines(keepends=True):
                if line.strip() == "[running] Original engine is running":
                    removed_heartbeats += 1
                elif "Extracting features" in line and line.lstrip().startswith("[engine_log]"):
                    # Rich refreshes the same progress bar many times. Retain
                    # the actual completion line, not every intermediate tick.
                    if "100%" not in line or line.strip() == last_feature_progress:
                        removed_progress_updates += 1
                    else:
                        kept.append(line)
                        last_feature_progress = line.strip()
                else:
                    kept.append(line)
                    if line.strip():
                        last_feature_progress = None
            return "".join(kept)
        for cell in code_cells:
            outputs = []
            for output in cell.get("outputs", []):
                value = copy.deepcopy(output)
                if "text" in value:
                    value["text"] = clean("".join(value["text"]))
                    if not value["text"]:
                        continue
                for mime in ("text/plain", "text/html"):
                    if mime in value.get("data", {}):
                        value["data"][mime] = clean("".join(value["data"][mime]))
                if (outputs and value.get("output_type") == "stream"
                        and outputs[-1].get("output_type") == "stream"
                        and value.get("name") == outputs[-1].get("name")):
                    outputs[-1]["text"] += value["text"]
                else:
                    outputs.append(value)
            cell["outputs"] = outputs
        header = f"""# Completed {version} example

Recorded notebook execution for **{case}**, dated {run_date}.
Version: **{version}**; refined D1 segmentation with separately obtained SAM3 base weights;
vision model: **Qwen3.6-27B-Q6_K**. The run used {len(slide_names)} checksum-verified source slides
and the pinned candidate ROIs. Anatomical site: **{known_site or 'not supplied'}**.

Code cells, execution counts and outputs are retained. Machine-specific output paths are
normalized, and repeated progress lines are condensed for display.

[View HTML report](https://zhaorui-tan.github.io/PathXAI/{version}/report.html) · [Report text](../{version}/generated_long_report.txt) ·
[Run your own case](../../PathXAI_report.ipynb)
"""
        nb["cells"].insert(0, {"cell_type":"markdown", "id":f"record-{version.lower()}",
                               "metadata":{}, "source":header})
        nb["cells"].append({"cell_type":"markdown", "id":f"saved-report-{version.lower()}",
                            "metadata":{}, "source":"## Saved generated report\n\nThe following text was copied from this run's `generated_long_report.txt`.\n\n```text\n" + report_text.read_text().rstrip() + "\n```\n"})
        nb["metadata"]["pathxai_example"] = {
            "version":version, "case_id":case, "source_executed_notebook_sha256":sha(notebook),
            "report_text_sha256":sha(report_text), "removed_heartbeat_lines":removed_heartbeats,
            "removed_progress_updates":removed_progress_updates,
            "status":"executed_successfully", "source_execution_counts":original_counts,
            "slide_names":slide_names, "known_site":known_site,
        }
        notebook_dest = out / "notebooks" / f"{version}_completed.ipynb"
        nbformat.write(nbformat.from_dict(nb), notebook_dest)
        nbformat.validate(nbformat.read(notebook_dest, as_version=4))
        entries.append({"version":version, "case_id":case,
                        "slide_names":slide_names, "known_site":known_site,
                        "source_executed_notebook_sha256":sha(notebook),
                        "source_report_html_sha256":sha(page),
                        "published_report_html_sha256":sha(dest / "report.html"),
                        "published_notebook_sha256":sha(notebook_dest),
                        "report_text_sha256":sha(report_text),
                        "removed_heartbeat_lines":removed_heartbeats,
                        "removed_progress_updates":removed_progress_updates,
                        "html_change":"Only assets/ references changed to ../assets/ for deduplication; all text unchanged.",
                        "image_bytes_changed":False})
    files = [{"path":p.relative_to(out).as_posix(), "bytes":p.stat().st_size, "sha256":sha(p)}
             for p in sorted(out.rglob("*")) if p.is_file()]
    manifest = {"schema":"pathxai-successful-examples-v1", "runs":entries, "files":files,
                "scope":"Public TCGA example outputs only; no original WSI, model weights or credentials."}
    (out / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Published {len(entries)} successful notebook records; {len(files)} files; {sum(f['bytes'] for f in files)/1e6:.1f} MB")

if __name__ == "__main__":
    main()
