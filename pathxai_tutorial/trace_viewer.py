"""Build one self-contained HTML page showing every LLM request in a run.

Reads ``cases/<case>/llm_trace.jsonl`` and shows, per request, the input images,
the prompt, any reasoning the model returned and its output.

The trace stores images only as hashes, so they are matched back to files by
SHA-256 and re-encoded with the recorded settings. Each image is then checked
against the recorded payload hash, and the page says whether it is verifiably
the image that was sent.

Usage: python -m pathxai_tutorial.trace_viewer outputs/<run_dir>
"""
import argparse
import base64
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re

from .paths import ROOT

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
_THINK = re.compile(r"<think>(.*?)</think>", re.S | re.I)
_TOKENS = re.compile(r"\b(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,})\b")

# First prompt line -> stage name. Anything else is shown under its own first line.
_STAGES = (
    ("You are selecting WSI ROI", "Region selection"),
    ("Classify the histogenetic BROAD LINEAGE", "Broad lineage"),
    ("Judge whether current ROI evidence is sufficient", "Subtype sufficiency"),
    ("You are a pathologist making a diagnosis", "Diagnosis"),
    ("Assess whether the current ROI evidence", "Evidence sufficiency"),
    ("Verify pathology claims", "Claim verification"),
    ("Generate a structured pathology report", "Report writing"),
    ("Rewrite this structured pathology report", "Report rewrite"),
)

_STYLE = """
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 system-ui,sans-serif;color:#1f2933;background:#f5f7fa}
nav{position:fixed;inset:0 auto 0 0;width:270px;overflow:auto;background:#fff;border-right:1px solid #d9e2ec;padding:12px}
nav a{display:block;padding:4px 8px;border-radius:4px;color:inherit;text-decoration:none;font-size:13px}
nav a:hover{background:#eef2f7}
nav small{color:#627d98}
main{margin-left:270px;padding:24px;max-width:1150px}
h1{font-size:22px;margin:0 0 8px}
h2{font-size:17px;margin:0}
section,.summary{background:#fff;border:1px solid #d9e2ec;border-radius:8px;padding:16px;margin-bottom:18px}
.meta{color:#627d98;font-size:13px;margin:4px 0 10px}
.note{color:#52606d;font-size:13px}
.notice{background:#fff8e1;border:1px solid #f0d58c;border-radius:6px;padding:8px 12px;font-size:13px;margin:10px 0 0}
details{border:1px solid #e4e7eb;border-radius:6px;margin-top:10px}
summary{cursor:pointer;font-weight:600;padding:8px 12px;background:#f8fafc;border-radius:6px}
pre{margin:0;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.45 ui-monospace,monospace;max-height:560px;overflow:auto}
.input summary{border-left:4px solid #3e7cb1}
.reasoning summary{border-left:4px solid #b7791f;background:#fff8e1}
.none summary{border-left:4px solid #cbd2d9;font-weight:400;color:#52606d}
.output summary{border-left:4px solid #2f855a}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px;padding:12px}
figure{margin:0;font-size:12px;color:#52606d;overflow-wrap:anywhere}
figure img{width:100%;display:block;border:1px solid #d9e2ec;border-radius:4px;cursor:zoom-in;background:#eef2f7;min-height:90px}
.missing{padding:24px 8px;text-align:center;border:1px dashed #b42318;color:#b42318;border-radius:4px}
.warn{color:#b42318}
button{font:inherit;padding:4px 10px;margin:8px 6px 0 0;cursor:pointer}
#zoom{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;cursor:zoom-out}
#zoom img{max-width:96vw;max-height:96vh}
:root{color-scheme:light;--series-text:#2a78d6;--series-images:#eb6834;--series-output:#1baf7a}
.legend{display:flex;gap:16px;font-size:13px;color:#52606d;margin:6px 0 10px;flex-wrap:wrap}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.ctx-row{display:grid;grid-template-columns:300px 1fr 66px;gap:10px;align-items:center;font-size:13px;padding:2px 0;color:inherit;text-decoration:none}
a.ctx-row:hover{background:#eef2f7}
.ctx-row .value{text-align:right;font-variant-numeric:tabular-nums}
.ctx-bar{display:flex;gap:2px;height:14px}
.ctx-bar span{display:block;min-width:2px;height:100%}
.ctx-bar span:first-child{border-radius:4px 0 0 4px}
.ctx-bar span:last-child{border-radius:0 4px 4px 0}
.ctx-bar span:only-child{border-radius:4px}
.seg-text{background:var(--series-text)}.seg-images{background:var(--series-images)}.seg-output{background:var(--series-output)}
.ctx-section{margin-top:10px;font-size:13px}
.ctx-section .ctx-bar{margin:4px 0}
table{border-collapse:collapse;font-size:13px;margin:12px;font-variant-numeric:tabular-nums}
th,td{padding:3px 10px;border-bottom:1px solid #e4e7eb;text-align:right}
th:nth-child(2),td:nth-child(2){text-align:left}
#tip{position:fixed;display:none;background:#fff;border:1px solid #cbd2d9;border-radius:6px;padding:6px 10px;font-size:13px;box-shadow:0 2px 8px rgba(0,0,0,.15);pointer-events:none;z-index:5}
#tip div{white-space:nowrap}
"""

_SCRIPT = """
const images = JSON.parse(document.getElementById('image-data').textContent);
document.querySelectorAll('img[data-image]').forEach(el => {
  el.src = 'data:' + images[el.dataset.image];
  el.onclick = () => {
    const zoom = document.getElementById('zoom');
    zoom.firstElementChild.src = el.src;
    zoom.style.display = 'flex';
  };
});
document.getElementById('zoom').onclick = e => { e.currentTarget.style.display = 'none'; };
const tip = document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el => {
  const show = e => {
    tip.replaceChildren(...JSON.parse(el.dataset.tip).map(line => {
      const row = document.createElement('div');
      row.textContent = line;
      return row;
    }));
    tip.style.display = 'block';
    const box = el.getBoundingClientRect();
    const x = e.clientX ?? box.left;
    const y = e.clientY ?? box.top;
    tip.style.left = Math.min(x + 12, innerWidth - tip.offsetWidth - 8) + 'px';
    tip.style.top = Math.min(y + 12, innerHeight - tip.offsetHeight - 8) + 'px';
  };
  const hide = () => { tip.style.display = 'none'; };
  el.addEventListener('pointermove', show);
  el.addEventListener('focus', show);
  el.addEventListener('pointerleave', hide);
  el.addEventListener('blur', hide);
});
function setAll(open) { document.querySelectorAll('details').forEach(d => { d.open = open; }); }
"""


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _index_images(wanted, roots):
    """Map each wanted source SHA-256 to a file, searching ``roots`` in order."""
    found = {}
    for root in roots:
        for folder, _, names in os.walk(root):
            for name in names:
                if len(found) == len(wanted):
                    return found
                path = Path(folder) / name
                if path.suffix.lower() in _IMAGE_SUFFIXES:
                    digest = _sha256(path.read_bytes())
                    if digest in wanted:
                        found.setdefault(digest, path)
    return found


def _as_sent(path, encoding):
    """Return image bytes encoded the way the LLM client encoded them."""
    max_edge = encoding.get("max_edge")
    if encoding.get("function") == "identity" or not max_edge:
        return path.read_bytes(), False
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(1.0, float(max_edge) / float(max(width, height)))
        if scale < 1.0:
            size = (max(1, int(width * scale)), max(1, int(height * scale)))
            image = image.resize(size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=encoding.get("jpeg_quality", 85), optimize=True)
    return buffer.getvalue(), True


def _mime(data):
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return "image/jpeg" if data[:2] == b"\xff\xd8" else "application/octet-stream"


def _reasoning(response):
    """Return every piece of reasoning the model returned, with where it was found."""
    found = []
    def add(source, value):
        if isinstance(value, str) and value.strip():
            found.append((source, value.strip()))
    add("response.reasoning", response.get("reasoning"))
    choices = (response.get("raw") or {}).get("choices") or [{}]
    message = choices[0].get("message") or {}
    add("message.reasoning_content", message.get("reasoning_content"))
    add("message.reasoning", message.get("reasoning"))
    add("message.thinking", message.get("thinking"))
    add("provider_specific_fields.reasoning", (message.get("provider_specific_fields") or {}).get("reasoning"))
    for block in _THINK.findall(str(response.get("content") or "")):
        add("<think> block in the output", block)
    unique, seen = [], set()
    for source, text in found:
        if text not in seen:
            seen.add(text)
            unique.append((source, text))
    return unique


def _pretty_output(response):
    content = _THINK.sub("", str(response.get("content") or "")).strip()
    try:
        return json.dumps(json.loads(content), indent=2, ensure_ascii=False)
    except ValueError:
        return content


def _context(row):
    """Return tokens held in context for one request, or None if not reported."""
    response = row["response"]
    usage = (response.get("raw") or {}).get("usage") or {}
    prompt, output = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if prompt is None or output is None:
        return None
    details = usage.get("prompt_tokens_details") or {}
    multimodal = details.get("multimodal_tokens")
    if isinstance(multimodal, dict):
        images = multimodal.get("image") or 0
    elif not row["request"].get("images"):
        images = 0  # nothing to split when the request had no images
    else:
        images = None  # some servers (e.g. llama.cpp) do not report image tokens separately
    return {"text": prompt - images if images is not None else prompt, "images": images or 0,
            "split": images is not None, "output": output,
            "cached": details.get("cached_tokens") or 0, "total": prompt + output}


def _context_bar(context, scale):
    """Stacked bar on a run-wide scale, so bars are comparable across requests."""
    tip = html.escape(json.dumps([
        f"{context['total']:,} tokens in context",
        f"{context['text']:,} prompt text" if context["split"] else f"{context['text']:,} prompt (text + images)",
        f"{context['images']:,} images" if context["split"] else "image tokens not reported separately",
        f"{context['output']:,} output"]), quote=True)
    parts = "".join(
        f'<span class="seg-{key}" style="flex:0 0 {100 * context[key] / scale:.3f}%"></span>'
        for key in ("text", "images", "output") if context[key])
    return f'<div class="ctx-bar" tabindex="0" data-tip="{tip}" style="padding-right:4px">{parts}</div>'


_LEGEND = ('<div class="legend"><span><i class="seg-text"></i>Prompt text</span>'
           '<span><i class="seg-images"></i>Images</span><span><i class="seg-output"></i>Output</span></div>')
_LEGEND_UNSPLIT = ('<div class="legend"><span><i class="seg-text"></i>Prompt (text + images)</span>'
                   '<span><i class="seg-output"></i>Output</span></div>')


def _stages(rows, has_final=False):
    """Label each request with its pipeline pass and stage.

    The orchestrator runs an initial pass, zero or more reinspection rounds and,
    when ``has_final`` (its final-pass checkpoints exist), a final pass. Each pass
    starts with the broad-lineage request, and its two diagnosis requests are the
    segmentation branch then the image branch, in the orchestrator's call order.
    """
    names = []
    for row in rows:
        prompt = row["request"].get("prompt") or ""
        lines = [line for line in prompt.splitlines() if line.strip() and line.strip() != "/no_think"]
        first = lines[0].strip() if lines else ""
        names.append(next((label for prefix, label in _STAGES if first.startswith(prefix)), first[:60] or "Request"))
    starts = [i for i, name in enumerate(names) if name == "Broad lineage"]
    labels, current, in_pair = [], None, False
    for i, (row, name) in enumerate(zip(rows, names)):
        if i in starts:
            number = starts.index(i)
            if number == 0:
                current = "Initial pass"
            elif has_final and number == len(starts) - 1:
                current = "Final pass"
            else:
                current = f"Reinspection round {number}"
            current += f" ({len(row['request'].get('images') or [])} regions)"
        if name == "Diagnosis":
            name += ": image branch" if in_pair else ": segmentation branch"
            in_pair = not in_pair
        else:
            in_pair = False
        if current and name not in ("Claim verification", "Report writing", "Report rewrite"):
            name = f"{current} · {name}"
        labels.append(name)
    return labels


def build_trace_html(run_dir, output=None, workspace=None, secrets=(), context_window=None):
    """Write the viewer for ``run_dir`` and return its path.

    ``context_window`` is the model's context limit in tokens. It is not in the
    trace, so usage is shown as a share of the limit only when it is supplied.
    """
    if context_window is None and os.environ.get("PATHXAI_CONTEXT_WINDOW", "").isdigit():
        context_window = int(os.environ["PATHXAI_CONTEXT_WINDOW"]) or None
    run = Path(run_dir).expanduser().resolve()
    traces = sorted(run.glob("cases/*/llm_trace.jsonl"))
    if not traces:
        raise FileNotFoundError(f"No llm_trace.jsonl under {run}")
    rows = [json.loads(line) for line in traces[0].read_text().splitlines() if line.strip()]
    try:
        info = json.loads((run / "tutorial_run.json").read_text())
    except (OSError, ValueError):
        info = {}

    secrets = [str(item) for item in secrets if item]
    def clean(value):
        text = str(value)
        for secret in secrets:
            text = text.replace(secret, "[redacted]")
        return _TOKENS.sub("[redacted]", text)
    def esc(value):
        return html.escape(clean(value))

    wanted = {im["source_sha256"] for row in rows for im in row["request"].get("images") or []}
    roots = [run, Path(workspace).expanduser() if workspace else ROOT / "workspace"]
    sources = _index_images(wanted, [root for root in roots if root.is_dir()])

    image_data, status = {}, {}
    for row in rows:
        for im in row["request"].get("images") or []:
            key = im.get("payload_sha256") or im["source_sha256"]
            if key in status:
                continue
            path = sources.get(im["source_sha256"])
            if path is None:
                status[key] = "missing"
                continue
            data, _ = _as_sent(path, im.get("encoding") or {})
            status[key] = "exact" if _sha256(data) == im.get("payload_sha256") else "unverified"
            image_data[key] = _mime(data) + ";base64," + base64.b64encode(data).decode()

    labels = _stages(rows, has_final=(traces[0].parent / "ckpt_08a_final_broad_hypotheses.json").exists())
    contexts = [_context(row) for row in rows]
    known = [item for item in contexts if item]
    # Reserve room so the bar and its 2px gaps never overflow the row.
    scale = max((item["total"] for item in known), default=1) * 1.02
    def share(tokens):
        return f" ({100 * tokens / context_window:.1f}% of {context_window:,})" if context_window else ""
    nav, sections, context_rows, table_rows = [], [], [], []
    totals = {"prompt": 0, "completion": 0, "reasoning": 0, "seconds": 0.0, "with_reasoning": 0}
    for row, label, context in zip(rows, labels, contexts):
        request, response = row["request"], row["response"]
        number = int(row["sequence"]) + 1
        raw = response.get("raw") or {}
        usage = raw.get("usage") or {}
        choice = (raw.get("choices") or [{}])[0]
        reasoning = _reasoning(response)
        reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        totals["prompt"] += usage.get("prompt_tokens") or 0
        totals["completion"] += usage.get("completion_tokens") or 0
        totals["reasoning"] += reasoning_tokens or 0
        totals["seconds"] += response.get("elapsed_sec") or 0
        totals["with_reasoning"] += bool(reasoning)

        figures = []
        for position, im in enumerate(request.get("images") or [], 1):
            key = im.get("payload_sha256") or im["source_sha256"]
            path = sources.get(im["source_sha256"])
            if status.get(key) == "missing":
                body = '<div class="missing">Image file not found</div>'
                where = f"source sha256 {im['source_sha256'][:16]}…"
            else:
                body = f'<img data-image="{esc(key)}" alt="Input image {position}">'
                try:
                    where = str(path.relative_to(run))
                except ValueError:
                    where = path.name
                if status[key] != "exact":
                    where += ' <span class="warn">(does not match the recorded hash of what was sent)</span>'
                    figures.append(f"<figure>{body}<figcaption>Image {position}: {where}</figcaption></figure>")
                    continue
                where = html.escape(where)
            figures.append(f"<figure>{body}<figcaption>Image {position}: {where}</figcaption></figure>")
        images_html = (f'<details class="input" open><summary>Input images ({len(figures)})</summary>'
                       f'<div class="grid">{"".join(figures)}</div></details>'
                       if figures else '<details class="none"><summary>Input images: none</summary></details>')

        extra_input = ""
        if request.get("system_prompt"):
            extra_input += ('<details class="input" open><summary>Input system prompt</summary>'
                            f'<pre>{esc(request["system_prompt"])}</pre></details>')
        if request.get("messages"):
            extra_input += ('<details class="input" open><summary>Input messages</summary><pre>'
                            f'{esc(json.dumps(request["messages"], indent=2, ensure_ascii=False))}</pre></details>')

        if reasoning:
            reasoning_html = "".join(
                f'<details class="reasoning" open><summary>REASONING (from {esc(source)})</summary>'
                f"<pre>{esc(text)}</pre></details>" for source, text in reasoning)
        else:
            detail = "" if reasoning_tokens is None else f" (reasoning tokens reported: {reasoning_tokens})"
            reasoning_html = f'<details class="none"><summary>Reasoning: none returned{detail}</summary></details>'

        meta = [f"model {request.get('generation_params', {}).get('model', '?')}",
                f"temperature {request.get('generation_params', {}).get('temperature', '?')}",
                f"thinking {request.get('generation_params', {}).get('enable_thinking', '?')}",
                f"{response.get('elapsed_sec', '?')} s",
                f"{usage.get('prompt_tokens', '?')} prompt + {usage.get('completion_tokens', '?')} output tokens",
                f"finish: {choice.get('finish_reason', '?')}"]
        if context:
            params = request.get("generation_params", {})
            context_html = (
                f'<div class="ctx-section"><b>Context used: {context["total"]:,} tokens{share(context["total"])}</b>'
                f'{_context_bar(context, scale)}'
                + (f'{context["text"]:,} prompt text · {context["images"]:,} images · ' if context["split"] else
                   f'{context["text"]:,} prompt (text + images; the server does not report image tokens separately) · ')
                + f'{context["output"]:,} output (output cap {params.get("max_tokens", "?")}) · '
                f'{context["cached"]:,} prompt tokens served from cache</div>')
            context_rows.append(
                f'<a class="ctx-row" href="#r{number}"><span>{number}. {esc(label)}</span>'
                f'{_context_bar(context, scale)}<span class="value">{context["total"]:,}</span></a>')
            images_cell = f'{context["images"]:,}' if context["split"] else "not reported"
            table_rows.append(
                f'<tr><td>{number}</td><td>{esc(label)}</td><td>{context["text"]:,}</td>'
                f'<td>{images_cell}</td>'
                f'<td>{context["output"]:,}</td><td>{context["total"]:,}</td><td>{context["cached"]:,}</td>'
                f'<td>{esc(choice.get("finish_reason", "?"))}</td></tr>')
        else:
            context_html = '<div class="ctx-section">Context used: not reported for this request</div>'
        nav.append(f'<a href="#r{number}">{number}. {esc(label)} <small>'
                   f'{len(figures)} img{" · reasoning" if reasoning else ""}</small></a>')
        sections.append(
            f'<section id="r{number}"><h2>Request {number} of {len(rows)}: {esc(label)}</h2>'
            f'<div class="meta">{esc(" · ".join(str(item) for item in meta))}</div>{context_html}{images_html}{extra_input}'
            f'<details class="input" open><summary>Input prompt</summary><pre>{esc(request.get("prompt") or "")}</pre></details>'
            f'{reasoning_html}<details class="output" open><summary>Model output</summary>'
            f"<pre>{esc(_pretty_output(response))}</pre></details></section>")

    counts = {name: sum(1 for value in status.values() if value == name) for name in ("exact", "unverified", "missing")}
    notices = []
    if counts["missing"] or counts["unverified"]:
        notices.append(f'{counts["missing"]} image(s) could not be found and {counts["unverified"]} could not be '
                       "confirmed as the exact image sent.")
    notices.append("Reasoning means text the model returned as reasoning. &quot;reason&quot; fields inside "
                   "a JSON output are part of the answer, not its reasoning, so they stay under Model output.")
    notices.append("Stage, pass and branch names are not recorded in the trace. They are inferred from each prompt, "
                   "the call order and the presence of the pipeline's final-pass checkpoint. Evidence sufficiency is "
                   "the step that decides whether to reinspect.")
    case = info.get("case") or {}
    peak = max(known, key=lambda item: item["total"], default=None)
    peak_number = contexts.index(peak) + 1 if peak else None
    if context_window:
        limit_line = f"The model's context limit is shown as {context_window:,} tokens, as supplied when this page was built."
    else:
        limit_line = ("The model's context limit is not recorded in the trace, so no percentage is shown. "
                      "Set PATHXAI_CONTEXT_WINDOW to add one.")
    context_html = "" if not known else (
        f'<div class="summary"><h2>Context used per request</h2><div class="meta">Tokens as reported by the endpoint. '
        f'Largest: request {peak_number}, {peak["total"]:,} tokens{share(peak["total"])}.</div>'
        + (_LEGEND if all(item["split"] for item in known) else _LEGEND_UNSPLIT)
        + f'{"".join(context_rows)}'
        '<div class="notice">Each request is sent separately, so these numbers do not add up to a running conversation. '
        "Earlier results reach a later request only if the pipeline writes them into its prompt. "
        + ("Prompt text is prompt tokens minus image tokens, so it includes the chat template. "
           if all(item["split"] for item in known) else
           "This server does not report image tokens separately, so the blue bar is text and images together. ")
        + f"{html.escape(limit_line)}</div>"
        '<details><summary>Table view</summary><table><tr><th>#</th><th>Stage</th><th>Prompt text</th><th>Images</th>'
        f'<th>Output</th><th>Total</th><th>Cached</th><th>Finish</th></tr>{"".join(table_rows)}</table></details></div>')
    summary = [("Case", case.get("case_id") or traces[0].parent.name), ("Version", info.get("version", "?")),
               ("Known site given", case.get("known_site") or "(none)"),
               ("Model", rows[0]["request"].get("generation_params", {}).get("model", "?") if rows else "?"),
               ("Requests", len(rows)),
               ("Distinct images", f'{len(status)} ({counts["exact"]} verified as the exact image sent)'),
               ("Requests with reasoning", f'{totals["with_reasoning"]} of {len(rows)}'),
               ("Largest context", f'{peak["total"]:,} tokens (request {peak_number}){share(peak["total"])}' if peak else "not reported"),
               ("Tokens across all requests", f'{totals["prompt"]:,} prompt, {totals["completion"]:,} output, {totals["reasoning"]:,} reasoning'),
               ("Time in LLM calls", f'{totals["seconds"]:.0f} s')]
    summary_html = "<br>".join(f"<b>{esc(key)}:</b> {esc(value)}" for key, value in summary)
    payload = json.dumps(image_data).replace("</", "<\\/")
    title = f"LLM trace: {case.get('case_id') or traces[0].parent.name} {info.get('version', '')}".strip()
    page = (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>{esc(title)}</title>'
        f"<style>{_STYLE}</style></head><body><nav><b>Requests</b>{''.join(nav)}</nav><main>"
        f'<div class="summary"><h1>{esc(title)}</h1>{summary_html}'
        f'{"".join(f"<div class=notice>{item}</div>" for item in notices)}'
        '<button onclick="setAll(true)">Expand all</button><button onclick="setAll(false)">Collapse all</button></div>'
        f'{context_html}{"".join(sections)}</main><div id="zoom"><img alt=""></div><div id="tip"></div>'
        f'<script id="image-data" type="application/json">{payload}</script><script>{_SCRIPT}</script></body></html>')
    target = Path(output).expanduser() if output else run / "llm_trace.html"
    target.write_text(page, encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir")
    parser.add_argument("--output")
    parser.add_argument("--workspace")
    parser.add_argument("--context-window", type=int, help="model context limit in tokens, if known")
    args = parser.parse_args()
    print(build_trace_html(args.run_dir, output=args.output, workspace=args.workspace,
                           context_window=args.context_window))


if __name__ == "__main__":
    main()
