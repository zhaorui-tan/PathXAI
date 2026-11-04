#!/usr/bin/env python3
"""Check published run records, unchanged assets and static report links."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]

class Resources(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if value and (key in {"src", "poster"} or (tag in {"a", "link"} and key == "href")):
                self.links.append(value)

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024**2), b""):
            digest.update(block)
    return digest.hexdigest()

def audit(root=ROOT / "examples"):
    manifest = json.loads((root / "manifest.json").read_text())
    errors = []
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.is_file() or sha(path) != row["sha256"]:
            errors.append("Example file mismatch: " + row["path"])
    for row in manifest["runs"]:
        version = row["version"]
        path = root / "notebooks" / f"{version}_completed.ipynb"
        nb = json.loads(path.read_text())
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        counts = [c.get("execution_count") for c in code]
        if counts != nb["metadata"]["pathxai_example"]["source_execution_counts"] or any(n is None for n in counts):
            errors.append(version + " execution counts were not preserved")
        if not any(c.get("outputs") for c in code):
            errors.append(version + " has no saved outputs")
        if any(o.get("output_type") == "error" for c in code for o in c.get("outputs", [])):
            errors.append(version + " contains execution errors")
        raw = path.read_text()
        if any(s in raw for s in ("/local/PathXAI_Agent_v2", "/home/tanzr3", "workspace/live_notebook_validation")):
            errors.append(version + " still contains machine-specific output paths")
        report = root / version / "report.html"
        if sha(report) != row["published_report_html_sha256"]:
            errors.append(version + " report checksum mismatch")
    for page in (root / "index.html", root / "comparison.html", root / "SO1/report.html", root / "SO4/report.html"):
        parser = Resources(); parser.feed(page.read_text())
        for link in parser.links:
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path:
                continue
            target = (page.parent / unquote(url.path)).resolve()
            if not target.is_relative_to(root.resolve()) or not target.is_file():
                errors.append(f"Broken static link in {page.relative_to(root)}: {link}")
    return errors

if __name__ == "__main__":
    errors = audit()
    if errors:
        print("\n".join(errors), file=sys.stderr); raise SystemExit(1)
    print("Verified two recorded notebooks, report checksums, original image bytes and static links")
