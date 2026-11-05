"""Run the packaged deterministic report-only evaluator on one report/reference pair."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__:
    from .vendor import eval_v4
    from .manuscript_metrics import manuscript_metrics
else:  # Allow the documented ``python PathXAI_tutorial/evaluation/run_evaluation.py`` form.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vendor import eval_v4
    from manuscript_metrics import manuscript_metrics


class _VisibleText(HTMLParser):
    """Extract readable text from report HTML without an extra dependency."""

    SKIP_TAGS = {"script", "style", "head", "noscript", "svg"}
    BREAK_TAGS = {"br", "p", "div", "section", "article", "header", "footer", "li", "tr", "h1", "h2", "h3", "h4"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        elif self._skip_depth == 0 and tag in self.BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif self._skip_depth == 0 and tag in self.BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)


class _PathXAIReport(HTMLParser):
    """Read final report rows only, excluding evidence, references and UI text."""
    VOID_TAGS = {"br", "img", "hr", "input", "meta", "link", "source", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.capture = None
        self.parts = []

    def handle_starttag(self, tag, attrs):
        classes = dict(attrs).get("class", "").split()
        if self.capture is None and any(c in classes for c in ("report-row-title", "report-row-content")):
            self.capture = self.depth
            self.parts.append("\n")
        if tag == "br" and self.capture is not None:
            self.parts.append("\n")
        if tag not in self.VOID_TAGS:
            self.depth += 1

    def handle_endtag(self, tag):
        if tag not in self.VOID_TAGS:
            self.depth -= 1
        if self.capture is not None and self.depth == self.capture:
            self.parts.append("\n")
            self.capture = None

    def handle_data(self, data):
        if self.capture is not None:
            self.parts.append(data)


def visible_report_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() not in {".html", ".htm"}:
        return raw.strip()
    parser = _PathXAIReport() if "report-row-content" in raw else _VisibleText()
    parser.feed(raw)
    return "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_reference(path: Path, fmt: str) -> tuple[str, dict[str, Any] | None]:
    if fmt == "text":
        return path.read_text(encoding="utf-8", errors="replace").strip(), None
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
        raise ValueError("JSON reference must be an object with a string field named 'text'")
    structured = obj.get("structured")
    if structured is not None and not isinstance(structured, dict):
        raise ValueError("JSON reference 'structured' must be an object")
    return obj["text"].strip(), structured


def evaluate_paths(report_path: Path, reference_path: Path, *, cancer_type: str | None = None,
                   reference_format: str = "text", diagnosis_only: bool = False,
                   site: str | None = None) -> dict[str, Any]:
    report = visible_report_text(report_path)
    ref_text, structured = _read_reference(reference_path, reference_format)
    if not report or not ref_text:
        raise ValueError("Both report and independent reference text must be non-empty")
    reference = eval_v4.reference(ref_text, structured=structured, cancer_type=cancer_type, site=site)
    result = eval_v4.evaluate(
        report,
        reference,
        reference_text=ref_text,
        diagnosis_only=diagnosis_only,
        narrative_reference=not diagnosis_only,
    )
    result["manuscript_metrics"] = manuscript_metrics(result)
    result["run_metadata"] = {
        "evaluator": eval_v4.VERSION,
        "terminology": result.get("terminology_version"),
        "report_path": str(report_path.resolve()),
        "reference_path": str(reference_path.resolve()),
        "report_sha256": _sha256(report_path),
        "reference_sha256": _sha256(reference_path),
        "evaluator_source_sha256": {
            name: _sha256(Path(__file__).resolve().parent / "vendor" / name)
            for name in ("eval_v3.py", "eval_v4.py", "diagnostic_text.py", "subtype_ontology_v4.py", "tcga_ground_truth.py")
        },
        "manuscript_metrics_source_sha256": _sha256(Path(__file__).resolve().parent / "manuscript_metrics.py"),
        "cancer_type_context": cancer_type,
        "site_context": site,
        "reference_format": reference_format,
        "diagnosis_only": diagnosis_only,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path, help="Generated report (.html, .htm, or plain text)")
    parser.add_argument("--reference", required=True, type=Path, help="Reference report (.txt) or JSON reference")
    parser.add_argument("--reference-format", choices=("text", "json"), default="text")
    parser.add_argument("--cancer-type", help="Optional TCGA project code, e.g. TCGA-BRCA; provides anatomy context only")
    parser.add_argument("--site", help="Optional anatomy context, e.g. breast or kidney")
    parser.add_argument("--diagnosis-only", action="store_true", help="Suppress narrative/finding metrics when only diagnosis was assessed")
    parser.add_argument("--output", required=True, type=Path, help="Path for detailed JSON result")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for label, path in (("report", args.report), ("reference", args.reference)):
        if not path.is_file():
            raise SystemExit(f"{label} file not found: {path}")
    result = evaluate_paths(args.report, args.reference, cancer_type=args.cancer_type,
                            reference_format=args.reference_format, diagnosis_only=args.diagnosis_only,
                            site=args.site)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics = result["metrics"]
    print(f"Evaluator: {result['schema']}")
    print(f"Diagnosis agreement: {metrics.get('diagnosis')}")
    print(f"Strict subtype agreement: {metrics.get('subtype')}")
    print(f"TCGA cohort compatibility: {metrics.get('cohort_compatibility')}")
    print(f"Cohort diagnosis composite: {metrics.get('diagnosis_cohort')}")
    findings = result["manuscript_metrics"]["findings"]
    print(f"Finding contradiction among asserted predictions (E/A): {findings['contradiction_asserted'] if findings else None}")
    print(f"Detailed result: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

