#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathxai_agent.agents.evidence_graph_agent import render_evidence_graph_html
from pathxai_agent.agents.staged_report_agent import render_report_html


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate report.html files from staged_trace.json without rerunning inference."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/reports/ablation_evidence_graph/A0_full"),
        help="Run directory or parent directory containing TCGA-*/*/staged_trace.json files.",
    )
    parser.add_argument("--out_name", type=str, default="report.html")
    parser.add_argument("--backup", action="store_true", help="Keep existing HTML as <out_name>.bak before overwriting.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    root = args.root
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    traces = find_traces(root)
    if args.limit:
        traces = traces[: args.limit]

    rows: list[dict] = []
    for trace_path in traces:
        case_dir = trace_path.parent
        out_path = case_dir / args.out_name
        row = {"case_dir": str(case_dir), "trace": str(trace_path), "out": str(out_path)}
        try:
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            html = render_html(payload)
            row.update({
                "status": "dry_run" if args.dry_run else "ok",
                "mode": payload.get("mode") or "staged",
                "figures": len(payload.get("figures") or []),
                "selected_rois": len(payload.get("selected_rois") or []),
                "bytes": len(html.encode("utf-8")),
            })
            if not args.dry_run:
                if args.backup and out_path.exists():
                    backup_path = out_path.with_name(out_path.name + ".bak")
                    backup_path.write_bytes(out_path.read_bytes())
                    row["backup"] = str(backup_path)
                out_path.write_text(html, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            row.update({"status": "error", "error": str(exc)})
        rows.append(row)
        print(f"[{len(rows)}/{len(traces)}] {row['status']} {case_dir}")

    summary = summarize(rows)
    summary_path = root / f"regenerate_html_summary_{args.out_name.replace('/', '_')}.json"
    if not args.dry_run:
        summary_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.dry_run:
        print(f"Wrote {summary_path}")


def find_traces(root: Path) -> list[Path]:
    if root.name == "staged_trace.json" and root.is_file():
        return [root]
    if (root / "staged_trace.json").exists():
        return [root / "staged_trace.json"]
    return sorted(root.glob("**/staged_trace.json"))


def render_html(payload: dict) -> str:
    mode = str(payload.get("mode") or "")
    if mode.startswith("case_evidence_graph_agent") or payload.get("evidence_graph") or payload.get("claim_verification"):
        return render_evidence_graph_html(payload)
    return render_report_html(payload)


def summarize(rows: list[dict]) -> dict:
    return {
        "total": len(rows),
        "ok": sum(1 for r in rows if r.get("status") in {"ok", "dry_run"}),
        "error": sum(1 for r in rows if r.get("status") == "error"),
        "total_figures": sum(int(r.get("figures") or 0) for r in rows),
        "total_selected_rois": sum(int(r.get("selected_rois") or 0) for r in rows),
    }


if __name__ == "__main__":
    main()
