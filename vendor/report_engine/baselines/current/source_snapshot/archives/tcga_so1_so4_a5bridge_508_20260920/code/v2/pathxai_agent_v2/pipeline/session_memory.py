"""Session memory — accumulates diagnostic context across pipeline stages.

Enables the agent to maintain state (what was observed, what decisions were made)
as it progresses through ROI selection → site → segmentation → morphology →
diagnosis → reinspection → report.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SessionMemory:
    """Accumulated diagnostic session state across pipeline stages."""

    version: str = "session_memory_v2"
    case_id: str = ""
    events: list[dict] = field(default_factory=list)
    stage_summaries: dict[str, dict] = field(default_factory=dict)

    # Accumulated evidence
    selected_roi_count: int = 0
    total_roi_count: int = 0
    reinspection_rounds: int = 0
    known_site: str = ""

    def record(
        self,
        stage: str,
        summary: dict[str, Any],
        roi_count: int = 0,
    ) -> None:
        """Record a stage completion event."""
        event = {
            "stage": stage,
            "timestamp": time.time(),
            "roi_count": roi_count,
            "summary": _safe_summary(summary),
        }
        self.events.append(event)
        self.stage_summaries[stage] = event
        self.selected_roi_count = max(self.selected_roi_count, roi_count)

    def get_context_for_stage(self, stage: str) -> str:
        """Build a natural language summary of the session so far for LLM prompts."""
        if not self.events:
            return ""

        lines = [f"Diagnostic session progress ({len(self.events)} stages completed):"]
        for ev in self.events:
            prev_stage = ev["stage"]
            prev_summary = ev.get("summary", {})
            lines.append(f"- Stage {prev_stage}: {_brief(prev_summary)}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "case_id": self.case_id,
            "events": self.events,
            "stage_summaries": self.stage_summaries,
            "selected_roi_count": self.selected_roi_count,
            "total_roi_count": self.total_roi_count,
            "reinspection_rounds": self.reinspection_rounds,
            "known_site": self.known_site,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMemory":
        return cls(
            version=data.get("version", "session_memory_v2"),
            case_id=data.get("case_id", ""),
            events=data.get("events", []),
            stage_summaries=data.get("stage_summaries", {}),
            selected_roi_count=data.get("selected_roi_count", 0),
            total_roi_count=data.get("total_roi_count", 0),
            reinspection_rounds=data.get("reinspection_rounds", 0),
            known_site=data.get("known_site", ""),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "SessionMemory":
        if path.exists():
            data = json.loads(path.read_text())
            return cls.from_dict(data)
        return cls()


def _safe_summary(data: dict, max_keys: int = 10) -> dict:
    """Truncate large summaries for the session log."""
    out = {}
    for k, v in list(data.items())[:max_keys]:
        if isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, dict)):
            s = json.dumps(v)
            out[k] = s[:200] + ("..." if len(s) > 200 else "")
        else:
            out[k] = str(v)[:200]
    return out


def _brief(summary: dict) -> str:
    """One-line summary of a stage's output."""
    parts = []
    for k, v in summary.items():
        if isinstance(v, str) and len(v) < 80:
            parts.append(f"{k}={v}")
    return ", ".join(parts[:5]) if parts else "completed"
