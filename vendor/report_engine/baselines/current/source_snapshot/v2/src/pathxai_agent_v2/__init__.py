"""pathxai_agent_v2 — clean re-architecture of the PathXAI agent pipeline.

This package is a fresh, incremental rewrite of the legacy ``pathxai_agent``
agents (``evidence_graph_agent.py``, ``staged_report_agent.py``,
``pathagent_qwen.py``).  The legacy package is intentionally left untouched so
existing ablation runs and checkpoints remain reproducible.

Design goals driving this rewrite (see ``README.md``):

1. No hardcoded paths / magic numbers in code — everything lives in
   :mod:`pathxai_agent_v2.config`.
2. A single LLM abstraction + one validated JSON parsing layer
   (:mod:`pathxai_agent_v2.llm`) instead of scattered ``raw.get("json") or {}``.
3. Centralized, auditable data-leakage control
   (:mod:`pathxai_agent_v2.leakage_guard`): ``known_site`` is a supported
   clinical prior; ``known_cancer_type`` (the ground-truth label) is *removed*.
4. Stages are small, testable units; the orchestrator only wires them together.
5. Prompts are separated from pipeline logic (:mod:`pathxai_agent_v2.prompts`).
6. Evaluation uses rule-based extraction by default — no LLM self-preference bias
   (:mod:`pathxai_agent_v2.eval`).

Quick start::

    from pathxai_agent_v2.config import load_settings
    from pathxai_agent_v2.run import run_single_case

    settings = load_settings()
    result = run_single_case(settings, "TCGA-XX-XXXX", candidate_rois, "outputs/")
    print(result.report.long_report)
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

