"""KEEP knowledge retrieval stage — searches external pathology knowledge base."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathxai_agent_v2.config import Settings
from pathxai_agent_v2.pipeline.base import (
    DiagnosisPrediction,
    PipelineStage,
    SitePrediction,
    StageResult,
    ok,
)


@dataclass
class KnowledgeOutput:
    available: bool = False
    disease_hits: list[dict] = field(default_factory=list)
    semantic_group_hits: list[dict] = field(default_factory=list)
    site_subtype_candidates: list[dict] = field(default_factory=list)
    hypothesis_hits: list[dict] = field(default_factory=list)
    phase: str = "broad"  # "broad" (site only) | "hypothesis" (subtype detail)
    query: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "disease_hits": self.disease_hits,
            "semantic_group_hits": self.semantic_group_hits,
            "site_subtype_candidates": self.site_subtype_candidates,
            "hypothesis_hits": self.hypothesis_hits,
            "phase": self.phase,
            "query": self.query,
            "error": self.error,
        }


class KnowledgeStage(PipelineStage[dict, KnowledgeOutput]):
    """Hierarchical KEEP retrieval, mirroring how a pathologist narrows down.

    Phase 1 (``broad``): only the specimen *site* is known. Retrieve the broad
    range of entities reported at that site (``site_subtype_candidates``) plus
    site-anchored disease/semantic hits. This gives the diagnosis stage the
    candidate space to reason ``broad \u2192 subtype`` \u2014 the site is a legitimate
    clinical input, only the diagnosis is the answer to predict.

    Phase 2 (``hypothesis``): once the diagnosis stage has committed to a
    subtype, retrieve fine-grained, hypothesis-anchored knowledge (definition,
    child subtypes, morphology captions) to support verification and reporting.
    """

    stage_name = "knowledge"

    def run(
        self,
        stage_input: dict,
        dry_run: bool = False,
    ) -> StageResult[KnowledgeOutput]:
        """Retrieve KEEP knowledge.

        stage_input expects:
            site_prediction: SitePrediction
            diagnosis_prediction: DiagnosisPrediction | None
                When None \u2192 phase 1 (broad, site-only).
                When set  \u2192 phase 2 (hypothesis-anchored subtype detail).
        """
        site_pred: SitePrediction | None = stage_input.get("site_prediction")
        diag: DiagnosisPrediction | None = stage_input.get("diagnosis_prediction")

        cfg = self.settings.knowledge
        if not cfg.enabled or dry_run:
            return ok(
                KnowledgeOutput(available=False, error="disabled" if not cfg.enabled else "dry_run"),
                self.stage_name,
            )

        try:
            import sys
            v1_src = str(Path(self.settings.paths.v1_src_root))
            if v1_src not in sys.path:
                sys.path.insert(0, v1_src)
            from pathxai_agent.knowledge.keep_retriever import (
                retrieve_keep_knowledge,
                retrieve_hypothesis_knowledge,
            )
        except ImportError:
            return ok(KnowledgeOutput(available=False, error="keep_retriever not importable"), self.stage_name)

        site = site_pred.predicted_site if site_pred else ""

        # ── Phase 1: broad, site-anchored retrieval ──────────────────────────
        # Query is built from site only (no diagnosis) so the candidate space is
        # not pre-narrowed. The site is a legitimate input; the diagnosis is not.
        try:
            result = retrieve_keep_knowledge(
                query=site,
                site=site,
                diagnosis="",
                keep_root=self.settings.paths.keep_root,
                max_disease=cfg.max_disease,
                max_semantic_groups=cfg.max_semantic_groups,
            )
        except Exception as exc:
            return ok(KnowledgeOutput(available=False, error=str(exc), phase="broad"), self.stage_name)

        out = KnowledgeOutput(
            available=result.get("available", False),
            disease_hits=result.get("disease_hits", []),
            semantic_group_hits=result.get("semantic_group_hits", []),
            site_subtype_candidates=result.get("site_subtype_candidates", []),
            phase="broad",
            query=site,
        )

        # ── Phase 2: hypothesis-anchored subtype detail ──────────────────────
        if diag is not None:
            hyp_names = [diag.subtype or diag.primary_diagnosis]
            hyp_names.extend(str(d) for d in (diag.differential or [])[:3])
            hypothesis_set = [{"diagnosis": n, "confidence": diag.confidence} for n in hyp_names if n]
            # Differential entries get lower confidence
            for i in range(1, len(hypothesis_set)):
                hypothesis_set[i]["confidence"] = "low"
            try:
                hyp = retrieve_hypothesis_knowledge(
                    hypothesis_set=hypothesis_set,
                    site=site,
                    keep_root=self.settings.paths.keep_root,
                )
                out.hypothesis_hits = hyp.get("per_hypothesis", [])
                out.phase = "hypothesis"
                out.query = " | ".join(n for n in hyp_names if n)
                if out.hypothesis_hits:
                    out.available = True
            except Exception as exc:
                out.error = f"phase2: {exc}"

        return ok(out, self.stage_name)
