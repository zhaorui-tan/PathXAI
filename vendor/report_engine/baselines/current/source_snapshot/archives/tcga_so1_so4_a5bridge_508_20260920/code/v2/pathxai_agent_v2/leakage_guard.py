"""Centralized data-leakage control for the agent.

This module is the *only* place allowed to translate dataset ground-truth into
priors that reach the agent. Keeping it in one small file makes the leakage
surface auditable.

Policy (decided for this rewrite):

* ``known_site`` — KEPT. The specimen organ/site is a legitimate clinical prior
  ("we know where the biopsy came from"). It may be derived from the TCGA
  cancer-type label via ``known_site_from="tcga"``, but that is an explicit,
  logged opt-in. Callers are reminded (:func:`site_metric_is_valid`) that
  ``site_accuracy`` is NOT a capability metric while a known site is supplied.

* ``known_cancer_type`` — REMOVED. The ground-truth cancer type/diagnosis must
  never be handed to the agent. The legacy pipeline passed it in and never used
  it, which was a dormant leak. There is intentionally no parameter, field, or
  code path for it here. :func:`assert_no_cancer_type_leak` exists so callers /
  tests can assert this invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .config import LeakageConfig

# Field name that must never be plumbed into agent inputs.
FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "known_cancer_type",
        "cancer_type",
        "cancer_type_tcga",
        "ground_truth_diagnosis",
        "reference_diagnosis",
        "gold_diagnosis",
    }
)


@dataclass(frozen=True)
class DiagnosisPriors:
    """Immutable set of label-derived priors permitted to reach the agent.

    Only ``known_site`` is representable. ``known_site_source`` records how it
    was obtained so it can be surfaced in the run manifest and in the site
    prediction payload (``source="known_site:<src>"``).
    """

    known_site: Optional[str] = None
    known_site_source: Optional[str] = None  # e.g. "tcga" | "user" | None

    @property
    def has_known_site(self) -> bool:
        return bool(self.known_site)

    def site_prediction_source(self) -> str:
        """Value for the site_prediction ``source`` field."""
        if not self.has_known_site:
            return "predicted"
        return f"known_site:{self.known_site_source or 'unspecified'}"


def resolve_diagnosis_priors(
    leakage: LeakageConfig,
    cancer_type: Optional[str],
    *,
    known_site_for_tcga: Optional[Callable[[Optional[str]], Optional[str]]] = None,
) -> DiagnosisPriors:
    """Build the allowed priors from config + dataset metadata.

    Parameters
    ----------
    leakage:
        The resolved :class:`LeakageConfig` (``known_site_from`` policy).
    cancer_type:
        The dataset's TCGA cancer-type label. It is used *only* to look up the
        organ/site when ``known_site_from == "tcga"``; it is never returned or
        forwarded as a diagnosis prior.
    known_site_for_tcga:
        Injected mapping ``cancer_type -> site`` (the legacy
        ``pathxai_agent.datasets.tcga_sites.known_site_for_tcga``). Injected
        rather than imported so this module stays dependency-light and testable.
    """
    if leakage.known_site_from == "none":
        return DiagnosisPriors()

    if leakage.known_site_from == "tcga":
        if known_site_for_tcga is None:
            raise ValueError(
                "known_site_from='tcga' requires a known_site_for_tcga mapping"
            )
        site = known_site_for_tcga(cancer_type)
        if not site:
            return DiagnosisPriors()
        return DiagnosisPriors(known_site=str(site), known_site_source="tcga")

    # LeakageConfig.__post_init__ already validates the enum, but guard anyway.
    raise ValueError(f"Unsupported known_site_from: {leakage.known_site_from!r}")


def site_metric_is_valid(priors: DiagnosisPriors) -> bool:
    """False when reporting site_accuracy would be circular.

    When a known site is supplied to the agent, it did not *predict* the site,
    so ``site_accuracy`` reflects the label, not model capability.
    """
    return not priors.has_known_site


def assert_no_cancer_type_leak(payload: Any, *, where: str = "agent input") -> None:
    """Fail loudly if a ground-truth cancer type sneaks into agent inputs.

    Recursively scans dict keys for any :data:`FORBIDDEN_LABEL_KEYS`. Intended
    for use at the agent boundary and in tests, so the removed leak cannot
    silently return.
    """
    offenders = sorted(_scan_forbidden_keys(payload))
    if offenders:
        raise AssertionError(
            f"Forbidden ground-truth label(s) present in {where}: {offenders}. "
            "The agent must not receive the cancer-type/diagnosis label."
        )


def _scan_forbidden_keys(obj: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and key in FORBIDDEN_LABEL_KEYS:
                found.add(key)
            found |= _scan_forbidden_keys(value)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            found |= _scan_forbidden_keys(item)
    return found
