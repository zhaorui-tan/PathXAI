"""Pipeline stage imports."""

from pathxai_agent_v2.pipeline.base import (
    DiagnosisPrediction,
    EvidenceSufficiency,
    PipelineStage,
    RoiInfo,
    SitePrediction,
    StageResult,
)
from pathxai_agent_v2.pipeline.roi_selection import (
    RoiSelectionInput,
    RoiSelectionOutput,
    RoiSelectionStage,
)
from pathxai_agent_v2.pipeline.site_prediction import (
    SitePredictionInput,
    SitePredictionStage,
)
from pathxai_agent_v2.pipeline.segmentation import SegmentationOutput, SegmentationStage
from pathxai_agent_v2.pipeline.morphology import MorphologyOutput, MorphologyStage
from pathxai_agent_v2.pipeline.knowledge import KnowledgeOutput, KnowledgeStage
from pathxai_agent_v2.pipeline.diagnosis import (
    DiagnosisInput,
    DiagnosisStage,
)
from pathxai_agent_v2.pipeline.subtype_sufficiency import (
    BroadHypothesesOutput,
    BroadHypothesesStage,
    SubtypeReasoningInput,
    SubtypeSufficiencyOutput,
    SubtypeSufficiencyStage,
    apply_subtype_sufficiency_guard,
)
from pathxai_agent_v2.pipeline.lineage_guard import apply_site_lineage_guard, find_site_lineage
from pathxai_agent_v2.pipeline.evidence_verification import (
    EvidenceInput,
    EvidenceVerificationStage,
)
from pathxai_agent_v2.pipeline.reinspection import ReinspectionPlan, ReinspectionStage
from pathxai_agent_v2.pipeline.evidence_graph import EvidenceGraphOutput, EvidenceGraphStage
from pathxai_agent_v2.pipeline.report_generation import (
    ReportGenerationStage,
    ReportInput,
    ReportOutput,
)
from pathxai_agent_v2.pipeline.session_memory import SessionMemory

__all__ = [
    # Base
    "PipelineStage", "StageResult", "RoiInfo",
    "SitePrediction", "DiagnosisPrediction", "EvidenceSufficiency",
    # Stages
    "RoiSelectionStage", "RoiSelectionInput", "RoiSelectionOutput",
    "SitePredictionStage", "SitePredictionInput",
    "SegmentationStage", "SegmentationOutput",
    "MorphologyStage", "MorphologyOutput",
    "KnowledgeStage", "KnowledgeOutput",
    "DiagnosisStage", "DiagnosisInput",
    "BroadHypothesesStage", "BroadHypothesesOutput",
    "SubtypeSufficiencyStage", "SubtypeSufficiencyOutput", "SubtypeReasoningInput",
    "EvidenceVerificationStage", "EvidenceInput",
    "ReinspectionStage", "ReinspectionPlan",
    "EvidenceGraphStage", "EvidenceGraphOutput",
    "ReportGenerationStage", "ReportInput", "ReportOutput",
    # Guards
    "apply_site_lineage_guard", "find_site_lineage", "apply_subtype_sufficiency_guard",
    # Memory
    "SessionMemory",
]
