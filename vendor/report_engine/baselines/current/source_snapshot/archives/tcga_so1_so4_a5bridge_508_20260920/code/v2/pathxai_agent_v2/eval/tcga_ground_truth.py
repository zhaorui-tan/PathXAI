"""TCGA cohort ground-truth anchors for hierarchical diagnosis evaluation.

Site (organ) is a legitimate, always-known input in this task — it is NOT
leakage. The *diagnosis* (broad lineage / subtype / malignancy) is the answer
the model must produce and is only used here on the reference side to score
predictions.

Each TCGA project code maps to its cohort-defining ground truth, structured as
three independent tiers:

    broad_type   — histogenetic lineage / category (carcinoma, sarcoma, ...)
    subtype      — the specific entity the cohort is defined by, plus aliases
    malignancy   — benign / borderline / malignant

Reference facts are anchored on this table rather than parsed from free text,
so a correct prediction is never penalised by synonym-coverage gaps in the
reference report. When the reference report contains a *more specific* subtype
than the cohort default, callers may refine the subtype from the report text.
"""

from __future__ import annotations

from typing import Any


# code -> {broad_type, subtype, subtype_aliases, malignancy}
TCGA_GROUND_TRUTH: dict[str, dict[str, Any]] = {
    "TCGA-ACC": {
        "broad_type": "carcinoma",
        "lineage": "adrenal cortical neoplasm",
        "subtype": "adrenocortical carcinoma",
        "subtype_aliases": ["adrenocortical carcinoma", "adrenal cortical carcinoma"],
        "malignancy": "malignant",
    },
    "TCGA-PCPG": {
        "broad_type": "neuroendocrine neoplasm",
        "lineage": "adrenal medullary / paraganglial neoplasm",
        "subtype": "pheochromocytoma/paraganglioma",
        "subtype_aliases": ["pheochromocytoma", "paraganglioma"],
        "malignancy": "borderline",
    },
    "TCGA-BLCA": {
        "broad_type": "carcinoma",
        "lineage": "urothelial neoplasm",
        "subtype": "urothelial carcinoma",
        "subtype_aliases": ["urothelial carcinoma", "transitional cell carcinoma"],
        "malignancy": "malignant",
    },
    "TCGA-BRCA": {
        "broad_type": "carcinoma",
        "lineage": "breast epithelial neoplasm",
        "subtype": "invasive breast carcinoma",
        "subtype_aliases": ["invasive ductal carcinoma", "invasive lobular carcinoma",
                             "breast carcinoma", "invasive carcinoma of no special type"],
        "malignancy": "malignant",
    },
    "TCGA-CESC": {
        "broad_type": "carcinoma",
        "lineage": "cervical carcinoma",
        "subtype": "cervical squamous cell carcinoma",
        "subtype_aliases": ["squamous cell carcinoma", "cervical adenocarcinoma"],
        "malignancy": "malignant",
    },
    "TCGA-COAD": {
        "broad_type": "carcinoma",
        "lineage": "colorectal epithelial neoplasm",
        "subtype": "colorectal adenocarcinoma",
        "subtype_aliases": ["colonic adenocarcinoma", "colorectal adenocarcinoma", "adenocarcinoma"],
        "malignancy": "malignant",
    },
    "TCGA-READ": {
        "broad_type": "carcinoma",
        "lineage": "colorectal epithelial neoplasm",
        "subtype": "rectal adenocarcinoma",
        "subtype_aliases": ["rectal adenocarcinoma", "colorectal adenocarcinoma", "adenocarcinoma"],
        "malignancy": "malignant",
    },
    "TCGA-ESCA": {
        "broad_type": "carcinoma",
        "lineage": "esophageal carcinoma",
        "subtype": "esophageal carcinoma",
        "subtype_aliases": ["esophageal squamous cell carcinoma", "esophageal adenocarcinoma"],
        "malignancy": "malignant",
    },
    "TCGA-STAD": {
        "broad_type": "carcinoma",
        "lineage": "gastric epithelial neoplasm",
        "subtype": "gastric adenocarcinoma",
        "subtype_aliases": ["gastric adenocarcinoma", "stomach adenocarcinoma", "adenocarcinoma"],
        "malignancy": "malignant",
    },
    "TCGA-HNSC": {
        "broad_type": "carcinoma",
        "lineage": "head and neck carcinoma",
        "subtype": "head and neck squamous cell carcinoma",
        "subtype_aliases": ["squamous cell carcinoma", "head and neck squamous cell carcinoma"],
        "malignancy": "malignant",
    },
    "TCGA-KICH": {
        "broad_type": "carcinoma",
        "lineage": "renal epithelial neoplasm",
        "subtype": "chromophobe renal cell carcinoma",
        "subtype_aliases": ["chromophobe renal cell carcinoma", "chromophobe rcc"],
        "malignancy": "malignant",
    },
    "TCGA-KIRC": {
        "broad_type": "carcinoma",
        "lineage": "renal epithelial neoplasm",
        "subtype": "clear cell renal cell carcinoma",
        "subtype_aliases": ["clear cell renal cell carcinoma", "clear cell rcc",
                             "conventional renal cell carcinoma"],
        "malignancy": "malignant",
    },
    "TCGA-KIRP": {
        "broad_type": "carcinoma",
        "lineage": "renal epithelial neoplasm",
        "subtype": "papillary renal cell carcinoma",
        "subtype_aliases": ["papillary renal cell carcinoma", "papillary rcc"],
        "malignancy": "malignant",
    },
    "TCGA-CHOL": {
        "broad_type": "carcinoma",
        "lineage": "biliary adenocarcinoma",
        "subtype": "cholangiocarcinoma",
        "subtype_aliases": ["cholangiocarcinoma", "bile duct adenocarcinoma"],
        "malignancy": "malignant",
    },
    "TCGA-LIHC": {
        "broad_type": "carcinoma",
        "lineage": "hepatocellular neoplasm",
        "subtype": "hepatocellular carcinoma",
        "subtype_aliases": ["hepatocellular carcinoma", "hcc"],
        "malignancy": "malignant",
    },
    "TCGA-LUAD": {
        "broad_type": "carcinoma",
        "lineage": "pulmonary epithelial neoplasm",
        "subtype": "lung adenocarcinoma",
        "subtype_aliases": ["lung adenocarcinoma", "pulmonary adenocarcinoma", "adenocarcinoma"],
        "malignancy": "malignant",
    },
    "TCGA-LUSC": {
        "broad_type": "carcinoma",
        "lineage": "pulmonary epithelial neoplasm",
        "subtype": "lung squamous cell carcinoma",
        "subtype_aliases": ["lung squamous cell carcinoma", "squamous cell carcinoma"],
        "malignancy": "malignant",
    },
    "TCGA-MESO": {
        "broad_type": "mesothelioma",
        "lineage": "mesothelial neoplasm",
        "subtype": "malignant mesothelioma",
        "subtype_aliases": ["mesothelioma", "malignant mesothelioma"],
        "malignancy": "malignant",
    },
    "TCGA-LGG": {
        "broad_type": "glioma",
        "lineage": "glioma",
        "subtype": "lower-grade glioma",
        "subtype_aliases": ["astrocytoma", "oligodendroglioma", "diffuse glioma", "lower-grade glioma"],
        "malignancy": "malignant",
    },
    "TCGA-GBM": {
        "broad_type": "glioma",
        "lineage": "glioma",
        "subtype": "glioblastoma",
        "subtype_aliases": ["glioblastoma", "gbm", "glioblastoma multiforme"],
        "malignancy": "malignant",
    },
    "TCGA-OV": {
        "broad_type": "carcinoma",
        "lineage": "ovarian epithelial neoplasm",
        "subtype": "high-grade serous ovarian carcinoma",
        "subtype_aliases": ["serous carcinoma", "ovarian carcinoma", "high-grade serous carcinoma"],
        "malignancy": "malignant",
    },
    "TCGA-UCEC": {
        "broad_type": "carcinoma",
        "lineage": "endometrial epithelial neoplasm",
        "subtype": "endometrial carcinoma",
        "subtype_aliases": ["endometrioid carcinoma", "endometrial adenocarcinoma",
                            "uterine serous carcinoma", "endometrial carcinoma"],
        "malignancy": "malignant",
    },
    "TCGA-UCS": {
        "broad_type": "carcinosarcoma",
        "lineage": "uterine mixed epithelial-mesenchymal neoplasm",
        "subtype": "uterine carcinosarcoma",
        "subtype_aliases": ["carcinosarcoma", "malignant mixed mullerian tumor", "mmmt"],
        "malignancy": "malignant",
    },
    "TCGA-PRAD": {
        "broad_type": "carcinoma",
        "lineage": "prostatic epithelial neoplasm",
        "subtype": "prostatic adenocarcinoma",
        "subtype_aliases": ["prostatic adenocarcinoma", "prostate adenocarcinoma", "acinar adenocarcinoma"],
        "malignancy": "malignant",
    },
    "TCGA-SKCM": {
        "broad_type": "melanoma",
        "lineage": "melanocytic neoplasm",
        "subtype": "cutaneous melanoma",
        "subtype_aliases": ["melanoma", "malignant melanoma", "cutaneous melanoma"],
        "malignancy": "malignant",
    },
    "TCGA-SARC": {
        "broad_type": "sarcoma",
        "lineage": "soft tissue sarcoma",
        "subtype": "soft tissue sarcoma",
        "subtype_aliases": ["leiomyosarcoma", "liposarcoma", "undifferentiated pleomorphic sarcoma",
                            "myxofibrosarcoma", "sarcoma"],
        "malignancy": "malignant",
    },
    "TCGA-TGCT": {
        "broad_type": "germ cell tumor",
        "lineage": "germ cell neoplasm",
        "subtype": "testicular germ cell tumor",
        "subtype_aliases": ["seminoma", "embryonal carcinoma", "teratoma", "germ cell tumor",
                            "mixed germ cell tumor", "yolk sac tumor"],
        "malignancy": "malignant",
    },
    "TCGA-THCA": {
        "broad_type": "carcinoma",
        "lineage": "thyroid epithelial neoplasm",
        "subtype": "papillary thyroid carcinoma",
        "subtype_aliases": ["papillary thyroid carcinoma", "papillary carcinoma", "follicular carcinoma"],
        "malignancy": "malignant",
    },
    "TCGA-THYM": {
        "broad_type": "thymic epithelial neoplasm",
        "lineage": "thymic epithelial neoplasm",
        "subtype": "thymoma",
        "subtype_aliases": ["thymoma", "thymic carcinoma"],
        "malignancy": "borderline",
    },
    "TCGA-UVM": {
        "broad_type": "melanoma",
        "lineage": "melanocytic neoplasm",
        "subtype": "uveal melanoma",
        "subtype_aliases": ["uveal melanoma", "choroidal melanoma", "melanoma"],
        "malignancy": "malignant",
    },
    "TCGA-DLBC": {
        "broad_type": "lymphoma",
        "lineage": "lymphoid neoplasm",
        "subtype": "diffuse large B-cell lymphoma",
        "subtype_aliases": ["diffuse large b-cell lymphoma", "dlbcl", "large b-cell lymphoma"],
        "malignancy": "malignant",
    },
}


def ground_truth_for(cancer_type: str) -> dict[str, Any] | None:
    """Return the ground-truth tiers for a TCGA project code, or None."""
    if not cancer_type:
        return None
    key = cancer_type.strip().upper()
    if not key.startswith("TCGA-"):
        key = f"TCGA-{key}"
    return TCGA_GROUND_TRUTH.get(key)
