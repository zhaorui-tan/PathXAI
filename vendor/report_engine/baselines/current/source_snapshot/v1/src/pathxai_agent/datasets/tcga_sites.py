from __future__ import annotations


TCGA_SITE_MAP = {
    "TCGA-ACC": "adrenal gland / adrenocortical tissue",
    "TCGA-PCPG": "adrenal medulla / paraganglia / pheochromocytoma tissue",
    "TCGA-BLCA": "bladder / urothelial tissue",
    "TCGA-BRCA": "breast tissue",
    "TCGA-CESC": "cervix / cervical tissue",
    "TCGA-COAD": "colon / colorectal mucosa",
    "TCGA-READ": "rectum / colorectal mucosa",
    "TCGA-ESCA": "esophagus / squamous or glandular mucosa",
    "TCGA-STAD": "stomach / gastric mucosa",
    "TCGA-HNSC": "head and neck squamous mucosa",
    "TCGA-KICH": "kidney / renal tissue",
    "TCGA-KIRC": "kidney / renal tissue",
    "TCGA-KIRP": "kidney / renal tissue",
    "TCGA-CHOL": "bile duct / cholangiocarcinoma tissue",
    "TCGA-LIHC": "liver / hepatic tissue",
    "TCGA-LUAD": "lung tissue",
    "TCGA-LUSC": "lung tissue",
    "TCGA-MESO": "pleura / mesothelial tissue",
    "TCGA-LGG": "brain / glial tissue",
    "TCGA-OV": "ovary / ovarian tissue",
    "TCGA-UCEC": "uterus / endometrium",
    "TCGA-UCS": "uterus / endometrium",
    "TCGA-PRAD": "prostate tissue",
    "TCGA-SKCM": "skin / melanocytic tissue",
    "TCGA-SARC": "soft tissue / sarcoma",
    "TCGA-TGCT": "testis / germ cell tissue",
    "TCGA-THCA": "thyroid tissue",
    "TCGA-THYM": "thymus tissue",
    "TCGA-UVM": "eye / uveal melanocytic tissue",
    "TCGA-DLBC": "lymphoid tissue",
}


TCGA_DIAGNOSIS_PRIORS = {
    "TCGA-ACC": {
        "site": "adrenal gland / adrenocortical tissue",
        "broad_type": "adrenal cortical neoplasm",
        "subtype_candidates": [
            "adrenocortical carcinoma",
            "adrenocortical neoplasm of uncertain/indeterminate malignant potential",
            "adrenocortical adenoma",
            "metastatic carcinoma to adrenal gland",
            "pheochromocytoma/paraganglioma mimic",
        ],
        "not_further_classifiable": "adrenocortical neoplasm, not further classifiable from ROI evidence",
    },
    "TCGA-BLCA": {
        "site": "bladder / urothelial tissue",
        "broad_type": "urothelial neoplasm",
        "subtype_candidates": [
            "urothelial carcinoma",
            "invasive urothelial carcinoma",
            "papillary urothelial carcinoma",
            "carcinoma in situ",
            "squamous/glandular divergent differentiation if supported",
        ],
        "not_further_classifiable": "urothelial carcinoma/neoplasm, not further classifiable from ROI evidence",
    },
    "TCGA-BRCA": {
        "site": "breast tissue",
        "broad_type": "breast epithelial neoplasm",
        "subtype_candidates": [
            "invasive ductal carcinoma / invasive carcinoma NST",
            "invasive lobular carcinoma",
            "ductal carcinoma in situ",
            "invasive breast carcinoma, not otherwise specified",
        ],
        "not_further_classifiable": "invasive breast carcinoma, not further classifiable from ROI evidence",
    },
    "TCGA-CESC": {
        "site": "cervix / cervical tissue",
        "broad_type": "cervical carcinoma",
        "subtype_candidates": [
            "squamous cell carcinoma of cervix",
            "endocervical adenocarcinoma",
            "adenosquamous carcinoma",
            "cervical carcinoma, not otherwise specified",
        ],
        "not_further_classifiable": "cervical carcinoma, not further classifiable from ROI evidence",
    },
    "TCGA-CHOL": {
        "site": "bile duct / cholangiocarcinoma tissue",
        "broad_type": "biliary adenocarcinoma",
        "subtype_candidates": [
            "cholangiocarcinoma",
            "intrahepatic cholangiocarcinoma",
            "extrahepatic bile duct adenocarcinoma",
            "biliary adenocarcinoma, not otherwise specified",
        ],
        "not_further_classifiable": "cholangiocarcinoma/biliary adenocarcinoma, not further classifiable from ROI evidence",
    },
    "TCGA-COAD": {
        "site": "colon / colorectal mucosa",
        "broad_type": "colorectal epithelial neoplasm",
        "subtype_candidates": [
            "colorectal adenocarcinoma",
            "mucinous adenocarcinoma",
            "poorly differentiated adenocarcinoma",
            "adenoma with high-grade dysplasia if invasion is not supported",
        ],
        "not_further_classifiable": "colorectal adenocarcinoma/neoplasm, not further classifiable from ROI evidence",
    },
    "TCGA-DLBC": {
        "site": "lymphoid tissue",
        "broad_type": "lymphoid neoplasm",
        "subtype_candidates": [
            "diffuse large B-cell lymphoma",
            "large B-cell lymphoma",
            "high-grade B-cell lymphoma",
            "lymphoma, not otherwise specified",
        ],
        "not_further_classifiable": "lymphoid neoplasm/lymphoma, not further classifiable from ROI evidence",
    },
    "TCGA-ESCA": {
        "site": "esophagus / squamous or glandular mucosa",
        "broad_type": "esophageal carcinoma",
        "subtype_candidates": [
            "esophageal adenocarcinoma",
            "esophageal squamous cell carcinoma",
            "poorly differentiated carcinoma",
            "esophageal carcinoma, not otherwise specified",
        ],
        "not_further_classifiable": "esophageal carcinoma, not further classifiable from ROI evidence",
    },
    "TCGA-HNSC": {
        "site": "head and neck squamous mucosa",
        "broad_type": "head and neck carcinoma",
        "subtype_candidates": [
            "squamous cell carcinoma",
            "keratinizing squamous cell carcinoma",
            "non-keratinizing squamous cell carcinoma",
            "head and neck carcinoma, not otherwise specified",
        ],
        "not_further_classifiable": "head and neck carcinoma, not further classifiable from ROI evidence",
    },
    "TCGA-KICH": {
        "site": "kidney / renal tissue",
        "broad_type": "renal epithelial neoplasm",
        "subtype_candidates": [
            "chromophobe renal cell carcinoma",
            "oncocytoma/chromophobe oncocytic renal neoplasm mimic",
            "renal cell carcinoma, not further classifiable",
        ],
        "not_further_classifiable": "renal epithelial neoplasm, not further classifiable from ROI evidence",
    },
    "TCGA-KIRC": {
        "site": "kidney / renal tissue",
        "broad_type": "renal epithelial neoplasm",
        "subtype_candidates": [
            "clear cell renal cell carcinoma",
            "renal cell carcinoma, clear cell type",
            "renal cell carcinoma, not further classifiable",
        ],
        "not_further_classifiable": "renal cell carcinoma, not further classifiable from ROI evidence",
    },
    "TCGA-KIRP": {
        "site": "kidney / renal tissue",
        "broad_type": "renal epithelial neoplasm",
        "subtype_candidates": [
            "papillary renal cell carcinoma",
            "papillary renal neoplasm",
            "renal cell carcinoma, not further classifiable",
        ],
        "not_further_classifiable": "renal cell carcinoma, not further classifiable from ROI evidence",
    },
    "TCGA-LGG": {
        "site": "brain / glial tissue",
        "broad_type": "glioma",
        "subtype_candidates": [
            "diffuse glioma",
            "astrocytoma",
            "oligodendroglioma",
            "oligoastrocytoma / mixed glioma",
            "glioma, not otherwise specified without molecular markers",
        ],
        "not_further_classifiable": "diffuse glioma, not further classifiable from ROI evidence without molecular markers",
    },
}


def known_site_for_tcga(cancer_type: str | None) -> str | None:
    if not cancer_type:
        return None
    return TCGA_SITE_MAP.get(str(cancer_type).strip().upper())


def diagnosis_prior_for_tcga(cancer_type: str | None) -> dict:
    if not cancer_type:
        return {}
    key = str(cancer_type).strip().upper()
    prior = TCGA_DIAGNOSIS_PRIORS.get(key)
    if prior:
        out = dict(prior)
        out["tcga_cancer_type"] = key
        return out
    site = known_site_for_tcga(key)
    return {
        "tcga_cancer_type": key,
        "site": site or "not specified",
        "broad_type": "not specified",
        "subtype_candidates": [],
        "not_further_classifiable": "not further classifiable from ROI evidence",
    }
