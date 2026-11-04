"""Versioned, method-blind terminology for report-subtype evaluation.

This is a bounded research terminology, not a complete WHO/NCIt ontology.
Equivalence, ancestry, neutral organ context and TCGA membership are separate.
No prediction is completed using the histology implied by a TCGA project.
"""
from __future__ import annotations

import re

from . import eval_v3 as legacy

VERSION = "subtype_terminology_20261016_r2"
SOURCES = (
    "https://www.cancer.gov/publications/dictionaries/cancer-terms/def/conventional-renal-cell-carcinoma",
    "https://www.cancer.gov/types/cervical",
    "https://www.cancer.gov/ccg/research/genome-sequencing/tcga/studied-cancers/breast-lobular-carcinoma-study",
    "https://training.seer.cancer.gov/breast/types.html",
)

ENTITIES: dict[str, dict] = {}


def _add(name, *aliases, parents=(), organs=(), specific=True):
    ENTITIES[name] = dict(aliases=tuple(dict.fromkeys((name, *aliases))),
                          parents=tuple(parents), organs=tuple(organs),
                          specific=bool(specific))


for _name, _aliases in legacy.ENTITIES.items():
    _add(_name, *_aliases)

_add("carcinoma", specific=False)
_add("adenocarcinoma", parents=("carcinoma",), specific=False)
_add("invasive carcinoma", parents=("carcinoma",), specific=False)
_add("breast carcinoma", "breast cancer", "mammary carcinoma",
     parents=("carcinoma",), organs=("breast",), specific=False)
_add("invasive breast carcinoma", "breast invasive carcinoma", "invasive breast cancer",
     "infiltrating breast carcinoma", parents=("breast carcinoma", "invasive carcinoma"),
     organs=("breast",), specific=False)
_add("invasive ductal carcinoma", "infiltrating ductal carcinoma", "infiltrating duct carcinoma",
     "invasive duct carcinoma", "ductal carcinoma nst", "idc", "invasive carcinoma nst",
     "invasive carcinoma of no special type", "invasive breast carcinoma no special type",
     "invasive breast carcinoma of no special type", "invasive ductal cancer",
     parents=("invasive breast carcinoma",), organs=("breast",))
_add("invasive lobular carcinoma", "infiltrating lobular carcinoma", "ilc",
     parents=("invasive breast carcinoma",), organs=("breast",))
_add("mixed ductal lobular carcinoma", "mixed invasive ductal and lobular carcinoma",
     "invasive carcinoma with ductal and lobular features",
     parents=("invasive breast carcinoma",), organs=("breast",))
_add("ductal carcinoma in situ", "dcis",
     parents=("breast carcinoma",), organs=("breast",))
_add("lobular carcinoma in situ", "lcis", parents=("breast carcinoma",), organs=("breast",))
_add("renal cell carcinoma", "rcc", "renal carcinoma", parents=("carcinoma",),
     organs=("kidney",), specific=False)
_add("clear cell renal cell carcinoma", "clear cell rcc", "ccrcc", "cc rcc",
     "conventional renal cell carcinoma", "clear cell renal carcinoma", "renal clear cell carcinoma",
     "renal cell carcinoma clear cell type", "renal cell carcinoma of clear cell type",
     "renal cell carcinoma clear cell subtype", "renal cell carcinoma clear cell",
     "clear cell carcinoma of the kidney", "clear cell renal cell cancer",
     parents=("renal cell carcinoma",), organs=("kidney",))
_add("papillary renal cell carcinoma", "papillary rcc", "renal papillary carcinoma",
     "renal cell carcinoma papillary type", "renal cell carcinoma of papillary type",
     "renal cell carcinoma papillary subtype", "papillary renal carcinoma",
     parents=("renal cell carcinoma",), organs=("kidney",))
_add("chromophobe renal cell carcinoma", "chromophobe rcc", "renal chromophobe carcinoma",
     "renal cell carcinoma chromophobe type", "renal cell carcinoma chromophobe subtype",
     parents=("renal cell carcinoma",), organs=("kidney",))
_add("collecting duct carcinoma", "bellini duct carcinoma", "carcinoma of collecting ducts",
     parents=("renal cell carcinoma",), organs=("kidney",))
_add("squamous cell carcinoma", "squamous carcinoma", "scc", "epidermoid carcinoma",
     "squamous cell cancer", parents=("carcinoma",))
_add("cervical adenocarcinoma", "endocervical adenocarcinoma", "cervical glandular adenocarcinoma", "adenocarcinoma of the cervix",
     "adenocarcinoma of cervix", "adenocarcinoma of the endocervix",
     parents=("adenocarcinoma",), organs=("cervix",))
_add("adenosquamous carcinoma", parents=("carcinoma",))
_add("small cell neuroendocrine carcinoma", "small cell carcinoma", "neuroendocrine small cell carcinoma",
     "small cell neuroendocrine cancer", parents=("carcinoma",))
_add("large cell neuroendocrine carcinoma", parents=("carcinoma",))
_add("neuroendocrine carcinoma", parents=("carcinoma",), specific=False)
_add("urothelial carcinoma", "transitional cell carcinoma", "urothelial cancer",
     parents=("carcinoma",), organs=("urinary",))
_add("prostate adenocarcinoma", "prostatic adenocarcinoma", "prostatic acinar adenocarcinoma",
     "acinar adenocarcinoma of the prostate", "prostate acinar adenocarcinoma",
     parents=("adenocarcinoma",), organs=("prostate",))
_add("benign prostatic hyperplasia", "bph", "prostatic glandular hyperplasia",
     "active hyperplasia of prostatic glands", "benign prostate hyperplasia", organs=("prostate",))
_add("benign prostate tissue", "benign prostatic tissue", organs=("prostate",))
_add("colorectal adenocarcinoma", "colonic adenocarcinoma", "colon adenocarcinoma",
     "rectal adenocarcinoma", "adenocarcinoma of the colon", "adenocarcinoma of the rectum",
     "adenocarcinoma of colon", "adenocarcinoma of rectum",
     parents=("adenocarcinoma",), organs=("colorectal",))
_add("lung adenocarcinoma", "pulmonary adenocarcinoma", "adenocarcinoma of the lung",
     "adenocarcinoma of lung", parents=("adenocarcinoma",), organs=("lung",))
_add("gastric adenocarcinoma", "stomach adenocarcinoma", "adenocarcinoma of the stomach",
     "adenocarcinoma of stomach", "adenocarcinoma of the gastric mucosa",
     parents=("adenocarcinoma",), organs=("stomach",))
_add("esophageal adenocarcinoma", "oesophageal adenocarcinoma", "adenocarcinoma of the esophagus",
     "adenocarcinoma of esophagus", parents=("adenocarcinoma",), organs=("esophagus",))
_add("pancreatic ductal adenocarcinoma", "pancreatic adenocarcinoma", "pdac",
     "ductal adenocarcinoma of the pancreas", parents=("adenocarcinoma",), organs=("pancreas",))
_add("endometrial adenocarcinoma", "adenocarcinoma of the endometrium",
     parents=("adenocarcinoma",), organs=("endometrium",))
_add("endometrial carcinoma", "uterine carcinoma", parents=("carcinoma",),
     organs=("endometrium",), specific=False)
_add("ovarian carcinoma", "ovarian cancer", parents=("carcinoma",), organs=("ovary",), specific=False)
_add("ovarian clear cell carcinoma", "clear cell carcinoma of the ovary", "ovarian clear cell adenocarcinoma",
     parents=("ovarian carcinoma",), organs=("ovary",))
_add("endometrial clear cell carcinoma", "clear cell carcinoma of the endometrium",
     parents=("endometrial carcinoma",), organs=("endometrium",))
_add("serous carcinoma", "serous adenocarcinoma", "high grade serous carcinoma",
     "high grade serous ovarian carcinoma", "ovarian serous carcinoma", "uterine serous carcinoma",
     parents=("carcinoma",))
_add("endometrioid carcinoma", "endometrioid adenocarcinoma", parents=("adenocarcinoma",))
_add("mucinous carcinoma", "mucinous adenocarcinoma", "colloid carcinoma", parents=("adenocarcinoma",))
_add("adrenocortical neoplasm", "adrenal cortical neoplasm", "cortical adrenal tumor", specific=False)
_add("adrenocortical carcinoma", "adrenal cortical carcinoma", parents=("adrenocortical neoplasm", "carcinoma"),
     organs=("adrenal",))
_add("hepatocellular carcinoma", "hcc", "hepatocellular cancer", parents=("carcinoma",), organs=("liver",))
_add("cholangiocarcinoma", "intrahepatic cholangiocarcinoma", "bile duct adenocarcinoma",
     "biliary adenocarcinoma", parents=("adenocarcinoma",), organs=("biliary",))
_add("papillary thyroid carcinoma", "thyroid papillary carcinoma", "papillary carcinoma of the thyroid",
     "papillary carcinoma of thyroid", "papillary thyroid cancer", parents=("carcinoma",), organs=("thyroid",))
_add("follicular thyroid carcinoma", "thyroid follicular carcinoma", "follicular carcinoma of thyroid",
     parents=("carcinoma",), organs=("thyroid",))
_add("medullary thyroid carcinoma", "medullary carcinoma of thyroid", parents=("carcinoma",), organs=("thyroid",))
_add("anaplastic thyroid carcinoma", "anaplastic carcinoma of thyroid", parents=("carcinoma",), organs=("thyroid",))
_add("glioma", "glial neoplasm", specific=False, organs=("cns",))
_add("diffuse glioma", parents=("glioma",), specific=False, organs=("cns",))
_add("lower grade glioma", "low grade glioma", parents=("glioma",), specific=False, organs=("cns",))
_add("astrocytoma", "diffuse astrocytoma", "anaplastic astrocytoma", parents=("diffuse glioma",), organs=("cns",))
_add("oligodendroglioma", "anaplastic oligodendroglioma", parents=("diffuse glioma",), organs=("cns",))
_add("glioblastoma", "glioblastoma multiforme", "gbm", parents=("glioma",), organs=("cns",))
_add("melanoma", "malignant melanoma", "cutaneous melanoma", "uveal melanoma", "choroidal melanoma", specific=False)
_add("nodular melanoma", "nodular malignant melanoma", parents=("melanoma",))
_add("superficial spreading melanoma", "superficial spreading malignant melanoma", parents=("melanoma",))
_add("spitzoid melanoma", "spitz like melanoma", parents=("melanoma",))
_add("lentigo maligna melanoma", parents=("melanoma",))
_add("acral lentiginous melanoma", parents=("melanoma",))
_add("sarcoma", specific=False)
for _name, _aliases in {
    "leiomyosarcoma": (), "liposarcoma": (), "myxofibrosarcoma": (),
    "undifferentiated pleomorphic sarcoma": ("malignant fibrous histiocytoma",),
    "synovial sarcoma": (), "angiosarcoma": (), "rhabdomyosarcoma": (),
    "malignant peripheral nerve sheath tumor": ("mpnst",),
}.items():
    _add(_name, *_aliases, parents=("sarcoma",))
_add("carcinosarcoma", "malignant mixed mullerian tumor", "mmmt", "uterine carcinosarcoma",
     parents=("carcinoma", "sarcoma"))
_add("germ cell tumor", "germ cell tumour", "germ cell neoplasm", specific=False)
_add("mixed germ cell tumor", "mixed germ cell tumour", parents=("germ cell tumor",))
for _name, _aliases in {"seminoma": (), "embryonal carcinoma": (), "yolk sac tumor": ("yolk sac tumour",),
                         "teratoma": ()}.items():
    _add(_name, *_aliases, parents=("germ cell tumor",))
_add("diffuse large b cell lymphoma", "dlbcl", "large b cell lymphoma", "diffuse large b cell non hodgkin lymphoma")
_add("mesothelioma", "malignant mesothelioma")
_add("thymoma", organs=("thymus",))
_add("thymic carcinoma", parents=("carcinoma",), organs=("thymus",))
_add("pheochromocytoma", "phaeochromocytoma", organs=("adrenal",))
_add("paraganglioma", organs=("paraganglia",))

# This mapping contains anatomy only: siblings share exactly the same context.
TCGA_ORGANS = dict(ACC="adrenal", PCPG="adrenal_paraganglia", BLCA="urinary", BRCA="breast",
                   CESC="cervix", COAD="colorectal", READ="colorectal", ESCA="esophagus",
                   STAD="stomach", HNSC="head_neck", KICH="kidney", KIRC="kidney", KIRP="kidney",
                   CHOL="biliary", LIHC="liver", LUAD="lung", LUSC="lung", MESO="pleura",
                   LGG="cns", GBM="cns", OV="ovary", UCEC="endometrium", UCS="endometrium",
                   PRAD="prostate", SKCM="skin", SARC="soft_tissue", TGCT="testis", THCA="thyroid",
                   THYM="thymus", UVM="eye", DLBC="lymphoid")

ORGAN_ALIASES = {
    "kidney": ("kidney", "renal"), "breast": ("breast", "mammary"),
    "cervix": ("cervix", "cervical", "endocervix", "endocervical"),
    "colorectal": ("colorectal", "colon", "colonic", "rectum", "rectal"),
    "lung": ("lung", "pulmonary"), "stomach": ("stomach", "gastric"),
    "esophagus": ("esophagus", "esophageal", "oesophagus", "oesophageal"),
    "prostate": ("prostate", "prostatic"), "ovary": ("ovary", "ovarian"),
    "endometrium": ("endometrium", "endometrial", "uterus", "uterine"),
    "thyroid": ("thyroid",), "pancreas": ("pancreas", "pancreatic"),
    "liver": ("liver", "hepatic", "hepatocellular"), "biliary": ("biliary", "bile duct"),
    "adrenal": ("adrenal", "adrenocortical"), "cns": ("brain", "cns", "cerebral"),
    "urinary": ("bladder", "urothelial", "urinary"), "skin": ("skin", "cutaneous"),
    "eye": ("eye", "uveal", "choroidal"), "pleura": ("pleura", "pleural"),
    "thymus": ("thymus", "thymic"), "testis": ("testis", "testicular"),
    "paraganglia": ("paraganglia", "paraganglial"), "soft_tissue": ("soft tissue",),
    "head_neck": ("head and neck", "head neck"), "lymphoid": ("lymph node", "lymphoid", "hematologic"),
}

CONTEXTUAL_ALIASES = {
    "breast": {"invasive carcinoma": "invasive breast carcinoma", "intraductal carcinoma": "ductal carcinoma in situ"},
    "cervix": {"adenocarcinoma": "cervical adenocarcinoma"},
    "colorectal": {"adenocarcinoma": "colorectal adenocarcinoma"},
    "lung": {"adenocarcinoma": "lung adenocarcinoma"},
    "stomach": {"adenocarcinoma": "gastric adenocarcinoma"},
    "esophagus": {"adenocarcinoma": "esophageal adenocarcinoma"},
    "prostate": {"adenocarcinoma": "prostate adenocarcinoma", "acinar carcinoma": "prostate adenocarcinoma"},
    "endometrium": {"adenocarcinoma": "endometrial adenocarcinoma"},
    "thyroid": {"papillary carcinoma": "papillary thyroid carcinoma", "follicular carcinoma": "follicular thyroid carcinoma"},
}

# Historical benchmark families are NOT synonym sets or patient histologies.
COHORT_MEMBERS = {
    "ACC": ("adrenocortical carcinoma",), "PCPG": ("pheochromocytoma", "paraganglioma"),
    "BLCA": ("urothelial carcinoma",), "BRCA": ("invasive breast carcinoma",),
    "CESC": ("squamous cell carcinoma", "cervical adenocarcinoma"),
    "COAD": ("colorectal adenocarcinoma",), "READ": ("colorectal adenocarcinoma",),
    "ESCA": ("squamous cell carcinoma", "esophageal adenocarcinoma"),
    "STAD": ("gastric adenocarcinoma",), "HNSC": ("squamous cell carcinoma",),
    "KIRC": ("clear cell renal cell carcinoma",), "KIRP": ("papillary renal cell carcinoma",),
    "KICH": ("chromophobe renal cell carcinoma",), "CHOL": ("cholangiocarcinoma",),
    "LIHC": ("hepatocellular carcinoma",), "LUAD": ("lung adenocarcinoma",),
    "LUSC": ("squamous cell carcinoma",), "MESO": ("mesothelioma",),
    "LGG": ("astrocytoma", "oligodendroglioma", "diffuse glioma", "lower grade glioma"),
    "GBM": ("glioblastoma",),
    "OV": ("ovarian carcinoma", "serous carcinoma"),
    "UCEC": ("endometrial carcinoma", "endometrial adenocarcinoma", "endometrioid carcinoma", "serous carcinoma"),
    "UCS": ("carcinosarcoma",), "PRAD": ("prostate adenocarcinoma",),
    "SKCM": ("melanoma",), "UVM": ("melanoma",), "SARC": ("sarcoma",),
    "TGCT": ("germ cell tumor",), "THCA": ("papillary thyroid carcinoma", "follicular thyroid carcinoma"),
    "THYM": ("thymoma", "thymic carcinoma"), "DLBC": ("diffuse large b cell lymphoma",),
}


def project_code(cancer_type):
    return str(cancer_type or "").strip().upper().removeprefix("TCGA-")


def organs_in(text):
    text = legacy.normalize(text)
    return {organ for organ, aliases in ORGAN_ALIASES.items()
            if any(legacy.contains(text, alias) for alias in aliases)}


def organ_context(cancer_type=None, site=None):
    code = project_code(cancer_type)
    if code in TCGA_ORGANS:
        return TCGA_ORGANS[code]
    found = organs_in(str(site or ""))
    return next(iter(found)) if len(found) == 1 else None


def ancestors(entity):
    result, pending = set(), list(ENTITIES.get(entity, {}).get("parents", ()))
    while pending:
        parent = pending.pop()
        if parent not in result:
            result.add(parent)
            pending.extend(ENTITIES.get(parent, {}).get("parents", ()))
    return result


def is_a(entity, parent):
    return entity == parent or parent in ancestors(entity)


def is_specific(entity):
    return bool(ENTITIES.get(entity, {}).get("specific", False))


def _compile_aliases():
    result = []
    for entity, item in ENTITIES.items():
        for alias in item["aliases"]:
            result.append((entity, alias, re.compile(r"(?<!\w)" + re.escape(legacy.normalize(alias)) + r"(?!\w)")))
    return tuple(result)


ALIASES = _compile_aliases()
