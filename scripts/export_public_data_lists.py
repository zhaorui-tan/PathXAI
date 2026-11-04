#!/usr/bin/env python3
"""Export identifiers only from the study's local source inventories.

Requires the publisher's original manifests, not model weights or inference.
Run with --project-root, --seg-root, --report-inventory and --output.
No source dataset, split, report or model file is modified.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

PIXEL = {"BCSS", "GlaS", "DigestPath2019", "WSSS4LUAD", "ICIAR2018_WSI"}
ORIGINAL = "PathXAI_Seg2Report/results/tcga_so1_so4_a5bridge_508_20260920/manifests/patients.jsonl"
EXPANSION = "PathXAI_Seg2Report/results/tcga_expansion_20260924_1200_d1/manifests/patients.jsonl"
D1_RUN = "medsam3_combined_runs_4096_fusion_fixed_data_fixed2/stage3_ablation_group_d/stage3_ablation_D1/run_20260619_144518_stage3_tissue"
REFINED_RUN = "baseline_experiments/medsam3/medsam3_d1_refined/d1_final/run_20260804_140335_stage3_tissue"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(value):
    value = str(value).replace("\\", "/")
    if "/local/datasets/public/" in value:
        return value.split("/local/datasets/public/", 1)[1]
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise ValueError("Unrecognized/non-public source path; refuse export")
    return value


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path, records, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--seg-root", type=Path, required=True)
    parser.add_argument("--report-inventory", type=Path, required=True)
    parser.add_argument("--source-links", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    sources = []

    def source(path, root, kind):
        sources.append({"collection": kind, "source": str(path.relative_to(root)),
                        "bytes": path.stat().st_size, "sha256": sha(path)})

    # Report: source-case WSI associations and their frozen cohort membership.
    original = rows(args.project_root / ORIGINAL)
    expansion = rows(args.project_root / EXPANSION)
    assert len(original) == 508 and len(expansion) == 1200
    source(args.project_root / ORIGINAL, args.project_root, "report")
    source(args.project_root / EXPANSION, args.project_root, "report")
    source(args.report_inventory, args.report_inventory.parent, "report_inventory")
    inventory = {r["case_id"]: r for r in rows(args.report_inventory)}
    patients, slides, summaries = [], [], []
    seen_cases = set()
    for cohort, records in [("original508_development", original), ("new1200_confirmation", expansion)]:
        start = len(slides)
        for row in records:
            case = row["case_id"]
            assert re.fullmatch(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", case) and case not in seen_cases
            seen_cases.add(case)
            candidates = {r["metadata"]["slide_path"] for r in row.get("candidate_rois", [])}
            paths = sorted(set(inventory[case]["metadata"]["slide_paths"])) if cohort.startswith("original") else list(dict.fromkeys(row["slide_paths"]))
            assert candidates <= set(paths)
            if cohort.startswith("new"):
                assert set(paths) == set(inventory[case]["metadata"]["slide_paths"])
            patients.append({"cohort": cohort, "case_id": case, "project_id": row["cancer_type"],
                             "source_slide_count": len(paths), "candidate_slide_count": len(candidates) if candidates else ""})
            for path in paths:
                p = Path(path)
                assert p.name.startswith(case + "-") and p.suffix.lower() == ".svs"
                file_id = p.parent.name
                assert re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", file_id)
                material = re.search(r"-(DX|TS)[A-Z0-9]*(?:\.|$)", p.name)
                slides.append({"cohort": cohort, "case_id": case, "project_id": row["cancer_type"],
                               "file_name": p.name, "gdc_file_id": file_id,
                               "slide_type": material.group(1) if material else "other",
                               "download_url": "https://api.gdc.cancer.gov/data/" + file_id,
                               "in_frozen_roi_candidates": int(path in candidates) if cohort.startswith("original") else "",
                               "recorded_primary_slide": int(path == row.get("slide_path")) if cohort.startswith("new") else "",
                               "list_scope": "case_association_inventory" if cohort.startswith("original") else "frozen_source_wsi_inventory"})
        subset = slides[start:]
        summaries.append({"cohort": cohort, "patients": len(records), "slides": len(subset),
                          "projects": len({r["cancer_type"] for r in records}),
                          "slide_types": dict(Counter(r["slide_type"] for r in subset))})
        p = out / "report" / (cohort + ".txt")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(sorted(r["case_id"] for r in records)) + "\n")
    assert len(slides) == 4160
    write_csv(out / "report/patients.csv", patients, list(patients[0]))
    write_csv(out / "report/slides.csv", slides, list(slides[0]))
    project_counts = Counter((r["cohort"], r["project_id"]) for r in patients)
    write_csv(out / "report/project_counts.csv", [{"cohort": c, "project_id": p, "patients": n}
              for (c, p), n in sorted(project_counts.items())], ["cohort", "project_id", "patients"])

    # Also disclose the complete public screening inventory, separately from
    # the confirmed report cohorts. Do not export source report text.
    inventory_patients, inventory_slides = [], []
    for case, row in sorted(inventory.items()):
        assert re.fullmatch(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", case)
        paths = sorted(set(row["metadata"].get("slide_paths", [])))
        inventory_patients.append({"case_id": case, "project_id": row["metadata"]["cancer_type"],
                                   "listed_slide_count": len(paths), "in_report_study": int(case in seen_cases)})
        for path in paths:
            p = Path(path)
            assert p.name.startswith(case + "-") and p.suffix.lower() == ".svs"
            file_id = p.parent.name
            if not re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", file_id):
                file_id = ""
            inventory_slides.append({"case_id": case, "project_id": row["metadata"]["cancer_type"],
                                     "file_name": p.name, "gdc_file_id": file_id,
                                     "in_report_study": int(case in seen_cases),
                                     "download_url": "https://api.gdc.cancer.gov/data/" + file_id if file_id else ""})
    write_csv(out / "report/source_inventory_patients.csv", inventory_patients, list(inventory_patients[0]))
    write_csv(out / "report/source_inventory_slides.csv", inventory_slides, list(inventory_slides[0]))

    # Segmentation: preserve all recorded train/held-out list entries and order.
    counts, split_sets = [], {}
    for split_file in sorted((args.seg_root / "datasets/splits").glob("*.txt")):
        match = re.fullmatch(r"(.+)_(train|test)\.txt", split_file.name)
        if not match:
            continue
        dataset, role = match.groups()
        split = "train_pool" if role == "train" else "heldout_pool"
        raw = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        values = [portable(value) for value in raw]
        assert len(values) == len(set(values)), (dataset, split)
        split_sets[dataset, split] = set(raw)
        p = out / "seg" / split / (dataset + ".txt")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(values) + "\n")
        counts.append({"split": split, "dataset": dataset, "entries": len(values),
                       "unit": "array_index" if dataset in {"CoNIC2022", "PanNuke"} else "source_image",
                       "supervision": "pixel-level" if dataset in PIXEL else "auxiliary image-level"})
        source(split_file, args.seg_root, "seg")
    for dataset, split in split_sets:
        assert not (split_sets[dataset, "train_pool"] & split_sets[dataset, "heldout_pool"]), dataset

    # Both D1 runs refer to byte-identical fixed validation caches.
    cache_suffix = "setup_cache/stage3_setup_c48121970c817995.json"
    caches = [args.seg_root / "medsam3_combined_runs_4096_fusion_fixed_data_fixed2/stage3_ablation_group_d" / cache_suffix,
              args.seg_root / "baseline_experiments/medsam3/medsam3_d1_refined" / cache_suffix]
    assert sha(caches[0]) == sha(caches[1])
    for p in caches:
        source(p, args.seg_root, "seg_validation_cache")
    mapped = args.seg_root / "PathXSeg/sample_id_splits/manifest.tsv"
    source(mapped, args.seg_root, "seg_validation_identity_mapping")
    with mapped.open() as stream:
        validation = [r for r in csv.DictReader(stream, delimiter="\t") if r["split"] in {"val_model_selection", "val_holdout_monitor"}]
    assert Counter(r["split"] for r in validation) == {"val_model_selection": 56, "val_holdout_monitor": 34}
    clean = []
    for row in validation:
        pool = "train_pool" if row["split"] == "val_model_selection" else "heldout_pool"
        assert row["image_path"] in split_sets[row["dataset"], pool]
        clean.append({"split": row["split"], "dataset": row["dataset"], "image_id": portable(row["image_path"]), "subset_of": pool})
    write_csv(out / "seg/fixed_validation.csv", clean, list(clean[0]))
    for split in ("val_model_selection", "val_holdout_monitor"):
        for dataset in sorted({r["dataset"] for r in clean if r["split"] == split}):
            counts.append({"split": split, "dataset": dataset, "entries": sum(r["split"] == split and r["dataset"] == dataset for r in clean), "unit": "source_image", "supervision": "pixel-level" if dataset in PIXEL else "auxiliary image-level"})

    for scope, name in [("test_internal", "internal_eval_manifest_124_cases.csv"),
                        ("test_external", "external_patch_roi_eval_manifest.csv")]:
        p = args.seg_root / "PathXSeg/results/evaluation_manifests" / name
        source(p, args.seg_root, "seg_aligned_evaluation")
        with p.open() as stream:
            clean = [{"dataset": r["dataset"], "eval_index": r["eval_index"], "image_id": portable(r["image_path"]),
                      "mask_id": portable(r["mask_path"])} for r in csv.DictReader(stream)]
        write_csv(out / "seg" / (scope + ".csv"), clean, list(clean[0]))
        for dataset, n in sorted(Counter(r["dataset"] for r in clean).items()):
            counts.append({"split": scope, "dataset": dataset, "entries": n, "unit": "source_image_or_window", "supervision": "pixel-level evaluation"})

    for run in (D1_RUN, REFINED_RUN):
        source(args.seg_root / run / "train_config.json", args.seg_root, "seg_run_config")

    aggc = args.seg_root / "PathXSeg/results/evaluation_manifests/aggc2022_test_tumor_20w_seed42.csv"
    source(aggc, args.seg_root, "seg_aggc_window_evaluation")
    with aggc.open() as stream:
        windows = list(csv.DictReader(stream))
    assert len(windows) == 2539
    for row in windows:
        row["image_path"] = portable(row["image_path"])
        row["annotation_dir"] = portable(row["annotation_dir"])
    write_csv(out / "seg/aggc_windows.csv", windows, list(windows[0]))
    specimens = sorted({(r["subset"], r["specimen_id"]) for r in windows})
    assert len(specimens) == 73
    write_csv(out / "seg/aggc_specimens.csv", [{"subset": subset, "specimen_id": specimen} for subset, specimen in specimens], ["subset", "specimen_id"])
    counts.append({"split": "test_external", "dataset": "AGGC2022_Tumor", "entries": len(windows), "unit": "window (73 specimens; 128 slide IDs)", "supervision": "pixel-level evaluation"})

    camelyon = args.seg_root / REFINED_RUN / "test_results/external_wsi_best_model_d1_final_20260804_221602_CAMELYON16_WSI_win200_stride2048/test_seg_per_image_nocond.csv"
    source(camelyon, args.seg_root, "seg_supplemental_window_id_recovery")
    fields = ["dataset", "slide", "window_idx", "x", "y", "w", "h", "native_mag"]
    with camelyon.open() as stream:
        recovered = sorted({tuple(row[k] for k in fields) for row in csv.DictReader(stream)})
    assert len(recovered) == 9800 and len({r[1] for r in recovered}) == 49
    write_csv(out / "seg/supplemental_camelyon16_windows.csv", [dict(zip(fields, r)) for r in recovered], fields)
    counts.append({"split": "supplemental", "dataset": "CAMELYON16_WSI", "entries": len(recovered), "unit": "window (49 slides)", "supervision": "pixel-level evaluation"})
    write_csv(out / "seg/counts.csv", counts, ["split", "dataset", "entries", "unit", "supervision"])

    aliases = {"ICIAR2018_BACH_ROI+WSI": ["ICIAR2018", "ICIAR2018_WSI"],
               "AGGC2022": ["AGGC2022_Tumor"], "CAMELYON16": ["CAMELYON16_WSI"],
               "TIGER": ["TIGER_WSI"], "TCGA/GDC": ["TCGA"]}
    catalog = []
    if args.source_links.suffix == ".csv":
        with args.source_links.open() as stream:
            links = list(csv.DictReader(stream))
    else:
        links = json.loads(args.source_links.read_text())
    for row in links:
        for dataset in aliases.get(row["dataset"], [row["dataset"]]):
            roles = sorted({r["split"] for r in counts if r["dataset"] == dataset})
            if dataset == "TCGA":
                roles = ["report_study", "source_inventory"]
            if dataset == "TIGER_WSI":
                roles = ["excluded_unaligned_protocol"]
            catalog.append({"dataset": dataset, "roles": ";".join(roles),
                            "source_url": row["source_url"] or "", "verification_note": row["verification_note"]})
    assert {r["dataset"] for r in counts} <= {r["dataset"] for r in catalog}
    write_csv(out / "sources.csv", sorted(catalog, key=lambda r: r["dataset"]), list(catalog[0]))
    result = {"schema": "pathxai-public-data-source-inventory-v1", "report": summaries,
              "seg_pool_counts": {split: sum(r["entries"] for r in counts if r["split"] == split)
                                  for split in ["train_pool", "heldout_pool"]},
              "seg_dataset_count": len({d for d, _ in split_sets}),
              "report_source_inventory": {"patients": len(inventory_patients), "listed_slides": len(inventory_slides), "in_study_patients": len(seen_cases)},
              "sources": sources}
    (out / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
