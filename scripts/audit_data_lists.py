#!/usr/bin/env python3
"""Check public list membership, counts, portable paths and release hashes."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data_lists"


def table(relative):
    with (ROOT / relative).open() as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def report_checks():
    patients = table("report/patients.csv")
    ids = {r["case_id"] for r in patients}
    expected = {"public_development456": 456, "public_evaluation1200": 1200}
    require(len(ids) == len(patients) == 1656, "Report release must have 1656 unique cases")
    require(Counter(r["cohort"] for r in patients) == expected, "Final report group counts differ")
    require("TCGA-AA-3939" not in ids, "Excluded corrupt case in final release")
    require({p.stem for p in (ROOT / "report").glob("*.txt")} == set(expected), "Unexpected or missing report case list")
    require(not list((ROOT / "report").glob("source_inventory*")), "Obsolete screening inventory remains")
    for cohort in expected:
        values = (ROOT / "report" / (cohort + ".txt")).read_text().splitlines()
        require(len(values) == len(set(values)) == expected[cohort], "Duplicate or incorrect list length")
        require(set(values) == {r["case_id"] for r in patients if r["cohort"] == cohort}, "TXT/CSV membership differs")
    slides = table("report/slides.csv")
    require(len(slides) == len({r["gdc_file_id"] for r in slides}) == 4055, "Listed WSI count/uniqueness differs")
    require({r["case_id"] for r in slides} == ids, "WSI case coverage differs")
    require(sum(r["in_frozen_roi_candidates"] == "1" for r in slides) == 745, "Development candidate-slide count differs")
    require(sum(r["recorded_primary_slide"] == "1" for r in slides) == 1200, "Evaluation primary-slide count differs")
    per_case = Counter(r["case_id"] for r in slides)
    indexed = {r["case_id"]: r for r in patients}
    for row in patients:
        require(per_case[row["case_id"]] == int(row["source_slide_count"]), "Patient WSI count differs")
        require(re.fullmatch(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", row["case_id"]), "Non-TCGA identifier")
    for row in slides:
        require(row["file_name"].startswith(row["case_id"] + "-"), "WSI filename/case identity differs")
        require(row["cohort"] == indexed[row["case_id"]]["cohort"], "Slide group mismatch")
        require(row["project_id"] == indexed[row["case_id"]]["project_id"], "Slide project mismatch")
        require(row["download_url"] == "https://api.gdc.cancer.gov/data/" + row["gdc_file_id"], "GDC URL differs from ID")
    actual_counts = {(r["cohort"],r["project_id"]): int(r["patients"]) for r in table("report/project_counts.csv")}
    require(actual_counts == Counter((r["cohort"],r["project_id"]) for r in patients), "Project counts differ")


def segmentation_checks():
    counts = table("seg/counts.csv")
    pools = {}
    for split, expected_total in [("train_pool", 360122), ("heldout_pool", 90040)]:
        files = list((ROOT / "seg" / split).glob("*.txt"))
        require(len(files) == 24, "Segmentation pool dataset count differs")
        total = 0
        for p in files:
            values = p.read_text().splitlines()
            require(len(values) == len(set(values)), "Duplicate split ID: " + p.name)
            pools[p.stem, split] = set(values)
            total += len(values)
            expected = [int(r["entries"]) for r in counts if r["split"] == split and r["dataset"] == p.stem]
            require(expected == [len(values)], "Split count table differs: " + p.name)
        require(total == expected_total, "Segmentation pool size differs")
    for dataset, split in pools:
        require(not pools[dataset, "train_pool"] & pools[dataset, "heldout_pool"], "Within-dataset train/heldout overlap: " + dataset)
    validation = table("seg/fixed_validation.csv")
    require(Counter(r["split"] for r in validation) == {"val_model_selection": 56, "val_holdout_monitor": 34}, "Fixed validation counts differ")
    for row in validation:
        require(row["image_id"] in pools[row["dataset"], row["subset_of"]], "Fixed validation outside declared pool")
    for name, size in [("test_internal", 124), ("test_external", 11016)]:
        records = table("seg/" + name + ".csv")
        require(len(records) == size, "Aligned evaluation count differs: " + name)
        require(len({(r["dataset"], r["image_id"]) for r in records}) == size, "Duplicate aligned image identifier")
        for dataset, n in Counter(r["dataset"] for r in records).items():
            require([int(r["entries"]) for r in counts if r["split"] == name and r["dataset"] == dataset] == [n], "Aligned count table differs")
    windows = table("seg/aggc_windows.csv")
    keys = ["subset", "specimen_id", "slide_id", "window_idx", "x", "y", "w", "h"]
    require(len(windows) == len({tuple(r[k] for k in keys) for r in windows}) == 2539, "AGGC window count/identity differs")
    specimens = {(r["subset"], r["specimen_id"]) for r in windows}
    require(len(specimens) == 73 and len({r["slide_id"] for r in windows}) == 128, "AGGC source units differ")
    require(specimens == {(r["subset"], r["specimen_id"]) for r in table("seg/aggc_specimens.csv")}, "AGGC specimen list differs")
    camelyon = table("seg/supplemental_camelyon16_windows.csv")
    require(len(camelyon) == len({tuple(r.values()) for r in camelyon}) == 9800, "CAMELYON window count/identity differs")
    require(set(Counter(r["slide"] for r in camelyon).values()) == {200} and len({r["slide"] for r in camelyon}) == 49, "CAMELYON slide coverage differs")
    require({r["dataset"] for r in counts} <= {r["dataset"] for r in table("sources.csv")}, "Dataset missing from source index")


def file_inventory(report_only=False):
    output = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == ROOT / "manifest.json":
            continue
        if report_only and path.relative_to(ROOT).parts[0] == "seg":
            continue
        require(not path.is_symlink(), "Unexpected symlink in public lists")
        data = path.read_bytes()
        text = data.decode("utf-8")
        require(not any(marker in text for marker in ("/local/", "/home/", "/mnt/", "172.20.", '"report_text":', '"full_report_text":')), "Machine/private content marker: " + str(path.relative_to(ROOT)))
        require(not re.search(r"gh[pousr]_[A-Za-z0-9]{20,}", text), "Credential-like string in data list")
        require(len(data) < 100 * 1024 * 1024, "File exceeds GitHub file limit")
        row = {"path": str(path.relative_to(ROOT)), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if path.suffix == ".csv":
            row["records"] = len(list(csv.DictReader(text.splitlines())))
        elif path.suffix == ".txt":
            row["records"] = len(text.splitlines())
            require(all(not line.startswith("/") and ".." not in Path(line).parts for line in text.splitlines()), "Non-portable sample ID")
        output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true", help="Publisher: record hashes after structural checks pass")
    parser.add_argument("--report-only", action="store_true", help="Check final public report lists without segmentation files")
    args = parser.parse_args()
    if args.report_only and args.write_manifest:
        parser.error("--write-manifest requires the complete checkout")
    report_checks()
    if not args.report_only:
        segmentation_checks()
    inventory = file_inventory(args.report_only)
    if args.write_manifest:
        (ROOT / "manifest.json").write_text(json.dumps({"schema": "pathxai-public-data-lists-v2", "files": inventory}, indent=2) + "\n")
    expected = json.loads((ROOT / "manifest.json").read_text())["files"]
    if args.report_only:
        expected = [r for r in expected if not r["path"].startswith("seg/")]
    require(expected == inventory, "Released list files differ from manifest")
    print(f"Verified {len(inventory)} public-list files, final report membership and identities{' plus segmentation checks' if not args.report_only else ''}.")


if __name__ == "__main__":
    main()
