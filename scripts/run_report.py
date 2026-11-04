#!/usr/bin/env python3
"""CLI equivalent of the Jupyter notebook, useful for durable long runs."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathxai_tutorial.inference import make_config, run_report, check_api

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="Non-secret make_config arguments as JSON; credentials come from environment")
    p.add_argument("--slide", action="append", required=True)
    p.add_argument("--case-id", required=True)
    p.add_argument("--version", choices=["SO1", "SO4"], default="SO1")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--known-site", default="")
    p.add_argument("--max-seconds", type=int, default=3600)
    p.add_argument("--check-api", action="store_true")
    args = p.parse_args()
    values = json.loads(args.config.read_text())
    if "api_key" in values or "hf_token" in values:
        p.error("Credentials must come from QWEN_API_KEY and HF_TOKEN, not a JSON file")
    config = make_config(**values)
    if args.check_api:
        probe = check_api(config)
        print(json.dumps(probe))
        if not probe["ok"]:
            return 1
    result = run_report(config, slide_paths=args.slide, case_id=args.case_id,
                        version=args.version, output_dir=args.output,
                        known_site=args.known_site, max_seconds=args.max_seconds)
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

