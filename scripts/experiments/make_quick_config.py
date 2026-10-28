"""Write a reduced copy of an experiment config for a fast test run.

The copy keeps the techniques and scenes but uses a short spp schedule, one
repetition, a low-spp reference and a separate output folder (<output_root>_quick).
Prints the path of the new config.

    python3 scripts/experiments/make_quick_config.py scripts/experiments/ablation_asteroids3_ppg.yaml
"""
import argparse
import tempfile
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--max-spp", type=int, default=16, help="largest spp of the schedule")
    ap.add_argument("--reference-spp", type=int, default=256)
    ap.add_argument("--out-dir", default=tempfile.gettempdir())
    args = ap.parse_args()

    path = Path(args.config)
    exp = yaml.safe_load(path.read_text(encoding="utf-8"))

    schedule = [s for s in exp.get("spp_schedule", []) if s <= args.max_spp] or [1]
    exp["spp_schedule"] = schedule
    exp["repetitions_per_spp"] = 1
    exp["output_root"] = str(exp["output_root"]).rstrip("/") + "_quick"
    if "reference_spp" in exp:
        exp["reference_spp"] = args.reference_spp
    ref = exp.get("reference") or {}
    if "spp" in ref and not ref.get("pre_rendered_exr"):
        ref["spp"] = args.reference_spp

    out = Path(args.out_dir) / (path.stem + "_quick.yaml")
    out.write_text(yaml.safe_dump(exp, sort_keys=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
