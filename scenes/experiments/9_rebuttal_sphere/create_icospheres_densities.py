"""
Creates a bunch of icosphere geometries with different densities and number of subdivisions, and saves them as .obj files. The resulting .obj files are used in the "9_rebuttal_sphere" scene to test the effect of geometry density on the rendering results.

"""

import argparse
import subprocess
import sys
from pathlib import Path

SUBDIVISIONS = [5, 10, 20, 40, 60, 100]
# AREA_RATIOS = [.05, .1, .2, .35, .5, .75, .9]
AREA_RATIOS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, .9]
# ^ log spaced area ratios to get a wider range of densities, especially in the lower density range.



if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Create icosphere geometries with different densities and number of subdivisions.")
    
    p.add_argument("-o", "--output-dir",  type=str,   default=".",   help="Directory for output files.")

    p.add_argument("-p", "--name-prefix", type=str,   default="hemisphere", help="Filename prefix.")

    args = p.parse_args()
    this_script_path = Path(__file__).resolve()
    generator_script = this_script_path.parent / "create_icosphere.py"

    for i, s in enumerate(SUBDIVISIONS):
        for j, a in enumerate(AREA_RATIOS):
            name_prefix = f"{args.name_prefix}_subdiv{s}_area{a:.2f}"
            cmd = [
                sys.executable,
                str(generator_script),
                "--frequency",
                str(s),
                "--ratio",
                str(a),
                "--output-dir",
                args.output_dir,
                "--name-prefix",
                name_prefix,
            ]
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

