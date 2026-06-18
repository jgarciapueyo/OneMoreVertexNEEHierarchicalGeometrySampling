

import sys
import os 
import regex
import argparse
from pathlib import Path


if __name__ == "__main__":
    # read all .xml files in the input directory
    p = argparse.ArgumentParser(description="run mitsuba for each xml in the input directory.")
    p.add_argument("-i", "--input-dir", type=str, default=".", help="Directory containing .xml files.")
    p.add_argument("-b", "--base", type=str, default=".", help="Base scene xml file.")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    

    for filename in os.listdir(args.input_dir):
        if filename.endswith(".xml"):
            xml_path = os.path.join(args.input_dir, filename)
            print(f"Running Mitsuba for {xml_path}...")
            os.system(f"mitsuba {args.base} -Dgeometry={xml_path}")