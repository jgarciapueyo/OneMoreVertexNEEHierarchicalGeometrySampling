"""
pip install opencv-python numpy

Download the oidn binaries from https://www.openimagedenoise.org/downloads.html, 
and point the oidn_bin variable below to the oidnDenoise executable.

"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import OpenEXR
import Imath
from argparse import ArgumentParser
import subprocess
from pathlib import Path

REQUIRED_FEATURE_CHANNELS = (
    "albedo.R", "albedo.G", "albedo.B",
    "normals.R", "normals.G", "normals.B",
)

def read_exr_channel(exr_file, channel_name):
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channel_str = exr_file.channel(channel_name, pt)
    arr = np.frombuffer(channel_str, dtype=np.float32)
    return np.reshape(arr, (height, width))

def validate_feature_channels(exr_file):
    channels = exr_file.header().get("channels", {})
    available = set(channels.keys())
    missing = [name for name in REQUIRED_FEATURE_CHANNELS if name not in available]
    if missing:
        raise ValueError(f"Feature EXR is missing channels: {missing}")

def load_features_exr(filename):
    """Loads albedo and normals, safely formatted for OpenCV"""
    filename = str(Path(filename))
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Feature EXR not found: {filename}")

    exr_file = OpenEXR.InputFile(filename)
    validate_feature_channels(exr_file)

    # Extract Albedo
    albedo_r = read_exr_channel(exr_file, "albedo.R")
    albedo_g = read_exr_channel(exr_file, "albedo.G")
    albedo_b = read_exr_channel(exr_file, "albedo.B")
    
    # Extract Normals
    normals_x = read_exr_channel(exr_file, "normals.R")
    normals_y = read_exr_channel(exr_file, "normals.G")
    normals_z = read_exr_channel(exr_file, "normals.B")

    # Stack for OpenCV (cv2.imwrite assumes BGR format)
    # This guarantees they will be written to the PFM correctly as RGB/XYZ
    albedo_img = np.dstack((albedo_b, albedo_g, albedo_r))
    normals_img = np.dstack((normals_z, normals_y, normals_x)) 

    # Sanitize invalid float values (NaN / Inf)
    albedo_img = np.nan_to_num(albedo_img, nan=0.0, posinf=1.0, neginf=0.0)
    normals_img = np.nan_to_num(normals_img, nan=0.0, posinf=1.0, neginf=-1.0)

    return albedo_img, normals_img


def split_features_exr(feature_exr, output_dir):
    """Split a features EXR into albedo.exr and normals.exr files."""
    albedo_img, normals_img = load_features_exr(feature_exr)
    output_dir.mkdir(parents=True, exist_ok=True)

    albedo_out = output_dir / "albedo.exr"
    normals_out = output_dir / "normals.exr"

    if not cv2.imwrite(str(albedo_out), np.ascontiguousarray(albedo_img, dtype=np.float32)):
        raise RuntimeError(f"Failed to write {albedo_out}")
    if not cv2.imwrite(str(normals_out), np.ascontiguousarray(normals_img, dtype=np.float32)):
        raise RuntimeError(f"Failed to write {normals_out}")

    print(f"Wrote split features:\n- {albedo_out}\n- {normals_out}")

def denoise_binary(input_exr, output_exr, input_albedonormals=None, verify_aux_usage=False, input_scale=5.0):
    feature_exr = input_albedonormals if input_albedonormals is not None else input_exr
    
    oidn_bin = "/home/nestor/bdpt_twopoints_mitsuba/oidn_bin/oidn-2.4.1.x86_64.linux/bin/oidnDenoise"
    if not os.path.exists(oidn_bin):
        raise FileNotFoundError(f"Could not find the oidnDenoise executable!")

    # 1. Load Noisy Image using OpenCV
    print(f"Loading noisy input: {input_exr}")
    img = cv2.imread(str(input_exr), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise FileNotFoundError(f"Could not load {input_exr}")
        
    # Do NOT convert to RGB here! Leave it as BGR so cv2.imwrite handles it correctly.
    # Just sanitize the path-tracer fireflies.
    img = np.nan_to_num(img, nan=0.0, posinf=65504.0, neginf=0.0)

    # 2. Load Features
    albedo, normals = load_features_exr(feature_exr)

    # Ensure float32 continuity
    img = np.ascontiguousarray(img, dtype=np.float32)
    albedo = np.ascontiguousarray(albedo, dtype=np.float32)
    normals = np.ascontiguousarray(normals, dtype=np.float32)

    temp_in = "temp_noisy.pfm"
    temp_albedo = "temp_albedo.pfm"
    temp_normals = "temp_normals.pfm"
    temp_out = "temp_clean.pfm"

    # Write temporary files (OpenCV will natively convert these BGR arrays to RGB in the PFM)
    cv2.imwrite(temp_in, img)
    cv2.imwrite(temp_albedo, albedo)
    cv2.imwrite(temp_normals, normals)

    print(f"Denoising via Intel CLI...")
    cmd = [
        oidn_bin, 
        "-hdr", temp_in, 
        "-alb", temp_albedo,
        "-nrm", temp_normals,
        "-o", temp_out
    ]

    print(f"Shapes of all images: input={img.shape}, albedo={albedo.shape}, normals={normals.shape}")
    
    # Add input scale if it's supported and not default
    if input_scale != 1.0:
        cmd.insert(3, "--is")
        cmd.insert(4, str(input_scale))

    try:
        subprocess.run(cmd, check=True, text=True, capture_output=True)

        print(f"Converting output to {output_exr}...")
        denoised_img = cv2.imread(temp_out, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        
        # Save final result. Because we want to save an EXR, OpenCV's default BGR handling is perfect here.
        cv2.imwrite(str(output_exr), denoised_img)
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"OIDN failed:\n{e.stderr}") from e
    finally:
        print("Cleaning up temporary files...")
        for temp_file in (temp_in, temp_albedo, temp_normals, temp_out):
            if os.path.exists(temp_file):
                os.remove(temp_file)
    print("Done!")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base", help="Input base folder")
    parser.add_argument("--input", help="Input technique name")
    parser.add_argument(
        "--features",
        help="Optional EXR containing albedo.* and normals.* channels (defaults to input EXR)",
        default=None,
    )
    parser.add_argument(
        "--verify-aux-usage",
        action="store_true",
        help="Run a second no-aux denoise and print output difference metrics",
    )
    parser.add_argument(
        "--input-scale",
        type=float,
        default=5.0,
        help="OIDN input scale for HDR values (default: 5.0)",
    )
    parser.add_argument(
        "--split-features-only",
        action="store_true",
        help="Split --features EXR into albedo.exr and normals.exr under this script's images/ folder and exit",
    )
    # parser.add_argument("--output", help="Output EXR file")
    args = parser.parse_args()

    features = Path(args.features) if args.features else None

    if args.split_features_only:
        if features is None:
            raise ValueError("--split-features-only requires --features <path-to-features.exr>")
        script_images_dir = Path(__file__).resolve().parent / "images"
        split_features_exr(features, script_images_dir)
        raise SystemExit(0)

    input_exr = Path(args.base) / args.input / "runs/spp_0008_rep_002.exr"
    output = Path(args.base) / args.input / "denoised_spp_0008_rep_002.exr" 

    print(f"Denoising {input_exr} to {output} using features from {features if features else 'input EXR'}...\n")
    denoise_binary(
        input_exr,
        output,
        features,
        verify_aux_usage=args.verify_aux_usage,
        input_scale=args.input_scale,
    )

