"""
pip install opencv-python numpy pandas matplotlib
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Enable OpenEXR in OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# --- CONFIGURATION ---
IGNORE_TOP_HALF = True  # Set to True to crop top half (e.g., ignore ceiling/sky)
SAVE_CROPS = False      # Save the cropped versions to disk for verification
HIST_BINS = 100         # Number of bins for histogram comparison
# ---------------------

def load_exr_images(folder_path):
    """
    Loads all EXR images from the specified folder.
    Returns:
        images: dict {filename: numpy_array}
        shape: tuple (H, W, C)
    """
    files = sorted(glob.glob(os.path.join(folder_path, "*.exr")) + 
                   glob.glob(os.path.join(folder_path, "*.EXR")))
    
    if not files:
        raise ValueError(f"No EXR files found in {folder_path}")

    print(f"Found {len(files)} images. Loading...")
    
    images = {}
    reference_shape = None
    
    for f in files:
        try:
            # Load as-is (floating point, likely BGR order)
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                print(f"Warning: Could not read {f}")
                continue
            
            # Crop if requested
            if IGNORE_TOP_HALF:
                h = img.shape[0]
                img = img[h//2:, :, :]
            
            images[os.path.basename(f)] = img
            
            # Store shape of the first successful load to compare others
            if reference_shape is None:
                reference_shape = img.shape
            elif img.shape != reference_shape:
                print(f"Skipping {f}: Dimension mismatch {img.shape} vs {reference_shape}")
                del images[os.path.basename(f)]
                
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not images:
        raise ValueError("No valid EXR images loaded.")
        
    return images, reference_shape

def to_grayscale(img_color):
    """
    Converts BGR float32 image to Grayscale using Rec.709 luma coefficients.
    """
    if len(img_color.shape) == 2: return img_color 
    return cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

def compute_histogram(target_gray, ref_hist_range):
    """
    Computes Histogram PDF.
    """
    # Density=True normalizes it to form a Probability Density Function (PDF)
    hist, bin_edges = np.histogram(target_gray, bins=HIST_BINS, range=ref_hist_range, density=True)
    return hist, bin_edges

def compute_kl_divergence(p, q):
    """
    Kullback-Leibler Divergence: D_KL(P || Q)
    """
    # Add epsilon to avoid log(0) or division by zero
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    
    # Re-normalize after adding epsilon
    p /= np.sum(p)
    q /= np.sum(q)
    
    return np.sum(p * np.log(p / q))

def analyze_convergence(folder_path, output_folder_name="analysis_results", reference_image=None):
    # 1. Load Data
    images_dict, shape = load_exr_images(folder_path)
    print(f"Loaded {len(images_dict)} images with shape {shape}.")
    filenames = list(images_dict.keys())

    # Create output path
    output_dir = os.path.join(folder_path, output_folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Stack images (N, H, W, C)
    print("Stacking images...")
    img_stack = np.stack([images_dict[f] for f in filenames]).astype(np.float32)
    
    if reference_image is None:
        # 2. Compute Reference (Mean)
        reference_img = np.mean(img_stack, axis=0).astype(np.float32)
        print("Computed reference image as mean of all techniques.")
    else:
        # Load provided reference image
        reference_img = cv2.imread(reference_image, cv2.IMREAD_UNCHANGED)
        if reference_img is None:
            raise ValueError(f"Could not read provided reference image: {reference_image}")
        
        # Crop if needed to match the loaded images
        if IGNORE_TOP_HALF:
            h = reference_img.shape[0]
            reference_img = reference_img[h//2:, :, :]
        
        # Ensure dimensions match
        if reference_img.shape != shape:
             raise ValueError(f"Reference image shape {reference_img.shape} does not match loaded images {shape}")

        print(f"Loaded provided reference image: {reference_image}")
        
    # Save Reference immediately
    ref_out_path = os.path.join(output_dir, "computed_reference.exr")
    cv2.imwrite(ref_out_path, reference_img)
    print(f"Saved reference to: {ref_out_path}")

    # Prepare for Histogram / KL Calc
    ref_gray = to_grayscale(reference_img)
    
    # Determine Histogram Range (0 to 99.5th percentile to ignore extreme fireflies)
    hist_min = 0.0
    hist_max = np.percentile(ref_gray, 99.5) 
    ref_hist_range = (hist_min, hist_max)
    
    ref_hist, _ = compute_histogram(ref_gray, ref_hist_range)

    # 3. Compute Metrics
    results = []
    print("Computing Bias-Variance decomposition & KL Divergence...")
    
    histograms = {} # Store hists for plotting later

    for i, filename in enumerate(filenames):
        img = img_stack[i]
        
        # --- Pixel-wise Error Metrics ---
        diff = img - reference_img
        
        # MSE (Mean Squared Error) = Bias^2 + Variance
        mse = np.mean(np.square(diff))
        
        # Bias (Systematic Error)
        # If bias != 0, the image is consistently darker/brighter than reference
        bias = np.mean(diff)
        
        # Variance (Stochastic Noise)
        # This isolates the noise by removing the bias shift
        variance = mse - (bias ** 2)
        
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(diff))
        
        # --- Distribution Metrics ---
        img_gray = to_grayscale(img)
        hist, bin_edges = compute_histogram(img_gray, ref_hist_range)
        
        # KL Divergence
        kl_div = compute_kl_divergence(ref_hist, hist)

        histograms[filename] = hist

        results.append({
            "Technique": filename,
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "Bias": bias,       # New Metric
            "Variance": variance, # New Metric (Noise)
            "MAE": mae,
            "KL_Divergence": kl_div
        })

    # 4. Rank and Organize
    df = pd.DataFrame(results)
    
    # Sort by KL_divergence (as requested in your version), 
    # but you might prefer sorting by MSE for strict convergence.
    df = df.sort_values(by="KL_Divergence", ascending=True).reset_index(drop=True)
    
    # 5. Generate Outputs
    
    # A. CSV Report
    csv_path = os.path.join(output_dir, "ranking_report.csv")
    df.to_csv(csv_path, index=False)
    
    # B. Stacked Bar Chart (Bias vs Variance) - NEW
    # Visualizes the composition of the error (MSE)
    plt.figure(figsize=(10, 6))
    
    # Sort by MSE for this specific chart to make the "Error" bars readable
    df_mse_sorted = df.sort_values(by="MSE", ascending=True)
    
    bias_sq = df_mse_sorted["Bias"] ** 2
    plt.barh(df_mse_sorted["Technique"], df_mse_sorted["Variance"], label='Variance (Noise)', color='#d62728', alpha=0.8)
    plt.barh(df_mse_sorted["Technique"], bias_sq, left=df_mse_sorted["Variance"], label='Bias² (Shift)', color='#1f77b4', alpha=0.8)
    
    plt.xlabel('Mean Squared Error (MSE)')
    plt.title('Error Decomposition: Noise vs Bias')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ranking_chart_bias_variance.png"))
    plt.close()

    # C. KL Divergence Chart 
    plt.figure(figsize=(10, 6))
    plt.bar(df["Technique"], df["KL_Divergence"], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Technique")
    plt.ylabel("KL Divergence (Lower is Better)")
    plt.title("Ranking by Histogram Similarity (KL Divergence)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ranking_chart_kl.png"))
    plt.close()

    # D. Histogram Comparison Plot
    # Plot Reference vs Best vs Worst (based on current sort order, which is KL)
    best_tech = df.iloc[0]["Technique"]
    worst_tech = df.iloc[-1]["Technique"]
    
    plt.figure(figsize=(8, 5))
    bins_x = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.plot(bins_x, ref_hist, color='black', linewidth=2, linestyle='--', label='Reference')
    plt.plot(bins_x, histograms[best_tech], color='green', alpha=0.7, label=f'Best: {best_tech}')
    plt.plot(bins_x, histograms[worst_tech], color='red', alpha=0.7, label=f'Worst: {worst_tech}')
    
    plt.xlabel(f'Pixel Intensity (0 - {hist_max:.2f})')
    plt.ylabel('Probability Density')
    plt.title('Intensity Distribution (Best vs Worst vs Ref)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "histogram_comparison.png"))
    plt.close()

    # E. All Histograms Plot
    plt.figure(figsize=(8, 4))
    for tech, hist in histograms.items():
        plt.plot(bins_x, hist, alpha=0.5, linewidth=1, label=tech)

    plt.plot(bins_x, ref_hist, color='black', linewidth=2, linestyle='--', label='Reference')
    plt.xlabel(f'Pixel Intensity')
    plt.ylabel('Probability Density')
    plt.title('All Distributions')
    # Move legend outside if too many items
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_histograms.png"))
    plt.close()

    # F. Save Cropped Images (Optional)
    if IGNORE_TOP_HALF and SAVE_CROPS:
        crop_dir = os.path.join(output_dir, "crops")
        if not os.path.exists(crop_dir):
            os.makedirs(crop_dir)
            
        for k in images_dict.keys():
            out_path = os.path.join(crop_dir, k)
            cv2.imwrite(out_path, images_dict[k])
        print(f"Saved cropped source images to: {crop_dir}/")

    # G. Console Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("TOP 5 PERFORMING TECHNIQUES (Sorted by KL_Divergence):")
    cols = ["Technique", "MSE", "Variance", "Bias", "KL_Divergence"]
    print(df[cols].to_string(index=False))
    print("-" * 60)
    print(f"Full results and plots saved to: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_exrs.py <input_folder> [optional_reference_image.exr]")
        sys.exit(1)

    input_folder = sys.argv[1]

    ref_image = None
    if len(sys.argv) >= 3:
        ref_image = sys.argv[2]
        print(f"Using provided reference image: {ref_image}")
    else:
        print("No reference image provided. Will compute reference as mean of all images.")
    
    try:
        analyze_convergence(input_folder, reference_image=ref_image)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()