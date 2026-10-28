""" 
Splits a set of EXR figures into vertical or horizontal slices, applies exposure and tonemapping, and exports a vector PDF with cleanly separated sections. 

Used for the lighthouse figure 
"""

import os
# Required in recent OpenCV versions to allow loading EXR files
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.patheffects as path_effects

def process_exr_to_ldr(path, exposure=2.5):
    """
    Loads an EXR file, applies exposure, and tonemaps to an LDR sRGB float array [0, 1].
    """
    # Load EXR as float32
    img_hdr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img_hdr is None:
        raise FileNotFoundError(f"Could not load EXR image: {path}")
    
    # OpenCV loads images in BGR format, convert to RGB
    img_hdr = cv2.cvtColor(img_hdr, cv2.COLOR_BGR2RGB)
    
    # 1. Apply Exposure
    # Multiplying by 2^exposure is the standard mathematical way to shift exposure EV
    img_exposed = img_hdr * (2.0 ** exposure)
    
    # 2. Convert to LDR (Tonemap and Gamma)
    # Clip values to [0.0, 1.0] to prevent clipping artifacts and invalid math
    img_clipped = np.clip(img_exposed, 0.0, 1.0)
    
    # Apply standard 2.2 gamma correction for sRGB display
    img_ldr = np.power(img_clipped, 1.0 / 2.2)
    
    return img_ldr

def create_exr_vector_split(exr_paths, output_path="output.pdf", exposure=2.5, slant_offset=None, horizontal_split=False):
    """
    Loads EXR images, applies exposure/gamma, and exports a vector PDF split.
    """
    # if len(exr_paths) != len(texts):
    #     print("Error: The number of images must match the number of texts.")
    #     return
    if len(exr_paths) < 2:
        print("Error: Need at least 2 images.")
        return

    # 1. Load the first image to establish the coordinate system
    try:
        img0 = process_exr_to_ldr(exr_paths[0], exposure)
        H, W = img0.shape[:2] # Numpy shape is (Height, Width, Channels)
    except Exception as e:
        print(f"Error processing {exr_paths[0]}: {e}")
        return

    n = len(exr_paths)
    slice_width = W / n
    slice_height = H / n

    if slant_offset is None:
        slant_offset = W / 10 

    # 2. Setup Matplotlib figure
    dpi = 30
    fig_width_inches = W / dpi
    fig_height_inches = H / dpi
    
    fig = plt.figure(figsize=(fig_width_inches, fig_height_inches), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    # 3. Process each EXR, clip it, and draw it
    for i, path in enumerate(exr_paths):
        try:
            img = process_exr_to_ldr(path, exposure)
            # Resize if dimensions don't match the base image
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"Error loading {path}, skipping: {e}")
            continue
        
        # Calculate polygon vertices for the slice.
        # Vertical mode: left/right slices with slanted separators.
        # Horizontal mode: top/bottom slices with slanted separators.
        if not horizontal_split:
            tl_x = 0 if i == 0 else (i * slice_width) + slant_offset
            tr_x = W if i == n - 1 else ((i + 1) * slice_width) + slant_offset
            br_x = W if i == n - 1 else ((i + 1) * slice_width) - slant_offset
            bl_x = 0 if i == 0 else (i * slice_width) - slant_offset

            vertices = [
                (tl_x, 0),
                (tr_x, 0),
                (br_x, H),
                (bl_x, H),
                (tl_x, 0)
            ]
        else:
            lt_y = 0 if i == 0 else (i * slice_height) + slant_offset
            rt_y = 0 if i == 0 else (i * slice_height) - slant_offset
            rb_y = H if i == n - 1 else ((i + 1) * slice_height) - slant_offset
            lb_y = H if i == n - 1 else ((i + 1) * slice_height) + slant_offset

            vertices = [
                (0, lt_y),
                (W, rt_y),
                (W, rb_y),
                (0, lb_y),
                (0, lt_y)
            ]
        
        # Create a vector clipping path
        codes = [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.LINETO, mpath.Path.LINETO, mpath.Path.CLOSEPOLY]
        clip_path = mpath.Path(vertices, codes)
        clip_patch = patches.PathPatch(clip_path, facecolor='none', edgecolor='none')
        ax.add_patch(clip_patch)

        # Plot the LDR image (Matplotlib natively accepts float arrays in [0, 1])
        im = ax.imshow(img, extent=[0, W, H, 0], aspect='auto')
        im.set_clip_path(clip_patch)

    # 4. Draw Vector Separator Lines
    for i in range(1, n):
        if not horizontal_split:
            x_top = (i * slice_width) + slant_offset
            x_bot = (i * slice_width) - slant_offset
            ax.plot([x_top, x_bot], [0, H], color='white', linewidth=3, solid_capstyle='butt')
        else:
            y_left = (i * slice_height) + slant_offset
            y_right = (i * slice_height) - slant_offset
            ax.plot([0, W], [y_left, y_right], color='white', linewidth=3, solid_capstyle='butt')

    # 5. Add Vector Typography
    fontsize_pt = min(36, fig_height_inches * 72 / max(5, n * 0.8))
    
    # for i, text in enumerate(texts):
    #     y_pos = H * 0.88
        
    #     x_base = (i + 0.5) * slice_width
    #     x_pos = x_base + slant_offset * (1 - 2 * (y_pos / H))

    #     txt = ax.text(x_pos, y_pos, text, color='white',
    #                   fontsize=fontsize_pt, fontweight='bold',
    #                   ha='center', va='center', fontfamily='sans-serif')
        
    #     txt.set_path_effects([
    #         path_effects.Stroke(linewidth=fontsize_pt/5, foreground='black'),
    #         path_effects.Normal()
    #     ])

    # 6. Save directly to PDF
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Success! Vector PDF saved to {output_path}")

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    my_exr_images = [
        # "thinlens_lighthouse_path_2_100kspp_36min.exr",
        # "thinlens_lighthouse_glinttracermis_2_2kspp_105s.exr",
        "lighthouse_path_2_4500spp_90s.exr",
        "lighthouse_glinttracermis_2_2kspp_90s.exr",
        
    ]
    
    # my_texts = [
    #     "PT, thinlens",
    #     "PT, glinttracer",
    #     "PT, glinttracer",
    #     "PT, glinttracer"

    # ]
    
    # Run the function, applying +2.5 EV
    create_exr_vector_split(my_exr_images, "comparison_figure.pdf", exposure=2.5, horizontal_split=True, slant_offset=-70)
# ==========================================
# Example Usage
# ==========================================