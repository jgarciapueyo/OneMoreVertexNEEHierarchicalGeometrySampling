"""
analyze_hemisphere_runs.py — Visualise hemisphere experiment results.

Folder layout expected under RUNS_DIR:
    runs/
      <geometry_name>/
        ours10/       render.exr  stdout.log
        pt10/         render.exr  stdout.log
        pt_equaltime/ render.exr  stdout.log
        ptref/        render.exr  stdout.log

Geometry names must contain tokens parseable by GEOMETRY_RE, e.g.:
    icosphere_512tris_0.75sa
    hemisphere_2048tris_1.00sa

Edit the CONFIG section at the bottom of this file to match your setup.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.axes import Axes

try:
    import OpenEXR  # type: ignore[import-not-found]
    import Imath  # type: ignore[import-not-found]
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False
    print("[warn] OpenEXR not installed — using .npy fallback if available.")

FORMAT = "svg" # or pdf 

# ---------------------------------------------------------------------------
# CONFIG — edit this section
# ---------------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent

RUNS_DIR = current_dir / "runs"

# Matches names like: hemisphere_subdiv20_area0.10_n394_ratio0.0995
GEOMETRY_RE = re.compile(
    r"subdiv(?P<subdiv>\d+)"
    r"_area(?P<area>[\d.]+)"
    r"_n(?P<n_triangles>\d+)"
    r"_ratio(?P<solid_angle_ratio>[\d.]+)"
)

# Image-space regions of interest (minx, miny, maxx, maxy) in pixel coordinates.
# Use None to evaluate the full image.
REGIONS: dict[str, tuple[int, int, int, int] | None] = {
    "full":   None,
    # "center": (200, 200, 456, 456),   # example — adjust to your resolution
    # "corner": (0,   0,   128, 128),
}

# Method directories
OUR_METHOD    = "ours10"       # our method (equal SPP to pt10)
REFERENCE     = "ptref"        # ground-truth reference
EQUAL_TIME    = "pt_equaltime" # PT at ~equal render time
EQUAL_SPP     = "pt10"         # PT at equal samples-per-pixel

# Plot output directory
OUT_DIR = current_dir / "plots"

# ---------------------------------------------------------------------------
# Visual style — single source of truth for all plots
# ---------------------------------------------------------------------------

# Each method maps to (color, marker, label)
METHOD_STYLE: dict[str, tuple[str, str, str]] = {
    OUR_METHOD:  ("#2166AC", "o", "Ours"),
    EQUAL_TIME:  ("#D6604D", "s", "PT (equal time)"),
    EQUAL_SPP:   ("#4DAC26", "D", "PT (equal SPP)"),
}

MARKER_SIZE   = 7
LINE_WIDTH    = 1.8
ALPHA_LINE    = 0.85
GRID_ALPHA    = 0.35

# Seaborn / matplotlib base theme
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
mpl.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         GRID_ALPHA,
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "#cccccc",
})


# ---------------------------------------------------------------------------
# EXR / image loading
# ---------------------------------------------------------------------------

def load_exr(path: Path) -> np.ndarray:
    """Return (H, W, 3) float32 RGB array from an EXR file."""
    if not HAS_OPENEXR:
        raise RuntimeError(
            "OpenEXR is required to load .exr files. "
            "Install it with:  pip install OpenEXR"
        )
    f = OpenEXR.InputFile(str(path))
    dw = f.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = [
        np.frombuffer(f.channel(c, pt), dtype=np.float32).reshape(h, w)
        for c in ("R", "G", "B")
    ]
    return np.stack(channels, axis=-1)


def find_render(folder: Path) -> Path | None:
    for ext in ("*.exr", "*.npy"):
        hits = sorted(folder.glob(ext))
        if hits:
            return hits[0]
    return None


def find_repetition_renders(folder: Path) -> list[Path]:
    hits: list[Path] = []
    for ext in ("render*.exr", "render*.npy"):
        hits.extend(sorted(folder.glob(ext)))
    return sorted(hits)


def load_render(folder: Path) -> np.ndarray | None:
    p = find_render(folder)
    if p is None:
        return None
    if p.suffix == ".npy":
        return np.load(p).astype(np.float32)
    return load_exr(p)


def load_renders(folder: Path) -> list[np.ndarray]:
    paths = find_repetition_renders(folder)
    if not paths:
        single = load_render(folder)
        return [] if single is None else [single]
    renders: list[np.ndarray] = []
    for p in paths:
        if p.suffix == ".npy":
            renders.append(np.load(p).astype(np.float32))
        else:
            renders.append(load_exr(p))
    return renders


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def crop(img: np.ndarray, region: tuple[int, int, int, int] | None) -> np.ndarray:
    if region is None:
        return img
    x0, y0, x1, y1 = region
    return img[y0:y1, x0:x1]


def relative_mse(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-4) -> float:
    diff = pred - ref
    denom = ref ** 2 + eps
    return float(np.mean(diff ** 2 / denom))


def compute_error(
    pred: np.ndarray,
    ref: np.ndarray,
    region: tuple[int, int, int, int] | None,
) -> float:
    return relative_mse(crop(pred, region), crop(ref, region))


def compute_variance(
    renders: list[np.ndarray],
    region: tuple[int, int, int, int] | None,
) -> float | None:
    if not renders:
        return None
    cropped = [crop(img, region) for img in renders]
    stack = np.stack(cropped, axis=0)
    ddof = 1 if stack.shape[0] > 1 else 0
    var = np.var(stack, axis=0, ddof=ddof)
    return float(np.mean(var))


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def parse_geometry(name: str) -> dict | None:
    m = GEOMETRY_RE.search(name)
    if not m:
        return None
    return {
        "subdiv":            int(m.group("subdiv")),
        "area":              float(m.group("area")),
        "n_triangles":       int(m.group("n_triangles")),
        "solid_angle_ratio": float(m.group("solid_angle_ratio")),
    }


def collect_data(
    runs_dir: Path,
    regions: dict[str, tuple[int, int, int, int] | None],
    our_method: str  = OUR_METHOD,
    reference: str   = REFERENCE,
    equal_time: str  = EQUAL_TIME,
    equal_spp: str   = EQUAL_SPP,
    use_variance: bool = False,
) -> pd.DataFrame:
    rows = []
    for geom_dir in sorted(runs_dir.iterdir()):
        if not geom_dir.is_dir():
            continue
        parsed = parse_geometry(geom_dir.name)
        if parsed is None:
            print(f"[skip] Cannot parse geometry from '{geom_dir.name}'")
            continue

        ref_img = None
        if not use_variance:
            ref_img = load_render(geom_dir / reference)
            if ref_img is None:
                print(f"[skip] No reference render in {geom_dir / reference}")
                continue

        ours_imgs    = load_renders(geom_dir / our_method)
        eq_time_imgs = load_renders(geom_dir / equal_time)
        eq_spp_imgs  = load_renders(geom_dir / equal_spp)

        for region_name, region_box in regions.items():
            row: dict = {
                "geometry":          geom_dir.name,
                "subdiv":            parsed["subdiv"],
                "area":              parsed["area"],
                "n_triangles":       parsed["n_triangles"],
                "solid_angle_ratio": parsed["solid_angle_ratio"],
                "region":            region_name,
            }
            for key, imgs in [
                (f"error_{our_method}", ours_imgs),
                (f"error_{equal_time}", eq_time_imgs),
                (f"error_{equal_spp}",  eq_spp_imgs),
            ]:
                if use_variance:
                    value = compute_variance(imgs, region_box)
                    if value is not None:
                        row[key] = value
                else:
                    if imgs:
                        row[key] = compute_error(imgs[0], ref_img, region_box)
            rows.append(row)

    if not rows:
        sys.exit("[error] No valid runs found — check RUNS_DIR and GEOMETRY_RE.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Low-level drawing helper
# ---------------------------------------------------------------------------

def _draw_method_lines(
    ax: Axes,
    df: pd.DataFrame,
    x_col: str,
    methods: list[str],
) -> None:
    """
    Draw one styled line per method directly (no seaborn hue magic),
    using METHOD_STYLE for color / marker / label.
    Aggregates over any duplicate x values with the mean.
    """
    for method in methods:
        col = f"error_{method}"
        if col not in df.columns:
            continue
        color, marker, label = METHOD_STYLE[method]
        agg = df.groupby(x_col)[col].mean().reset_index()
        ax.plot(
            agg[x_col], agg[col],
            color=color, marker=marker, label=label,
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
            alpha=ALPHA_LINE, zorder=3,
        )


def _method_legend(ax: Axes, methods: list[str], **kwargs) -> None:
    """Add a clean legend with the canonical method labels and styles."""
    handles = []
    for method in methods:
        color, marker, label = METHOD_STYLE[method]
        handles.append(
            mlines.Line2D([], [], color=color, marker=marker,
                          markersize=MARKER_SIZE, linewidth=LINE_WIDTH,
                          label=label)
        )
    ax.legend(handles=handles, fontsize=8, title_fontsize=8, **kwargs)


def _add_method_legend(ax: Axes, methods: list[str], **kwargs) -> None:
    handles = [
        mlines.Line2D(
            [], [],
            color=METHOD_STYLE[m][0],
            marker=METHOD_STYLE[m][1],
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=METHOD_STYLE[m][2],
        )
        for m in methods
    ]
    ax.legend(handles=handles, fontsize=8, title="Method",
              title_fontsize=8, **kwargs)


# ---------------------------------------------------------------------------
# Plot family: error vs area  (one subplot per subdiv)
# ---------------------------------------------------------------------------

def _plot_vs_area_single(
    df: pd.DataFrame,
    methods: list[str],
    out_path: Path,
    family_title: str,
    y_label: str,
) -> None:
    subdiv_vals = sorted(df["subdiv"].unique())
    n_rows = len(subdiv_vals)

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(6.5, 3.5 * n_rows),
        sharex=True, sharey=False, squeeze=False,
    )
    fig.suptitle(
        f"{family_title}\nError vs area ratio  |  region: {df['region'].iloc[0]}",
        fontsize=12, y=1.01,
    )

    for row_idx, subdiv in enumerate(subdiv_vals):
        ax = axes[row_idx, 0]
        tdf = df[df["subdiv"] == subdiv]
        _draw_method_lines(ax, tdf, "area", methods)
        ax.set_title(f"Subdivisions = {subdiv}", fontsize=10, pad=4)
        ax.set_ylabel(y_label)
        ax.tick_params(labelsize=8)
        _add_method_legend(ax, methods, loc="best")
        if row_idx == n_rows - 1:
            ax.set_xlabel("Area ratio")
        else:
            ax.set_xlabel("")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Plot family: error vs subdivisions  (one subplot per area)
# ---------------------------------------------------------------------------

def _plot_vs_subdiv_single(
    df: pd.DataFrame,
    methods: list[str],
    out_path: Path,
    family_title: str,
    y_label: str,
) -> None:
    area_vals = sorted(df["area"].unique())
    n_rows = len(area_vals)

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(6.5, 3.5 * n_rows),
        sharex=True, sharey=False, squeeze=False,
    )
    fig.suptitle(
        f"{family_title}\nError vs subdivisions  |  region: {df['region'].iloc[0]}",
        fontsize=12, y=1.01,
    )

    for row_idx, area in enumerate(area_vals):
        ax = axes[row_idx, 0]
        sdf = df[df["area"] == area]
        _draw_method_lines(ax, sdf, "subdiv", methods)
        ax.set_title(f"Area ratio = {area:.4f}", fontsize=10, pad=4)
        ax.set_ylabel(y_label)
        ax.tick_params(labelsize=8)
        _add_method_legend(ax, methods, loc="best")
        if row_idx == n_rows - 1:
            ax.set_xlabel("Subdivisions")
        else:
            ax.set_xlabel("")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Public API: three comparison families
# ---------------------------------------------------------------------------

def _ensure(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- Equal-time comparison (ours vs PT equal-time) -------------------------

def plot_vs_area_equaltime(df: pd.DataFrame, out_dir: Path, y_label: str) -> None:
    methods = [OUR_METHOD, EQUAL_TIME]
    sub = _ensure(out_dir / "equal_time" / "by_area")
    for region_name, rdf in df.groupby("region"):
        _plot_vs_area_single(
            rdf, methods,
            sub / f"area__{region_name}.{FORMAT}",
            family_title="Equal-time comparison",
            y_label=y_label,
        )


def plot_vs_subdiv_equaltime(df: pd.DataFrame, out_dir: Path, y_label: str) -> None:
    methods = [OUR_METHOD, EQUAL_TIME]
    sub = _ensure(out_dir / "equal_time" / "by_subdiv")
    for region_name, rdf in df.groupby("region"):
        _plot_vs_subdiv_single(
            rdf, methods,
            sub / f"subdiv__{region_name}.{FORMAT}",
            family_title="Equal-time comparison",
            y_label=y_label,
        )


# --- Equal-SPP comparison (ours vs PT equal-SPP) ---------------------------

def plot_vs_area_equalspp(df: pd.DataFrame, out_dir: Path, y_label: str) -> None:
    methods = [OUR_METHOD, EQUAL_SPP]
    sub = _ensure(out_dir / "equal_spp" / "by_area")
    for region_name, rdf in df.groupby("region"):
        _plot_vs_area_single(
            rdf, methods,
            sub / f"area__{region_name}.{FORMAT}",
            family_title="Equal-SPP comparison",
            y_label=y_label,
        )


def plot_vs_subdiv_equalspp(df: pd.DataFrame, out_dir: Path, y_label: str) -> None:
    methods = [OUR_METHOD, EQUAL_SPP]
    sub = _ensure(out_dir / "equal_spp" / "by_subdiv")
    for region_name, rdf in df.groupby("region"):
        _plot_vs_subdiv_single(
            rdf, methods,
            sub / f"subdiv__{region_name}.{FORMAT}",
            family_title="Equal-SPP comparison",
            y_label=y_label,
        )


# --- Summary: all three methods, both axes, all regions --------------------

def plot_summary(df: pd.DataFrame, out_dir: Path, y_label: str) -> None:
    """
    One PDF with all three methods on every subplot.
    Rows = regions, columns = (vs area ratio | vs subdivisions).
    Each subplot has one line per method; subplots are faceted by the
    *other* parameter value (one subplot per subdiv or per area).

    Because the number of subplots per region can vary, we produce
    separate per-region PDFs and one combined overview.
    """
    _ensure(out_dir)
    methods = [OUR_METHOD, EQUAL_TIME, EQUAL_SPP]

    regions = sorted(df["region"].unique())

    for region_name in regions:
        rdf = df[df["region"] == region_name]
        subdiv_vals = sorted(rdf["subdiv"].unique())
        area_vals   = sorted(rdf["area"].unique())

        n_rows = max(len(subdiv_vals), len(area_vals))
        fig, axes = plt.subplots(
            n_rows, 2,
            figsize=(13, 3.5 * n_rows),
            squeeze=False,
        )
        fig.suptitle(
            f"Summary — region: {region_name}\n"
            f"● = {METHOD_STYLE[OUR_METHOD][2]}   "
            f"■ = {METHOD_STYLE[EQUAL_TIME][2]}   "
            f"◆ = {METHOD_STYLE[EQUAL_SPP][2]}",
            fontsize=12, y=1.01,
        )

        # Left column: error vs area ratio, one subplot per subdiv
        for row_idx, subdiv in enumerate(subdiv_vals):
            ax = axes[row_idx, 0]
            tdf = rdf[rdf["subdiv"] == subdiv]
            _draw_method_lines(ax, tdf, "area", methods)
            ax.set_title(f"Subdivisions = {subdiv}", fontsize=9, pad=3)
            ax.set_ylabel(y_label, fontsize=8)
            ax.tick_params(labelsize=7)
            _add_method_legend(ax, methods, loc="best")
            ax.set_xlabel(
                "Area ratio" if row_idx == len(subdiv_vals) - 1 else "",
                fontsize=8,
            )
        # Hide unused rows in left column
        for row_idx in range(len(subdiv_vals), n_rows):
            axes[row_idx, 0].set_visible(False)

        # Right column: error vs subdivisions, one subplot per area
        for row_idx, area in enumerate(area_vals):
            ax = axes[row_idx, 1]
            sdf = rdf[rdf["area"] == area]
            _draw_method_lines(ax, sdf, "subdiv", methods)
            ax.set_title(f"Area ratio = {area:.4f}", fontsize=9, pad=3)
            ax.set_ylabel(y_label, fontsize=8)
            ax.tick_params(labelsize=7)
            _add_method_legend(ax, methods, loc="best")
            ax.set_xlabel(
                "Subdivisions" if row_idx == len(area_vals) - 1 else "",
                fontsize=8,
            )
        # Hide unused rows in right column
        for row_idx in range(len(area_vals), n_rows):
            axes[row_idx, 1].set_visible(False)

        # Column headers
        axes[0, 0].annotate(
            "Error vs area ratio",
            xy=(0.5, 1.08), xycoords="axes fraction",
            ha="center", fontsize=10, fontweight="bold",
        )
        axes[0, 1].annotate(
            "Error vs subdivisions",
            xy=(0.5, 1.08), xycoords="axes fraction",
            ha="center", fontsize=10, fontweight="bold",
        )

        fig.tight_layout()
        fname = out_dir / f"summary__{region_name}.{FORMAT}"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fname}")


import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch

def plot_vs_area_with_thumbnails(
    df: pd.DataFrame, 
    runs_dir: Path, 
    out_path: Path, 
    target_subdiv: int = 100,
    loglog: bool = True
) -> None:
    """
    Generates a composite figure containing the main Relative MSE line plot on top
    and a row of representative geometry renders aligned along the bottom.
    """
    # Filter and sort data for the specified subdivision level
    sub_df = df[df["subdiv"] == target_subdiv].sort_values("area")
    if loglog:
        # Log scale cannot display non-positive values; remove them up front.
        sub_df = sub_df[sub_df["area"] > 0]
    unique_areas = sub_df["area"].unique()
    
    if len(unique_areas) == 0:
        print(f"[warn] No data found for subdivisions = {target_subdiv}")
        return

    # Select up to 5 representative area ratios to display as thumbnails
    if len(unique_areas) <= 5:
        selected_areas = unique_areas
    else:
        # Select endpoints and evenly distributed intermediate steps
        indices = np.linspace(0, len(unique_areas) - 1, 4, dtype=int)
        selected_areas = [unique_areas[i] for i in indices]
        
    num_images = len(selected_areas)
    
    # Scale width with number of thumbnails to minimize wasted horizontal space.
    fig_w = max(4.8, 1.4 * num_images)
    fig = plt.figure(figsize=(fig_w, 5))
    
    # Create a 2-row grid: Row 0 for the main plot, Row 1 for thumbnails
    gs = fig.add_gridspec(2, num_images, height_ratios=[2.5, 1], hspace=0.4, wspace=0.05)
    fig.subplots_adjust(left=0.08, right=0.98)
    
    # -----------------------------------------------------------------------
    # 1. Main Plot Axis (Spans all columns in Row 0)
    # -----------------------------------------------------------------------
    ax_main = fig.add_subplot(gs[0, :])
    methods = [OUR_METHOD, EQUAL_TIME, EQUAL_SPP]
    _draw_method_lines(ax_main, sub_df, "area", methods)
    
    ax_main.set_title(f"Relative MSE vs Area Ratio ({target_subdiv} subdivisions)", fontsize=12, pad=10)
    ax_main.set_xlabel("Area Ratio", fontsize=10)
    ax_main.set_ylabel("Relative MSE", fontsize=10)
    
    if loglog:
        ax_main.set_yscale("log") 
        ax_main.set_xscale("log") 
    _add_method_legend(ax_main, methods, loc="upper right")

    # Start arrows slightly above the x-axis to dodge tick labels.
    x_data_y_axes = ax_main.get_xaxis_transform()
    
    # -----------------------------------------------------------------------
    # 2. Thumbnail Generation & Alignment (Row 1)
    # -----------------------------------------------------------------------
    for i, area in enumerate(selected_areas):
        ax_img = fig.add_subplot(gs[1, i])
        
        # Extract folder mapping from dataframe matching this specific point
        row_match = sub_df[sub_df["area"] == area].iloc[0]
        geom_folder_name = row_match["geometry"]
        
        # Load the reference render to best highlight geometric variation
        img = load_render(runs_dir.parent / "runs_outside_log" / geom_folder_name / REFERENCE)
        
        if img is not None:
            img = img * 8  # exposure
            img = 4 * img / (img + 1.0) # reinhard tonemapping
            # Simple HDR linear-to-sRGB exposure mapping fallback (Gamma 2.2)
            img_display = np.clip(img ** (1.0 / 2.2), 0, 1)
            ax_img.imshow(img_display)
        else:
            # Visual placeholder if file handling fails
            ax_img.text(0.5, 0.5, "Missing\nRender", ha="center", va="center", fontsize=8, color="gray")
            
        ax_img.text(
            0.03, 0.97,
            f"Ratio: {area:.3f}",
            transform=ax_img.transAxes,
            ha="left", va="top",
            fontsize=9,
            color="white",
            bbox={"facecolor": (0, 0, 0, 0.3), "edgecolor": "none", "pad": 2},
        )
        ax_img.axis("off")

        
        # -----------------------------------------------------------------------
        # 3. Connection Lines (Connects Graph X-axis to Thumbnail top-center)
        # -----------------------------------------------------------------------
        # Origin point: (x = area ratio value, y = 0 baseline on main plot)
        # Destination point: (x = 0.5 center, y = 1.0 top boundary of thumbnail bounding box)
        con = ConnectionPatch(
            xyA=(area, -0.09),
            xyB=(0.5, 1.0), 
            coordsA=x_data_y_axes,
            coordsB="axes fraction",
            axesA=ax_main, 
            axesB=ax_img, 
            color="#999999", 
            linestyle=":", 
            linewidth=1.2,
            zorder=1
        )
        fig.add_artist(con)
        
    
    # Save cleanly packed layout
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved custom composite figure to {out_path}")


def plot_vs_subdiv_with_thumbnails(
    df: pd.DataFrame,
    runs_dir: Path,
    out_path: Path,
    target_area: float = 0.16,
    logy: bool = True,
) -> None:
    """
    Composite figure: error vs subdivisions with thumbnails for a fixed area ratio.
    """
    sub_df = df[np.isclose(df["area"], target_area)].sort_values("subdiv")
    unique_subdivs = sub_df["subdiv"].unique()

    if len(unique_subdivs) == 0:
        print(f"[warn] No data found for area ratio = {target_area}")
        return

    if len(unique_subdivs) <= 5:
        selected_subdivs = unique_subdivs
    else:
        indices = np.linspace(0, len(unique_subdivs) - 1, 5, dtype=int)
        selected_subdivs = [unique_subdivs[i] for i in indices]

    num_images = len(selected_subdivs)
    fig_w = max(4.8, 1.4 * num_images)
    fig = plt.figure(figsize=(fig_w, 5))

    gs = fig.add_gridspec(2, num_images, height_ratios=[2.5, 1], hspace=0.4, wspace=0.05)
    fig.subplots_adjust(left=0.08, right=0.98)

    ax_main = fig.add_subplot(gs[0, :])
    methods = [OUR_METHOD, EQUAL_TIME, EQUAL_SPP]
    _draw_method_lines(ax_main, sub_df, "subdiv", methods)

    ax_main.set_title(f"Relative MSE vs Subdivisions ({target_area:.3f} area)", fontsize=12, pad=10)
    ax_main.set_xlabel("Subdivisions", fontsize=10)
    ax_main.set_ylabel("Relative MSE", fontsize=10)

    if logy:
        ax_main.set_yscale("log")
    _add_method_legend(ax_main, methods, loc="upper right")

    x_data_y_axes = ax_main.get_xaxis_transform()

    for i, subdiv in enumerate(selected_subdivs):
        ax_img = fig.add_subplot(gs[1, i])

        row_match = sub_df[sub_df["subdiv"] == subdiv].iloc[0]
        geom_folder_name = row_match["geometry"]

        img = load_render(runs_dir.parent / "runs_outside_log" / geom_folder_name / REFERENCE)

        if img is not None:
            img = img * 8  # exposure
            img = 4 * img / (img + 1.0)  # reinhard tonemapping
            img_display = np.clip(img ** (1.0 / 2.2), 0, 1)
            ax_img.imshow(img_display)
        else:
            ax_img.text(0.5, 0.5, "Missing\nRender", ha="center", va="center", fontsize=8, color="gray")

        ax_img.text(
            0.03, 0.97,
            f"Subdivisions: {subdiv}",
            transform=ax_img.transAxes,
            ha="left", va="top",
            fontsize=9,
            color="white",
            bbox={"facecolor": (0, 0, 0, 0.3), "edgecolor": "none", "pad": 2},
        )
        ax_img.axis("off")

        con = ConnectionPatch(
            xyA=(subdiv, -0.09),
            xyB=(0.5, 1.0),
            coordsA=x_data_y_axes,
            coordsB="axes fraction",
            axesA=ax_main,
            axesB=ax_img,
            color="#999999",
            linestyle=":",
            linewidth=1.2,
            zorder=1,
        )
        fig.add_artist(con)

    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved custom composite figure to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze hemisphere runs.")
    parser.add_argument(
        "--variance",
        action="store_true",
        help="Plot variance across repeated renders instead of Relative MSE.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(RUNS_DIR),
        help=f"Directory containing run subfolders (default: {RUNS_DIR})",
    )
    # parser.add_argument(
    #     "--loglog",
    #     action="store_true",
    #     help="Use log-log scale for the area ratio vs error plot (recommended for wide area ratio range).",
    # )
    args = parser.parse_args()

    y_label = "Variance" if args.variance else "Relative MSE"
    

    runs_dir = Path(args.input)

    # get the name of the input directory to use as a suffix for output files, e.g. "runs_emitter2"
    name_input = runs_dir.name

    out_dir = OUT_DIR / name_input

    print(f"Collecting data from '{runs_dir}' ...")
    df = collect_data(runs_dir, REGIONS, use_variance=args.variance)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"DataFrame saved to {csv_path}  ({len(df)} rows)")
    print(df.head(), "\n")

    # ---- Equal-time plots --------------------------------------------------
    print("--- Equal-time: error vs area ratio ---")
    plot_vs_area_equaltime(df, out_dir, y_label=y_label)

    print("--- Equal-time: error vs subdivisions ---")
    plot_vs_subdiv_equaltime(df, out_dir, y_label=y_label)

    # ---- Equal-SPP plots ---------------------------------------------------
    print("--- Equal-SPP: error vs area ratio ---")
    plot_vs_area_equalspp(df, out_dir, y_label=y_label)

    print("--- Equal-SPP: error vs subdivisions ---")
    plot_vs_subdiv_equalspp(df, out_dir, y_label=y_label)

    # ---- Summary (all three methods) ----------------------------------------
    print("--- Summary (all methods, all regions) ---")
    plot_summary(df, out_dir, y_label=y_label)

    print("--- Summary (all methods, all regions) ---")
    plot_summary(df, out_dir, y_label=y_label)

    # NEW: Generate the specialized composite breakdown layout
    print("--- Custom Composite Layout: Error vs Area + Thumnails ---")
    composite_path = out_dir / f"composite_subdiv20.{FORMAT}"
    plot_vs_area_with_thumbnails(df, runs_dir, composite_path, target_subdiv=20, loglog=False)
    composite_path = out_dir / f"composite_subdiv20_loglog.{FORMAT}"
    plot_vs_area_with_thumbnails(df, runs_dir, composite_path, target_subdiv=20, loglog=True)

    composite_path = out_dir / f"composite_area0.16.{FORMAT}"
    plot_vs_subdiv_with_thumbnails(df, runs_dir, composite_path, target_area=0.16, logy=False)
    composite_path = out_dir / f"composite_area0.16_logy.{FORMAT}"
    plot_vs_subdiv_with_thumbnails(df, runs_dir, composite_path, target_area=0.16, logy=True)
    

    print("\nDone.  Output layout:")
    print("  plots/")
    print("    results.csv")
    print("    equal_time/by_area/          ← ours vs PT equal-time")
    print("    equal_time/by_subdiv/")
    print("    equal_spp/by_area/           ← ours vs PT equal-SPP")
    print("    equal_spp/by_subdiv/")
    print(f"    summary__<region>.{FORMAT}        ← all three methods")


if __name__ == "__main__":
    main()
    