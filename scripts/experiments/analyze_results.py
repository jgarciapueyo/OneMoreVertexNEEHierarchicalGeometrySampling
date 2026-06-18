import argparse
import csv
import json
import math
import re
from pathlib import Path

import yaml
import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except Exception:
    plt = None
    patches = None

try:
    import flip_evaluator as flip_eval
except Exception:
    flip_eval = None
    print("[WARN] flip_evaluator not found — FLIP metrics will be skipped. Install with: pip install flip-evaluator")


LATEX_DEFAULT_PLOTS = [
    ("mean_flip", "spp"),
    ("mean_flip", "time"),
    ("mean_E", "spp"),
]


DISPLAY_NAME_OVERRIDES = {}
EXR_ASPECT_CROP = None


def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, dynamic_ncols=True, **kwargs)


def _display_name(raw_name: str):
    if raw_name in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[raw_name]
    lowered = str(raw_name).lower()
    if lowered == "bdpt" or "bdpt" in lowered:
        return "BDPT"
    if lowered == "baseline_path" or "baseline_path" in lowered:
        return "PT"
    return raw_name


def _crop_to_aspect(img: np.ndarray, aspect_x: float, aspect_y: float, anchor_y: str = "center"):
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img

    target = float(aspect_x) / float(aspect_y)
    src = float(w) / float(h)
    if abs(src - target) < 1e-12:
        return img

    if src > target:
        new_w = max(1, min(w, int(math.floor(h * target))))
        x0 = (w - new_w) // 2
        return img[:, x0:x0 + new_w]

    new_h = max(1, min(h, int(math.floor(w / target))))
    if anchor_y == "bottom":
        y0 = h - new_h - int(h*.1)
    else:
        y0 = (h - new_h) // 2
    return img[y0:y0 + new_h, :]


def _parse_aspect_ratio_crop(analysis_cfg):
    raw = analysis_cfg.get("aspect_ratio", None)

    if raw is None and ("aspect_x" in analysis_cfg or "aspect_y" in analysis_cfg):
        enabled = bool(analysis_cfg.get("aspect_ratio_enabled", True))
        if not enabled:
            return None
        raw = {
            "aspect_x": analysis_cfg.get("aspect_x"),
            "aspect_y": analysis_cfg.get("aspect_y"),
        }

    if raw is None:
        return None

    aspect_x = None
    aspect_y = None
    anchor_y = "center"

    if isinstance(raw, dict):
        enabled = raw.get("enabled", None)
        if enabled is False:
            return None
        aspect_x = raw.get("aspect_x", raw.get("x", None))
        aspect_y = raw.get("aspect_y", raw.get("y", None))
        anchor_y = str(raw.get("anchor_y", raw.get("anchor", "center"))).strip().lower()
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        aspect_x, aspect_y = raw[0], raw[1]
    else:
        raise ValueError(
            "analysis.aspect_ratio must be a dict with aspect_x/aspect_y (and optional enabled), "
            "or a 2-element list/tuple."
        )

    if aspect_x is None or aspect_y is None:
        return None

    aspect_x = float(aspect_x)
    aspect_y = float(aspect_y)
    if (not np.isfinite(aspect_x)) or (not np.isfinite(aspect_y)) or aspect_x <= 0 or aspect_y <= 0:
        raise ValueError(
            f"Invalid analysis.aspect_ratio values: aspect_x={aspect_x}, aspect_y={aspect_y}. "
            "Both must be finite and > 0."
        )

    if anchor_y in ("middle",):
        anchor_y = "center"
    if anchor_y not in ("center", "bottom"):
        raise ValueError(
            "Invalid analysis.aspect_ratio.anchor_y value. Use 'center' (default) or 'bottom'."
        )

    return aspect_x, aspect_y, anchor_y


def load_img(path: Path, cache=True):
    if cache:
        # Simple in-memory cache to speed up repeated loads of the same image (e.g. reference)
        if not hasattr(load_img, "_cache"):
            load_img._cache = {}
        if path in load_img._cache:
            return load_img._cache[path]

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[WARN] Failed to load image: {path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    if EXR_ASPECT_CROP is not None and str(path).lower().endswith(".exr"):
        img = _crop_to_aspect(img, EXR_ASPECT_CROP[0], EXR_ASPECT_CROP[1], EXR_ASPECT_CROP[2])

    if cache:
        load_img._cache[path] = img

    return img


def mse(a, b):
    d = a - b
    return float(np.mean(d * d))


def _to_float(x):
    try:
        if x in ("", None):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _time_value(row):
    mt = _to_float(row.get("mitsuba_time_s"))
    wt = _to_float(row.get("wall_time_s"))
    if np.isfinite(mt):
        return mt
    return wt


def _parse_config_dirname(dirname: str):
    with_hash = re.match(r"^(.+)_([0-9a-f]{10})$", dirname)
    if with_hash:
        return None, with_hash.group(1), with_hash.group(2)
    return None, dirname, None


def _parse_bbox_normalized(raw_bbox):
    if raw_bbox is None:
        return None
    try:
        if len(raw_bbox) == 2 and isinstance(raw_bbox[0], list):
            x0, y0, x1, y1 = raw_bbox[0][0], raw_bbox[0][1], raw_bbox[1][0], raw_bbox[1][1]
        elif len(raw_bbox) == 4:
            x0, y0, x1, y1 = raw_bbox
        else:
            return None
        x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
    except Exception:
        return None

    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    if (x1 - x0) < 1e-8 or (y1 - y0) < 1e-8:
        return None
    return x0, y0, x1, y1


def _bbox_pixel_window_for_shape(shape_hw, bbox_norm):
    if bbox_norm is None:
        return None
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if h <= 0 or w <= 0: return None
    x0n, y0n, x1n, y1n = bbox_norm
    x0, x1 = max(0, min(w - 1, int(x0n * w))), max(0, min(w, int(x1n * w)))
    y0, y1 = max(0, min(h - 1, int(y0n * h))), max(0, min(h, int(y1n * h)))
    if x1 <= x0: x1 = min(w, x0 + 1)
    if y1 <= y0: y1 = min(h, y0 + 1)
    return x0, y0, x1, y1


def iter_config_dirs(output_root: Path):
    dirs = [p for p in sorted(output_root.iterdir()) if p.is_dir() and (p / "config.json").exists()]
    if dirs: return dirs
    return [p for p in sorted(output_root.glob("config_*")) if p.is_dir()]


def parse_run_meta_filename(name: str):
    m = re.match(r"^spp_(\d+)(?:_rep_(\d+))?\.meta\.json$", name)
    if not m: return None
    return int(m.group(1)), int(m.group(2)) if m.group(2) is not None else 1


def _finite_stats(values):
    arr = np.array([_to_float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return float(np.mean(arr)), float(np.var(arr)), float(np.sqrt(np.var(arr))), int(arr.size)


def summarize_rows_by_spp(rows):
    by_spp = {}
    for row in rows:
        by_spp.setdefault(int(row["spp"]), []).append(row)

    summary = []
    for spp in sorted(by_spp.keys()):
        rr = by_spp[spp]
        mean_wall, var_wall, _, _ = _finite_stats([r.get("wall_time_s") for r in rr])
        mean_mitsuba, var_mitsuba, _, _ = _finite_stats([r.get("mitsuba_time_s") for r in rr])
        mean_eff, var_eff, _, _ = _finite_stats([_time_value(r) for r in rr])

        mean_mse, var_mse, std_mse, _ = _finite_stats([r.get("mse") for r in rr])
        mean_rmse, var_rmse, std_rmse, _ = _finite_stats([r.get("rmse") for r in rr])
        mean_rel_mse, var_rel_mse, _, _ = _finite_stats([r.get("rel_mse") for r in rr])
        mean_rel_rmse, var_rel_rmse, _, _ = _finite_stats([r.get("rel_rmse") for r in rr])
        mean_E, var_E, _, _ = _finite_stats([r.get("E") for r in rr])
        mean_spatial_var, _, _, _ = _finite_stats([r.get("spatial_var") for r in rr])
        mean_flip, var_flip, std_flip, _ = _finite_stats([r.get("flip") for r in rr])

        summary.append({
            "spp": spp,
            "n_repeats": len(rr),
            "n_success": sum(1 for r in rr if _to_float(r.get("returncode")) == 0),
            "mean_wall_time_s": mean_wall, "var_wall_time_s": var_wall,
            "mean_mitsuba_time_s": mean_mitsuba, "var_mitsuba_time_s": var_mitsuba,
            "mean_effective_time_s": mean_eff, "var_effective_time_s": var_eff,
            "mean_mse": mean_mse, "var_mse": var_mse, "std_mse": std_mse,
            "mean_rmse": mean_rmse, "var_rmse": var_rmse, "std_rmse": std_rmse,
            "mean_rel_mse": mean_rel_mse, "var_rel_mse": var_rel_mse,
            "mean_rel_rmse": mean_rel_rmse, "var_rel_rmse": var_rel_rmse,
            "mean_E": mean_E, "var_E": var_E,
            "mean_spatial_var": mean_spatial_var,
            "mean_flip": mean_flip, "var_flip": var_flip, "std_flip": std_flip,
        })
    return summary


def build_per_config_metrics(output_root: Path, metrics_bbox=None):
    bbox_norm = _parse_bbox_normalized(metrics_bbox)

    config_dirs = iter_config_dirs(output_root)
    for cdir in _progress(config_dirs, desc="metrics configs", unit="cfg"):
        ref = cdir / "reference.exr"
        runs_dir = cdir / "runs"
        if not ref.exists() or not runs_dir.exists(): continue

        ref_img = load_img(ref)
        ref_window = _bbox_pixel_window_for_shape(ref_img.shape[:2], bbox_norm) if (ref_img is not None and bbox_norm) else None
        ref_eval_img = ref_img[ref_window[1]:ref_window[3], ref_window[0]:ref_window[2]] if ref_window else ref_img

        rows = []
        run_meta_files = sorted(runs_dir.glob("spp_*.meta.json"))
        for p in _progress(run_meta_files, desc=f"metrics runs [{cdir.name}]", unit="run", leave=False):
            parsed = parse_run_meta_filename(p.name)
            if not parsed: continue
            spp, repeat_idx = parsed
            exr = runs_dir / (f"spp_{spp:04d}_rep_{repeat_idx:03d}.exr" if "_rep_" in p.name else f"spp_{spp:04d}.exr")
            meta = json.loads(p.read_text(encoding="utf-8"))

            row = {
                "spp": spp, "repeat": repeat_idx,
                "returncode": meta.get("returncode"),
                "wall_time_s": meta.get("wall_time_s"),
                "mitsuba_time_s": meta.get("mitsuba_time_s"),
                "effective_time_s": _time_value(meta),
                "mse": "", "rmse": "", "rel_mse": "", "rel_rmse": "", 
                "E": "", "spatial_var": "", "flip": ""
            }

            if ref_eval_img is not None and exr.exists():
                img = load_img(exr)
                if img is not None and img.shape == ref_img.shape:
                    eval_img = img[ref_window[1]:ref_window[3], ref_window[0]:ref_window[2]] if ref_window else img
                    d = eval_img - ref_eval_img
                    
                    m = mse(eval_img, ref_eval_img)
                    rel_m = float(np.mean((d * d) / (ref_eval_img * ref_eval_img + 1e-3)))
                    
                    row["mse"] = m
                    row["rmse"] = math.sqrt(m)
                    row["rel_mse"] = rel_m
                    row["rel_rmse"] = math.sqrt(rel_m)
                    row["E"] = float(np.mean(d))
                    row["spatial_var"] = float(np.var(d))

                    if flip_eval is not None:
                        try:
                            ref_c, test_c = np.ascontiguousarray(ref_eval_img, dtype=np.float32), np.ascontiguousarray(eval_img, dtype=np.float32)
                            _, row["flip"], _ = flip_eval.evaluate(ref_c, test_c, dynamicRangeString="HDR")
                        except Exception as e:
                            pass

            rows.append(row)

        if not rows: continue
        rows.sort(key=lambda r: (int(r["spp"]), int(r.get("repeat", 1))))

        metrics_csv = cdir / "metrics.csv"
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        summary_rows = summarize_rows_by_spp(rows)
        with (cdir / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)


def _parse_run_stdout_for_timing(stdout_log_path: Path):
    if not stdout_log_path.exists():
        return {
            "render_time_s": float("nan"),
            "primitive_count": float("nan"),
            "bvh_build_time_s": float("nan"),
        }

    text = stdout_log_path.read_text(encoding="utf-8", errors="replace")

    render_time_s = float("nan")
    primitive_count = float("nan")
    bvh_build_time_s = float("nan")

    render_matches = re.findall(r"\[RenderJob\]\s+Render time:\s*([0-9]+(?:\.[0-9]+)?)s", text)
    if render_matches:
        render_time_s = _to_float(render_matches[-1])

    primitive_patterns = [
        r"Constructing a SAH kd-tree \((\d+) primitives\)",
        r"GeometryBVH:\s+Collecting\s+(\d+)\s+triangles",
        r"GeometryBVH built:\s+\d+ nodes,\s+(\d+) primitives",
    ]
    for pat in primitive_patterns:
        m = re.search(pat, text)
        if m:
            primitive_count = _to_float(m.group(1))
            break

    m_build = re.search(r"Geometry BVH build time:\s*([0-9]+(?:\.[0-9]+)?)s", text)
    if m_build:
        bvh_build_time_s = _to_float(m_build.group(1))

    return {
        "render_time_s": render_time_s,
        "primitive_count": primitive_count,
        "bvh_build_time_s": bvh_build_time_s,
    }


def _format_nan(v, fmt=".6f"):
    if not np.isfinite(_to_float(v)):
        return "nan"
    return format(float(v), fmt)


def _load_time_rows_per_config(output_root: Path):
    configs = []
    for cdir in _progress(iter_config_dirs(output_root), desc="time mode configs", unit="cfg"):
        if not (cdir / "config.json").exists():
            continue

        runs_dir = cdir / "runs"
        if not runs_dir.exists():
            continue

        params = json.loads((cdir / "config.json").read_text(encoding="utf-8"))
        rows = []
        for p in _progress(sorted(runs_dir.glob("spp_*.meta.json")), desc=f"time runs [{cdir.name}]", unit="run", leave=False):
            parsed = parse_run_meta_filename(p.name)
            if not parsed:
                continue
            spp, repeat_idx = parsed
            meta = json.loads(p.read_text(encoding="utf-8"))

            stdout_log = runs_dir / p.name.replace(".meta.json", ".stdout.log")
            parsed_log = _parse_run_stdout_for_timing(stdout_log)

            total_time_s = _time_value(meta)
            wall_time_s = _to_float(meta.get("wall_time_s"))
            render_time_s = parsed_log["render_time_s"]
            build_overhead_s = float("nan")

            # Prefer wall-clock subtraction for overhead: total process time minus pure render time.
            if np.isfinite(wall_time_s) and np.isfinite(render_time_s):
                build_overhead_s = wall_time_s - render_time_s

            # If subtraction is unavailable or non-positive, fall back to explicit BVH build time from the log.
            bvh_build_time_s = parsed_log["bvh_build_time_s"]
            if (not np.isfinite(build_overhead_s) or build_overhead_s <= 0.0) and np.isfinite(bvh_build_time_s):
                build_overhead_s = bvh_build_time_s

            render_time_per_sample_s = (render_time_s / float(spp)) if (np.isfinite(render_time_s) and spp > 0) else float("nan")

            row = {
                "spp": spp,
                "repeat": repeat_idx,
                "returncode": meta.get("returncode"),
                "wall_time_s": meta.get("wall_time_s"),
                "mitsuba_time_s": meta.get("mitsuba_time_s"),
                "effective_time_s": total_time_s,
                "render_time_s": render_time_s,
                "build_overhead_s": build_overhead_s,
                "render_time_per_sample_s": render_time_per_sample_s,
                "primitive_count": parsed_log["primitive_count"],
                "bvh_build_time_s": parsed_log["bvh_build_time_s"],
                "stdout_log": str(stdout_log),
            }
            rows.append(row)

        rows.sort(key=lambda r: (int(r["spp"]), int(r.get("repeat", 1))))
        if rows:
            configs.append({
                "cdir": cdir,
                "name": cdir.name,
                "params": params,
                "rows": rows,
            })

    return configs


def _compute_time_overhead_report(configs, baseline_config_name="baseline_path"):
    baseline_cfg = _find_baseline_config(configs, baseline_config_name=baseline_config_name)
    if baseline_cfg is None:
        raise ValueError(
            "--measure-time requires a PT baseline configuration (default name contains 'baseline_path')."
        )

    baseline_by_spp = {}
    for spp in sorted({int(r["spp"]) for r in baseline_cfg["rows"]}):
        rr = [r for r in baseline_cfg["rows"] if int(r["spp"]) == spp]
        mean_pt_tps, _, _, n_pt = _finite_stats([r.get("render_time_per_sample_s") for r in rr])
        baseline_by_spp[spp] = {
            "pt_render_time_per_sample_s": mean_pt_tps,
            "pt_n": n_pt,
        }

    report_rows = []
    for cfg in configs:
        by_spp = {}
        for r in cfg["rows"]:
            by_spp.setdefault(int(r["spp"]), []).append(r)

        for spp in sorted(by_spp.keys()):
            rr = by_spp[spp]
            mean_total, _, _, _ = _finite_stats([r.get("effective_time_s") for r in rr])
            mean_render, _, _, _ = _finite_stats([r.get("render_time_s") for r in rr])
            mean_build, _, _, _ = _finite_stats([r.get("build_overhead_s") for r in rr])
            mean_tps, _, _, n = _finite_stats([r.get("render_time_per_sample_s") for r in rr])
            mean_bvh_build_s, _, _, _ = _finite_stats([r.get("bvh_build_time_s") for r in rr])

            prim_vals = [_to_float(r.get("primitive_count")) for r in rr]
            prim_vals = [v for v in prim_vals if np.isfinite(v)]
            primitive_count = prim_vals[0] if prim_vals else float("nan")

            pt_tps = baseline_by_spp.get(spp, {}).get("pt_render_time_per_sample_s", float("nan"))
            if np.isfinite(mean_tps) and np.isfinite(pt_tps):
                render_overhead_ratio = mean_tps / pt_tps
                render_overhead_delta_s = mean_tps - pt_tps
            else:
                render_overhead_ratio = float("nan")
                render_overhead_delta_s = float("nan")

            report_rows.append({
                "config": cfg["name"],
                "display_name": _display_name(cfg["name"]),
                "spp": spp,
                "n_repeats": n,
                "primitive_count": primitive_count,
                "mean_total_time_s": mean_total,
                "mean_render_time_s": mean_render,
                "mean_build_overhead_s": mean_build,
                "mean_bvh_build_time_s": mean_bvh_build_s,
                "mean_render_time_per_sample_s": mean_tps,
                "pt_render_time_per_sample_s": pt_tps,
                "render_overhead_vs_pt_ratio": render_overhead_ratio,
                "render_overhead_vs_pt_delta_s_per_sample": render_overhead_delta_s,
            })

    report_rows.sort(key=lambda r: (str(r["config"]), int(r["spp"])))
    return report_rows, baseline_cfg


def _write_measure_time_outputs(output_root: Path, report_rows):
    if not report_rows:
        return None

    out_csv = output_root / "measure_time_overheads.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)

    return out_csv


def _print_measure_time_report(report_rows, baseline_cfg):
    print("\n" + "=" * 72)
    print("MEASURE-TIME REPORT")
    print("=" * 72)
    print(f"Baseline PT config: {baseline_cfg['name']}")

    grouped = {}
    for r in report_rows:
        grouped.setdefault(r["config"], []).append(r)

    for cfg_name in sorted(grouped.keys()):
        rows = sorted(grouped[cfg_name], key=lambda x: int(x["spp"]))
        disp = rows[0].get("display_name", cfg_name)
        prim = rows[0].get("primitive_count", float("nan"))
        prim_str = str(int(prim)) if np.isfinite(_to_float(prim)) else "unknown"
        print(f"\n[{disp}] ({cfg_name})")
        print(f"  primitive_count: {prim_str}")
        print("  SPP | build_overhead_s | render_spp_s | PT_render_spp_s | ratio_vs_PT")
        for r in rows:
            ratio = r.get("render_overhead_vs_pt_ratio")
            ratio_str = _format_nan(ratio, ".4f")
            print(
                "  {spp:>3d} | {build:>16} | {rsps:>12} | {pt:>15} | {ratio:>10}".format(
                    spp=int(r["spp"]),
                    build=_format_nan(r.get("mean_build_overhead_s")),
                    rsps=_format_nan(r.get("mean_render_time_per_sample_s")),
                    pt=_format_nan(r.get("pt_render_time_per_sample_s")),
                    ratio=ratio_str,
                )
            )

    print("=" * 72 + "\n")


def _write_measure_time_ratio_plot(output_root: Path, report_rows, baseline_cfg, image_format="png"):
    if plt is None or not report_rows:
        return None

    image_ext = str(image_format).lower()
    if image_ext not in ("png", "svg"):
        raise ValueError(f"Unsupported image format '{image_format}'. Use 'png' or 'svg'.")

    grouped = {}
    for r in report_rows:
        cfg = str(r.get("config", ""))
        grouped.setdefault(cfg, []).append(r)

    baseline_name = str(baseline_cfg.get("name", "")).lower()

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    plotted = 0
    for cfg_name in sorted(grouped.keys()):
        lowered = cfg_name.lower()
        if lowered == baseline_name or "baseline_pt" in lowered or "baseline_path" in lowered:
            continue

        rows = sorted(grouped[cfg_name], key=lambda x: int(x["spp"]))
        xs = []
        ys = []
        for r in rows:
            y = _to_float(r.get("render_overhead_vs_pt_ratio"))
            if not np.isfinite(y):
                continue
            xs.append(int(r["spp"]))
            ys.append(y)

        if not xs:
            continue

        label = _display_name(rows[0].get("display_name", cfg_name))
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.4, label=label)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Samples per pixel (SPP)")
    ax.set_ylabel("Render overhead vs PT (time/sample ratio)")
    ax.set_title("ratio_vs_pt vs spp")
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)

    out_dir = output_root / "measure_time"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ratio_vs_pt_vs_spp.{image_ext}"

    save_kwargs = {"bbox_inches": "tight"}
    if image_ext != "svg":
        save_kwargs["dpi"] = 180

    fig.tight_layout()
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)
    return out_path


def _write_measure_time_tps_plot(output_root: Path, report_rows, image_format="png"):
    if plt is None or not report_rows:
        return None

    image_ext = str(image_format).lower()
    if image_ext not in ("png", "svg"):
        raise ValueError(f"Unsupported image format '{image_format}'. Use 'png' or 'svg'.")

    grouped = {}
    for r in report_rows:
        cfg = str(r.get("config", ""))
        grouped.setdefault(cfg, []).append(r)

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    plotted = 0
    for cfg_name in sorted(grouped.keys()):
        rows = sorted(grouped[cfg_name], key=lambda x: int(x["spp"]))
        xs = []
        ys = []
        for r in rows:
            y = _to_float(r.get("mean_render_time_per_sample_s"))
            if not np.isfinite(y) or y <= 0:
                continue
            xs.append(int(r["spp"]))
            ys.append(y)

        if not xs:
            continue

        label = _display_name(rows[0].get("display_name", cfg_name))
        marker = "s" if "baseline" in cfg_name.lower() else "o"
        ax.plot(xs, ys, marker=marker, markersize=3, linewidth=1.4, label=label)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Samples per pixel (SPP)")
    ax.set_ylabel("Render time per sample (s/sample)")
    ax.set_title("time_per_spp vs spp")
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)

    out_dir = output_root / "measure_time"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"time_per_spp_vs_spp.{image_ext}"

    save_kwargs = {"bbox_inches": "tight"}
    if image_ext != "svg":
        save_kwargs["dpi"] = 180

    fig.tight_layout()
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)
    return out_path


def _extract_bvhdepth_from_config(config_name, params):
    m = re.search(r"bvhdepth[_-]?(\d+)", str(config_name).lower())
    if m:
        return int(m.group(1))

    def _search(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = str(k).lower()
                # Only treat BVH-specific keys as BVH depth.
                # Integrator maxDepth is unrelated and can otherwise inject
                # baseline/path-tracer points into BVH-depth plots.
                if key in ("bvhdepth", "bvh_depth", "bvhmaxdepth", "bvh_max_depth"):
                    fv = _to_float(v)
                    if np.isfinite(fv):
                        return int(round(fv))
                found = _search(v)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for it in obj:
                found = _search(it)
                if found is not None:
                    return found
        return None

    return _search(params)


def _write_measure_time_build_plots(output_root: Path, configs_time, image_format="png"):
    if plt is None or not configs_time:
        return None

    image_ext = str(image_format).lower()
    if image_ext not in ("png", "svg"):
        raise ValueError(f"Unsupported image format '{image_format}'. Use 'png' or 'svg'.")

    points = []
    for cfg in configs_time:
        depth = _extract_bvhdepth_from_config(cfg.get("name", ""), cfg.get("params", {}))
        build_values = [_to_float(r.get("bvh_build_time_s")) for r in cfg.get("rows", [])]
        build_values = [v for v in build_values if np.isfinite(v) and v > 0]
        if not build_values:
            build_values = [_to_float(r.get("build_overhead_s")) for r in cfg.get("rows", [])]
            build_values = [v for v in build_values if np.isfinite(v) and v > 0]
        if not build_values:
            continue

        prim_values = [_to_float(r.get("primitive_count")) for r in cfg.get("rows", [])]
        prim_values = [v for v in prim_values if np.isfinite(v) and v > 0]
        primitive_count = float(np.median(prim_values)) if prim_values else float("nan")

        points.append({
            "name": cfg.get("name", ""),
            "display_name": _display_name(cfg.get("name", "")),
            "bvhdepth": depth,
            "primitive_count": primitive_count,
            "build_time_s": float(np.mean(build_values)),
        })

    if not points:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax_depth, ax_prim = axes

    # Panel 1: build time vs bvh depth (log y)
    depth_points = [p for p in points if p["bvhdepth"] is not None]
    depth_points = sorted(depth_points, key=lambda p: (int(p["bvhdepth"]), p["name"]))
    if depth_points:
        xs = [int(p["bvhdepth"]) for p in depth_points]
        ys = [p["build_time_s"] for p in depth_points]
        ax_depth.plot(xs, ys, marker="o", linewidth=1.2)
    ax_depth.set_yscale("log")
    ax_depth.set_xlabel("BVH depth")
    ax_depth.set_ylabel("Build time (s)")
    ax_depth.set_title("build_time vs bvhdepth")
    ax_depth.grid(True, which="both", linestyle="--", alpha=0.25)

    # Panel 2: build time vs number of primitives (log-log)
    prim_points = [p for p in points if np.isfinite(_to_float(p["primitive_count"])) and p["primitive_count"] > 0]
    if prim_points:
        xs = [p["primitive_count"] for p in prim_points]
        ys = [p["build_time_s"] for p in prim_points]
        ax_prim.scatter(xs, ys, s=20)
    ax_prim.set_xscale("log")
    ax_prim.set_yscale("log")
    ax_prim.set_xlabel("Primitive count")
    ax_prim.set_ylabel("Build time (s)")
    ax_prim.set_title("build_time vs primitives")
    ax_prim.grid(True, which="both", linestyle="--", alpha=0.25)

    out_dir = output_root / "measure_time"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"build_time_vs_bvhdepth_and_primitives.{image_ext}"

    save_kwargs = {"bbox_inches": "tight"}
    if image_ext != "svg":
        save_kwargs["dpi"] = 180

    fig.tight_layout()
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)
    return out_path


def _write_measure_time_tps_vs_bvhdepth_plot(output_root: Path, configs_time, image_format="png"):
    if plt is None or not configs_time:
        return None

    image_ext = str(image_format).lower()
    if image_ext not in ("png", "svg"):
        raise ValueError(f"Unsupported image format '{image_format}'. Use 'png' or 'svg'.")

    min_spp_for_tps = 2 ** 4
    points = []
    for cfg in configs_time:
        depth = _extract_bvhdepth_from_config(cfg.get("name", ""), cfg.get("params", {}))
        if depth is None:
            continue

        eligible_rows = [r for r in cfg.get("rows", []) if int(_to_float(r.get("spp"))) >= min_spp_for_tps]
        tps_values = [_to_float(r.get("render_time_per_sample_s")) for r in eligible_rows]
        tps_values = [v for v in tps_values if np.isfinite(v) and v > 0]
        if not tps_values:
            continue

        points.append({
            "name": cfg.get("name", ""),
            "display_name": _display_name(cfg.get("name", "")),
            "bvhdepth": int(depth),
            "time_per_sample_s": float(np.mean(tps_values)),
        })

    if not points:
        return None

    points = sorted(points, key=lambda p: (p["bvhdepth"], p["name"]))
    xs = [p["bvhdepth"] for p in points]
    ys = [p["time_per_sample_s"] for p in points]
    # # select only depth>5 
    # xs = [x for x, y in zip(xs, ys) if x > 10 and np.isfinite(y) and y > 0]
    # ys = [y for x, y in zip(xs, ys) if x > 10 and np.isfinite(y) and y > 0]

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(xs, ys, marker="o", linewidth=1.2, label="Measured")

    fit_points = [p for p in points if 10 <= int(p["bvhdepth"]) <= 20]
    if len(fit_points) >= 2:
        fit_x = np.array([float(p["bvhdepth"]) for p in fit_points], dtype=float)
        fit_y = np.array([float(p["time_per_sample_s"]) for p in fit_points], dtype=float)
        slope, intercept = np.polyfit(fit_x, fit_y, 1)

        x_fit_min = int(np.min(fit_x))
        x_fit_max = int(np.max(fit_x))
        fit_line_x = np.array([x_fit_min, x_fit_max], dtype=float)
        fit_line_y = slope * fit_line_x + intercept
        # ax.plot(fit_line_x, fit_line_y, linestyle="--", linewidth=1.3, color="black", label="Fit (depth 10-20)")

        print(
            "time_per_spp_vs_bvhdepth fit [10..20]: "
            f"slope={slope:.6e}, intercept={intercept:.6e}"
        )
    else:
        print(
            "[WARN] time_per_spp_vs_bvhdepth fit skipped: "
            "need at least 2 points with BVH depth in [10, 20]."
        )

    # ax.set_yscale("log")
    ax.set_xlabel("BVH depth")
    ax.set_ylabel("Render time (s/sample)")
    ax.set_title("Time per sample")
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    # set ylim to 0, max*1.5
    if ys:
        ax.set_ylim(0, max(ys) * 1.5)
    # xlim to 0, max+5
    if xs:
        ax.set_xlim(0, max(xs) + 1)
    # ax.legend(loc="best", fontsize=7, frameon=False)

    out_dir = output_root / "measure_time"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"time_per_spp_vs_bvhdepth.{image_ext}"

    save_kwargs = {"bbox_inches": "tight"}
    if image_ext != "svg":
        save_kwargs["dpi"] = 180

    fig.tight_layout()
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)
    return out_path


def load_config_table(output_root: Path):
    table = []
    config_dirs = iter_config_dirs(output_root)
    for cdir in _progress(config_dirs, desc="load configs", unit="cfg"):
        if not (cdir / "config.json").exists() or not (cdir / "metrics_summary.csv").exists():
            continue
        params = json.loads((cdir / "config.json").read_text(encoding="utf-8"))
        rows = []
        with (cdir / "metrics_summary.csv").open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows.append({k: _to_float(v) for k, v in row.items()})
        rows.sort(key=lambda x: x["spp"])
        table.append({"cdir": cdir, "name": cdir.name, "params": params, "rows": rows})
    return table


def normalize_and_order_ablation_configs(configs, ablation_order, ablation_alias):
    if len(ablation_order) != len(ablation_alias):
        raise ValueError("ablation_order and ablation_alias must have the same length")

    canonical_map = {name.lower(): name for name in ablation_order}
    alias_to_canonical = {
        alias.lower(): canonical
        for alias, canonical in zip(ablation_alias, ablation_order)
    }

    selected = {}
    ignored_names = []
    for c in configs:
        raw_name = str(c.get("name", ""))
        lowered = raw_name.lower()

        canonical_name = None
        if lowered in canonical_map:
            canonical_name = canonical_map[lowered]
        elif lowered in alias_to_canonical:
            canonical_name = alias_to_canonical[lowered]

        if canonical_name is None:
            ignored_names.append(raw_name)
            continue

        c2 = dict(c)
        c2["name"] = canonical_name

        # Prefer an exact canonical-name match over an alias duplicate.
        existing = selected.get(canonical_name)
        if existing is None:
            selected[canonical_name] = c2
        else:
            existing_was_exact = existing.get("_ablation_exact_match", False)
            this_is_exact = lowered == canonical_name.lower()
            if this_is_exact and not existing_was_exact:
                selected[canonical_name] = c2

        selected[canonical_name]["_ablation_exact_match"] = (lowered == canonical_name.lower())

    ordered = []
    for name in ablation_order:
        cfg = selected.get(name)
        if cfg is not None:
            cfg.pop("_ablation_exact_match", None)
            ordered.append(cfg)

    return ordered, ignored_names


def detect_bias(configs):
    print("\n" + "="*40)
    print("BIAS DETECTION (Highest SPP Analysis)")
    print("="*40)
    found = False
    for c in configs:
        if not c["rows"]: continue
        best = c["rows"][-1]
        E, spatial_var, mse = best.get("mean_E", 0), best.get("mean_spatial_var", 0), best.get("mean_mse", 0)
        
        # Bias-Variance decomposition logic: If systematic error squared overtakes the noise/spatial variance
        if E**2 > spatial_var and mse > 1e-6:
            print(f"[WARN] Technique '{_display_name(c['name'])}' exhibits likely bias at SPP={int(best['spp'])}.")
            print(f"       Systematic Error (E^2): {E**2:.2e}  |  Spatial Noise Var: {spatial_var:.2e}")
            found = True
            
    if not found:
        print("No significantly biased techniques detected.")
    print("="*40 + "\n")

def export_summary_grid(configs, output_root, inset_bbox, mode="spp", target=8, exposure_scale=1.0, output_dir=None, image_ext="png", output_name=None):
    if plt is None or patches is None: return
    
    picks = []
    for c in _progress(configs, desc=f"grid picks ({mode})", unit="cfg"):
        runs_dir = c["cdir"] / "runs"
        valid = []
        for p in _progress(list(runs_dir.glob("spp_*.meta.json")), desc=f"grid scan [{c['name']}]", unit="run", leave=False):
            spp, _ = parse_run_meta_filename(p.name)
            t = _time_value(json.loads(p.read_text()))
            exr = runs_dir / p.name.replace(".meta.json", ".exr")
            if exr.exists(): valid.append({"spp": spp, "time": t, "path": exr})
            
        if not valid: continue
        
        candidates = [r for r in valid if r["spp"] == target] if mode == "spp" else [r for r in valid if r["time"] <= target]
        if not candidates: continue
        candidates.sort(key=lambda r: (r["spp"], -r["time"]), reverse=True) # Max SPP, then min time
        
        picks.append((_display_name(c["name"]), candidates[0]))
        
    if not picks: 
        return

    # Match subplot geometry to image/crop aspect to avoid empty margins for non-square images.
    ref_img = load_img(picks[0][1]["path"])
    if ref_img is None:
        return
    ref_h, ref_w = ref_img.shape[:2]
    if ref_h <= 0 or ref_w <= 0:
        return

    main_box_aspect = float(ref_h) / float(ref_w)
    if inset_bbox:
        x0n, y0n, x1n, y1n = inset_bbox
        x0_ref = int(x0n * ref_w)
        y0_ref = int(y0n * ref_h)
        x1_ref = int(x1n * ref_w)
        y1_ref = int(y1n * ref_h)
        zoom_w = max(1, x1_ref - x0_ref)
        zoom_h = max(1, y1_ref - y0_ref)
        zoom_box_aspect = float(zoom_h) / float(zoom_w)
    else:
        zoom_box_aspect = main_box_aspect

    ncols = len(picks)
    cell_w_in = 2.8
    fig_width = max(8.0, cell_w_in * ncols)
    main_h_in = cell_w_in * main_box_aspect
    zoom_h_in = cell_w_in * zoom_box_aspect
    fig_height = max(1.0, main_h_in + zoom_h_in)

    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(fig_width, fig_height),
        gridspec_kw={"wspace": 0.01, "hspace": 0.0, "height_ratios": [main_box_aspect, zoom_box_aspect]},
    )
    if len(picks) == 1:
        axes = np.array(axes).reshape(2, 1)
    
    for i, (name, pick) in enumerate(_progress(picks, desc=f"grid render ({mode})", unit="img")):
        img = load_img(pick["path"])
        disp = np.maximum(img, 0.0)
        h, w = disp.shape[:2]
        
        # Determine bbox coordinates
        if inset_bbox:
            x0n, y0n, x1n, y1n = inset_bbox
            x0, y0, x1, y1 = int(x0n*w), int(y0n*h), int(x1n*w), int(y1n*h)
            crop = disp[y0:y1, x0:x1]
            # Compute exposure scaling strictly inside the bbox
            p99 = np.percentile(crop, 99.0) if crop.size > 0 else 1.0
        else:
            p99 = np.percentile(disp, 99.0) if disp.size > 0 else 1.0
            
        # Apply Tone Mapping using the bbox-derived exposure
        if p99 > 0: 
            disp = exposure_scale * disp / p99
        disp = np.clip(np.power(disp / (1.0 + disp), 1/2.2), 0, 1)
        
        ax_main = axes[0, i]
        ax_zoom = axes[1, i]
        ax_main.set_box_aspect(main_box_aspect)
        ax_zoom.set_box_aspect(zoom_box_aspect)
        ax_main.imshow(disp)
        ax_main.set_axis_off()
        ax_main.set_title(f"{name}\nSPP={pick['spp']} | {pick['time']:.2f}s", fontsize=18, pad=1, y=1.01)
        
        if inset_bbox:
            rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor="yellow", facecolor="none")
            ax_main.add_patch(rect)
            ax_zoom.imshow(disp[y0:y1, x0:x1])
        else:
            ax_zoom.imshow(disp) # Fallback to full if no bbox
            
        ax_zoom.set_axis_off()

    if inset_bbox:
        axes[1, 0].set_ylabel("Zoom", fontsize=8)

    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.0, top=1.0, wspace=0.01, hspace=0.0)
    if output_dir is None:
        output_dir = output_root
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_ext = str(image_ext).lower()
    if output_name is None:
        output_name = f"summary_grid_{mode}_{target}"

    save_kwargs = {"bbox_inches": "tight"}
    if image_ext != "svg":
        save_kwargs["dpi"] = 180

    out_path = output_dir / f"{output_name}.{image_ext}"
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)
    return out_path

def _latex_include_line(rel_path: str, image_ext: str, width="0.98\\columnwidth"):
    if image_ext == "svg":
        rel_no_ext = rel_path[:-4] if rel_path.endswith(".svg") else rel_path
        return f"  \\includesvg[width={width}]{{{rel_no_ext}}}"
    return f"  \\includegraphics[width={width}]{{{rel_path}}}"


def _write_convergence_latex_file(latex_dir: Path, test_name: str, image_ext: str, selected_plots=None):
    tex_path = latex_dir / f"{test_name}_figures.tex"
    if selected_plots is None:
        selected_plots = LATEX_DEFAULT_PLOTS

    fig_prefix = f"figs/{test_name}"
    lines = [
        "% Auto-generated by scripts/experiments/analyze_results.py.",
        "% This file is overwritten on every analysis run",
        "",
        f"Figure \\ref{{fig:{test_name}:qualitative:spp}} shows qualitative results at equal samples (8 SPP), and Figure \\ref{{fig:{test_name}:qualitative:time}} shows qualitative results at approximately equal render time.",
        "",
        f"Figures \\ref{{fig:{test_name}:quant:mean_flip:spp}}, \\ref{{fig:{test_name}:quant:mean_flip:time}}, and \\ref{{fig:{test_name}:quant:mean_E:spp}} summarize the quantitative behavior for the selected metrics.",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        _latex_include_line(f"{fig_prefix}/summary_grid_spp.{image_ext}", image_ext=image_ext),
        f"  \\caption{{TODO: Qualitative comparison at equal samples (8 SPP) for {test_name}.}}",
        f"  \\label{{fig:{test_name}:qualitative:spp}}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "  \\centering",
        _latex_include_line(f"{fig_prefix}/summary_grid_time.{image_ext}", image_ext=image_ext),
        f"  \\caption{{TODO: Qualitative comparison at approximately equal render time for {test_name}.}}",
        f"  \\label{{fig:{test_name}:qualitative:time}}",
        "\\end{figure}",
        "",
    ]

    metric_label_map = {
        "mean_mse": "MSE",
        "mean_rmse": "RMSE",
        "mean_rel_mse": "rel MSE",
        "mean_rel_rmse": "rel RMSE",
        "mean_E": "Mean Error",
        "mean_spatial_var": "Spatial Variance of Error",
        "var_mse": "Variance of MSE",
        "var_rmse": "Variance of RMSE",
        "mean_flip": "FLIP",
        "var_flip": "Variance of FLIP",
    }

    for metric_key, x_axis in selected_plots:
        metric_label = metric_label_map.get(metric_key, metric_key)
        axis_label = "SPP" if x_axis == "spp" else "time"
        rel_path = f"{fig_prefix}/logscale_y/{metric_key}_vs_{x_axis}.{image_ext}"
        lines.extend([
            "\\begin{figure}[htbp]",
            "  \\centering",
            _latex_include_line(rel_path, image_ext=image_ext),
            f"  \\caption{{TODO: {metric_label} versus {axis_label}.}}",
            f"  \\label{{fig:{test_name}:quant:{metric_key}:{x_axis}}}",
            "\\end{figure}",
            "",
        ])

    tex_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Written LaTeX figure file: {tex_path}")


def _find_baseline_config(configs, baseline_config_name=None):
    if baseline_config_name:
        exact = next((c for c in configs if c["name"] == baseline_config_name), None)
        if exact is not None:
            return exact

    fallback_names = ["baseline_path", "baseline_pt", "baseline"]
    for key in fallback_names:
        match = next((c for c in configs if key in c["name"].lower()), None)
        if match is not None:
            return match
    return None


def _find_bdpt_config(configs):
    exact = next((c for c in configs if c["name"].lower() == "bdpt"), None)
    if exact is not None:
        return exact
    return next((c for c in configs if "bdpt" in c["name"].lower()), None)


def _find_best_non_pinned_name(configs, baseline_config_name=None, rank_spp=8):
    baseline_cfg = _find_baseline_config(configs, baseline_config_name=baseline_config_name)
    bdpt_cfg = _find_bdpt_config(configs)

    pinned_names = set()
    if baseline_cfg is not None:
        pinned_names.add(baseline_cfg["name"])
    if bdpt_cfg is not None:
        pinned_names.add(bdpt_cfg["name"])

    ranked = []
    for c in configs:
        if c["name"] in pinned_names:
            continue

        valid_rows = [r for r in c["rows"] if np.isfinite(_to_float(r.get("mean_mse")))]
        if not valid_rows:
            continue

        row_at_rank_spp = next(
            (
                r for r in valid_rows
                if int(round(_to_float(r.get("spp")))) == int(rank_spp)
            ),
            None,
        )
        metric_row = row_at_rank_spp if row_at_rank_spp is not None else valid_rows[-1]
        ranked.append((_to_float(metric_row.get("mean_mse")), c["name"]))

    ranked.sort(key=lambda t: t[0])
    return ranked[0][1] if ranked else None


def select_top_techniques(configs, n_top_techniques=None, baseline_config_name=None, rank_spp=8):
    if n_top_techniques is None:
        return list(configs)

    n_top_techniques = int(n_top_techniques)
    if n_top_techniques < 0:
        raise ValueError(f"Invalid n_top_techniques={n_top_techniques}. It must be >= 0.")

    baseline_cfg = _find_baseline_config(configs, baseline_config_name=baseline_config_name)
    bdpt_cfg = _find_bdpt_config(configs)

    pinned_names = set()
    if baseline_cfg is not None:
        pinned_names.add(baseline_cfg["name"])
    if bdpt_cfg is not None:
        pinned_names.add(bdpt_cfg["name"])

    ranked = []
    for c in configs:
        if c["name"] in pinned_names:
            continue

        valid_rows = [r for r in c["rows"] if np.isfinite(_to_float(r.get("mean_mse")))]
        if not valid_rows:
            continue

        row_at_rank_spp = next(
            (
                r for r in valid_rows
                if int(round(_to_float(r.get("spp")))) == int(rank_spp)
            ),
            None,
        )
        metric_row = row_at_rank_spp if row_at_rank_spp is not None else valid_rows[-1]
        ranked.append((_to_float(metric_row.get("mean_mse")), c["name"]))

    ranked.sort(key=lambda t: t[0])
    selected_names = set(name for _, name in ranked[:n_top_techniques])
    selected_names.update(pinned_names)

    selected = [c for c in configs if c["name"] in selected_names]

    print(f"Selected {len(selected)} techniques for plotting (PT + BDPT if present + top {n_top_techniques} by MSE):")
    for c in selected:
        print(f"  - {_display_name(c['name'])}")

    return selected


def write_convergence_plots(data_root: Path, output_dir: Path = None, logspace=True, image_format="png", configs=None):
    if plt is None:
        return

    if output_dir is None:
        output_dir = data_root
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_ext = str(image_format).lower()
    if image_ext not in ("png", "svg"):
        raise ValueError(f"Unsupported image format '{image_format}'. Use 'png' or 'svg'.")

    if configs is None:
        configs = load_config_table(data_root)
    if not configs:
        return

    # Build a stable per-technique color mapping using a colorblind-friendly colormap.
    technique_names = [c["name"] for c in configs]
    
    # Use a qualitative colormap designed for discrete categories
    cmap = plt.get_cmap("tab20")
    
    # Assign colors dynamically, wrapping around if there are >20 techniques
    technique_colors = {
        name: cmap(i % cmap.N) for i, name in enumerate(technique_names)
    }   
    metrics = [
        ("mean_mse",  "MSE"),
        ("mean_rmse", "RMSE"),
        ("mean_rel_mse", "Relative MSE"),
        ("mean_rel_rmse", "Relative RMSE"),
        ("mean_E", "Mean Error (Bias)"),
        ("mean_spatial_var", "Spatial Variance of Error"),
        ("var_mse",   "Variance of MSE"),
        ("var_rmse",  "Variance of RMSE"),
        ("mean_flip", "FLIP"),
        ("var_flip",  "Variance of FLIP"),
    ]

    fig_save_kwargs = {"bbox_inches": "tight"}
    if image_ext != "svg":
        fig_save_kwargs["dpi"] = 180

    # Dimensions tuned for two-column LaTeX documents.
    fig_size = (4.2, 2.6)

    plot_desc = "plots log" if logspace else "plots linear"
    for metric_key, metric_label in _progress(metrics, desc=plot_desc, unit="metric"):
        # vs SPP
        fig, ax = plt.subplots(figsize=fig_size)
        for c in _progress(configs, desc=f"{metric_key} vs spp", unit="cfg", leave=False):
            valid_rows = [r for r in c["rows"] if np.isfinite(_to_float(r.get(metric_key)))]
            if not valid_rows:
                continue

            is_baseline_path = "baseline_path" in c["name"].lower()
            marker = "s" if is_baseline_path else "o"
            
            xs = [r["spp"] for r in valid_rows]
            ys = [_to_float(r.get(metric_key)) for r in valid_rows]
            
            if metric_key == "mean_E":
                # Compute standard deviation for error bars
                yerr = [math.sqrt(max(0.0, _to_float(r.get("var_E", 0)))) for r in valid_rows]
                ax.errorbar(xs, ys, yerr=yerr, marker=marker, markersize=3, linewidth=1.4, 
                            label=_display_name(c["name"]), capsize=3, color=technique_colors[c["name"]])
            else:
                ax.plot(xs, ys, marker=marker, markersize=3, linewidth=1.4, 
                        label=_display_name(c["name"]), color=technique_colors[c["name"]])
            
        if logspace:
            if metric_key == "mean_E":
                ax.set_yscale("symlog", linthresh=1e-6)
            else:
                ax.set_yscale("log")
            
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Samples per pixel (SPP)")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} vs SPP")
        ax.grid(True, which="both", linestyle="--", alpha=0.25)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric_key}_vs_spp.{image_ext}", **fig_save_kwargs)
        plt.close(fig)

        # vs time
        fig, ax = plt.subplots(figsize=fig_size)
        for c in _progress(configs, desc=f"{metric_key} vs time", unit="cfg", leave=False):
            valid_rows = [r for r in c["rows"] 
                          if np.isfinite(_to_float(r.get("mean_effective_time_s"))) 
                          and np.isfinite(_to_float(r.get(metric_key)))]
            if not valid_rows:
                continue

            is_baseline_path = "baseline_path" in c["name"].lower()
            marker = "s" if is_baseline_path else "o"
                
            xs = [_to_float(r.get("mean_effective_time_s")) for r in valid_rows]
            ys = [_to_float(r.get(metric_key)) for r in valid_rows]
            
            if metric_key == "mean_E":
                yerr = [math.sqrt(max(0.0, _to_float(r.get("var_E", 0)))) for r in valid_rows]
                ax.errorbar(xs, ys, yerr=yerr, marker=marker, markersize=3, linewidth=1.4, 
                            label=_display_name(c["name"]), capsize=3, color=technique_colors[c["name"]])
            else:
                ax.plot(xs, ys, marker=marker, markersize=3, linewidth=1.4, 
                        label=_display_name(c["name"]), color=technique_colors[c["name"]])
            
        if logspace:
            if metric_key == "mean_E":
                ax.set_yscale("symlog", linthresh=1e-5)
            else:
                ax.set_yscale("log")
            
        ax.set_xscale("log")
        ax.set_xlabel("Effective render time (s)")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} vs time")
        ax.grid(True, which="both", linestyle="--", alpha=0.25)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric_key}_vs_time.{image_ext}", **fig_save_kwargs)
        plt.close(fig)

    return output_dir


def select_grid_plot_configs(configs, rank_spp=16, n_best=4, baseline_config_name=None):
    baseline_cfg = _find_baseline_config(configs, baseline_config_name=baseline_config_name)
    bdpt_cfg = _find_bdpt_config(configs)

    pinned_names = set()
    if baseline_cfg is not None:
        pinned_names.add(baseline_cfg["name"])
    if bdpt_cfg is not None:
        pinned_names.add(bdpt_cfg["name"])

    ranked = []
    for c in configs:
        if c["name"] in pinned_names:
            continue

        row_at_spp = next(
            (
                r for r in c["rows"]
                if int(round(_to_float(r.get("spp")))) == int(rank_spp)
                and np.isfinite(_to_float(r.get("mean_mse")))
            ),
            None,
        )
        if row_at_spp is None:
            continue

        ranked.append((_to_float(row_at_spp.get("mean_mse")), c))

    ranked.sort(key=lambda t: t[0])

    selected = []
    if baseline_cfg is not None:
        selected.append(baseline_cfg)
    if bdpt_cfg is not None and (baseline_cfg is None or bdpt_cfg["name"] != baseline_cfg["name"]):
        selected.append(bdpt_cfg)
    selected.extend([cfg for _, cfg in ranked[:n_best]])

    print(f"Selected {len(selected)} techniques for grid plots (PT + BDPT if present + top {n_best} at SPP={rank_spp}):")
    for c in selected:
        print(f"  - {_display_name(c['name'])}")

    return selected

def print_top_and_worst(configs):
    print("\n" + "="*40)
    print("PERFORMANCE RANKING (Lowest rel MSE at Highest SPP)")
    print("="*40)
    
    ranking = []
    for c in configs:
        valid_rows = [r for r in c["rows"] if np.isfinite(_to_float(r.get("mean_rel_mse")))]
        if not valid_rows: continue
        best_row = valid_rows[-1] # Rows are already sorted by SPP
        ranking.append({
            "name": c["name"], 
            "spp": int(best_row["spp"]), 
            "relmse": _to_float(best_row["mean_rel_mse"])
        })
        
    if not ranking:
        print("No valid data for ranking.")
        return
        
    ranking.sort(key=lambda x: x["relmse"]) # Sort ascending (lower MSE is better)
    
    print("TOP 5 TECHNIQUES:")
    for i, res in enumerate(ranking[:5], 1):
        print(f"  {i}. {_display_name(res['name'])} (relMSE: {res['relmse']:.2e} @ SPP={res['spp']})")
        
    print("\nWORST 5 TECHNIQUES:")
    worst = ranking[-5:]
    worst.reverse() # Print absolute worst first
    for i, res in enumerate(worst, 1):
        print(f"  {i}. {_display_name(res['name'])} (relMSE: {res['relmse']:.2e} @ SPP={res['spp']})")
    print("="*40 + "\n")


from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
def export_composite_pdf_figure(configs, output_root, inset_bbox, target_spp, target_time, 
                                exposure_scale=1.0, output_name="composite_figure", 
                                metric_key="mean_flip", fallback_metric="mean_mse"):
    if plt is None or patches is None: return
    
    has_metric = any(np.isfinite(_to_float(r.get(metric_key))) for c in configs for r in c["rows"])
    if not has_metric:
        metric_key = fallback_metric

    metric_labels = {
        "mean_flip": "FLIP vs ref",
        "mean_mse": "MSE vs ref"
    }
    metric_label = metric_labels.get(metric_key, metric_key)

    def get_picks(mode, target):
        picks = []
        for c in _progress(configs, desc=f"composite picks ({mode})", unit="cfg", leave=False):
            runs_dir = c["cdir"] / "runs"
            valid = []
            for p in _progress(list(runs_dir.glob("spp_*.meta.json")), desc=f"composite scan [{c['name']}]", unit="run", leave=False):
                spp, _ = parse_run_meta_filename(p.name)
                t = _time_value(json.loads(p.read_text()))
                exr = runs_dir / p.name.replace(".meta.json", ".exr")
                if exr.exists(): valid.append({"spp": spp, "time": t, "path": exr})
            if not valid: continue
            
            if mode == "spp":
                candidates = [r for r in valid if r["spp"] == target]
            else:
                candidates = [r for r in valid if r["time"] <= target]
            
            if not candidates: continue
            candidates.sort(key=lambda r: (r["spp"], -r["time"]), reverse=True)
            picks.append((_display_name(c["name"]), candidates[0]))
        return picks

    picks_spp = get_picks("spp", target_spp)
    picks_time = get_picks("time", target_time)
    
    if not picks_spp or not picks_time:
        print("[WARN] Not enough data to generate composite PDF figure.")
        return

    # Slightly wider figure to give the right column more breathing room
    fig = plt.figure(figsize=(5.6, 3.2))
    
    # Outer GridSpec: Give the left column slightly more width ratio
    # wspace is smaller because we are manually adding a left-margin to the plots
    gs_outer = gridspec.GridSpec(1, 2, width_ratios=[1.7, 1.15], wspace=0.15)
    
    # Inner Left GridSpec: Keep image rows close together
    gs_left = gs_outer[0].subgridspec(2, 1, hspace=0.25)
    
    # Inner Right GridSpec: Set hspace to 0.0 because vertical margins are handled per-plot below
    gs_right = gs_outer[1].subgridspec(2, 1, hspace=0.0) 
    
    box_color_top = "#A99ECE" # Purple-ish
    box_color_bot = "#A2C195" # Green-ish
    colors = {"PT": "#1f77b4", "BDPT": "#aec7e8", "Ours": "#ff7f0e"}
    
    def draw_row(picks, target_val, gs_img_cell, gs_plot_cell, box_color, mode):
        # 1. Background axis for the colored bounding box around the image grid
        ax_bg = fig.add_subplot(gs_img_cell)
        ax_bg.axis("off")
        rect = FancyBboxPatch((-0.03, 0.01), 1.06, 1.21, boxstyle="round,pad=0.02", 
                              edgecolor=box_color, facecolor="none", linewidth=1.5, 
                              transform=ax_bg.transAxes, clip_on=False)
        ax_bg.add_patch(rect)
        
        # 2. Subgridspec for the images
        gs_img = gs_img_cell.subgridspec(2, 3, wspace=0.03, hspace=0.03)
        for i, (name, pick) in enumerate(picks):
            if i >= 3: break
            
            img = load_img(pick["path"])
            disp = np.maximum(img, 0.0)
            h, w = disp.shape[:2]
            
            x0, y0, x1, y1 = 0, 0, w, h
            if inset_bbox:
                x0n, y0n, x1n, y1n = inset_bbox
                x0, y0, x1, y1 = int(x0n*w), int(y0n*h), int(x1n*w), int(y1n*h)
                crop = disp[y0:y1, x0:x1]
                p99 = np.percentile(crop, 99.0) if crop.size > 0 else 1.0
            else:
                p99 = np.percentile(disp, 99.0) if disp.size > 0 else 1.0
                
            if p99 > 0: disp = exposure_scale * disp / p99
            disp = np.clip(np.power(disp / (1.0 + disp), 1/2.2), 0, 1)
            
            # Full Image
            ax_main = fig.add_subplot(gs_img[0, i])
            ax_main.imshow(disp)
            ax_main.set_axis_off()
            
            # Larger text inside the narrow space
            time_str = f"{pick['time']:.2f}s" if pick['time'] < 10 else f"{int(pick['time'])}s"
            ax_main.set_title(f"{name}\n{time_str} | {pick['spp']} spp", fontsize=6, pad=2)
            
            # Zoom Image
            ax_zoom = fig.add_subplot(gs_img[1, i])
            if inset_bbox:
                rect_zoom = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=1, edgecolor="yellow", facecolor="none")
                ax_main.add_patch(rect_zoom)
                ax_zoom.imshow(disp[y0:y1, x0:x1])
            else:
                ax_zoom.imshow(disp)
            ax_zoom.set_axis_off()

        # 3. Line Plot on the right
        # --- FIX: Create an inner subgridspec to act as structural margins ---
        # This explicitly restricts the plot size and leaves safe zones for the labels
        gs_plot_margin = gs_plot_cell.subgridspec(
            3, 3, 
            width_ratios=[0.25, 0.70, 0.05], # Left space (ylabel), Plot width, Right space
            height_ratios=[0.10, 0.65, 0.25] # Top space, Plot height, Bottom space (xlabel)
        )
        ax_plot = fig.add_subplot(gs_plot_margin[1, 1]) # Place plot strictly in the center cell
        
        for c in configs:
            cname = _display_name(c["name"])
            valid_rows = [r for r in c["rows"] if np.isfinite(_to_float(r.get(metric_key)))]
            if not valid_rows: continue
            
            if mode == "spp":
                xs = [r["spp"] for r in valid_rows]
                ax_plot.set_xlabel("Samples per pixel (SPP)", fontsize=6, labelpad=2)
            else:
                xs = [_to_float(r.get("mean_effective_time_s")) for r in valid_rows]
                ax_plot.set_xlabel("Effective time (s)", fontsize=6, labelpad=2)
                
            ys = [_to_float(r.get(metric_key)) for r in valid_rows]
            col = colors.get(cname, None)
            marker = "s" if "PT" in cname else "o"
            
            ax_plot.plot(xs, ys, marker=marker, markersize=3, linewidth=1.2, label=cname, color=col)

        # Draw the target line
        ax_plot.axvline(x=target_val, color=box_color, linestyle="--", linewidth=2.5, alpha=0.9)
        
        # Configure axes for narrow column
        ax_plot.set_yscale("log")
        if mode == "spp": ax_plot.set_xscale("log", base=2)
        else: ax_plot.set_xscale("log")
        
        ax_plot.set_ylabel(metric_label, fontsize=6, labelpad=2) 
        ax_plot.grid(True, which="both", linestyle="--", alpha=0.3)
        ax_plot.tick_params(axis='both', which='major', labelsize=6.5, pad=2)
        
        # Embed legend entirely inside the plot box
        ax_plot.legend(loc="lower left", fontsize=5, framealpha=0.8, handlelength=1.5, handletextpad=0.4, borderpad=0.3)

    # Pass cells from the separate left/right layouts explicitly
    draw_row(picks_spp, target_spp, gs_left[0], gs_right[0], box_color_top, "spp")
    draw_row(picks_time, target_time, gs_left[1], gs_right[1], box_color_bot, "time")

    out_path = output_root / f"{output_name}.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print(f"Exported composite PDF figure to {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_file")
    ap.add_argument("--measure-time", action="store_true", default=None,
                    help="Timing-only mode: skip EXR loading/figures and report BVH build/render overheads.")
    ap.add_argument("--is-ablation", action="store_true", default=None,
                    help="Enable ablation mode: normalize aliases and force ablation_order plotting")
    ap.add_argument("--figure-format", choices=["png", "svg"], default=None,
                    help="Output format for convergence plots (default: from YAML analysis.figure_format or png)")
    ap.add_argument("--exposure-scale", type=float, default=None,
                    help="Exposure scale for summary grid tonemapping (default: from YAML analysis.exposure_scale or 1.0)")
    args = ap.parse_args()

    exp = yaml.safe_load(Path(args.exp_file).read_text(encoding="utf-8"))
    output_root = Path(exp["output_root"])

    analysis_cfg = exp.get("analysis", {})
    figure_format = (args.figure_format or analysis_cfg.get("figure_format", "png")).lower()
    measure_time = bool(analysis_cfg.get("measure_time", False)) if args.measure_time is None else bool(args.measure_time)

    if measure_time:
        baseline_config_name = analysis_cfg.get("baseline_config_name", "baseline_path")
        configs_time = _load_time_rows_per_config(output_root)
        if not configs_time:
            raise ValueError(f"No timing run data found under: {output_root}")

        report_rows, baseline_cfg = _compute_time_overhead_report(
            configs_time,
            baseline_config_name=baseline_config_name,
        )
        if not report_rows:
            raise ValueError("No timing overhead rows could be computed.")

        out_csv = _write_measure_time_outputs(output_root, report_rows)
        out_plot = _write_measure_time_ratio_plot(
            output_root,
            report_rows,
            baseline_cfg,
            image_format=figure_format,
        )
        out_tps_plot = _write_measure_time_tps_plot(
            output_root,
            report_rows,
            image_format=figure_format,
        )
        out_build_plot = _write_measure_time_build_plots(
            output_root,
            configs_time,
            image_format=figure_format,
        )
        out_tps_vs_depth_plot = _write_measure_time_tps_vs_bvhdepth_plot(
            output_root,
            configs_time,
            image_format=figure_format,
        )
        _print_measure_time_report(report_rows, baseline_cfg)
        if out_csv is not None:
            print(f"Wrote timing overhead CSV: {out_csv}")
        if out_plot is not None:
            print(f"Wrote timing overhead plot: {out_plot}")
        if out_tps_plot is not None:
            print(f"Wrote time-per-sample plot: {out_tps_plot}")
        if out_build_plot is not None:
            print(f"Wrote build-time plot: {out_build_plot}")
        if out_tps_vs_depth_plot is not None:
            print(f"Wrote time-per-sample vs BVH-depth plot: {out_tps_vs_depth_plot}")
        print(f"Measure-time analysis done: {output_root}")
        return

    global EXR_ASPECT_CROP
    EXR_ASPECT_CROP = _parse_aspect_ratio_crop(analysis_cfg)
    if EXR_ASPECT_CROP is not None:
        ax, ay, anchor_y = EXR_ASPECT_CROP
        print(f"Aspect-ratio pre-crop enabled for EXRs: aspect_x={ax:g}, aspect_y={ay:g}, anchor_y={anchor_y}")

    metrics_bbox = analysis_cfg.get("average_bbox", [0.7, 0.7, 0.75, 0.75])
    metrics_in_bbox = bool(analysis_cfg.get("metrics_in_bbox", False))
    is_ablation = bool(analysis_cfg.get("is_ablation", False)) if args.is_ablation is None else bool(args.is_ablation)
    baseline_config_name = analysis_cfg.get("baseline_config_name", "baseline_path")
    n_top_techniques = analysis_cfg.get("n_top_techniques", None)
    exposure_scale = args.exposure_scale if args.exposure_scale is not None else analysis_cfg.get("exposure_scale", 1.0)
    exposure_scale = float(exposure_scale)
    if not np.isfinite(exposure_scale) or exposure_scale <= 0:
        raise ValueError(f"Invalid exposure_scale={exposure_scale}. It must be a finite value > 0.")
    if n_top_techniques is not None:
        n_top_techniques = int(n_top_techniques)
        if n_top_techniques < 0:
            raise ValueError(f"Invalid n_top_techniques={n_top_techniques}. It must be >= 0.")

    ablation_order = ["PT", "Ours", "no_gaussian", "no_vmf", "no_Le", "no_bsdf", "all_ablation"]
    ablation_alias = ["baseline_path", "ours", "nogaussian", "novmf", "nocosinexe", "nocosinexs", "allablation"]


    # 1. Process Metrics
    metrics_bbox_for_eval = metrics_bbox if metrics_in_bbox else None
    build_per_config_metrics(output_root, metrics_bbox=metrics_bbox_for_eval)
    
    # 2. Load Table
    configs = load_config_table(output_root)
    if is_ablation:
        configs, ignored_ablation_configs = normalize_and_order_ablation_configs(configs, ablation_order, ablation_alias)
        if not configs:
            raise ValueError(
                "is_ablation=True but no configuration matched ablation_order/ablation_alias"
            )
        baseline_config_name = "PT"

        if ignored_ablation_configs:
            print("Ablation mode: ignoring configs not listed in ablation_order/ablation_alias:")
            for name in sorted(set(ignored_ablation_configs)):
                print(f"  - {name}")

        print("Ablation mode enabled. Plotting configurations in fixed order:")
        for c in configs:
            print(f"  - {_display_name(c['name'])}")

    if n_top_techniques == 1 and not is_ablation:
        ours_name = _find_best_non_pinned_name(
            configs,
            baseline_config_name=baseline_config_name,
            rank_spp=8,
        )
        if ours_name is not None:
            DISPLAY_NAME_OVERRIDES[ours_name] = "Ours"

    if is_ablation:
        plot_configs = list(configs)
    else:
        plot_configs = select_top_techniques(
            configs,
            n_top_techniques=n_top_techniques,
            baseline_config_name=baseline_config_name,
            rank_spp=8,
        )
    
    # 3. Detect Bias
    detect_bias(configs)

    # 4. Generate Convergence Plots (SPP & Time)
    test_name = output_root.name
    latex_dir = output_root / "latex"
    latex_fig_root = latex_dir / "figs" / test_name
    logscale_dir = latex_fig_root / "logscale_y"
    linscale_dir = latex_fig_root / "linscale_y"
    os.makedirs(logscale_dir, exist_ok=True)
    os.makedirs(linscale_dir, exist_ok=True)
    write_convergence_plots(
        output_root,
        logscale_dir,
        logspace=True,
        image_format=figure_format,
        configs=plot_configs,
    )
    write_convergence_plots(
        output_root,
        linscale_dir,
        logspace=False,
        image_format=figure_format,
        configs=plot_configs,
    )

    # 5. Grid Renderings
    if is_ablation:
        grid_configs = list(plot_configs)
        print("Grid plot techniques (ablation fixed order):")
        for c in grid_configs:
            print(f"  - {_display_name(c['name'])}")
    else:
        grid_n_best = n_top_techniques if n_top_techniques is not None else 4
        grid_configs = select_grid_plot_configs(
            plot_configs,
            rank_spp=8,
            n_best=grid_n_best,
            baseline_config_name=baseline_config_name,
        )
        if grid_configs:
            print("Grid plot techniques (baseline + best at SPP=8):")
            for c in grid_configs:
                print(f"  - {_display_name(c['name'])}")
        else:
            print("[WARN] No techniques available for grid plot subset at SPP=8; using all techniques.")
            grid_configs = plot_configs

    export_summary_grid(
        grid_configs,
        output_root,
        _parse_bbox_normalized(metrics_bbox),
        mode="spp",
        target=8,
        exposure_scale=exposure_scale,
        output_dir=latex_fig_root,
        image_ext=figure_format,
        output_name="summary_grid_8spp",
    )  
    export_summary_grid(
        grid_configs,
        output_root,
        _parse_bbox_normalized(metrics_bbox),
        mode="spp",
        target=16,
        exposure_scale=exposure_scale,
        output_dir=latex_fig_root,
        image_ext=figure_format,
        output_name="summary_grid_16spp",
    )  
    maxspp = max(
        (int(round(_to_float(r.get("spp")))) for c in grid_configs for r in c["rows"] if np.isfinite(_to_float(r.get("spp")))),
        default=8
    )
    export_summary_grid(
        grid_configs,
        output_root,
        _parse_bbox_normalized(metrics_bbox),
        mode="spp",
        target=maxspp,
        exposure_scale=exposure_scale,
        output_dir=latex_fig_root,
        image_ext=figure_format,
        output_name="summary_grid_maxspp",
    )
    
    # Calculate target time based on 110% of baseline_path's maximum time
    target_time = 10.0  # Fallback
    baseline_cfg = _find_baseline_config(plot_configs, baseline_config_name=baseline_config_name)
    if baseline_cfg and baseline_cfg["rows"]:
        valid_times = [r["mean_effective_time_s"] for r in baseline_cfg["rows"] if np.isfinite(r.get("mean_effective_time_s", float("nan")))]
        if valid_times:
            scale = 1.
            target_time = max(valid_times) * scale
            print(f"Using target_time={target_time:.2f}s for time-based grid plot ({scale*100:.0f}% of baseline max time).")
    export_summary_grid(
        grid_configs,
        output_root,
        _parse_bbox_normalized(metrics_bbox),
        mode="time",
        target=target_time,
        exposure_scale=exposure_scale,
        output_dir=latex_fig_root,
        image_ext=figure_format,
        output_name="summary_grid_time",
    )

    os.makedirs(latex_dir, exist_ok=True)
    _write_convergence_latex_file(
        latex_dir=latex_dir,
        test_name=test_name,
        image_ext=figure_format,
        selected_plots=LATEX_DEFAULT_PLOTS,
    )

    # export_composite_pdf_figure(
    #     configs=grid_configs,
    #     output_root=latex_fig_root,
    #     inset_bbox=_parse_bbox_normalized(metrics_bbox),
    #     target_spp=8,
    #     target_time=target_time,
    #     exposure_scale=exposure_scale,
    #     output_name="composite_figure",
    #     metric_key="mean_rel_rmse"
    # )

    print_top_and_worst(configs)

    print(f"Analysis done: {output_root}")


if __name__ == "__main__":
    main()
    