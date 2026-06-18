import argparse
import csv
import json
import math
import re
from pathlib import Path

import yaml
import numpy as np



import os 
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

try:
    import matplotlib.pyplot as plt
    from matplotlib import patches
except Exception:
    plt = None
    patches = None


def load_img(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[WARN] Failed to load image: {path}")
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    return img.astype(np.float32)  # Ensure it's float32 for consistency



def mse(a, b):
    d = a - b
    return float(np.mean(d * d))


def psnr_from_mse(m):
    if m <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / m)


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


def _short_text(text: str, max_len: int) -> str:
    s = str(text)
    if len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[:max_len]
    return s[: max_len - 3] + "..."


def _parse_config_dirname(dirname: str):
    legacy = re.match(r"^config_(\d+)_(.+)_([0-9a-f]{10})$", dirname)
    if legacy:
        return int(legacy.group(1)), legacy.group(2), legacy.group(3)

    with_hash = re.match(r"^(.+)_([0-9a-f]{10})$", dirname)
    if with_hash:
        return None, with_hash.group(1), with_hash.group(2)

    if dirname:
        return None, dirname, None
    return None, None, None


def _config_plot_title(dirname: str) -> str:
    idx, named, _ = _parse_config_dirname(dirname)
    if idx is None:
        return _short_text(named if named else dirname, 60)
    label = named if named else dirname
    return _short_text(label, 60)


def _parse_bbox_normalized(raw_bbox):
    if raw_bbox is None:
        return None
    try:
        if len(raw_bbox) == 2 and isinstance(raw_bbox[0], list):
            x0, y0 = raw_bbox[0]
            x1, y1 = raw_bbox[1]
        elif len(raw_bbox) == 4:
            x0, y0, x1, y1 = raw_bbox
        else:
            return None

        x0 = float(x0)
        y0 = float(y0)
        x1 = float(x1)
        y1 = float(y1)
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

    h = int(shape_hw[0])
    w = int(shape_hw[1])
    if h <= 0 or w <= 0:
        return None

    x0n, y0n, x1n, y1n = bbox_norm
    x0 = int(x0n * w)
    x1 = int(x1n * w)
    y0 = int(y0n * h)
    y1 = int(y1n * h)

    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h, y1))

    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return x0, y0, x1, y1


def _crop_with_window(img, window):
    if img is None or window is None:
        return img
    x0, y0, x1, y1 = window
    return img[y0:y1, x0:x1]


def iter_config_dirs(output_root: Path):
    dirs = []
    if not output_root.exists():
        return dirs

    for p in sorted(output_root.iterdir()):
        if p.is_dir() and (p / "config.json").exists():
            dirs.append(p)

    if dirs:
        return dirs

    return [p for p in sorted(output_root.glob("config_*")) if p.is_dir()]


def parse_run_meta_filename(name: str):
    m = re.match(r"^spp_(\d+)(?:_rep_(\d+))?\.meta\.json$", name)
    if not m:
        return None
    spp = int(m.group(1))
    repeat_idx = int(m.group(2)) if m.group(2) is not None else 1
    return spp, repeat_idx


def _finite_stats(values):
    arr = np.array([_to_float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(arr))
    var = float(np.var(arr))
    std = float(np.sqrt(var))
    return mean, var, std, int(arr.size)


def summarize_rows_by_spp(rows):
    by_spp = {}
    for row in rows:
        spp = int(row["spp"])
        by_spp.setdefault(spp, []).append(row)

    summary = []
    for spp in sorted(by_spp.keys()):
        rr = by_spp[spp]
        n_repeats = len(rr)
        n_success = sum(1 for r in rr if _to_float(r.get("returncode")) == 0)

        mean_wall, var_wall, _, _ = _finite_stats([r.get("wall_time_s") for r in rr])
        mean_mitsuba, var_mitsuba, _, _ = _finite_stats([r.get("mitsuba_time_s") for r in rr])
        mean_eff, var_eff, _, _ = _finite_stats([_time_value(r) for r in rr])

        mean_mse, var_mse, std_mse, _ = _finite_stats([r.get("mse") for r in rr])
        mean_rmse, var_rmse, std_rmse, _ = _finite_stats([r.get("rmse") for r in rr])
        mean_psnr, var_psnr, std_psnr, _ = _finite_stats([r.get("psnr") for r in rr])

        summary.append(
            {
                "spp": spp,
                "n_repeats": n_repeats,
                "n_success": n_success,
                "mean_wall_time_s": mean_wall,
                "var_wall_time_s": var_wall,
                "mean_mitsuba_time_s": mean_mitsuba,
                "var_mitsuba_time_s": var_mitsuba,
                "mean_effective_time_s": mean_eff,
                "var_effective_time_s": var_eff,
                "mean_mse": mean_mse,
                "var_mse": var_mse,
                "std_mse": std_mse,
                "mean_rmse": mean_rmse,
                "var_rmse": var_rmse,
                "std_rmse": std_rmse,
                "mean_psnr": mean_psnr,
                "var_psnr": var_psnr,
                "std_psnr": std_psnr,
            }
        )

    return summary


def build_per_config_metrics(output_root: Path, metrics_bbox=None):
    bbox_norm = _parse_bbox_normalized(metrics_bbox)
    if metrics_bbox is not None and bbox_norm is None:
        print("[WARN] analysis.average_bbox is invalid. Falling back to full-frame metrics.")

    for cdir in iter_config_dirs(output_root):
        ref = cdir / "reference.exr"
        runs_dir = cdir / "runs"
        if not ref.exists() or not runs_dir.exists():
            continue

        ref_img = load_img(ref)
        ref_window = None
        ref_eval_img = ref_img
        if ref_img is not None and bbox_norm is not None:
            ref_window = _bbox_pixel_window_for_shape(ref_img.shape[:2], bbox_norm)
            ref_eval_img = _crop_with_window(ref_img, ref_window)

        rows = []

        for p in sorted(runs_dir.glob("spp_*.meta.json")):
            parsed = parse_run_meta_filename(p.name)
            if parsed is None:
                continue
            spp, repeat_idx = parsed
            if "_rep_" in p.name:
                exr = runs_dir / f"spp_{spp:04d}_rep_{repeat_idx:03d}.exr"
            else:
                exr = runs_dir / f"spp_{spp:04d}.exr"
            meta = json.loads(p.read_text(encoding="utf-8"))

            row = {
                "spp": spp,
                "repeat": repeat_idx,
                "returncode": meta.get("returncode"),
                "wall_time_s": meta.get("wall_time_s"),
                "mitsuba_time_s": meta.get("mitsuba_time_s"),
                "effective_time_s": "",
                "mse": "",
                "rmse": "",
                "psnr": "",
            }
            row["effective_time_s"] = _time_value(row)

            if ref_eval_img is not None and exr.exists():
                img = load_img(exr)
                if img is not None and ref_img is not None and img.shape == ref_img.shape:
                    eval_img = _crop_with_window(img, ref_window) if ref_window is not None else img
                    if eval_img is not None and eval_img.shape == ref_eval_img.shape:
                        m = mse(eval_img, ref_eval_img)
                        row["mse"] = m
                        row["rmse"] = math.sqrt(m)
                        row["psnr"] = psnr_from_mse(m)

            rows.append(row)

        if not rows:
            continue

        rows.sort(key=lambda r: (int(r["spp"]), int(r.get("repeat", 1))))

        metrics_csv = cdir / "metrics.csv"
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["spp", "repeat", "returncode", "wall_time_s", "mitsuba_time_s", "effective_time_s", "mse", "rmse", "psnr"],
            )
            w.writeheader()
            w.writerows(rows)

        summary_rows = summarize_rows_by_spp(rows)
        metrics_summary_csv = cdir / "metrics_summary.csv"
        with metrics_summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "spp",
                    "n_repeats",
                    "n_success",
                    "mean_wall_time_s",
                    "var_wall_time_s",
                    "mean_mitsuba_time_s",
                    "var_mitsuba_time_s",
                    "mean_effective_time_s",
                    "var_effective_time_s",
                    "mean_mse",
                    "var_mse",
                    "std_mse",
                    "mean_rmse",
                    "var_rmse",
                    "std_rmse",
                    "mean_psnr",
                    "var_psnr",
                    "std_psnr",
                ],
            )
            w.writeheader()
            w.writerows(summary_rows)

        if plt is not None:
            xs = [int(r["spp"]) for r in summary_rows if np.isfinite(_to_float(r.get("mean_psnr")))]
            ys = [_to_float(r.get("mean_psnr")) for r in summary_rows if np.isfinite(_to_float(r.get("mean_psnr")))]
            yerr = [_to_float(r.get("std_psnr")) for r in summary_rows if np.isfinite(_to_float(r.get("mean_psnr")))]
            if xs:
                xs_arr = np.array(xs, dtype=float)
                ys_arr = np.array(ys, dtype=float)
                yerr_arr = np.array(yerr, dtype=float)

                fig, ax = plt.subplots(figsize=(8.2, 4.8))
                ax.plot(xs_arr, ys_arr, marker="o", linewidth=1.8, markersize=4)

                err_mask = np.isfinite(yerr_arr) & (yerr_arr > 0)
                if np.any(err_mask):
                    lo = ys_arr - np.where(err_mask, yerr_arr, 0.0)
                    hi = ys_arr + np.where(err_mask, yerr_arr, 0.0)
                    ax.fill_between(xs_arr, lo, hi, alpha=0.18)

                ax.set_xscale("log", base=2)
                ax.set_xticks(sorted(set(xs)))
                ax.set_xlabel("Samples per pixel (SPP)")
                ax.set_ylabel("PSNR vs reference (dB)")
                ax.set_title(_config_plot_title(cdir.name))
                ax.grid(True, which="both", linestyle="--", alpha=0.25)

                fig.tight_layout()
                fig.savefig(cdir / "convergence_psnr.png", dpi=180)
                plt.close(fig)


def load_config_table(output_root: Path):
    table = []
    for cdir in iter_config_dirs(output_root):
        cfg_json = cdir / "config.json"
        metrics_csv = cdir / "metrics.csv"
        metrics_summary_csv = cdir / "metrics_summary.csv"
        if not cfg_json.exists() or not metrics_csv.exists():
            continue

        params = json.loads(cfg_json.read_text(encoding="utf-8"))
        rows = []
        if metrics_summary_csv.exists():
            with metrics_summary_csv.open("r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    row2 = dict(row)
                    row2["spp"] = int(row["spp"])
                    row2["psnr"] = _to_float(row.get("mean_psnr"))
                    row2["mse"] = _to_float(row.get("mean_mse"))
                    row2["rmse"] = _to_float(row.get("mean_rmse"))
                    row2["effective_time_s"] = _to_float(row.get("mean_effective_time_s"))
                    row2["var_mse"] = _to_float(row.get("var_mse"))
                    row2["var_psnr"] = _to_float(row.get("var_psnr"))
                    row2["var_effective_time_s"] = _to_float(row.get("var_effective_time_s"))
                    rows.append(row2)
        else:
            raw_rows = []
            with metrics_csv.open("r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    row2 = dict(row)
                    row2["spp"] = int(row["spp"])
                    raw_rows.append(row2)
            for row in summarize_rows_by_spp(raw_rows):
                rows.append(
                    {
                        "spp": int(row["spp"]),
                        "psnr": _to_float(row.get("mean_psnr")),
                        "mse": _to_float(row.get("mean_mse")),
                        "rmse": _to_float(row.get("mean_rmse")),
                        "effective_time_s": _to_float(row.get("mean_effective_time_s")),
                        "var_mse": _to_float(row.get("var_mse")),
                        "var_psnr": _to_float(row.get("var_psnr")),
                        "var_effective_time_s": _to_float(row.get("var_effective_time_s")),
                    }
                )

        rows.sort(key=lambda x: x["spp"])
        table.append({"cdir": cdir, "name": cdir.name, "params": params, "rows": rows})
    return table


def _named_config_from_dirname(dirname: str):
    _, named, _ = _parse_config_dirname(dirname)
    return named


def _stable_value_repr(v):
    try:
        return json.dumps(v, sort_keys=True)
    except Exception:
        return str(v)


def build_config_display_labels(configs, max_keys=3):
    if not configs:
        return {}

    all_keys = sorted({k for c in configs for k in c.get("params", {}).keys()})
    varying_keys = []
    for k in all_keys:
        vals = {_stable_value_repr(c.get("params", {}).get(k, None)) for c in configs}
        if len(vals) > 1:
            varying_keys.append(k)

    priority = [
        "integrator",
        "leafSamplingMode",
        "maxLeafSize",
        "leaf_sample_mode",
        "enable_lightcuts",
        "bvh_max_primitives_per_leaf",
        "useBVH",
        "disableNEE",
    ]
    ordered = [k for k in priority if k in varying_keys] + [k for k in varying_keys if k not in priority]
    selected_keys = ordered[:max_keys]

    raw_labels = {}
    for c in configs:
        idx, named, h = _parse_config_dirname(c["name"])

        named_is_compact = bool(named) and ("=" not in str(named)) and (len(str(named)) <= 40)
        if named_is_compact and named not in ("sweep", "named"):
            detail = str(named)
        else:
            parts = []
            params = c.get("params", {})
            for k in selected_keys:
                if k in params:
                    parts.append(f"{k}={params[k]}")
            if parts:
                detail = ", ".join(parts)
            elif named:
                detail = str(named)
            else:
                detail = c["name"]

        detail = _short_text(detail, 72)
        raw_labels[c["name"]] = detail

    counts = {}
    for label in raw_labels.values():
        counts[label] = counts.get(label, 0) + 1

    labels = {}
    for c in configs:
        cfg_name = c["name"]
        idx, _, h = _parse_config_dirname(cfg_name)
        label = raw_labels.get(cfg_name, cfg_name)
        if counts.get(label, 0) <= 1:
            labels[cfg_name] = label
            continue

        suffix = f"C{idx:03d}" if idx is not None else (h[:6] if h else cfg_name)
        labels[cfg_name] = f"{label} ({suffix})"

    return labels


def choose_path_baseline(configs):
    path_cfgs = [c for c in configs if str(c["params"].get("integrator", "")).lower() == "path"]
    if not path_cfgs:
        return None
    for c in path_cfgs:
        if "baseline_path" in c["name"]:
            return c
    path_cfgs.sort(key=lambda x: x["name"])
    return path_cfgs[0]


def choose_baseline(configs, baseline_config_name=None):
    if baseline_config_name:
        exact_named = [c for c in configs if _named_config_from_dirname(c["name"]) == baseline_config_name]
        if exact_named:
            exact_named.sort(key=lambda x: x["name"])
            if len(exact_named) > 1:
                print(
                    f"[WARN] Multiple configs matched analysis.baseline_config_name='{baseline_config_name}'. "
                    f"Using {exact_named[0]['name']}"
                )
            return exact_named[0]

        partial = [c for c in configs if baseline_config_name in c["name"]]
        if partial:
            partial.sort(key=lambda x: x["name"])
            print(
                f"[WARN] No exact named-config match for analysis.baseline_config_name='{baseline_config_name}'. "
                f"Using partial match {partial[0]['name']}"
            )
            return partial[0]

        print(
            f"[WARN] analysis.baseline_config_name='{baseline_config_name}' did not match any config directory. "
            "Falling back to automatic path baseline selection."
        )

    return choose_path_baseline(configs)


def interp_psnr_at_times(cfg_rows, query_times):
    return interp_metric_at_times(cfg_rows, query_times, "psnr")


def interp_metric_at_times(cfg_rows, query_times, metric_key):
    pts = [
        (_to_float(r.get("effective_time_s")), _to_float(r.get(metric_key)))
        for r in cfg_rows
        if np.isfinite(_to_float(r.get("effective_time_s"))) and np.isfinite(_to_float(r.get(metric_key)))
    ]
    if len(pts) < 2:
        return np.full_like(query_times, np.nan, dtype=float)

    pts = sorted(set((float(t), float(_to_float(v))) for t, v in pts), key=lambda x: x[0])
    t = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)

    tmin, tmax = t.min(), t.max()
    out = np.full_like(query_times, np.nan, dtype=float)
    mask = (query_times >= tmin) & (query_times <= tmax)
    if np.any(mask):
        out[mask] = np.interp(query_times[mask], t, y)
    return out


def time_to_reach_psnr(cfg_rows, target_psnr):
    pts = [(r["effective_time_s"], r["psnr"]) for r in cfg_rows if np.isfinite(r["effective_time_s"]) and np.isfinite(r["psnr"])]
    if len(pts) < 2:
        return float("nan")
    pts = sorted(pts, key=lambda x: x[0])
    t = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)

    y = np.maximum.accumulate(y)  # enforce monotonic improvement for inverse lookup
    if target_psnr > y[-1]:
        return float("nan")
    idx = np.searchsorted(y, target_psnr, side="left")
    if idx == 0:
        return t[0]
    y0, y1 = y[idx - 1], y[idx]
    t0, t1 = t[idx - 1], t[idx]
    if y1 <= y0:
        return t1
    alpha = (target_psnr - y0) / (y1 - y0)
    return t0 + alpha * (t1 - t0)


def write_cross_config(output_root: Path, baseline_config_name=None):
    configs = load_config_table(output_root)
    if not configs:
        return

    label_map = build_config_display_labels(configs)
    baseline = choose_baseline(configs, baseline_config_name=baseline_config_name)
    if baseline is None:
        print("[WARN] No baseline with integrator=path found. Skipping cross-config comparison.")
        return

    # Equal SPP
    base_by_spp = {r["spp"]: r for r in baseline["rows"] if np.isfinite(r["psnr"])}
    eq_spp_rows = []
    for c in configs:
        cfg_by_spp = {r["spp"]: r for r in c["rows"] if np.isfinite(r["psnr"])}
        common = sorted(set(base_by_spp.keys()) & set(cfg_by_spp.keys()))
        for spp in common:
            b = base_by_spp[spp]
            x = cfg_by_spp[spp]
            eq_spp_rows.append(
                {
                    "baseline_config": baseline["name"],
                    "baseline_label": label_map.get(baseline["name"], baseline["name"]),
                    "config": c["name"],
                    "config_label": label_map.get(c["name"], c["name"]),
                    "spp": spp,
                    "baseline_psnr": b["psnr"],
                    "config_psnr": x["psnr"],
                    "delta_psnr": x["psnr"] - b["psnr"],
                    "baseline_mse": _to_float(b.get("mse")),
                    "config_mse": _to_float(x.get("mse")),
                    "delta_mse": _to_float(x.get("mse")) - _to_float(b.get("mse")),
                    "mse_improvement_ratio": (_to_float(b.get("mse")) / _to_float(x.get("mse"))) if np.isfinite(_to_float(b.get("mse"))) and np.isfinite(_to_float(x.get("mse"))) and _to_float(x.get("mse")) > 0 else float("nan"),
                    "baseline_var_mse": _to_float(b.get("var_mse")),
                    "config_var_mse": _to_float(x.get("var_mse")),
                    "delta_var_mse": _to_float(x.get("var_mse")) - _to_float(b.get("var_mse")),
                    "baseline_var_psnr": _to_float(b.get("var_psnr")),
                    "config_var_psnr": _to_float(x.get("var_psnr")),
                    "delta_var_psnr": _to_float(x.get("var_psnr")) - _to_float(b.get("var_psnr")),
                    "baseline_time_s": b["effective_time_s"],
                    "config_time_s": x["effective_time_s"],
                    "speedup_time": (b["effective_time_s"] / x["effective_time_s"]) if np.isfinite(b["effective_time_s"]) and np.isfinite(x["effective_time_s"]) and x["effective_time_s"] > 0 else float("nan"),
                }
            )

    eq_spp_csv = output_root / "cross_config_equal_spp.csv"
    with eq_spp_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "baseline_config",
                "baseline_label",
                "config",
                "config_label",
                "spp",
                "baseline_psnr",
                "config_psnr",
                "delta_psnr",
                "baseline_mse",
                "config_mse",
                "delta_mse",
                "mse_improvement_ratio",
                "baseline_var_mse",
                "config_var_mse",
                "delta_var_mse",
                "baseline_var_psnr",
                "config_var_psnr",
                "delta_var_psnr",
                "baseline_time_s",
                "config_time_s",
                "speedup_time",
            ],
        )
        w.writeheader()
        w.writerows(eq_spp_rows)

    # Equal Time (query at baseline times)
    base_pts = [(r["effective_time_s"], r["psnr"]) for r in baseline["rows"] if np.isfinite(r["effective_time_s"]) and np.isfinite(r["psnr"])]
    base_pts = sorted(base_pts, key=lambda x: x[0])
    base_t = np.array([p[0] for p in base_pts], dtype=float)
    base_y = np.array([p[1] for p in base_pts], dtype=float)
    base_mse = interp_metric_at_times(baseline["rows"], base_t, "mse")
    base_var_mse = interp_metric_at_times(baseline["rows"], base_t, "var_mse")
    base_var_psnr = interp_metric_at_times(baseline["rows"], base_t, "var_psnr")

    eq_time_rows = []
    for c in configs:
        yq = interp_psnr_at_times(c["rows"], base_t)
        mse_q = interp_metric_at_times(c["rows"], base_t, "mse")
        var_mse_q = interp_metric_at_times(c["rows"], base_t, "var_mse")
        var_psnr_q = interp_metric_at_times(c["rows"], base_t, "var_psnr")
        for t, bpsnr, cpsnr, bmse, cmse, bvmse, cvmse, bvpsnr, cvpsnr in zip(base_t, base_y, yq, base_mse, mse_q, base_var_mse, var_mse_q, base_var_psnr, var_psnr_q):
            if not np.isfinite(cpsnr):
                continue
            eq_time_rows.append(
                {
                    "baseline_config": baseline["name"],
                    "baseline_label": label_map.get(baseline["name"], baseline["name"]),
                    "config": c["name"],
                    "config_label": label_map.get(c["name"], c["name"]),
                    "time_s": t,
                    "baseline_psnr": bpsnr,
                    "config_psnr": cpsnr,
                    "delta_psnr": cpsnr - bpsnr,
                    "baseline_mse": bmse,
                    "config_mse": cmse,
                    "delta_mse": cmse - bmse if np.isfinite(cmse) and np.isfinite(bmse) else float("nan"),
                    "baseline_var_mse": bvmse,
                    "config_var_mse": cvmse,
                    "delta_var_mse": cvmse - bvmse if np.isfinite(cvmse) and np.isfinite(bvmse) else float("nan"),
                    "baseline_var_psnr": bvpsnr,
                    "config_var_psnr": cvpsnr,
                    "delta_var_psnr": cvpsnr - bvpsnr if np.isfinite(cvpsnr) and np.isfinite(bvpsnr) else float("nan"),
                }
            )

    eq_time_csv = output_root / "cross_config_equal_time.csv"
    with eq_time_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "baseline_config",
                "baseline_label",
                "config",
                "config_label",
                "time_s",
                "baseline_psnr",
                "config_psnr",
                "delta_psnr",
                "baseline_mse",
                "config_mse",
                "delta_mse",
                "baseline_var_mse",
                "config_var_mse",
                "delta_var_mse",
                "baseline_var_psnr",
                "config_var_psnr",
                "delta_var_psnr",
            ],
        )
        w.writeheader()
        w.writerows(eq_time_rows)

    # Summary
    target_psnr = float(base_y[-1]) if len(base_y) else float("nan")
    base_t_target = time_to_reach_psnr(baseline["rows"], target_psnr)

    summary_rows = []
    for c in configs:
        spp_deltas = [r["delta_psnr"] for r in eq_spp_rows if r["config"] == c["name"] and np.isfinite(_to_float(r["delta_psnr"]))]
        spp_mse_deltas = [r["delta_mse"] for r in eq_spp_rows if r["config"] == c["name"] and np.isfinite(_to_float(r["delta_mse"]))]
        spp_var_mse_deltas = [r["delta_var_mse"] for r in eq_spp_rows if r["config"] == c["name"] and np.isfinite(_to_float(r["delta_var_mse"]))]
        spp_var_psnr_deltas = [r["delta_var_psnr"] for r in eq_spp_rows if r["config"] == c["name"] and np.isfinite(_to_float(r["delta_var_psnr"]))]
        time_deltas = [r["delta_psnr"] for r in eq_time_rows if r["config"] == c["name"] and np.isfinite(_to_float(r["delta_psnr"]))]
        time_mse_deltas = [r["delta_mse"] for r in eq_time_rows if r["config"] == c["name"] and np.isfinite(_to_float(r["delta_mse"]))]

        cfg_t_target = time_to_reach_psnr(c["rows"], target_psnr)
        speedup_to_target = (
            (base_t_target / cfg_t_target)
            if np.isfinite(base_t_target) and np.isfinite(cfg_t_target) and cfg_t_target > 0
            else float("nan")
        )

        summary_rows.append(
            {
                "baseline_config": baseline["name"],
                "baseline_label": label_map.get(baseline["name"], baseline["name"]),
                "config": c["name"],
                "config_label": label_map.get(c["name"], c["name"]),
                "integrator": c["params"].get("integrator", ""),
                "mean_delta_psnr_equal_spp": float(np.mean(spp_deltas)) if spp_deltas else float("nan"),
                "mean_delta_psnr_equal_time": float(np.mean(time_deltas)) if time_deltas else float("nan"),
                "mean_delta_mse_equal_spp": float(np.mean(spp_mse_deltas)) if spp_mse_deltas else float("nan"),
                "mean_delta_mse_equal_time": float(np.mean(time_mse_deltas)) if time_mse_deltas else float("nan"),
                "mean_delta_var_mse_equal_spp": float(np.mean(spp_var_mse_deltas)) if spp_var_mse_deltas else float("nan"),
                "mean_delta_var_psnr_equal_spp": float(np.mean(spp_var_psnr_deltas)) if spp_var_psnr_deltas else float("nan"),
                "target_psnr_from_baseline": target_psnr,
                "baseline_time_to_target_s": base_t_target,
                "config_time_to_target_s": cfg_t_target,
                "speedup_to_target_psnr": speedup_to_target,
            }
        )

    summary_csv = output_root / "cross_config_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "baseline_config",
                "baseline_label",
                "config",
                "config_label",
                "integrator",
                "mean_delta_psnr_equal_spp",
                "mean_delta_psnr_equal_time",
                "mean_delta_mse_equal_spp",
                "mean_delta_mse_equal_time",
                "mean_delta_var_mse_equal_spp",
                "mean_delta_var_psnr_equal_spp",
                "target_psnr_from_baseline",
                "baseline_time_to_target_s",
                "config_time_to_target_s",
                "speedup_to_target_psnr",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)

    # Optional plot: equal-time delta PSNR
    if plt is not None and eq_time_rows:
        fig, ax = plt.subplots(figsize=(11.5, 6.2))
        by_cfg = {}
        for r in eq_time_rows:
            by_cfg.setdefault(r["config"], []).append(r)

        for cfg, rr in sorted(by_cfg.items(), key=lambda kv: kv[0]):
            rr = sorted(rr, key=lambda x: x["time_s"])
            x = np.array([z["time_s"] for z in rr], dtype=float)
            y = np.array([z["delta_psnr"] for z in rr], dtype=float)
            is_baseline = (cfg == baseline["name"])
            ax.plot(
                x,
                y,
                marker="o",
                markersize=4 if is_baseline else 3,
                linewidth=2.4 if is_baseline else 1.6,
                alpha=1.0 if is_baseline else 0.9,
                label=label_map.get(cfg, cfg),
                zorder=3 if is_baseline else 2,
            )

        ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("Effective render time (s)")
        ax.set_ylabel("ΔPSNR vs baseline (dB)")
        ax.set_title(f"Equal-time PSNR delta (baseline: {label_map.get(baseline['name'], baseline['name'])})")
        ax.grid(True, which="both", linestyle="--", alpha=0.25)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)

        fig.tight_layout()
        fig.savefig(output_root / "cross_config_equal_time_delta_psnr.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _finite_cfg_rows(cfg):
    rows = []
    for r in cfg.get("rows", []):
        spp = int(r.get("spp"))
        t = _to_float(r.get("effective_time_s"))
        p = _to_float(r.get("psnr"))
        if np.isfinite(t) and np.isfinite(p):
            rows.append(
                {
                    "spp": spp,
                    "effective_time_s": float(t),
                    "psnr": float(p),
                    "mse": _to_float(r.get("mse")),
                    "rmse": _to_float(r.get("rmse")),
                    "var_mse": _to_float(r.get("var_mse")),
                    "var_psnr": _to_float(r.get("var_psnr")),
                    "var_effective_time_s": _to_float(r.get("var_effective_time_s")),
                }
            )
    rows.sort(key=lambda x: (x["spp"], x["effective_time_s"]))
    return rows


def _choose_top_configs_for_plots(configs, overview_rows, baseline_name, top_k):
    row_by_name = {r["config"]: r for r in overview_rows}

    def rank_key(cfg_name):
        r = row_by_name.get(cfg_name, {})
        speed = _to_float(r.get("speedup_to_baseline_target_psnr"))
        peak = _to_float(r.get("peak_psnr_db"))
        final_psnr = _to_float(r.get("final_psnr_db"))
        score_speed = speed if np.isfinite(speed) else -1.0
        score_peak = peak if np.isfinite(peak) else -1e9
        score_final = final_psnr if np.isfinite(final_psnr) else -1e9
        return (score_speed, score_peak, score_final)

    names = [c["name"] for c in configs]
    names.sort(key=rank_key, reverse=True)
    out = names[: max(1, int(top_k))]
    if baseline_name and baseline_name not in out and baseline_name in names:
        if len(out) >= max(1, int(top_k)):
            out = out[:-1] + [baseline_name]
        else:
            out.append(baseline_name)
    return out


def _compute_pareto_front(rows):
    pts = []
    for r in rows:
        t = _to_float(r.get("time_to_peak_s"))
        p = _to_float(r.get("peak_psnr_db"))
        if np.isfinite(t) and np.isfinite(p):
            pts.append((r, t, p))

    pareto = []
    for i, (ri, ti, pi) in enumerate(pts):
        dominated = False
        for j, (_, tj, pj) in enumerate(pts):
            if i == j:
                continue
            if (tj <= ti and pj >= pi) and (tj < ti or pj > pi):
                dominated = True
                break
        if not dominated:
            pareto.append(ri)
    return pareto


def _parse_inset_bbox(raw_bbox):
    default_bbox = (0.7, 0.7, 0.75, 0.75)
    if raw_bbox is None:
        return default_bbox
    try:
        vals = [float(x) for x in raw_bbox]
        if len(vals) != 4:
            return default_bbox
        x0, y0, x1, y1 = vals
        x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
        y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
        if (x1 - x0) < 1e-4 or (y1 - y0) < 1e-4:
            return default_bbox
        return x0, y0, x1, y1
    except Exception:
        return default_bbox


def _to_display_rgb(arr, exposure_scale=None):
    if arr is None:
        return None

    img = np.asarray(arr, dtype=np.float32)

    # Collapse singleton batch/depth dimensions that some EXR backends return
    # (e.g. shape (1, H, W, 3)).
    while img.ndim > 3 and any(s == 1 for s in img.shape):
        squeeze_axis = next(i for i, s in enumerate(img.shape) if s == 1)
        img = np.squeeze(img, axis=squeeze_axis)

    # Fallback: if still higher-dimensional, keep the first frame/slice.
    while img.ndim > 3:
        img = img[0]

    # Handle channel-first tensors like (3, H, W).
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
        img = np.moveaxis(img, 0, -1)

    if img.ndim == 2:
        img = img[..., None]

    if img.ndim != 3:
        return None

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    elif img.shape[-1] >= 3:
        img = img[..., :3]
    else:
        return None

    img = np.where(np.isfinite(img), img, 0.0)
    img = np.maximum(img, 0.0)

    if exposure_scale is not None and np.isfinite(float(exposure_scale)) and float(exposure_scale) > 0:
        img = img * float(exposure_scale)
    else:
        p99 = float(np.percentile(img, 99.0)) if img.size else 1.0
        if not np.isfinite(p99) or p99 <= 0:
            p99 = 1.0
        img = img / p99

    img = img / (1.0 + img)
    img = np.clip(img, 0.0, 1.0)
    img = np.power(img, 1.0 / 2.2)
    return img


def _mean_luminance_in_bbox(img, bbox_norm):
    if img is None:
        return float("nan")

    arr = np.asarray(img, dtype=np.float32)
    while arr.ndim > 3 and any(s == 1 for s in arr.shape):
        squeeze_axis = next(i for i, s in enumerate(arr.shape) if s == 1)
        arr = np.squeeze(arr, axis=squeeze_axis)
    while arr.ndim > 3:
        arr = arr[0]

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        return float("nan")

    if arr.shape[-1] == 1:
        rgb = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] >= 3:
        rgb = arr[..., :3]
    else:
        return float("nan")

    rgb = np.where(np.isfinite(rgb), rgb, 0.0)
    rgb = np.maximum(rgb, 0.0)

    if bbox_norm is not None:
        win = _bbox_pixel_window_for_shape(rgb.shape[:2], bbox_norm)
        if win is not None:
            rgb = _crop_with_window(rgb, win)

    if rgb.size == 0:
        return float("nan")

    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return float(np.mean(lum)) if lum.size else float("nan")


def _reference_exposure_scale_from_bbox(configs, selected_names, baseline_name, bbox_norm, target_luminance=0.18):
    if bbox_norm is None:
        return None

    by_name = {c["name"]: c for c in configs}
    ordered_names = []
    if baseline_name and baseline_name in by_name:
        ordered_names.append(baseline_name)
    ordered_names.extend([name for name in selected_names if name not in ordered_names and name in by_name])

    for cfg_name in ordered_names:
        cdir = by_name[cfg_name]["cdir"]
        ref_path = cdir / "reference.exr"
        if not ref_path.exists():
            continue
        ref_img = load_img(ref_path)
        mean_lum = _mean_luminance_in_bbox(ref_img, bbox_norm)
        if np.isfinite(mean_lum) and mean_lum > 1e-12:
            return float(target_luminance / mean_lum)

    return None


def _resolve_image_path(image_path_text, cdir: Path, spp: int, repeat_idx: int):
    candidates = []
    if image_path_text:
        p = Path(str(image_path_text))
        candidates.append(p)
        if not p.is_absolute():
            candidates.append((Path.cwd() / p).resolve())
            candidates.append((cdir / p).resolve())

    candidates.append(cdir / "runs" / f"spp_{int(spp):04d}_rep_{int(repeat_idx):03d}.exr")
    candidates.append(cdir / "runs" / f"spp_{int(spp):04d}.exr")

    for cand in candidates:
        try:
            if cand.exists():
                return cand
        except Exception:
            continue
    return None


def load_config_run_rows(cdir: Path):
    timings_csv = cdir / "timings.csv"
    metrics_csv = cdir / "metrics.csv"
    if not timings_csv.exists() or not metrics_csv.exists():
        return []

    timing_map = {}
    with timings_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            spp = int(row.get("spp", 0))
            repeat_idx = int(row.get("repeat", 1) or 1)
            key = (spp, repeat_idx)
            timing_map[key] = {
                "effective_time_s": _time_value(row),
                "image_path_text": row.get("image_path", ""),
                "returncode": _to_float(row.get("returncode")),
            }

    out = []
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            spp = int(row.get("spp", 0))
            repeat_idx = int(row.get("repeat", 1) or 1)
            key = (spp, repeat_idx)
            trow = timing_map.get(key, {})
            img_path = _resolve_image_path(trow.get("image_path_text", ""), cdir, spp, repeat_idx)
            if img_path is None:
                continue
            out.append(
                {
                    "spp": spp,
                    "repeat": repeat_idx,
                    "effective_time_s": _to_float(trow.get("effective_time_s")),
                    "psnr": _to_float(row.get("psnr")),
                    "mse": _to_float(row.get("mse")),
                    "rmse": _to_float(row.get("rmse")),
                    "image_path": img_path,
                }
            )
    return out


def _select_row_equal_spp(run_rows, target_spp):
    rr = [r for r in run_rows if int(r.get("spp", -1)) == int(target_spp)]
    if not rr:
        return None
    rr.sort(key=lambda r: (_to_float(r.get("psnr")), -_to_float(r.get("effective_time_s"))), reverse=True)
    return rr[0]


def _select_row_equal_time(run_rows, target_time_s):
    rr = [r for r in run_rows if np.isfinite(_to_float(r.get("effective_time_s")))]
    if not rr:
        return None
    rr.sort(key=lambda r: abs(_to_float(r.get("effective_time_s")) - float(target_time_s)))
    return rr[0]


def write_render_summary_figure(configs, selected_names, label_map, baseline_name, inset_bbox, mode, target_value, output_path):
    if plt is None or patches is None:
        return

    selected = [c for c in configs if c["name"] in selected_names]
    if not selected:
        return

    exposure_scale = _reference_exposure_scale_from_bbox(
        configs,
        selected_names,
        baseline_name,
        inset_bbox,
        target_luminance=0.08,
    )
    if exposure_scale is None:
        print("[WARN] Could not derive reference-based exposure for inset figures. Using per-image auto exposure.")

    picks = []
    for c in selected:
        run_rows = load_config_run_rows(c["cdir"])
        if not run_rows:
            continue
        if mode == "equal_spp":
            pick = _select_row_equal_spp(run_rows, int(target_value))
        else:
            pick = _select_row_equal_time(run_rows, float(target_value))
        if pick is None:
            continue
        img = load_img(pick["image_path"])
        disp = _to_display_rgb(img, exposure_scale=exposure_scale)
        if disp is None:
            continue
        picks.append((c, pick, disp))

    if not picks:
        return

    n = len(picks)
    fig_height = max(2.4, 2.35 * n)
    fig_width = 6.0
    fig, axes = plt.subplots(
        nrows=n,
        ncols=2,
        figsize=(fig_width, fig_height),
        gridspec_kw={"wspace": 0.01, "hspace": 0.01},
    )
    if n == 1:
        axes = np.array([axes])

    x0n, y0n, x1n, y1n = inset_bbox

    for i, (cfg, pick, disp) in enumerate(picks):
        h, w = disp.shape[:2]
        x0 = int(round(x0n * (w - 1)))
        x1 = int(round(x1n * (w - 1)))
        y0 = int(round(y0n * (h - 1)))
        y1 = int(round(y1n * (h - 1)))
        x0, x1 = sorted((max(0, min(w - 1, x0)), max(0, min(w - 1, x1))))
        y0, y1 = sorted((max(0, min(h - 1, y0)), max(0, min(h - 1, y1))))
        if x1 <= x0:
            x1 = min(w - 1, x0 + 1)
        if y1 <= y0:
            y1 = min(h - 1, y0 + 1)

        crop = disp[y0:y1, x0:x1, :]

        ax_full = axes[i, 0]
        ax_zoom = axes[i, 1]

        ax_full.imshow(disp)
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.6, edgecolor="yellow", facecolor="none")
        ax_full.add_patch(rect)
        ax_full.set_axis_off()

        title = label_map.get(cfg["name"], cfg["name"])
        if mode == "equal_spp":
            subtitle = f"SPP={int(pick['spp'])} | t={_to_float(pick['effective_time_s']):.3f}s"
        else:
            subtitle = f"t={_to_float(pick['effective_time_s']):.3f}s | SPP={int(pick['spp'])}"
        if cfg["name"] == baseline_name:
            subtitle += " | baseline"
        ax_full.text(
            0.01,
            0.02,
            f"{title} | {subtitle}",
            transform=ax_full.transAxes,
            fontsize=7,
            color="white",
            ha="left",
            va="bottom",
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.5, "edgecolor": "none"},
        )

        ax_zoom.imshow(crop)
        ax_zoom.set_axis_off()
        ax_zoom.text(
            0.01,
            0.02,
            "Inset",
            transform=ax_zoom.transAxes,
            fontsize=7,
            color="white",
            ha="left",
            va="bottom",
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.2, "edgecolor": "none"},
        )

    mode_title = (
        f"Rendered summary (equal samples, SPP={int(target_value)})"
        if mode == "equal_spp"
        else f"Rendered summary (equal time, target={float(target_value):.3f}s)"
    )
    fig.text(0.005, 0.995, mode_title, fontsize=9, ha="left", va="top")
    fig.subplots_adjust(left=0.003, right=0.997, bottom=0.003, top=0.975, wspace=0.01, hspace=0.01)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_grouped_comparison(
    output_root: Path,
    baseline_config_name=None,
    top_k=6,
    time_grid_points=10,
    inset_bbox=None,
    summary_equal_spp=None,
    summary_equal_time_s=None,
):
    configs = load_config_table(output_root)
    if not configs:
        return

    label_map = build_config_display_labels(configs)
    baseline = choose_baseline(configs, baseline_config_name=baseline_config_name)
    baseline_name = baseline["name"] if baseline is not None else ""

    meta = {}
    for c in configs:
        idx, named, h = _parse_config_dirname(c["name"])
        meta[c["name"]] = {
            "config_index": idx if idx is not None else "",
            "named_config": named if named is not None else "",
            "hash": h if h is not None else "",
            "integrator": str(c.get("params", {}).get("integrator", "")),
            "params_json": json.dumps(c.get("params", {}), sort_keys=True),
        }

    base_rows = _finite_cfg_rows(baseline) if baseline is not None else []
    base_by_spp = {r["spp"]: r for r in base_rows}

    target_psnr = float(base_rows[-1]["psnr"]) if len(base_rows) else float("nan")
    base_t_target = time_to_reach_psnr(baseline["rows"], target_psnr) if baseline is not None and np.isfinite(target_psnr) else float("nan")

    grouped_long_rows = []
    for c in configs:
        cmeta = meta[c["name"]]
        for r in _finite_cfg_rows(c):
            b = base_by_spp.get(r["spp"])
            delta_psnr = r["psnr"] - b["psnr"] if b is not None else float("nan")
            speedup = (
                b["effective_time_s"] / r["effective_time_s"]
                if b is not None and np.isfinite(b["effective_time_s"]) and r["effective_time_s"] > 0
                else float("nan")
            )
            grouped_long_rows.append(
                {
                    "config": c["name"],
                    "label": label_map.get(c["name"], c["name"]),
                    "config_index": cmeta["config_index"],
                    "named_config": cmeta["named_config"],
                    "integrator": cmeta["integrator"],
                    "spp": r["spp"],
                    "psnr_db": r["psnr"],
                    "mse": _to_float(r.get("mse")),
                    "rmse": _to_float(r.get("rmse")),
                    "var_mse": _to_float(r.get("var_mse")),
                    "var_psnr": _to_float(r.get("var_psnr")),
                    "effective_time_s": r["effective_time_s"],
                    "delta_psnr_vs_baseline_equal_spp": delta_psnr,
                    "delta_mse_vs_baseline_equal_spp": _to_float(r.get("mse")) - _to_float(b.get("mse")) if b is not None else float("nan"),
                    "mse_improvement_ratio_vs_baseline_equal_spp": (_to_float(b.get("mse")) / _to_float(r.get("mse"))) if b is not None and np.isfinite(_to_float(b.get("mse"))) and np.isfinite(_to_float(r.get("mse"))) and _to_float(r.get("mse")) > 0 else float("nan"),
                    "delta_var_mse_vs_baseline_equal_spp": _to_float(r.get("var_mse")) - _to_float(b.get("var_mse")) if b is not None else float("nan"),
                    "delta_var_psnr_vs_baseline_equal_spp": _to_float(r.get("var_psnr")) - _to_float(b.get("var_psnr")) if b is not None else float("nan"),
                    "speedup_vs_baseline_equal_spp": speedup,
                    "is_baseline": int(c["name"] == baseline_name),
                    "params_json": cmeta["params_json"],
                }
            )

    grouped_long_rows.sort(key=lambda x: (int(x["spp"]), str(x["config"])))
    write_csv(
        output_root / "grouped_metrics_long.csv",
        [
            "config",
            "label",
            "config_index",
            "named_config",
            "integrator",
            "spp",
            "psnr_db",
            "mse",
            "rmse",
            "var_mse",
            "var_psnr",
            "effective_time_s",
            "delta_psnr_vs_baseline_equal_spp",
            "delta_mse_vs_baseline_equal_spp",
            "mse_improvement_ratio_vs_baseline_equal_spp",
            "delta_var_mse_vs_baseline_equal_spp",
            "delta_var_psnr_vs_baseline_equal_spp",
            "speedup_vs_baseline_equal_spp",
            "is_baseline",
            "params_json",
        ],
        grouped_long_rows,
    )

    by_spp = {}
    for r in grouped_long_rows:
        by_spp.setdefault(int(r["spp"]), []).append(r)

    best_by_spp_rows = []
    wins_spp = {}
    for spp in sorted(by_spp.keys()):
        rr = sorted(by_spp[spp], key=lambda x: _to_float(x["psnr_db"]), reverse=True)
        if not rr:
            continue
        best = rr[0]
        second_psnr = _to_float(rr[1]["psnr_db"]) if len(rr) > 1 else float("nan")
        best_psnr = _to_float(best["psnr_db"])
        best_cfg = best["config"]
        wins_spp[best_cfg] = wins_spp.get(best_cfg, 0) + 1
        best_by_spp_rows.append(
            {
                "spp": spp,
                "best_config": best_cfg,
                "best_label": label_map.get(best_cfg, best_cfg),
                "best_psnr_db": best_psnr,
                "best_mse": _to_float(best.get("mse")),
                "second_psnr_db": second_psnr,
                "gap_to_second_db": (best_psnr - second_psnr) if np.isfinite(second_psnr) else float("nan"),
                "best_time_s": _to_float(best.get("effective_time_s")),
            }
        )

    write_csv(
        output_root / "grouped_best_by_spp.csv",
        ["spp", "best_config", "best_label", "best_psnr_db", "best_mse", "second_psnr_db", "gap_to_second_db", "best_time_s"],
        best_by_spp_rows,
    )

    best_mse_by_spp_rows = []
    wins_mse_spp = {}
    for spp in sorted(by_spp.keys()):
        rr = [row for row in by_spp[spp] if np.isfinite(_to_float(row.get("mse")))]
        if not rr:
            continue
        rr.sort(key=lambda x: _to_float(x["mse"]))
        best = rr[0]
        second_mse = _to_float(rr[1]["mse"]) if len(rr) > 1 else float("nan")
        best_mse = _to_float(best["mse"])
        best_cfg = best["config"]
        wins_mse_spp[best_cfg] = wins_mse_spp.get(best_cfg, 0) + 1
        best_mse_by_spp_rows.append(
            {
                "spp": spp,
                "best_config": best_cfg,
                "best_label": label_map.get(best_cfg, best_cfg),
                "best_mse": best_mse,
                "second_mse": second_mse,
                "improvement_to_second": (second_mse - best_mse) if np.isfinite(second_mse) else float("nan"),
                "best_time_s": _to_float(best.get("effective_time_s")),
            }
        )

    write_csv(
        output_root / "grouped_best_mse_by_spp.csv",
        ["spp", "best_config", "best_label", "best_mse", "second_mse", "improvement_to_second", "best_time_s"],
        best_mse_by_spp_rows,
    )

    all_times = np.array([_to_float(r["effective_time_s"]) for r in grouped_long_rows], dtype=float)
    all_times = all_times[np.isfinite(all_times)]

    if len(base_rows) >= 2:
        time_query = np.array([r["effective_time_s"] for r in base_rows], dtype=float)
    elif all_times.size >= 2:
        tmin = float(np.min(all_times))
        tmax = float(np.max(all_times))
        n = max(4, int(time_grid_points))
        if tmin > 0 and tmax > tmin:
            time_query = np.geomspace(tmin, tmax, num=n)
        else:
            time_query = np.linspace(tmin, tmax, num=n)
    else:
        time_query = np.array([], dtype=float)

    best_by_time_rows = []
    wins_time = {}
    if time_query.size > 0:
        psnr_at_time = {}
        for c in configs:
            psnr_at_time[c["name"]] = interp_psnr_at_times(c["rows"], time_query)

        for idx_t, t in enumerate(time_query):
            vals = []
            for c in configs:
                cfg_name = c["name"]
                p = _to_float(psnr_at_time[cfg_name][idx_t])
                if np.isfinite(p):
                    vals.append((cfg_name, p))
            if not vals:
                continue

            vals.sort(key=lambda z: z[1], reverse=True)
            best_cfg, best_psnr = vals[0]
            second_psnr = vals[1][1] if len(vals) > 1 else float("nan")
            wins_time[best_cfg] = wins_time.get(best_cfg, 0) + 1
            best_by_time_rows.append(
                {
                    "time_s": float(t),
                    "best_config": best_cfg,
                    "best_label": label_map.get(best_cfg, best_cfg),
                    "best_psnr_db": float(best_psnr),
                    "second_psnr_db": float(second_psnr) if np.isfinite(second_psnr) else float("nan"),
                    "gap_to_second_db": float(best_psnr - second_psnr) if np.isfinite(second_psnr) else float("nan"),
                }
            )

    write_csv(
        output_root / "grouped_best_by_time.csv",
        ["time_s", "best_config", "best_label", "best_psnr_db", "second_psnr_db", "gap_to_second_db"],
        best_by_time_rows,
    )

    overview_rows = []
    for c in configs:
        cfg_name = c["name"]
        cmeta = meta[cfg_name]
        rr = _finite_cfg_rows(c)

        if rr:
            peak = max(rr, key=lambda z: z["psnr"])
            final = max(rr, key=lambda z: (z["spp"], z["effective_time_s"]))
            mse_rows = [z for z in rr if np.isfinite(_to_float(z.get("mse")))]
            min_mse_row = min(mse_rows, key=lambda z: _to_float(z.get("mse"))) if mse_rows else None
            var_mse_vals = np.array([_to_float(z.get("var_mse")) for z in rr], dtype=float)
            var_psnr_vals = np.array([_to_float(z.get("var_psnr")) for z in rr], dtype=float)
            var_mse_vals = var_mse_vals[np.isfinite(var_mse_vals)]
            var_psnr_vals = var_psnr_vals[np.isfinite(var_psnr_vals)]

            peak_psnr = peak["psnr"]
            peak_spp = peak["spp"]
            time_to_peak = peak["effective_time_s"]
            final_psnr = final["psnr"]
            final_spp = final["spp"]
            final_time = final["effective_time_s"]
            final_mse = _to_float(final.get("mse"))
            min_mse = _to_float(min_mse_row.get("mse")) if min_mse_row is not None else float("nan")
            min_mse_spp = min_mse_row.get("spp") if min_mse_row is not None else ""
            time_to_min_mse = _to_float(min_mse_row.get("effective_time_s")) if min_mse_row is not None else float("nan")
            mean_var_mse = float(np.mean(var_mse_vals)) if var_mse_vals.size else float("nan")
            mean_var_psnr = float(np.mean(var_psnr_vals)) if var_psnr_vals.size else float("nan")
            auc_psnr_time = float(np.trapz([z["psnr"] for z in rr], x=[z["effective_time_s"] for z in rr])) if len(rr) >= 2 else float("nan")
        else:
            peak_psnr = float("nan")
            peak_spp = ""
            time_to_peak = float("nan")
            final_psnr = float("nan")
            final_spp = ""
            final_time = float("nan")
            final_mse = float("nan")
            min_mse = float("nan")
            min_mse_spp = ""
            time_to_min_mse = float("nan")
            mean_var_mse = float("nan")
            mean_var_psnr = float("nan")
            auc_psnr_time = float("nan")

        t_to_target = time_to_reach_psnr(c["rows"], target_psnr) if np.isfinite(target_psnr) else float("nan")
        speedup_target = (
            (base_t_target / t_to_target)
            if np.isfinite(base_t_target) and np.isfinite(t_to_target) and t_to_target > 0
            else float("nan")
        )

        overview_rows.append(
            {
                "config": cfg_name,
                "label": label_map.get(cfg_name, cfg_name),
                "config_index": cmeta["config_index"],
                "named_config": cmeta["named_config"],
                "integrator": cmeta["integrator"],
                "peak_psnr_db": peak_psnr,
                "peak_spp": peak_spp,
                "time_to_peak_s": time_to_peak,
                "final_psnr_db": final_psnr,
                "final_spp": final_spp,
                "final_time_s": final_time,
                "final_mse": final_mse,
                "min_mse": min_mse,
                "min_mse_spp": min_mse_spp,
                "time_to_min_mse_s": time_to_min_mse,
                "mean_var_mse": mean_var_mse,
                "mean_var_psnr": mean_var_psnr,
                "auc_psnr_time": auc_psnr_time,
                "wins_best_psnr_by_spp": int(wins_spp.get(cfg_name, 0)),
                "wins_best_psnr_by_time": int(wins_time.get(cfg_name, 0)),
                "wins_best_mse_by_spp": int(wins_mse_spp.get(cfg_name, 0)),
                "target_psnr_from_baseline": target_psnr,
                "time_to_baseline_target_psnr_s": t_to_target,
                "speedup_to_baseline_target_psnr": speedup_target,
                "is_baseline": int(cfg_name == baseline_name),
                "params_json": cmeta["params_json"],
            }
        )

    def _overview_sort_key(row):
        speed = _to_float(row.get("speedup_to_baseline_target_psnr"))
        peak = _to_float(row.get("peak_psnr_db"))
        final_psnr = _to_float(row.get("final_psnr_db"))
        return (
            speed if np.isfinite(speed) else -1.0,
            peak if np.isfinite(peak) else -1e9,
            final_psnr if np.isfinite(final_psnr) else -1e9,
        )

    overview_rows.sort(key=_overview_sort_key, reverse=True)
    for rank, row in enumerate(overview_rows, start=1):
        row["overall_rank"] = rank

    write_csv(
        output_root / "grouped_config_overview.csv",
        [
            "overall_rank",
            "config",
            "label",
            "config_index",
            "named_config",
            "integrator",
            "peak_psnr_db",
            "peak_spp",
            "time_to_peak_s",
            "final_psnr_db",
            "final_spp",
            "final_time_s",
            "final_mse",
            "min_mse",
            "min_mse_spp",
            "time_to_min_mse_s",
            "mean_var_mse",
            "mean_var_psnr",
            "auc_psnr_time",
            "wins_best_psnr_by_spp",
            "wins_best_psnr_by_time",
            "wins_best_mse_by_spp",
            "target_psnr_from_baseline",
            "time_to_baseline_target_psnr_s",
            "speedup_to_baseline_target_psnr",
            "is_baseline",
            "params_json",
        ],
        overview_rows,
    )

    if plt is None:
        return

    top_names = _choose_top_configs_for_plots(configs, overview_rows, baseline_name, top_k=top_k)
    inset_bbox = _parse_inset_bbox(inset_bbox)

    selected_cfgs = [c for c in configs if c["name"] in top_names]
    selected_spp_sets = []
    for c in selected_cfgs:
        spp_set = {int(r["spp"]) for r in _finite_cfg_rows(c)}
        if spp_set:
            selected_spp_sets.append(spp_set)

    if summary_equal_spp is None:
        if selected_spp_sets:
            common = set.intersection(*selected_spp_sets)
            if common:
                summary_equal_spp = max(common)
            else:
                summary_equal_spp = max(max(s) for s in selected_spp_sets)
        elif base_rows:
            summary_equal_spp = int(max(r["spp"] for r in base_rows))

    if summary_equal_time_s is None:
        if base_rows:
            summary_equal_time_s = float(max(r["effective_time_s"] for r in base_rows))
        else:
            all_t = []
            for c in selected_cfgs:
                for r in _finite_cfg_rows(c):
                    all_t.append(_to_float(r.get("effective_time_s")))
            all_t = [t for t in all_t if np.isfinite(t)]
            if all_t:
                summary_equal_time_s = float(np.median(np.array(all_t, dtype=float)))

    if summary_equal_spp is not None:
        write_render_summary_figure(
            configs,
            top_names,
            label_map,
            baseline_name,
            inset_bbox,
            mode="equal_spp",
            target_value=int(summary_equal_spp),
            output_path=output_root / "render_summary_equal_spp.png",
        )
        
        top_names_8 = ["baseline_path", "ours_64"]
        write_render_summary_figure(
            configs,
            top_names_8,
            label_map,
            baseline_name,
            inset_bbox,
            mode="equal_spp",
            target_value=8,
            output_path=output_root / "render_summary_equal_spp_8.png",
        )

    if summary_equal_time_s is not None:
        write_render_summary_figure(
            configs,
            top_names,
            label_map,
            baseline_name,
            inset_bbox,
            mode="equal_time",
            target_value=float(summary_equal_time_s),
            output_path=output_root / "render_summary_equal_time.png",
        )

    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    for c in configs:
        if c["name"] not in top_names:
            continue
        rr = _finite_cfg_rows(c)
        if not rr:
            continue
        x = np.array([z["spp"] for z in rr], dtype=float)
        y = np.array([z["psnr"] for z in rr], dtype=float)
        is_baseline = (c["name"] == baseline_name)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.4 if is_baseline else 1.8,
            markersize=4,
            alpha=1.0 if is_baseline else 0.92,
            label=label_map.get(c["name"], c["name"]),
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Samples per pixel (SPP)")
    ax.set_ylabel("PSNR vs reference (dB)")
    ax.set_title("Top configurations: PSNR vs SPP")
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_root / "grouped_top_psnr_vs_spp.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    for c in configs:
        if c["name"] not in top_names:
            continue
        rr = _finite_cfg_rows(c)
        if not rr:
            continue
        x = np.array([z["effective_time_s"] for z in rr], dtype=float)
        y = np.array([z["psnr"] for z in rr], dtype=float)
        is_baseline = (c["name"] == baseline_name)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.4 if is_baseline else 1.8,
            markersize=4,
            alpha=1.0 if is_baseline else 0.92,
            label=label_map.get(c["name"], c["name"]),
        )
    ax.set_xscale("log")
    ax.set_xlabel("Effective render time (s)")
    ax.set_ylabel("PSNR vs reference (dB)")
    ax.set_title("Top configurations: PSNR vs time")
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_root / "grouped_top_psnr_vs_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    has_nonpositive_mse = False
    for c in configs:
        if c["name"] not in top_names:
            continue
        rr = [z for z in _finite_cfg_rows(c) if np.isfinite(_to_float(z.get("mse")))]
        if not rr:
            continue
        x = np.array([z["spp"] for z in rr], dtype=float)
        y = np.array([_to_float(z.get("mse")) for z in rr], dtype=float)
        if np.any(y <= 0):
            has_nonpositive_mse = True
        is_baseline = (c["name"] == baseline_name)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.4 if is_baseline else 1.8,
            markersize=4,
            alpha=1.0 if is_baseline else 0.92,
            label=label_map.get(c["name"], c["name"]),
        )
    ax.set_xscale("log", base=2)
    if not has_nonpositive_mse:
        ax.set_yscale("log")
    ax.set_xlabel("Samples per pixel (SPP)")
    ax.set_ylabel("MSE vs reference")
    ax.set_title("Top configurations: MSE vs SPP")
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_root / "grouped_top_mse_vs_spp.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    var_rows = [r for r in overview_rows if np.isfinite(_to_float(r.get("mean_var_mse"))) or np.isfinite(_to_float(r.get("mean_var_psnr")))]
    if var_rows:
        var_rows = sorted(var_rows, key=lambda z: int(z.get("overall_rank", 10**9)))[: max(1, min(len(var_rows), int(top_k)))]
        fig, ax = plt.subplots(figsize=(10.8, 5.0))
        x = np.arange(len(var_rows))
        var_mse_vals = np.array([_to_float(r.get("mean_var_mse")) for r in var_rows], dtype=float)
        var_psnr_vals = np.array([_to_float(r.get("mean_var_psnr")) for r in var_rows], dtype=float)
        ax.bar(x - 0.18, var_mse_vals, width=0.36, label="Mean variance (MSE)")
        ax.bar(x + 0.18, var_psnr_vals, width=0.36, label="Mean variance (PSNR)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(r.get("label", "")).split(" ")[0] for r in var_rows], rotation=0)
        ax.set_ylabel("Variance")
        ax.set_title("Variance summary (top configs)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_root / "grouped_variance_summary.png", dpi=180)
        plt.close(fig)

    win_cfgs = sorted(
        {*(wins_spp.keys()), *(wins_time.keys())},
        key=lambda cfg_name: (wins_spp.get(cfg_name, 0) + wins_time.get(cfg_name, 0), wins_spp.get(cfg_name, 0)),
        reverse=True,
    )
    if win_cfgs:
        fig, ax = plt.subplots(figsize=(10.8, 5.0))
        x = np.arange(len(win_cfgs))
        spp_vals = np.array([wins_spp.get(c, 0) for c in win_cfgs], dtype=float)
        time_vals = np.array([wins_time.get(c, 0) for c in win_cfgs], dtype=float)
        ax.bar(x - 0.18, spp_vals, width=0.36, label="Best at equal SPP")
        ax.bar(x + 0.18, time_vals, width=0.36, label="Best at equal time")
        ax.set_xticks(x)
        ax.set_xticklabels([label_map.get(c, c).split(" ")[0] for c in win_cfgs], rotation=0)
        ax.set_ylabel("#Wins")
        ax.set_title("Best-configuration wins")
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_root / "grouped_best_config_wins.png", dpi=180)
        plt.close(fig)

    pareto_rows = _compute_pareto_front(overview_rows)
    scatter_rows = [r for r in overview_rows if np.isfinite(_to_float(r.get("time_to_peak_s"))) and np.isfinite(_to_float(r.get("peak_psnr_db")))]
    if scatter_rows:
        fig, ax = plt.subplots(figsize=(10.6, 5.6))
        x_all = np.array([_to_float(r["time_to_peak_s"]) for r in scatter_rows], dtype=float)
        y_all = np.array([_to_float(r["peak_psnr_db"]) for r in scatter_rows], dtype=float)
        ax.scatter(x_all, y_all, s=36, alpha=0.35, label="All configs")

        if pareto_rows:
            x_p = np.array([_to_float(r["time_to_peak_s"]) for r in pareto_rows], dtype=float)
            y_p = np.array([_to_float(r["peak_psnr_db"]) for r in pareto_rows], dtype=float)
            order = np.argsort(x_p)
            ax.plot(x_p[order], y_p[order], linewidth=2.0, label="Pareto front")
            ax.scatter(x_p, y_p, s=56, alpha=0.95)

        for r in sorted(scatter_rows, key=lambda z: int(z.get("overall_rank", 10**9)))[: min(6, len(scatter_rows))]:
            ax.annotate(
                str(r["label"]).split("[")[0].strip(),
                (_to_float(r["time_to_peak_s"]), _to_float(r["peak_psnr_db"])),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )

        ax.set_xscale("log")
        ax.set_xlabel("Time to peak PSNR (s)")
        ax.set_ylabel("Peak PSNR (dB)")
        ax.set_title("Peak quality-speed trade-off")
        ax.grid(True, which="both", linestyle="--", alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_root / "grouped_pareto_peak_psnr_time.png", dpi=180)
        plt.close(fig)

def analyze_bbox_average(output_root: Path, analysis_cfg: dict):
    average_bbox = analysis_cfg.get("average_bbox")
    bbox_norm = _parse_bbox_normalized(average_bbox)
    if bbox_norm is None:
        return

    x0, y0, x1, y1 = bbox_norm
         
    configs = load_config_table(output_root)
    if not configs:
        return
    
    # 1. Find and process the reference image
    ref_val = None
    for cdir_info in configs:
        ref_path = cdir_info["cdir"] / "reference.exr"
        if ref_path.exists():
            ref_img = load_img(ref_path)
            if ref_img is not None:
                if ref_img.ndim == 4 and ref_img.shape[0] == 1:
                    ref_img = ref_img[0]
                print(f"Loaded reference image from {ref_path} with shape {ref_img.shape} and dtype {ref_img.dtype}")
                H, W = ref_img.shape[:2]
                px0, px1 = int(x0 * W), int(x1 * W)
                py0, py1 = int(y0 * H), int(y1 * H)
                
                ref_img_cropped = ref_img[py0:py1, px0:px1]
                
                
                out_path = output_root / "bbox_reference_cropped.exr"
                out_img = np.ascontiguousarray(ref_img_cropped, dtype=np.float32)# Convert RGB to BGR for OpenCV
                if out_img.ndim == 3:
                    if out_img.shape[-1] == 3:
                        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                    elif out_img.shape[-1] == 4:
                        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGBA2BGRA)

                # Set EXR compression type (optional, but good practice)
                # cv2.IMWRITE_EXR_TYPE_HALF sets it to 16-bit float to save space, 
                # use IMWRITE_EXR_TYPE_FLOAT for 32-bit if needed.
                params = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]
                cv2.imwrite(str(out_path), out_img, params)
                print(f"Successfully saved to {out_path}")

                print("will write cropped reference image to:", out_path, " Some stats: shape:", out_img.shape, " dtype:", out_img.dtype, " min:", np.min(out_img), " max:", np.max(out_img), " mean:", np.mean(out_img))
                # iio.imwrite(str(out_path), out_img)
                # # Downsample 4x, make contiguous, and save to root
                # ref_img_down = ref_img_cropped[::4, ::4]
                
                # if iio is not None:
                #     try:
                #         out_path = output_root / "bbox_reference_cropped_4x.exr"
                #         out_img = np.ascontiguousarray(ref_img_down, dtype=np.float32)
                #         print(out_img.shape, out_img.dtype)
                #         print(ref_img_cropped.shape, ref_img_cropped.dtype)
                #         print(ref_img_down.shape, ref_img_down.dtype)
                #         iio.imwrite(str(out_path), out_img)
                #     except Exception as e:
                #         print(f"[WARN] Failed to save cropped exr: {e}")
                
                # Compute luminance (Rec. 709)
                if ref_img_cropped.ndim == 3 and ref_img_cropped.shape[-1] >= 3:
                    lum = 0.2126 * ref_img_cropped[..., 0] + 0.7152 * ref_img_cropped[..., 1] + 0.0722 * ref_img_cropped[..., 2]
                else:
                    lum = ref_img_cropped
                ref_val = float(np.mean(lum))
                break
                
    if ref_val is None:
        print("[WARN] Could not compute reference average for bbox.")
        return

    label_map = build_config_display_labels(configs)

    if plt is None:
        return

    # 2. Setup Figures
    fig_spp, ax_spp = plt.subplots(figsize=(10, 6))
    fig_time, ax_time = plt.subplots(figsize=(10, 6))
    
    ax_spp.axhline(ref_val, color="k", linestyle="--", linewidth=1.5, label="Reference")
    ax_time.axhline(ref_val, color="k", linestyle="--", linewidth=1.5, label="Reference")
    
    # 3. Process each configuration
    for cdir_info in configs:
        cdir = cdir_info["cdir"]
        runs_dir = cdir / "runs"
        if not runs_dir.exists():
            continue
            
        # Group repetitions by SPP
        spp_data = {}
        for p in sorted(runs_dir.glob("spp_*.meta.json")):
            parsed = parse_run_meta_filename(p.name)
            if not parsed:
                continue
            spp, repeat_idx = parsed
            
            exr_name = f"spp_{spp:04d}_rep_{repeat_idx:03d}.exr" if "_rep_" in p.name else f"spp_{spp:04d}.exr"
            exr_path = runs_dir / exr_name
                
            meta = json.loads(p.read_text(encoding="utf-8"))
            eff_time = _time_value(meta)
            
            if exr_path.exists():
                img = load_img(exr_path)
                if img is not None:
                    img_cropped = img[py0:py1, px0:px1]
                    
                    if img_cropped.ndim == 3 and img_cropped.shape[-1] >= 3:
                        lum = 0.2126 * img_cropped[..., 0] + 0.7152 * img_cropped[..., 1] + 0.0722 * img_cropped[..., 2]
                    else:
                        lum = img_cropped
                    val = float(np.mean(lum))
                    
                    spp_data.setdefault(spp, {"vals": [], "times": []})
                    spp_data[spp]["vals"].append(val)
                    spp_data[spp]["times"].append(eff_time)

        if not spp_data:
            continue
            
        spps = sorted(spp_data.keys())
        mean_vals, std_vals, mean_times = [], [], []
        
        for spp in spps:
            vals = spp_data[spp]["vals"]
            times = spp_data[spp]["times"]
            
            mean_vals.append(np.mean(vals))
            std_vals.append(np.std(vals))
            mean_times.append(np.mean(times))
            
        mean_vals = np.array(mean_vals)
        std_vals = np.array(std_vals)
        mean_times = np.array(mean_times)
        spps = np.array(spps)
        
        label = label_map.get(cdir.name, cdir.name)
        
        # Plot SPP with +/- 1 STD shaded area
        line_spp = ax_spp.plot(spps, mean_vals, marker="o", markersize=4, label=label)[0]
        ax_spp.fill_between(spps, mean_vals - std_vals, mean_vals + std_vals, alpha=0.15, color=line_spp.get_color())
        
        # Plot Time with +/- 1 STD shaded area
        line_time = ax_time.plot(mean_times, mean_vals, marker="o", markersize=4, label=label)[0]
        ax_time.fill_between(mean_times, mean_vals - std_vals, mean_vals + std_vals, alpha=0.15, color=line_time.get_color())

    # Finalize SPP Plot
    ax_spp.set_xscale("log", base=2)
    ax_spp.set_xlabel("Samples per pixel (SPP)")
    ax_spp.set_ylabel("Average Luminance")
    ax_spp.set_title("BBox Average Luminance vs SPP")
    ax_spp.grid(True, which="both", linestyle="--", alpha=0.25)
    ax_spp.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig_spp.tight_layout()
    fig_spp.savefig(output_root / "bbox_average_vs_spp.png", dpi=180, bbox_inches="tight")
    plt.close(fig_spp)

    # Finalize Time Plot
    ax_time.set_xscale("log")
    ax_time.set_xlabel("Effective render time (s)")
    ax_time.set_ylabel("Average Luminance")
    ax_time.set_title("BBox Average Luminance vs Time")
    ax_time.grid(True, which="both", linestyle="--", alpha=0.25)
    ax_time.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig_time.tight_layout()
    fig_time.savefig(output_root / "bbox_average_vs_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig_time)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_file")
    args = ap.parse_args()

    exp = yaml.safe_load(Path(args.exp_file).read_text(encoding="utf-8"))
    output_root = Path(exp["output_root"])
    analysis_cfg = exp.get("analysis", {}) or {}
    baseline_config_name = analysis_cfg.get("baseline_config_name")
    top_k_configs = int(analysis_cfg.get("top_k_configs", 10))
    time_grid_points = int(analysis_cfg.get("time_grid_points", 10))
    # inset_bbox = analysis_cfg.get("inset_bbox", [0.7, 0.7, 0.75, 0.75])
    summary_equal_spp = analysis_cfg.get("summary_equal_spp")
    summary_equal_time_s = analysis_cfg.get("summary_equal_time_s")
    metrics_bbox = analysis_cfg.get("average_bbox")

    build_per_config_metrics(output_root, metrics_bbox=metrics_bbox)
    write_cross_config(output_root, baseline_config_name=baseline_config_name)
    write_grouped_comparison(
        output_root,
        baseline_config_name=baseline_config_name,
        top_k=top_k_configs,
        time_grid_points=time_grid_points,
        inset_bbox=metrics_bbox,
        summary_equal_spp=summary_equal_spp,
        summary_equal_time_s=summary_equal_time_s,
    )

    analyze_bbox_average(output_root, analysis_cfg) # to make sure all are unbiased

    print(f"Analysis done: {output_root}")


if __name__ == "__main__":
    main()
