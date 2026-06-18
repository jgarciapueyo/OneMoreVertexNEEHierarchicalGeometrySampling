import argparse
import csv
import hashlib
import itertools
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import yaml


def _norm(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def cfg_hash(params: dict) -> str:
    s = json.dumps(params, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _sanitize_name_token(text: str) -> str:
    token = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text))
    token = token.strip("._-")
    return token or "cfg"


def cfg_name(prefix: str, params: dict) -> str:
    return _sanitize_name_token(prefix)


def _value_token(v):
    if isinstance(v, bool):
        v = "true" if v else "false"
    return _sanitize_name_token(v)


def sweep_config_base_name(sweep_keys, combo_dict):
    tokens = []
    for k in sweep_keys:
        if k not in combo_dict:
            continue
        tok = _value_token(combo_dict[k])
        if tok:
            tokens.append(tok)

    if not tokens:
        return "sweep"
    return "_".join(tokens[:4])


def strip_reserved_params(params: dict):
    return {k: v for k, v in params.items() if not str(k).startswith("__")}


def build_configs(exp: dict):
    defaults = exp.get("global_defaults", {})
    out = []

    for item in exp.get("named_configs", []):
        p = dict(defaults)
        p.update(item.get("params", {}))
        out.append((item.get("name", "named"), p))

    sweep = exp.get("sweep", {})
    if sweep:
        sweep_keys = list(sweep.keys())
        vals = [sweep[k] for k in sweep_keys]
        for combo in itertools.product(*vals):
            combo_dict = dict(zip(sweep_keys, combo))
            p = dict(defaults)
            p.update(combo_dict)
            out.append((sweep_config_base_name(sweep_keys, combo_dict), p))

    seen = set()
    unique = []
    for base, p in out:
        h = cfg_hash(p)
        if h not in seen:
            seen.add(h)
            unique.append((base, p))
    return unique


def parse_mitsuba_time(stdout: str):
    patterns = [
        r"render time[^0-9]*([0-9.]+)\s*(ms|s|m)",
        r"\btime[^0-9]*([0-9.]+)\s*(ms|s|m)",
    ]
    for pat in patterns:
        m = re.search(pat, stdout, flags=re.IGNORECASE)
        if m:
            val = float(m.group(1))
            unit = m.group(2).lower()
            if unit == "ms":
                return val / 1000.0
            elif unit == "m":
                return val * 60.0
            else:
                return val
    return None


def _tail(text: str, n: int = 20):
    if not text:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _cmd_to_str(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def build_reference_params(reference_cfg: dict, config_params: dict):
    inherit = bool(reference_cfg.get("inherit_config_params", True))
    params = strip_reserved_params(config_params) if inherit else {}

    # Ensure reference render still receives geometry from the active config.
    geometry_value = config_params.get("geometry")

    exclude_params = reference_cfg.get("exclude_params", []) or []
    for k in exclude_params:
        params.pop(str(k), None)

    params.update(reference_cfg.get("params", {}) or {})

    # If not explicitly set in reference.params, carry geometry through.
    if "geometry" not in params and geometry_value is not None:
        params["geometry"] = geometry_value

    return params


def print_render_failure(
    stage: str,
    config_name: str,
    spp,
    result: dict,
    stdout_log: Path = None,
    stderr_log: Path = None,
):
    spp_text = "" if spp is None else f", spp={spp}"
    print(f"[ERROR] Mitsuba {stage} failed for config={config_name}{spp_text}")
    print(
        f"        returncode={result.get('returncode')} "
        f"wall_time_s={result.get('wall_time_s')} "
        f"mitsuba_time_s={result.get('mitsuba_time_s')}"
    )

    cmd = result.get("cmd")
    if cmd:
        print(f"        cmd: {_cmd_to_str(cmd)}")

    if stdout_log is not None:
        print(f"        stdout log: {stdout_log}")
    if stderr_log is not None:
        print(f"        stderr log: {stderr_log}")

    stderr_tail = _tail(result.get("stderr", ""), n=20)
    stdout_tail = _tail(result.get("stdout", ""), n=10)

    if stderr_tail:
        print("        ---- stderr tail ----")
        print(stderr_tail)
    elif stdout_tail:
        print("        ---- stdout tail ----")
        print(stdout_tail)


# Integrators provided by the PPG build (mitsuba-ppg) rather than the main build.
_PPG_INTEGRATORS = {"guided_path"}


def _resolve_mitsuba(
    params: dict,
    setpath_script: str,
    ppg_setpath_script: Optional[str],
):
    """Return (exe_name, setpath_script) for the given config params."""
    integrator = str(params.get("integrator", ""))
    if integrator in _PPG_INTEGRATORS:
        if not ppg_setpath_script:
            raise ValueError(
                f"Integrator '{integrator}' requires docker.ppg_setpath_script to be set in the YAML."
            )
        return "mitsuba-ppg", ppg_setpath_script
    return "mitsuba", setpath_script


def build_docker_mitsuba_cmd(
    docker_bin: str,
    container: str,
    workdir: str,
    setpath_script: str,
    scene: str,
    out_exr_rel: str,
    spp: int,
    params: dict,
    mitsuba_exe: str = "mitsuba",
):
    mitsuba_cmd = [mitsuba_exe, scene, "-o", out_exr_rel, f"-Dspp={spp}"]
    for k, v in sorted(params.items()):
        mitsuba_cmd.append(f"-D{k}={_norm(v)}")

    shell_cmd = f"source {shlex.quote(setpath_script)} && {' '.join(shlex.quote(x) for x in mitsuba_cmd)}"
    return [docker_bin, "exec", "-w", workdir, container, "bash", "-lc", shell_cmd]


def run_render(
    docker_bin: str,
    container: str,
    workdir: str,
    setpath_script: str,
    scene: str,
    out_exr_rel: str,
    spp: int,
    params: dict,
    mitsuba_exe: str = "mitsuba",
):
    cmd = build_docker_mitsuba_cmd(docker_bin, container, workdir, setpath_script, scene, out_exr_rel, spp, params, mitsuba_exe)

    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        wall_s = time.perf_counter() - t0
        mitsuba_s = parse_mitsuba_time(p.stdout or "")

        return {
            "cmd": cmd,
            "returncode": p.returncode,
            "wall_time_s": wall_s,
            "mitsuba_time_s": mitsuba_s,
            "stdout": p.stdout or "",
            "stderr": p.stderr or "",
        }
    except Exception as e:
        wall_s = time.perf_counter() - t0
        return {
            "cmd": cmd,
            "returncode": 127,
            "wall_time_s": wall_s,
            "mitsuba_time_s": None,
            "stdout": "",
            "stderr": f"Runner exception while invoking Mitsuba: {type(e).__name__}: {e}",
        }


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_run_paths(runs_dir: Path, spp: int, repeat_idx: int, repetitions_per_spp: int):
    if repetitions_per_spp <= 1:
        stem = f"spp_{int(spp):04d}"
    else:
        stem = f"spp_{int(spp):04d}_rep_{int(repeat_idx):03d}"
    return {
        "out_exr": runs_dir / f"{stem}.exr",
        "meta_json": runs_dir / f"{stem}.meta.json",
        "out_log": runs_dir / f"{stem}.stdout.log",
        "err_log": runs_dir / f"{stem}.stderr.log",
    }


def resolve_reference_source_path(path_value: str, exp_file_path: Path) -> Path:
    src = Path(str(path_value))
    if src.is_absolute():
        return src

    src_rel_to_exp = (exp_file_path.parent / src).resolve()
    if src_rel_to_exp.exists():
        return src_rel_to_exp

    return src.resolve()


def _same_target(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except Exception:
        return False


def materialize_reference_exr(reference_source: Path, target_ref_exr: Path) -> str:
    target_ref_exr.parent.mkdir(parents=True, exist_ok=True)

    if target_ref_exr.exists() or target_ref_exr.is_symlink():
        if _same_target(target_ref_exr, reference_source):
            return "existing"
        target_ref_exr.unlink()

    try:
        rel_src = os.path.relpath(reference_source, start=target_ref_exr.parent)
        target_ref_exr.symlink_to(rel_src)
        return "symlink"
    except Exception:
        pass

    try:
        os.link(reference_source, target_ref_exr)
        return "hardlink"
    except Exception:
        pass

    shutil.copy2(reference_source, target_ref_exr)
    return "copy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_file")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--container", default=None)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--setpath-script", default=None)
    ap.add_argument("--docker-bin", default="docker")
    args = ap.parse_args()

    exp_file_path = Path(args.exp_file).resolve()
    exp = yaml.safe_load(exp_file_path.read_text(encoding="utf-8"))
    
    # Extract and validate config parameters
    # Docker
    docker_cfg = exp.get("docker", {})
    container = args.container or docker_cfg.get("container")
    workdir = args.workdir or docker_cfg.get("workdir")
    setpath_script = args.setpath_script or docker_cfg.get("setpath_script") or "/home/mitsuba/setpath.sh"
    ppg_setpath_script = docker_cfg.get("ppg_setpath_script")
    if not container or not workdir:
        raise ValueError("Missing Docker config. Provide --container/--workdir or set 'docker.container' and 'docker.workdir' in YAML.")

    # Scene and output
    scene = exp["scene"]  # container-relative (or absolute in container)
    output_root = Path(exp["output_root"])  # host path, expected bind-mounted in container workdir
    
    # Reference config
    reference_cfg = exp.get("reference", {}) or {}
    reference_scene = reference_cfg.get("scene") or exp.get("reference_scene") or scene
    pre_rendered_reference_exr = reference_cfg.get("pre_rendered_exr")
    
    reference_source_path = None
    if pre_rendered_reference_exr:
        reference_source_path = resolve_reference_source_path(pre_rendered_reference_exr, exp_file_path)
        if not reference_source_path.exists():
            raise FileNotFoundError(
                f"reference.pre_rendered_exr does not exist: {reference_source_path}"
            )
        print(f"[INFO] Using pre-rendered reference EXR: {reference_source_path}")

    reference_spp = None
    if reference_source_path is None:
        if "spp" in reference_cfg:
            reference_spp = int(reference_cfg["spp"])
        else:
            raise ValueError(
                "Missing reference spp. Set reference.spp (preferred)"
                "or provide reference.pre_rendered_exr to reuse an existing reference image."
            )
    
    # Main schedule for tested runs
    spp_schedule = exp["spp_schedule"]
    repetitions_per_spp = int(exp.get("repetitions_per_spp", exp.get("spp_repetitions", 1)))
    if repetitions_per_spp < 1:
        raise ValueError("repetitions_per_spp (or spp_repetitions) must be >= 1")

    output_root.mkdir(parents=True, exist_ok=True)
    configs = build_configs(exp)

    summary_rows = []
    used_name_counts = {}

    for i, (base_name, params) in enumerate(configs, start=1):
        base_cname = cfg_name(base_name, params)
        nth = used_name_counts.get(base_cname, 0) + 1
        used_name_counts[base_cname] = nth
        cname = base_cname if nth == 1 else f"{base_cname}_{nth}"
        print(f"[INFO] [{i}/{len(configs)}] Running config: {cname}")
        scene_for_config = str(params.get("__scene", scene))
        run_params = strip_reserved_params(params)
        cdir = output_root / cname
        runs_dir = cdir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        (cdir / "config.json").write_text(json.dumps(run_params, indent=2), encoding="utf-8")

        ref_exr = cdir / "reference.exr"
        ref_exr_rel = ref_exr.as_posix()
        reference_meta_json = cdir / "reference.meta.json"
        if reference_source_path is not None:
            link_mode = "existing" if (args.resume and ref_exr.exists()) else materialize_reference_exr(reference_source_path, ref_exr)
            if not (args.resume and reference_meta_json.exists()):
                reference_meta_json.write_text(
                    json.dumps(
                        {
                            "mode": "pre_rendered",
                            "source_exr": str(reference_source_path),
                            "materialization": link_mode,
                            "returncode": 0,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        else:
            if not (args.resume and ref_exr.exists()):
                ref_params = build_reference_params(reference_cfg, run_params)
                ref_exe, ref_setpath = _resolve_mitsuba(ref_params, setpath_script, ppg_setpath_script)
                r = run_render(
                    args.docker_bin,
                    container,
                    workdir,
                    ref_setpath,
                    reference_scene,
                    ref_exr_rel,
                    reference_spp,
                    ref_params,
                    ref_exe,
                )
                ref_stdout_log = cdir / "stdout_reference.log"
                ref_stderr_log = cdir / "stderr_reference.log"
                write_text(ref_stdout_log, r["stdout"])
                write_text(ref_stderr_log, r["stderr"])
                reference_meta_json.write_text(
                    json.dumps({k: v for k, v in r.items() if k not in ("stdout", "stderr")}, indent=2),
                    encoding="utf-8",
                )
                if r["returncode"] != 0:
                    print_render_failure(
                        stage="reference",
                        config_name=cname,
                        spp=reference_spp,
                        result=r,
                        stdout_log=ref_stdout_log,
                        stderr_log=ref_stderr_log,
                    )
                    continue

        timing_csv = cdir / "timings.csv"
        first_write = not timing_csv.exists()
        with timing_csv.open("a", newline="", encoding="utf-8") as f:
            timing_fields = ["spp", "returncode", "wall_time_s", "mitsuba_time_s", "image_path"]
            if repetitions_per_spp > 1:
                timing_fields = ["spp", "repeat", "returncode", "wall_time_s", "mitsuba_time_s", "image_path"]
            w = csv.DictWriter(
                f,
                fieldnames=timing_fields,
            )
            if first_write:
                w.writeheader()

            for spp in spp_schedule:
                for repeat_idx in range(1, repetitions_per_spp + 1):
                    run_paths = build_run_paths(runs_dir, int(spp), repeat_idx, repetitions_per_spp)
                    out_exr = run_paths["out_exr"]
                    out_exr_rel = out_exr.as_posix()
                    meta_json = run_paths["meta_json"]
                    out_log = run_paths["out_log"]
                    err_log = run_paths["err_log"]

                    if args.resume and out_exr.exists() and meta_json.exists():
                        continue

                    run_exe, run_setpath = _resolve_mitsuba(run_params, setpath_script, ppg_setpath_script)
                    r = run_render(
                        args.docker_bin,
                        container,
                        workdir,
                        run_setpath,
                        scene_for_config,
                        out_exr_rel,
                        int(spp),
                        run_params,
                        run_exe,
                    )
                    write_text(out_log, r["stdout"])
                    write_text(err_log, r["stderr"])
                    meta_json.write_text(
                        json.dumps({k: v for k, v in r.items() if k not in ("stdout", "stderr")}, indent=2),
                        encoding="utf-8",
                    )

                    if r["returncode"] != 0:
                        print_render_failure(
                            stage=f"run (repeat {repeat_idx}/{repetitions_per_spp})",
                            config_name=cname,
                            spp=int(spp),
                            result=r,
                            stdout_log=out_log,
                            stderr_log=err_log,
                        )

                    row = {
                        "spp": int(spp),
                        "returncode": r["returncode"],
                        "wall_time_s": r["wall_time_s"],
                        "mitsuba_time_s": r["mitsuba_time_s"],
                        "image_path": str(out_exr),
                    }
                    if repetitions_per_spp > 1:
                        row["repeat"] = repeat_idx
                    w.writerow(row)

        summary_rows.append({"config_dir": str(cdir), "config_name": cname})

    if summary_rows:
        with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["config_name", "config_dir"])
            w.writeheader()
            w.writerows(summary_rows)

    print(f"Finished. Results at: {output_root}")


if __name__ == "__main__":
    main()