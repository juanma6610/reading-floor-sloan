"""
run_seeds.py — Run the four federated experiments under multiple seeds.

Each (aggregation, partition, seed) triple is one `flwr run .` invocation.
After each run the per-mode output files written by server_app.py are
renamed to include both the partition strategy and the seed, so the
matrix of runs doesn't clobber itself.

Output layout (in <project>/results/federated/):
  federated_<agg>_<partition>_seed<S>_metrics.csv   (per-round global-test curve)
  xgb_federated_<agg>_<partition>_seed<S>_model.json  (final round)
  run_<agg>_<partition>_seed<S>.log                   (flwr stdout/stderr)

Then run evaluate_federated.py to select each run's round on federated
validation loss and produce every reported number.

NOTE: the global test holdout in task.py is intentionally NOT seeded by
this — it always uses RANDOM_STATE = 42 from the module constant. That
means across seeds you get error bars on the SAME test set, which is
what you want for a thesis-style mean±std table.

Usage:
  cd src/federated
  python run_seeds.py                              # all configs, all seeds
  python run_seeds.py --seeds 42 7 123             # custom seed list
  python run_seeds.py --only bagging_iid           # one config, all seeds
  python run_seeds.py --only bagging_iid bagging_team --seeds 42 7
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
FED_DIR      = SCRIPT_DIR
OUTPUT_DIR   = PROJECT_ROOT / "results" / "federated"


# ──────────────────────────────────────────────────────────────
# Experiment matrix and default seeds
# ──────────────────────────────────────────────────────────────
# (label, aggregation, partition_strategy, num_rounds)
EXPERIMENTS = [
    ("bagging_iid",  "bagging", "iid",   50),
    ("bagging_team", "bagging", "team",  50),
    ("cyclic_iid",   "cyclic",  "iid",  1500),
    ("cyclic_team",  "cyclic",  "team", 1500),
]

DEFAULT_SEEDS = [42, 7, 123, 2024, 99]   # 5 seeds is a common thesis sweet spot


def _run_config_string(num_rounds: int, partition: str, aggregation: str, seed: int) -> str:
    return (
        f"num-server-rounds={num_rounds} "
        f"strategy='{partition}' "
        f"aggregation='{aggregation}' "
        f"seed={seed}"
    )


def _server_outputs(aggregation: str) -> dict[str, str]:
    """Server-written files (keyed only on aggregation), mapped to per-run names."""
    return {
        f"federated_{aggregation}_metrics.csv":    "federated_{label}_seed{seed}_metrics.csv",
        f"xgb_federated_{aggregation}_model.json": "xgb_federated_{label}_seed{seed}_model.json",
    }


def _clear_stale_server_outputs(aggregation: str) -> None:
    """
    Remove any server-written output files from a previous run BEFORE we
    launch a new one. Without this, a still-running prior simulation can
    leave half-written files that the next run's rename step then picks
    up as if they belonged to it.
    """
    for src_name in _server_outputs(aggregation).keys():
        p = OUTPUT_DIR / src_name
        if p.exists():
            try:
                p.unlink()
                print(f"  cleared stale: {p.name}")
            except OSError as e:
                print(f"  WARNING: could not remove {p}: {e}")


def _wait_for_run_to_finish(
    aggregation: str,
    num_rounds: int,
    timeout_min: float,
    stability_s: float = 20.0,
    poll_s: float = 5.0,
) -> bool:
    """
    Because `flwr run` is fire-and-forget, wait until the actual federated
    simulation has finished writing its outputs.

    A run is considered finished when ALL of the following hold:
      (i)   the metrics CSV exists and has >= num_rounds data rows;
      (ii)  the final booster JSON exists and is non-empty;
      (iii) none of those three files has changed size for `stability_s`
            consecutive seconds (i.e. the server has stopped writing).

    Returns True on success, False on timeout.
    """
    import csv

    metrics_csv = OUTPUT_DIR / f"federated_{aggregation}_metrics.csv"
    model_json  = OUTPUT_DIR / f"xgb_federated_{aggregation}_model.json"
    watched = [metrics_csv, model_json]

    print(f"  waiting for outputs to finish writing "
          f"(>= {num_rounds} rounds, stable for {stability_s:.0f}s, "
          f"timeout {timeout_min:.0f} min)...")

    deadline = time.time() + timeout_min * 60
    last_sizes: dict[Path, int] = {p: -1 for p in watched}
    stable_since = None
    last_progress_print = 0.0

    while time.time() < deadline:
        time.sleep(poll_s)

        # (i) metrics CSV row count
        n_rows = 0
        if metrics_csv.exists():
            try:
                with open(metrics_csv, newline="") as f:
                    n_rows = max(0, sum(1 for _ in csv.reader(f)) - 1)  # minus header
            except OSError:
                n_rows = 0

        # (ii) booster JSONs present and non-empty
        sizes = {p: (p.stat().st_size if p.exists() else 0) for p in watched}
        nonempty_jsons = sizes[model_json] > 200

        # Heartbeat
        now = time.time()
        if now - last_progress_print > 30:
            print(f"    [waiting] rounds in CSV = {n_rows}/{num_rounds},  "
                  f"sizes(metrics, model) = "
                  f"({sizes[metrics_csv]}, {sizes[model_json]}) B")
            last_progress_print = now

        # (iii) stability
        if n_rows >= num_rounds and nonempty_jsons:
            if sizes == last_sizes:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= stability_s:
                    print(f"    outputs stable; finished. rounds={n_rows}, "
                          f"sizes={list(sizes.values())} B")
                    return True
            else:
                stable_since = None
        last_sizes = sizes

    print(f"  [TIMEOUT] outputs did not stabilize within {timeout_min:.0f} min.")
    return False


def run_one(label: str, aggregation: str, partition: str, num_rounds: int, seed: int) -> bool:
    tag = f"{label} (seed={seed})"
    banner = f"  {tag}  -  {aggregation} / {partition} / {num_rounds} rounds"
    print("\n" + "=" * (len(banner) + 4))
    print("=" + banner)
    print("=" * (len(banner) + 4))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / f"run_{label}_seed{seed}.log"

    # Wipe any stale server-output files from a previous run that may still
    # be sitting on disk. Without this the watcher below can match a leftover
    # file and "finish" before the new simulation has even started writing.
    _clear_stale_server_outputs(aggregation)

    overrides = _run_config_string(num_rounds, partition, aggregation, seed)
    cmd = ["flwr", "run", ".", "--run-config", overrides]
    print(f"$ {' '.join(cmd)}")
    print(f"  log -> {log_path}\n")

    t0 = time.time()
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(FED_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log_f.write(line)
            proc.wait()
        except KeyboardInterrupt:
            print("\n[interrupted] terminating flwr run...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
    dt_submit = time.time() - t0

    if proc.returncode != 0:
        print(f"\n[FAIL] {tag} `flwr run` exited with code {proc.returncode} "
              f"after {dt_submit/60:.1f} min")
        print(f"       see {log_path} for full output")
        return False

    # `flwr run` is fire-and-forget in modern Flower: the CLI submits the run
    # and exits immediately while the actual simulation continues in the
    # background under the SuperLink. Block here until the simulation has
    # actually finished writing its outputs — otherwise the rename step
    # below races the next run.
    # Cyclic runs do 1500 rounds so allow much more wall time.
    timeout_min = 30 if aggregation == "bagging" else 180
    finished = _wait_for_run_to_finish(
        aggregation=aggregation,
        num_rounds=num_rounds,
        timeout_min=timeout_min,
    )
    dt = time.time() - t0

    if not finished:
        print(f"\n[FAIL] {tag} simulation did not finish within timeout "
              f"({timeout_min} min)")
        return False

    # Rename server outputs to include partition strategy AND seed.
    rename_map = _server_outputs(aggregation)
    moved = 0
    for src_name, dst_template in rename_map.items():
        src = OUTPUT_DIR / src_name
        dst = OUTPUT_DIR / dst_template.format(label=label, seed=seed)
        if src.exists():
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
            print(f"  renamed: {src.name}  ->  {dst.name}")
            moved += 1
        else:
            print(f"  WARNING: expected output not found: {src}")

    print(f"\n[OK] {tag} done in {dt/60:.1f} min  ({moved}/{len(rename_map)} files saved)")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                    help=f"seeds to run (default: {DEFAULT_SEEDS})")
    ap.add_argument("--only", type=str, nargs="+", default=None,
                    metavar="LABEL", help=f"subset of experiment labels (default: all of {[e[0] for e in EXPERIMENTS]})")
    args = ap.parse_args()

    plan = [e for e in EXPERIMENTS if args.only is None or e[0] in args.only]
    if not plan:
        print(f"no experiments matched {args.only}; valid labels: {[e[0] for e in EXPERIMENTS]}")
        return 1

    print(f"Project root  : {PROJECT_ROOT}")
    print(f"Federated dir : {FED_DIR}")
    print(f"Output dir    : {OUTPUT_DIR}")
    print(f"Experiments   : {[e[0] for e in plan]}")
    print(f"Seeds         : {args.seeds}")
    print(f"Total runs    : {len(plan) * len(args.seeds)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_t0 = time.time()
    results: dict[tuple[str, int], str] = {}

    try:
        for seed in args.seeds:
            for label, agg, part, rounds in plan:
                ok = run_one(label, agg, part, rounds, seed)
                results[(label, seed)] = "ok" if ok else "FAILED"
    except KeyboardInterrupt:
        results.setdefault(("(interrupted)", -1), "INTERRUPTED")

    total_min = (time.time() - overall_t0) / 60
    print("\n" + "=" * 60)
    print(f"  SUMMARY  (total wall time: {total_min:.1f} min)")
    print("=" * 60)
    for (label, seed), status in results.items():
        print(f"  {label:20s} seed={seed:>4}  {status}")

    return 0 if all(s == "ok" for s in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
