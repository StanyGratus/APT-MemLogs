"""
APT Combiner
============
Loops over TIME_POINTS dict (folder → label), finds the results/ subfolder
inside each time-point directory, calls run_pipeline() on each, concatenates,
and saves master_apt_dataset.csv.

Edit TIME_POINTS below or pass a JSON config path as the first CLI argument.

Usage:
    python apt_combiner.py
    python apt_combiner.py time_points.json
"""

import os
import sys
import json
import glob
import pandas as pd
from tabulate import tabulate         # pip install tabulate  (optional, falls back to plain print)

from apt_baseline_pipeline import run_pipeline

# ─────────────────────────────────────────────────────────────────────────────
#  TIME_POINTS: map each dump folder name → integer label (0=benign, 1=malicious, -1=unknown)
#  Keys are the top-level folders; the script automatically finds the results/ subfolder.
#
#  Example:
#    TIME_POINTS = {
#        "/data/dumps/H.D6RU1AR2MTM4S": 1,
#        "/data/dumps/H.CLEAN001":       0,
#        "/data/dumps/H.UNKNOWN":       -1,
#    }
# ─────────────────────────────────────────────────────────────────────────────
MAGICRAT_ROOT = r"C:\Users\stany\OneDrive\Desktop\APTs\Real APT logs\cobaltStrike"# ← only line you edit

def discover_time_points(root: str) -> dict:
    label_map = {"dataset_pre": 0, "dataset_during": 1, "dataset_post": 1}
    time_points = {}
    for phase, label in label_map.items():
        phase_path = os.path.join(root, phase)
        if not os.path.isdir(phase_path):
            continue
        for snapshot in sorted(os.listdir(phase_path)):
            snapshot_path = os.path.join(phase_path, snapshot)
            if not os.path.isdir(snapshot_path):
                continue
            for machine in os.listdir(snapshot_path):
                machine_path = os.path.join(snapshot_path, machine)
                if os.path.isdir(machine_path):
                    time_points[machine_path] = label
    return time_points

OUTPUT_FILE = "master_apt_dataset.csv"


def find_results_folder(base_folder: str) -> str:
    """
    Locate the results/ subfolder inside base_folder.
    Velociraptor stores exports in:
        <base_folder>/results/<random_subfolder>/
    We walk one level deep to find a directory named 'results', then
    return the first randomly-named subfolder inside it.
    If the folder itself contains NDJSON files, return it directly.
    """
    for root, dirs, files in os.walk(base_folder):
        if any(f.endswith(".json") and not f in ("client_info.json","log.json","collection_context.json","requests.json","uploads.json") for f in files):
            return root
    return base_folder


def main():
    global TIME_POINTS

    # Allow passing a JSON config file as CLI argument
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        if not os.path.exists(config_path):
            print(f"[ERROR] Config file not found: {config_path}")
            sys.exit(1)
        with open(config_path) as fh:
            TIME_POINTS = json.load(fh)
        print(f"Loaded TIME_POINTS from {config_path}: {len(TIME_POINTS)} entries")

    TIME_POINTS = discover_time_points(MAGICRAT_ROOT)
    if not TIME_POINTS:
        print("[ERROR] No snapshots found. Check MAGICRAT_ROOT path.")
        sys.exit(1)

    all_results = []
    summary_rows = []

    for base_folder, label in TIME_POINTS.items():
        # e.g. "dataset_pre__snapshot_000001__MDC04"
        parts = base_folder.replace(MAGICRAT_ROOT, "").strip("/").replace("/", "__")
        dump_id = parts
        results_folder = find_results_folder(base_folder)
        print(f"  Results folder: {results_folder}")

        print(f"\n{'#'*70}")
        print(f"  Processing: {dump_id}  (label={label})")
        print(f"  Results folder: {results_folder}")
        print(f"{'#'*70}")

        if not os.path.isdir(results_folder):
            print(f"  [SKIP] Folder not found: {results_folder}")
            summary_rows.append({
                "dump_id": dump_id, "label": label,
                "status": "MISSING", "rows": 0, "heuristic_positives": 0
            })
            continue

        try:
            df = run_pipeline(
                folder=results_folder,
                dump_id=dump_id,
                dump_label=label,
                save_output=False,   # combiner saves the master file itself
            )
        except Exception as exc:
            print(f"  [ERROR] Pipeline failed for {dump_id}: {exc}")
            summary_rows.append({
                "dump_id": dump_id, "label": label,
                "status": "ERROR", "rows": 0, "heuristic_positives": 0
            })
            continue

        if df is None or df.empty:
            print(f"  [WARN] Empty result for {dump_id}")
            summary_rows.append({
                "dump_id": dump_id, "label": label,
                "status": "EMPTY", "rows": 0, "heuristic_positives": 0
            })
            continue

        all_results.append(df)
        summary_rows.append({
            "dump_id":             dump_id,
            "label":               label,
            "status":              "OK",
            "rows":                len(df),
            "heuristic_positives": int(df.get("heuristic_label", pd.Series(0)).sum()),
            "high_risk_procs":     int((df.get("apt_risk_score", pd.Series(0)) >= 5).sum()),
        })

    # ── Concatenate and save ──────────────────────────────────
    if not all_results:
        print("\n[ERROR] No data collected. Check folder paths in TIME_POINTS.")
        sys.exit(1)

    master = pd.concat(all_results, ignore_index=True)
    master.to_csv(OUTPUT_FILE, index=False)
    print(f"\n{'='*70}")
    print(f"  Master dataset saved → {OUTPUT_FILE}")
    print(f"  Total rows : {len(master)}")
    print(f"  Total cols : {master.shape[1]}")

    # ── Per-dump breakdown ────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Per-dump breakdown:")
    print(f"{'='*70}")
    summary_df = pd.DataFrame(summary_rows)

    try:
        print(tabulate(summary_df, headers="keys", tablefmt="rounded_outline", showindex=False))
    except ImportError:
        # tabulate not installed — plain fallback
        print(summary_df.to_string(index=False))

    # Label distribution
    if "dump_label" in master.columns:
        label_counts = master["dump_label"].value_counts().sort_index()
        print("\n  Label distribution (row level):")
        for lbl, cnt in label_counts.items():
            tag = {0: "benign", 1: "malicious", -1: "unknown"}.get(lbl, str(lbl))
            print(f"    {lbl:>3}  ({tag:<10}) : {cnt:>6} rows")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
