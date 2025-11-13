#!/usr/bin/env python
"""
stage2_pairwise_eval.py

Scan trajectories exported by stage1 (trajectories_all.csv) and
find close approaches between objects using OBB distance.

Config (matches your request):
  - OBB distance: ON (via obb_distance extension)
  - Pair-type policy: user-configurable (D), default = "all"
      * "all": all object-type pairs EXCEPT (Sign, Sign) when both stationary
      * "VV" : Vehicle–Vehicle only
  - Thresholds:
      center <= 10.0 m
      OBB    <= 2.5 m

Inputs:
  data/samples/trajectories_all.csv

Outputs:
  data/samples/workload_pairs_min.csv   (per-pair min distances, one row per pair)
  data/samples/closeness_summary.csv    (per-file summary)
"""

import os
import csv
import math
import argparse
from collections import defaultdict, Counter

import numpy as np

# Try to use OBB extension
try:
    import obb_distance
    USE_OBB = True
except Exception:
    print("⚠️  obb_distance extension not found, falling back to center distance only.")
    USE_OBB = False


# -------------------------------------------------------------------
# Paths / config defaults
# -------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data", "samples")

TRAJ_CSV_DEFAULT = os.path.join(DATA_DIR, "trajectories_all.csv")
PAIRS_OUT_DEFAULT = os.path.join(DATA_DIR, "workload_pairs_min.csv")
SUMMARY_OUT_DEFAULT = os.path.join(DATA_DIR, "closeness_summary.csv")


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------

def load_trajectories(csv_path):
    """
    Load trajectories_all.csv into:
      trajs[file][oid] = list of dicts sorted by time
    """
    trajs = defaultdict(lambda: defaultdict(list))
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Trajectory CSV not found: {csv_path}")

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {h: i for i, h in enumerate(header)}

        required = ["file", "oid", "type", "t", "cx", "cy",
                    "cz", "heading", "length", "width", "height",
                    "speed", "is_stationary"]
        for r in required:
            if r not in idx:
                raise ValueError(f"Missing column '{r}' in {csv_path}")

        for row in reader:
            file_id = row[idx["file"]]
            oid = row[idx["oid"]]
            rec = {
                "file": file_id,
                "oid": oid,
                "type": row[idx["type"]],
                "t": float(row[idx["t"]]),
                "cx": float(row[idx["cx"]]),
                "cy": float(row[idx["cy"]]),
                "cz": float(row[idx["cz"]]),
                "heading": float(row[idx["heading"]]),
                "length": float(row[idx["length"]]),
                "width": float(row[idx["width"]]),
                "height": float(row[idx["height"]]),
                "speed": float(row[idx["speed"]]),
                "is_stationary": int(row[idx["is_stationary"]]),
            }
            trajs[file_id][oid].append(rec)

    # Sort each track by time
    for file_id in trajs:
        for oid in trajs[file_id]:
            trajs[file_id][oid].sort(key=lambda r: r["t"])

    return trajs


def center_distance(a, b):
    dx = a["cx"] - b["cx"]
    dy = a["cy"] - b["cy"]
    return math.hypot(dx, dy)


def obb_distance_between(a, b):
    if not USE_OBB:
        return center_distance(a, b)
    return obb_distance.obb_min_distance(
        a["cx"], a["cy"], a["length"], a["width"], a["heading"],
        b["cx"], b["cy"], b["length"], b["width"], b["heading"]
    )


def pair_type_allowed(a, b, policy):
    """
    policy:
      - "VV"  : Vehicle–Vehicle only
      - "all" : all pairs EXCEPT (Sign, Sign) when both stationary
    """
    ta, tb = a["type"], b["type"]
    sa, sb = a["is_stationary"], b["is_stationary"]

    if policy == "VV":
        return (ta == "Vehicle") and (tb == "Vehicle")

    # default "all": all pairs, but exclude Sign–Sign when both stationary
    if ta == "Sign" and tb == "Sign" and sa == 1 and sb == 1:
        return False
    return True


def build_time_index(tracks, time_window):
    """
    tracks: dict[oid] -> [rec,...] for one file
    time_window: window used to group nearly-simultaneous records

    Returns:
      list of (t_center, [records_active_at_t])
    """
    # Unique times from all records
    all_times = sorted({r["t"] for arr in tracks.values() for r in arr})
    if not all_times:
        return []

    half_win = time_window / 2.0
    time_index = []

    for t in all_times:
        snapshot = []
        for oid, arr in tracks.items():
            # Simple nearest-in-time search
            best = None
            best_dt = 1e9
            for rec in arr:
                dt = abs(rec["t"] - t)
                if dt < best_dt:
                    best_dt = dt
                    best = rec
            if best is not None and best_dt <= half_win:
                snapshot.append(best)
        if snapshot:
            time_index.append((t, snapshot))

    return time_index


# -------------------------------------------------------------------
# Main pairwise scanner
# -------------------------------------------------------------------

def scan_file(file_id, tracks, center_thresh, obb_thresh, pair_policy, time_window):
    """
    tracks: dict[oid] -> [rec,...] for a single file
    Returns:
      best_pairs: dict[(oid_i, oid_j)] -> dict(...)
      stats: dict summary
    """
    # Quick stats
    n_tracks = len(tracks)
    n_samples = sum(len(v) for v in tracks.values())
    type_counter = Counter(r["type"] for arr in tracks.values() for r in arr)

    print(f"  Tracks: {n_tracks}, samples: {n_samples}")
    print(f"  Type histogram: {dict(type_counter)}")

    # Build time index
    time_index = build_time_index(tracks, time_window)
    print(f"  Time bins: {len(time_index)}")

    best_pairs = {}  # key = (oid_i, oid_j), value = dict with metrics

    for t, snapshot in time_index:
        # For debugging: how many vehicles, min center distance, etc.
        veh_snap = [r for r in snapshot if r["type"] == "Vehicle"]
        if veh_snap:
            min_center = None
            for i in range(len(veh_snap)):
                for j in range(i + 1, len(veh_snap)):
                    d = center_distance(veh_snap[i], veh_snap[j])
                    if (min_center is None) or (d < min_center):
                        min_center = d
            # Uncomment if you want per-time logging:
            # print(f"[t={t:.3f}] vehicles={len(veh_snap)}, min(center)≈{min_center:.2f} m")

        # Pairwise on snapshot
        for i in range(len(snapshot)):
            a = snapshot[i]
            for j in range(i + 1, len(snapshot)):
                b = snapshot[j]

                if not pair_type_allowed(a, b, pair_policy):
                    continue

                d_center = center_distance(a, b)
                if d_center > center_thresh:
                    continue

                d_obb = obb_distance_between(a, b)
                if d_obb > obb_thresh:
                    continue

                # Keep best (min obb distance) per pair
                key = tuple(sorted((a["oid"], b["oid"])))
                prev = best_pairs.get(key)
                if (prev is None) or (d_obb < prev["d_obb_min"]):
                    best_pairs[key] = {
                        "file": file_id,
                        "oid_i": key[0],
                        "oid_j": key[1],
                        "type_i": a["type"],
                        "type_j": b["type"],
                        "t_at": t,
                        "d_center_min": d_center,
                        "d_obb_min": d_obb,
                        "L_i": a["length"],
                        "W_i": a["width"],
                        "H_i": a["height"],
                        "L_j": b["length"],
                        "W_j": b["width"],
                        "H_j": b["height"],
                        "speed_i": a["speed"],
                        "speed_j": b["speed"],
                        "is_stationary_i": a["is_stationary"],
                        "is_stationary_j": b["is_stationary"],
                    }

    stats = {
        "file": file_id,
        "tracks": n_tracks,
        "samples": n_samples,
        "pairs_found": len(best_pairs),
        "type_histogram": dict(type_counter),
    }

    print(
        f"  → {len(best_pairs)} pairs within "
        f"{center_thresh:.1f} m center & {obb_thresh:.2f} m OBB."
    )

    return best_pairs, stats


def write_pairs_csv(pairs_dicts, out_path):
    if not pairs_dicts:
        print("⚠️  No pairs to write (workload_pairs_min.csv).")
        return

    fieldnames = [
        "file",
        "oid_i", "oid_j",
        "type_i", "type_j",
        "t_at",
        "d_center_min", "d_obb_min",
        "L_i", "W_i", "H_i",
        "L_j", "W_j", "H_j",
        "speed_i", "speed_j",
        "is_stationary_i", "is_stationary_j",
    ]

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in pairs_dicts:
            w.writerow(row)
    print(f"✅ Pairs written → {out_path}")


def write_summary_csv(summary_list, out_path):
    if not summary_list:
        print("⚠️  No summary to write.")
        return

    # flatten type_histogram into JSON-like string
    fieldnames = ["file", "tracks", "samples", "pairs_found", "type_histogram"]

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summary_list:
            row = dict(s)
            row["type_histogram"] = repr(row["type_histogram"])
            w.writerow(row)
    print(f"✅ Summary written → {out_path}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Scan trajectories_all.csv for close approaches using OBB distance."
    )
    ap.add_argument(
        "--traj-csv",
        default=TRAJ_CSV_DEFAULT,
        help=f"Trajectory CSV (default: {TRAJ_CSV_DEFAULT})",
    )
    ap.add_argument(
        "--pairs-out",
        default=PAIRS_OUT_DEFAULT,
        help=f"Output CSV for per-pair minima (default: {PAIRS_OUT_DEFAULT})",
    )
    ap.add_argument(
        "--summary-out",
        default=SUMMARY_OUT_DEFAULT,
        help=f"Output CSV for per-file summary (default: {SUMMARY_OUT_DEFAULT})",
    )
    ap.add_argument(
        "--center-thresh",
        type=float,
        default=10.0,
        help="Max center-to-center distance (meters) for candidate pairs (default: 10.0)",
    )
    ap.add_argument(
        "--obb-thresh",
        type=float,
        default=2.5,
        help="Max OBB-to-OBB distance (meters) for candidate pairs (default: 2.5)",
    )
    ap.add_argument(
        "--time-window",
        type=float,
        default=0.15,
        help="Time window (seconds) to group records as simultaneous (default: 0.15)",
    )
    ap.add_argument(
        "--pair-types",
        choices=["VV", "all"],
        default="all",
        help=(
            "Pair-type policy: 'VV' = Vehicle–Vehicle only; "
            "'all' = all types except stationary Sign–Sign (default: all)"
        ),
    )

    args = ap.parse_args()

    print(f"▶ Loading trajectories from {args.traj_csv} ...")
    trajs_by_file = load_trajectories(args.traj_csv)
    print(f"  Files in CSV: {len(trajs_by_file)}")

    all_pairs = []
    summary = []

    for file_id, tracks in trajs_by_file.items():
        print(f"\n▶ Processing file: {file_id}")
        pairs_dict, stats = scan_file(
            file_id=file_id,
            tracks=tracks,
            center_thresh=args.center_thresh,
            obb_thresh=args.obb_thresh,
            pair_policy=args.pair_types,
            time_window=args.time_window,
        )
        all_pairs.extend(pairs_dict.values())
        summary.append(stats)

    if not all_pairs:
        print(
            f"\n❌ No pairs found under center≤{args.center_thresh} m "
            f"& OBB≤{args.obb_thresh} m, policy={args.pair_types}."
        )
    else:
        write_pairs_csv(all_pairs, args.pairs_out)

    write_summary_csv(summary, args.summary_out)


if __name__ == "__main__":
    main()
