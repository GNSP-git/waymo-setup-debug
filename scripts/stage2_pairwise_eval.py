#!/usr/bin/env python
# stage2_pairwise_eval.py
#
# Stage 2: From trajectories_all.csv → find close pairs (Vehicle–Vehicle only, Option A)
# and write a compact workload file for Ganaka / visualization.
#
# - Uses OBB distance if obb_distance extension is present
# - Falls back to center distance otherwise
# - Filters strictly to Vehicle–Vehicle pairs (Option A)
#
# Outputs:
#   /home/gns/waymo_work/data/samples/workload_pairs_min.csv
#   /home/gns/waymo_work/data/samples/closeness_summary.csv

import os
import csv
import math
from collections import defaultdict, Counter

import numpy as np

# -------- CONFIG --------
BASE_DIR   = "/home/gns/waymo_work"
SAVE_DIR   = os.path.join(BASE_DIR, "data", "samples")
TRAJ_CSV   = os.path.join(SAVE_DIR, "trajectories_all.csv")
PAIRS_CSV  = os.path.join(SAVE_DIR, "workload_pairs_min.csv")
SUMMARY_CSV = os.path.join(SAVE_DIR, "closeness_summary.csv")

# thresholds
CENTER_THRESH = 10.0   # meters (center-to-center cutoff)
OBB_THRESH    = 2.5    # meters (min OBB distance cutoff)

# time binning (for sampling snapshots)
TIME_BIN = 0.1         # seconds per bin

# mode = "A" (Vehicle–Vehicle only); B,C reserved for later
MODE = "A"

# -------- OBB distance extension (optional) --------
USE_OBB = True
try:
    import obb_distance
except Exception:
    USE_OBB = False
    print("ℹ️ obb_distance extension not found; falling back to center distance only.")


def pair_obb_distance(a, b):
    """Return OBB min distance if extension is available, else center distance."""
    if USE_OBB:
        return obb_distance.obb_min_distance(
            a["cx"], a["cy"], a["length"], a["width"], a["heading"],
            b["cx"], b["cy"], b["length"], b["width"], b["heading"]
        )
    # fallback: Euclidean center distance
    dx = a["cx"] - b["cx"]
    dy = a["cy"] - b["cy"]
    return math.hypot(dx, dy)


def pair_center_distance(a, b):
    dx = a["cx"] - b["cx"]
    dy = a["cy"] - b["cy"]
    return math.hypot(dx, dy)


def include_pair(a, b, mode="A"):
    """
    Pair filter by mode.
    Mode A: Vehicle–Vehicle only.
    (Modes B/C reserved for later – for now they behave like 'all except stationary Sign–Sign').
    """
    ta, tb = a["type"], b["type"]

    if mode == "A":
        # Strict Vehicle–Vehicle only
        return (ta == "Vehicle" and tb == "Vehicle")

    # --- Future: B / C behavior (for now: all except stationary Sign–Sign) ---
    both_sign = (ta == "Sign" and tb == "Sign")
    if both_sign and a["is_stationary"] and b["is_stationary"]:
        return False
    return True


def load_trajectories(csv_path):
    """
    Load trajectories_all.csv and return:
      traj_by_file: {file -> {oid -> [records...]}}
    Each record:
      dict(t, cx, cy, cz, heading, length, width, height, type, speed, is_stationary)
    """
    traj_by_file = defaultdict(lambda: defaultdict(list))
    files_seen = set()
    print(f"▶ Loading trajectories from {csv_path} ...")

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {h: i for i, h in enumerate(header)}

        for row in reader:
            fid = row[idx["file"]]
            oid = row[idx["oid"]]
            typ = row[idx["type"]]

            rec = {
                "file": fid,
                "oid": oid,
                "type": typ,
                "t": float(row[idx["t"]]),
                "cx": float(row[idx["cx"]]),
                "cy": float(row[idx["cy"]]),
                "cz": float(row[idx["cz"]]),
                "heading": float(row[idx["heading"]]),
                "length": float(row[idx["length"]]),
                "width": float(row[idx["width"]]),
                "height": float(row[idx["height"]]),
                "speed": float(row[idx["speed"]]),
                "is_stationary": bool(int(row[idx["is_stationary"]])),
            }
            traj_by_file[fid][oid].append(rec)
            files_seen.add(fid)

    # sort each trajectory by time
    for fid in traj_by_file:
        for oid in traj_by_file[fid]:
            traj_by_file[fid][oid].sort(key=lambda r: r["t"])

    print(f"  Files in CSV: {len(files_seen)}")
    return traj_by_file


def build_time_bins(trajs_for_file):
    """
    Given {oid -> [records]} for a single file, return array of bin centers.
    """
    all_ts = [r["t"] for arr in trajs_for_file.values() for r in arr]
    if not all_ts:
        return np.array([])
    t_min = min(all_ts)
    t_max = max(all_ts)
    # inclusive of t_max by padding slightly
    n_bins = max(1, int(math.ceil((t_max - t_min) / TIME_BIN)))
    bins = t_min + (np.arange(n_bins) + 0.5) * TIME_BIN
    return bins


def snapshot_at_time(traj_dict, t_query):
    """
    For a given file's trajectories, build a snapshot:
      list of records closest in time to t_query (within TIME_BIN/2).
    """
    half = TIME_BIN / 2.0
    snap = []
    for oid, arr in traj_dict.items():
        # arr is sorted by time, do a small linear scan (arrays are short)
        best = None
        best_dt = 1e18
        for r in arr:
            dt = abs(r["t"] - t_query)
            if dt < best_dt:
                best_dt = dt
                best = r
            else:
                # since sorted, once dt starts growing we can break
                if r["t"] > t_query:
                    break
        if best is not None and best_dt <= half:
            snap.append(best)
    return snap


def evaluate_pairs_for_file(fname, trajs_for_file, mode="A"):
    """
    For a single file:
      - Build time bins
      - For each bin, get snapshot
      - For each snapshot, compute pair distances
      - Keep best (min OBB distance) per pair over time
    Returns:
      pairs: list of dict(file, oid_i, oid_j, type_i, type_j, t_at, d_center_min, d_obb_min)
      meta:  dict(num_tracks, num_samples, type_hist, num_bins)
    """
    # meta info
    num_tracks = len(trajs_for_file)
    num_samples = sum(len(v) for v in trajs_for_file.values())
    type_hist = Counter(r["type"] for arr in trajs_for_file.values() for r in arr)

    time_bins = build_time_bins(trajs_for_file)
    print(f"  Tracks: {num_tracks}, samples: {num_samples}")
    print(f"  Type histogram: {dict(type_hist)}")
    print(f"  Time bins: {len(time_bins)}")

    best_for_pair = {}  # (oid_i, oid_j) -> (d_obb, d_center, t_at, rec_i, rec_j)

    for t_bin in time_bins:
        snapshot = snapshot_at_time(trajs_for_file, t_bin)
        n = len(snapshot)
        if n < 2:
            continue

        for i in range(n):
            a = snapshot[i]
            for j in range(i + 1, n):
                b = snapshot[j]

                if not include_pair(a, b, mode=mode):
                    continue

                d_center = pair_center_distance(a, b)
                if d_center > CENTER_THRESH:
                    continue

                d_obb = pair_obb_distance(a, b)
                if d_obb > OBB_THRESH:
                    continue

                key = tuple(sorted((a["oid"], b["oid"])))
                prev = best_for_pair.get(key)
                if (prev is None) or (d_obb < prev[0]):
                    best_for_pair[key] = (d_obb, d_center, t_bin, a, b)

    # convert dict → list
    pairs = []
    for (oid_i, oid_j), (d_obb, d_center, t_at, a, b) in best_for_pair.items():
        pairs.append({
            "file": fname,
            "oid_i": oid_i,
            "oid_j": oid_j,
            "type_i": a["type"],
            "type_j": b["type"],
            "t_at": t_at,
            "d_center_min": d_center,
            "d_obb_min": d_obb,
        })

    # sort by OBB distance
    pairs.sort(key=lambda x: x["d_obb_min"])

    print(f"  → {len(pairs)} pairs within {CENTER_THRESH:.1f} m center & {OBB_THRESH:.2f} m OBB.")
    return pairs, {
        "num_tracks": num_tracks,
        "num_samples": num_samples,
        "type_hist": dict(type_hist),
        "num_bins": len(time_bins),
    }


def main():
    traj_by_file = load_trajectories(TRAJ_CSV)

    all_pairs = []
    summary_rows = []

    for fname, trajs_for_file in traj_by_file.items():
        print(f"\n▶ Processing file: {os.path.basename(fname)}")
        pairs, meta = evaluate_pairs_for_file(fname, trajs_for_file, mode=MODE)
        all_pairs.extend(pairs)

        summary_rows.append({
            "file": fname,
            "num_tracks": meta["num_tracks"],
            "num_samples": meta["num_samples"],
            "num_bins": meta["num_bins"],
            "num_pairs": len(pairs),
            "center_thresh": CENTER_THRESH,
            "obb_thresh": OBB_THRESH,
            "mode": MODE,
        })

    # write pairs CSV
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(PAIRS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "file",
            "oid_i", "oid_j",
            "type_i", "type_j",
            "t_at",
            "d_center_min",
            "d_obb_min",
        ])
        for p in all_pairs:
            w.writerow([
                p["file"],
                p["oid_i"], p["oid_j"],
                p["type_i"], p["type_j"],
                f"{p['t_at']:.6f}",
                f"{p['d_center_min']:.6f}",
                f"{p['d_obb_min']:.6f}",
            ])
    print(f"✅ Pairs written → {PAIRS_CSV}")

    # write summary CSV
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "file",
            "num_tracks",
            "num_samples",
            "num_bins",
            "num_pairs",
            "center_thresh",
            "obb_thresh",
            "mode",
        ])
        for row in summary_rows:
            w.writerow([
                row["file"],
                row["num_tracks"],
                row["num_samples"],
                row["num_bins"],
                row["num_pairs"],
                row["center_thresh"],
                row["obb_thresh"],
                row["mode"],
            ])
    print(f"✅ Summary written → {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
