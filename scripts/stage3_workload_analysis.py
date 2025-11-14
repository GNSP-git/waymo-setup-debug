#!/usr/bin/env python3
"""
stage3_workload_analysis.py

Stage 3: Fine-time workload analysis for OBB distance checks, based on:
  - trajectories_all.csv  (from Stage 1)
  - workload_pairs_min.csv (from Stage 2, coarse close-approach pairs)

Features:
  1. Histogram of:
       - per-track sampling intervals (Δt)
       - per-track durations
     → saved to data/samples/track_stats.png

  2. Heuristic EGO detection (Option 2):
       For each file, choose the Vehicle track with the longest duration.

  3. For each dt in [0.001, 0.01, 0.1, 0.2], and for each pair from
     workload_pairs_min.csv:
       - Sweep time only over the overlap of the two tracks.
       - At each fine step, evaluate OBB distance (if obb_distance is
         available; else center distance).
       - Count:
           checks_all, checks_ego,
           near_all, near_ego
         and accumulate overlap durations for all pairs and EGO pairs.

     Results saved to:
       data/samples/fine_checks_per_dt.csv
"""

import os
import csv
import math
import bisect
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt

# ---------- PATHS ----------
BASE_DIR = "/home/gns/waymo_work"
SAVE_DIR = os.path.join(BASE_DIR, "data", "samples")
os.makedirs(SAVE_DIR, exist_ok=True)

TRAJ_CSV = os.path.join(SAVE_DIR, "trajectories_all.csv")
PAIRS_CSV = os.path.join(SAVE_DIR, "workload_pairs_min.csv")

# ---------- CONFIG ----------
DT_LIST = [0.001, 0.01, 0.1, 0.2]   # fine time steps (s)
NEAR_OBB_THRESH = 2.5              # meters, for "near" checks
CENTER_HORIZON = 10.0              # meters, coarse filter horizon (for interpretability only)

# ---------- Try OBB extension ----------
USE_OBB = True
try:
    import obb_distance
except Exception:
    USE_OBB = False
    print("ℹ️ obb_distance not found; using center distance instead.")


# ---------- HELPERS ----------

def pair_obb_distance(a, b):
    """
    a, b: dicts with keys: cx, cy, length, width, heading
    """
    if USE_OBB:
        return obb_distance.obb_min_distance(
            a["cx"], a["cy"], a["length"], a["width"], a["heading"],
            b["cx"], b["cy"], b["length"], b["width"], b["heading"]
        )
    dx = a["cx"] - b["cx"]
    dy = a["cy"] - b["cy"]
    return math.hypot(dx, dy)


def load_trajectories(path):
    """
    Load trajectories_all.csv into a nested dict:

      trajs[file][oid] = [
          {
              "t": ...,
              "cx": ...,
              "cy": ...,
              "cz": ...,
              "heading": ...,
              "length": ...,
              "width": ...,
              "height": ...,
              "speed": ...,
              "is_stationary": ...,
              "type": "Vehicle" / "Pedestrian" / ...
          }, ...
      ]

    Also returns:
      type_index[file][oid] = type_string
    """
    print(f"▶ Loading trajectories from {path} ...")
    trajs = {}
    type_index = defaultdict(dict)

    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {h: i for i, h in enumerate(header)}

        for row in reader:
            fname = row[idx["file"]]
            oid = row[idx["oid"]]
            typ = row[idx["type"]]

            rec = {
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
                "type": typ,
            }

            trajs.setdefault(fname, {}).setdefault(oid, []).append(rec)
            type_index[fname][oid] = typ

    # sort by time
    file_counts = {}
    for fname, objs in trajs.items():
        total = 0
        for oid, recs in objs.items():
            recs.sort(key=lambda r: r["t"])
            total += len(recs)
        file_counts[fname] = (len(objs), total)

    print(f"  Files in CSV: {len(trajs)}")
    for fname, (n_tracks, n_samples) in file_counts.items():
        print(f"  - {os.path.basename(fname)}: {n_tracks} tracks, {n_samples} samples")

    return trajs, type_index


def compute_track_stats(trajs):
    """
    Compute:
      - list of all Δt samples (time between successive samples in each track)
      - list of all track durations
    """
    dt_list = []
    dur_list = []

    for file_trajs in trajs.values():
        for track in file_trajs.values():
            if len(track) < 2:
                continue
            times = [r["t"] for r in track]
            # per-track Δt
            for i in range(1, len(times)):
                dt = times[i] - times[i - 1]
                if dt > 0:  # ignore zero / degenerate
                    dt_list.append(dt)
            # duration
            dur = times[-1] - times[0]
            if dur > 0:
                dur_list.append(dur)

    return dt_list, dur_list


def plot_track_stats(dt_list, dur_list, save_path):
    """
    Make a figure with 2 subplots:
      - histogram of sampling intervals Δt
      - histogram of track durations
    """
    if not dt_list or not dur_list:
        print("⚠️ Not enough data to plot track stats.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Δt histogram
    ax = axes[0]
    ax.hist(dt_list, bins=40, color="tab:blue", alpha=0.7)
    ax.set_xlabel("Δt between samples (s)")
    ax.set_ylabel("Count")
    ax.set_title("Per-track sampling intervals")
    ax.grid(True, linestyle="--", alpha=0.4)

    # duration histogram
    ax = axes[1]
    ax.hist(dur_list, bins=40, color="tab:green", alpha=0.7)
    ax.set_xlabel("Track duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Track durations")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✅ Track statistics figure saved → {save_path}")


def detect_ego_ids(trajs, type_index):
    """
    Heuristic EGO detection (Option 2):

      For each file, among all 'Vehicle' tracks, pick the one with
      the longest duration.

    Returns:
      ego_per_file: dict[file] = ego_oid
    """
    ego_per_file = {}
    for fname, file_trajs in trajs.items():
        best_oid = None
        best_dur = -1.0
        for oid, track in file_trajs.items():
            typ = type_index.get(fname, {}).get(oid, "Unknown")
            if typ != "Vehicle":
                continue
            if len(track) < 2:
                continue
            t0 = track[0]["t"]
            t1 = track[-1]["t"]
            dur = t1 - t0
            if dur > best_dur:
                best_dur = dur
                best_oid = oid
        if best_oid is not None:
            ego_per_file[fname] = best_oid
            print(f"  EGO for {os.path.basename(fname)}: {best_oid} (duration {best_dur:.2f} s)")
        else:
            print(f"  EGO for {os.path.basename(fname)}: NONE (no Vehicle tracks)")
    return ego_per_file


def load_pairs(path):
    """
    Load workload_pairs_min.csv.

    Expected columns include at least:
      file, oid_i, oid_j, t_at, type_i, type_j, d_obb_min, d_center_min?

    We don't need all fields, but we keep d_obb_min for sanity if present.
    """
    print(f"▶ Loading coarse pairs from {path} ...")
    pairs = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {h: i for i, h in enumerate(header)}

        def get_opt(row, key, default=None):
            i = idx.get(key)
            return row[i] if i is not None else default

        for row in reader:
            fname = row[idx["file"]]
            oid_i = row[idx["oid_i"]]
            oid_j = row[idx["oid_j"]]
            t_at = float(row[idx["t_at"]])
            type_i = row[idx["type_i"]]
            type_j = row[idx["type_j"]]

            d_obb_min = None
            if "d_obb_min" in idx:
                d_obb_min = float(row[idx["d_obb_min"]])

            d_center_min = None
            if "d_center_min" in idx:
                d_center_min = float(row[idx["d_center_min"]])

            pairs.append({
                "file": fname,
                "oid_i": oid_i,
                "oid_j": oid_j,
                "t_at": t_at,
                "type_i": type_i,
                "type_j": type_j,
                "d_obb_min": d_obb_min,
                "d_center_min": d_center_min,
            })

    print(f"  Loaded {len(pairs)} coarse pairs from Stage 2.")
    return pairs


def build_time_index(track):
    """
    For one track = list of records sorted by t,
    return separate list of times for fast bisect.
    """
    return [r["t"] for r in track]


def get_state_at(track, times, t):
    """
    Approximate state at time t by nearest recorded sample.
    We do NOT interpolate; for workload counting this is sufficient.

    track: list of records
    times: sorted list of same length
    """
    if not track:
        return None
    # clamp
    if t <= times[0]:
        return track[0]
    if t >= times[-1]:
        return track[-1]

    idx = bisect.bisect_left(times, t)
    if idx == 0:
        return track[0]
    if idx >= len(times):
        return track[-1]

    before = track[idx - 1]
    after = track[idx]
    if abs(before["t"] - t) <= abs(after["t"] - t):
        return before
    else:
        return after


def analyze_workload(trajs, pairs, ego_per_file, dt_list, near_obb_thresh, output_csv):
    """
    Main workload analysis:

      For each dt in dt_list:
        For each pair in pairs:
          - get two tracks
          - determine overlap [t0, t1]
          - sweep t from t0 to t1 in steps of dt
          - count checks_all, checks_ego, near_all, near_ego
          - accumulate total overlap durations (all, ego)

      Write per-dt summary to output_csv.
    """
    print("\n=== STAGE 3: Fine-time workload analysis (Method B) ===")
    # Prebuild "time index" for each track to speed queries
    time_index = {}
    for fname, file_trajs in trajs.items():
        time_index[fname] = {}
        for oid, track in file_trajs.items():
            time_index[fname][oid] = build_time_index(track)

    # Precompute durations for reporting
    total_track_duration = 0.0
    for fname, file_trajs in trajs.items():
        for track in file_trajs.values():
            if len(track) >= 2:
                total_track_duration += track[-1]["t"] - track[0]["t"]
    print(f"  Total summed track duration (all tracks, all files): {total_track_duration:.2f} s\n")

    # For collecting results per dt
    rows = []

    for dt in dt_list:
        print(f"--- dt = {dt} s ---")
        checks_all = 0
        checks_ego = 0
        near_all = 0
        near_ego = 0
        duration_all = 0.0
        duration_ego = 0.0

        for p in pairs:
            fname = p["file"]
            oid_i = p["oid_i"]
            oid_j = p["oid_j"]

            file_trajs = trajs.get(fname)
            if file_trajs is None:
                continue
            track_i = file_trajs.get(oid_i)
            track_j = file_trajs.get(oid_j)
            if not track_i or not track_j:
                continue
            # time overlap
            t0 = max(track_i[0]["t"], track_j[0]["t"])
            t1 = min(track_i[-1]["t"], track_j[-1]["t"])
            if t1 <= t0:
                continue

            duration = t1 - t0
            duration_all += duration

            ego_id = ego_per_file.get(fname)
            is_ego_pair = (ego_id is not None) and (ego_id in (oid_i, oid_j))
            if is_ego_pair:
                duration_ego += duration

            times_i = time_index[fname][oid_i]
            times_j = time_index[fname][oid_j]

            # number of fine steps
            n_steps = int(math.floor(duration / dt)) + 1
            # simple guard
            if n_steps <= 0:
                continue

            base_t = t0
            for k in range(n_steps):
                t = base_t + k * dt
                if t > t1:
                    break

                rec_i = get_state_at(track_i, times_i, t)
                rec_j = get_state_at(track_j, times_j, t)
                if rec_i is None or rec_j is None:
                    continue

                d = pair_obb_distance(
                    {
                        "cx": rec_i["cx"],
                        "cy": rec_i["cy"],
                        "length": rec_i["length"],
                        "width": rec_i["width"],
                        "heading": rec_i["heading"],
                    },
                    {
                        "cx": rec_j["cx"],
                        "cy": rec_j["cy"],
                        "length": rec_j["length"],
                        "width": rec_j["width"],
                        "heading": rec_j["heading"],
                    },
                )

                checks_all += 1
                if is_ego_pair:
                    checks_ego += 1

                if d <= near_obb_thresh:
                    near_all += 1
                    if is_ego_pair:
                        near_ego += 1

        # compute rates
        rate_all = checks_all / duration_all if duration_all > 0 else 0.0
        rate_ego = checks_ego / duration_ego if duration_ego > 0 else 0.0
        near_rate_all = near_all / duration_all if duration_all > 0 else 0.0
        near_rate_ego = near_ego / duration_ego if duration_ego > 0 else 0.0

        print(f"  Total fine-time OBB checks (all pairs): {checks_all}")
        print(f"  Total fine-time OBB checks (EGO only):  {checks_ego}")
        print(f"  Near checks (≤ {near_obb_thresh:.2f} m) – all pairs: {near_all}")
        print(f"  Near checks (≤ {near_obb_thresh:.2f} m) – EGO only:  {near_ego}")
        print(f"  Overlap time all pairs: {duration_all:.3f} s")
        print(f"  Overlap time EGO pairs: {duration_ego:.3f} s")
        print(f"  Check rate all pairs: {rate_all:.1f} checks/s")
        print(f"  Check rate EGO pairs: {rate_ego:.1f} checks/s")
        print(f"  Near-check rate all pairs: {near_rate_all:.1f} checks/s")
        print(f"  Near-check rate EGO pairs: {near_rate_ego:.1f} checks/s\n")

        rows.append({
            "dt_s": dt,
            "checks_all": checks_all,
            "checks_ego": checks_ego,
            "near_all": near_all,
            "near_ego": near_ego,
            "duration_all_s": duration_all,
            "duration_ego_s": duration_ego,
            "checks_per_sec_all": rate_all,
            "checks_per_sec_ego": rate_ego,
            "near_checks_per_sec_all": near_rate_all,
            "near_checks_per_sec_ego": near_rate_ego,
        })

    # write CSV
    fieldnames = [
        "dt_s",
        "checks_all",
        "checks_ego",
        "near_all",
        "near_ego",
        "duration_all_s",
        "duration_ego_s",
        "checks_per_sec_all",
        "checks_per_sec_ego",
        "near_checks_per_sec_all",
        "near_checks_per_sec_ego",
    ]
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"✅ Fine-time workload summary written → {output_csv}")


# ---------- MAIN ----------

def main():
    # 1. Load trajectories
    trajs, type_index = load_trajectories(TRAJ_CSV)

    # 2. Track statistics (Δt and duration histograms)
    dt_list, dur_list = compute_track_stats(trajs)
    stats_png = os.path.join(SAVE_DIR, "track_stats.png")
    plot_track_stats(dt_list, dur_list, stats_png)

    # 3. Detect EGO per file using heuristic 2
    print("\n▶ Detecting EGO tracks (longest Vehicle per file) ...")
    ego_per_file = detect_ego_ids(trajs, type_index)

    # 4. Load coarse close pairs (Stage 2 output)
    pairs = load_pairs(PAIRS_CSV)

    # 5. Fine-time workload analysis (Method B)
    output_csv = os.path.join(SAVE_DIR, "fine_checks_per_dt.csv")
    analyze_workload(trajs, pairs, ego_per_file, DT_LIST, NEAR_OBB_THRESH, output_csv)


if __name__ == "__main__":
    main()
