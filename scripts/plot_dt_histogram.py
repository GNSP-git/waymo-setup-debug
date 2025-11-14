#!/usr/bin/env python
# plot_dt_histogram.py
#
# Visualize:
#   (1) Sampling rate histogram (Δt for each track)
#   (2) Track duration histogram
#
# Reads:
#   /home/gns/waymo_work/data/samples/trajectories_all.csv
#
# Outputs:
#   dt_and_duration_hist.png in data/samples/

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

BASE_DIR  = "/home/gns/waymo_work"
SAVE_DIR  = os.path.join(BASE_DIR, "data", "samples")
TRAJ_CSV  = os.path.join(SAVE_DIR, "trajectories_all.csv")


def load_trajectories(path):
    print(f"Loading: {path}")
    trajs = defaultdict(lambda: defaultdict(list))

    with open(path, "r") as f:
        r = csv.reader(f)
        hdr = next(r)
        idx = {h:i for i,h in enumerate(hdr)}

        for row in r:
            file_id = row[idx["file"]]
            oid     = row[idx["oid"]]
            t       = float(row[idx["t"]])
            trajs[file_id][oid].append(t)

    # sort timestamps
    for f in trajs:
        for oid in trajs[f]:
            trajs[f][oid].sort()

    return trajs


def compute_dt_and_duration(trajs):
    """
    Compute:
        global_dts: list of all Δt
        durations:  list of all (max t - min t)
    """
    global_dts = []
    durations  = []

    for file_id, tracks in trajs.items():
        for oid, ts in tracks.items():
            if len(ts) < 2:
                continue

            # Δt values for the track
            dts = np.diff(ts)
            global_dts.extend(dts)

            # track duration
            dur = ts[-1] - ts[0]
            durations.append(dur)

    return np.array(global_dts), np.array(durations)


def main():
    trajs = load_trajectories(TRAJ_CSV)
    dts, durations = compute_dt_and_duration(trajs)

    print("=== Δt STATISTICS ===")
    print(f"Count: {len(dts)}")
    print(f"Min Δt: {dts.min():.6f} s")
    print(f"Mean Δt:{dts.mean():.6f} s")
    print(f"Median Δt:{np.median(dts):.6f} s")
    print(f"Max Δt: {dts.max():.6f} s")

    print("\n=== TRACK DURATION STATISTICS ===")
    print(f"Tracks: {len(durations)}")
    print(f"Min duration: {durations.min():.3f} s")
    print(f"Mean duration:{durations.mean():.3f} s")
    print(f"Median duration:{np.median(durations):.3f} s")
    print(f"Max duration: {durations.max():.3f} s")

    # --- FIGURE: Δt histogram + duration histogram ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # 1) Δt histogram
    bins_dt = np.linspace(0, 0.5, 100)  # Waymo Lidar sampling ~0.1s
    axes[0].hist(dts, bins=bins_dt, color="steelblue", alpha=0.85)
    axes[0].set_xlabel("Δt (s)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Sampling Interval Histogram (Δt)")
    axes[0].grid(True)

    # 2) Track duration histogram
    bins_dur = np.linspace(0, durations.max(), 100)
    axes[1].hist(durations, bins=bins_dur, color="darkorange", alpha=0.85)
    axes[1].set_xlabel("Track Duration (s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Histogram of Track Durations")
    axes[1].grid(True)

    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "dt_and_duration_hist.png")
    plt.savefig(out, dpi=150)
    print(f"✔ Saved combined histogram → {out}")

    plt.show()


if __name__ == "__main__":
    main()
