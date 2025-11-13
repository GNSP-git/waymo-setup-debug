#!/usr/bin/env python3
import os
import glob
import csv
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ================== CONFIG ==================

# Directory where your trajectory .npz files live
TRAJ_DIR = os.path.join(os.path.expanduser("~"), "waymo_work", "data", "samples")

# Output CSV with all close approaches
OUT_CSV = os.path.join(TRAJ_DIR, "closeness_summary.csv")

# Only consider object pairs whose minimum distance is below this (meters)
MAX_REPORT_DIST_M = 10.0

# "Near-synchronous" time tolerance when comparing samples (seconds)
T_SYNC_TOL = 0.20

# Minimum overlapping points considered to trust the min distance
MIN_OVERLAP_SAMPLES = 3

# How many top pairs (per file) to plot in 3D space-time
TOPK_PLOT = 5

# =============================================

def load_trajectories_from_npz(path: str):
    """
    Load trajectories when each object ID is a separate key in the .npz file.
    Returns a dict: {object_id: np.ndarray of shape (N,3) [t, x, y]}.
    """
    data = np.load(path, allow_pickle=True)
    trajs = {}

    for k in data.files:
        arr = np.asarray(data[k])
        if arr.ndim == 2 and arr.shape[1] >= 3:
            trajs[k] = arr[:, :3]
        elif arr.ndim == 1 and arr.size % 3 == 0:
            trajs[k] = arr.reshape(-1, 3)
        else:
            print(f"  ⚠️ Skipping {k}, unexpected shape {arr.shape}")

    return trajs

'''
def load_trajectories_from_npz(path: str) -> Dict[str, np.ndarray]:
    """
    Load trajectories from a .npz saved by our previous scripts.

    Expected format:
      npz["trajectories"] is either:
        - a dict-like (np.object_) mapping id -> list[(t, x, y)], or
        - an array of (id, list) pairs.

    Returns:
      dict: {object_id: np.ndarray of shape (N, 3) with columns [t, x, y]}
    """
    data = np.load(path, allow_pickle=True)
    if "trajectories" not in data:
        raise ValueError(f"{path} missing 'trajectories' key")

    raw = data["trajectories"].item() if hasattr(data["trajectories"], "item") else data["trajectories"]

    trajs = {}
    if isinstance(raw, dict):
        iterator = raw.items()
    else:
        # Fallback: assume iterable of (id, arr/list)
        iterator = raw

    for oid, arr in iterator:
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] < 3:
            continue
        # Ensure sorted by time
        arr = arr[np.argsort(arr[:, 0])]
        trajs[str(oid)] = arr[:, :3]  # keep t, x, y

    return trajs
'''

def min_distance_between_trajs(
    a: np.ndarray,
    b: np.ndarray,
    t_sync_tol: float
) -> Tuple[float, float, int]:
    """
    Compute minimum spatial distance between two trajectories a,b
    using a linear-time merge over time with tolerance.

    a, b: (N,3) arrays [t, x, y], times in seconds, sorted by t.
    t_sync_tol: only compare points with |t_a - t_b| <= t_sync_tol.

    Returns:
      (min_dist, t_at_min, overlap_count)
      If no overlapping times, returns (inf, 0.0, 0)
    """
    i = j = 0
    na, nb = a.shape[0], b.shape[0]
    best_d2 = np.inf
    best_t = 0.0
    overlap = 0

    while i < na and j < nb:
        ta, xa, ya = a[i]
        tb, xb, yb = b[j]

        dt = ta - tb
        if abs(dt) <= t_sync_tol:
            dx = xa - xb
            dy = ya - yb
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_t = 0.5 * (ta + tb)
            overlap += 1

            # Advance the earlier one to explore neighbors
            if ta <= tb:
                i += 1
            else:
                j += 1

        elif ta < tb:
            i += 1
        else:
            j += 1

    if overlap == 0 or not np.isfinite(best_d2):
        return float("inf"), 0.0, 0

    return float(np.sqrt(best_d2)), float(best_t), overlap


def analyze_file(npz_path: str):
    """
    Analyze one trajectory file.
    Returns:
      list of dict rows for CSV.
      list of (row_dict) for topK plotting.
    """
    basename = os.path.basename(npz_path)
    print(f"\n▶ Analyzing {basename} ...")

    try:
        trajs = load_trajectories_from_npz(npz_path)
    except Exception as e:
        print(f"  ❌ Failed to load {basename}: {e}")
        return [], []

    ids = sorted(trajs.keys())
    n = len(ids)
    if n < 2:
        print(f"  ⚠️ Only {n} tracks, skipping.")
        return [], []

    rows = []
    # Compute pairwise min distance
    for i in range(n):
        for j in range(i + 1, n):
            oid1, oid2 = ids[i], ids[j]
            a, b = trajs[oid1], trajs[oid2]
            dmin, tmin, overlap = min_distance_between_trajs(a, b, T_SYNC_TOL)

            if overlap >= MIN_OVERLAP_SAMPLES and dmin <= MAX_REPORT_DIST_M:
                rows.append({
                    "file": basename,
                    "obj1": oid1,
                    "obj2": oid2,
                    "min_dist_m": dmin,
                    "t_at_min_s": tmin,
                    "overlap_samples": overlap
                })

    if not rows:
        print("  ℹ️ No close approaches under threshold.")
        return [], []

    # Sort by distance ascending
    rows_sorted = sorted(rows, key=lambda r: r["min_dist_m"])
    print(f"  ✅ Found {len(rows_sorted)} close pairs (min_dist <= {MAX_REPORT_DIST_M} m).")
    print("  Top 5:")
    for r in rows_sorted[:5]:
        print(f"    {r['obj1']}–{r['obj2']}: {r['min_dist_m']:.2f} m at t={r['t_at_min_s']:.2f}s (n={r['overlap_samples']})")

    # Prepare top-K for plotting
    topk = rows_sorted[:TOPK_PLOT]
    return rows_sorted, topk, trajs


def plot_topk_space_time(npz_basename: str, topk_rows, trajs: Dict[str, np.ndarray]):
    """
    For the given file, create a 3D plot of top-K closest pairs in (x,y,t).
    Saves as PNG; no interactive requirement.
    """
    if not topk_rows:
        return

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for r in topk_rows:
        for key, color in [(r["obj1"], "tab:blue"), (r["obj2"], "tab:red")]:
            traj = trajs[key]
            t = traj[:, 0]
            x = traj[:, 1]
            y = traj[:, 2]
            ax.plot(x, y, t, alpha=0.7, label=f"{key}" if key == r["obj1"] else None, color=color)

        # Mark the closest-approach point (approx at t_at_min_s)
        t_star = r["t_at_min_s"]
        for key, color in [(r["obj1"], "tab:blue"), (r["obj2"], "tab:red")]:
            traj = trajs[key]
            # Closest index in time
            idx = int(np.argmin(np.abs(traj[:, 0] - t_star)))
            ax.scatter(traj[idx, 1], traj[idx, 2], traj[idx, 0],
                       s=30, color=color, marker="o")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Time (s)")
    ax.set_title(f"Top {len(topk_rows)} Closest Pairs (Space–Time)\n{npz_basename}")
    ax.grid(True)

    out_path = os.path.join(TRAJ_DIR, f"{os.path.splitext(npz_basename)[0]}_space_time_topk.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  🖼 Saved 3D space-time figure: {out_path}")


def main():
    npz_files = sorted(glob.glob(os.path.join(TRAJ_DIR, "*.npz")))
    if not npz_files:
        print(f"No .npz files found in {TRAJ_DIR}")
        return

    all_rows = []

    for path in npz_files:
        rows, topk, trajs = analyze_file(path)
        all_rows.extend(rows)
        if topk:
            plot_topk_space_time(os.path.basename(path), topk, trajs)

    if not all_rows:
        print("No close approaches found in any file.")
        return

    # Write CSV
    fieldnames = ["file", "obj1", "obj2", "min_dist_m", "t_at_min_s", "overlap_samples"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"\n✅ Wrote closeness summary CSV to: {OUT_CSV}")
    print("   Columns: file, obj1, obj2, min_dist_m, t_at_min_s, overlap_samples")


if __name__ == "__main__":
    main()
