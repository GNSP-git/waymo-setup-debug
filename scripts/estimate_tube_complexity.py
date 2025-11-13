import os
import numpy as np
import math
from collections import Counter, defaultdict

# Adjust if your file lives elsewhere
DATA_DIR = "/home/gns/waymo_work/data/samples"
NPZ_NAME = "trajectories_preview.npz"

# Error / complexity thresholds (tune these!)
MAX_POS_ERR = 0.5      # [m] max allowed deviation from segment model
MAX_HEADING_JUMP = 10  # [deg] max heading change inside one tube
MIN_SEG_DURATION = 0.5 # [s] don't create microscopic segments

def compute_heading_dxdy(x0, y0, x1, y1):
    return math.degrees(math.atan2(y1 - y0, x1 - x0)) if (x1 != x0 or y1 != y0) else 0.0

def segment_error(times, xs, ys, use_accel=False):
    """
    Fit a simple motion model on [0..T] and compute max Euclidean error.
    times: [N], xs: [N], ys: [N], times in seconds (monotone)
    """
    if len(times) <= 2:
        return 0.0

    t0 = times[0]
    dt = times - t0
    x0, y0 = xs[0], ys[0]

    if not use_accel or len(times) < 4:
        # Constant velocity model: linear least squares on x(t), y(t)
        A = np.vstack([dt, np.ones_like(dt)]).T
        vx, _ = np.linalg.lstsq(A, xs - x0, rcond=None)[0]
        vy, _ = np.linalg.lstsq(A, ys - y0, rcond=None)[0]
        x_fit = x0 + vx * dt
        y_fit = y0 + vy * dt
    else:
        # Optional: constant acceleration model (rarely needed)
        A = np.vstack([0.5 * dt**2, dt, np.ones_like(dt)]).T
        ax, vx, _ = np.linalg.lstsq(A, xs - x0, rcond=None)[0]
        ay, vy, _ = np.linalg.lstsq(A, ys - y0, rcond=None)[0]
        x_fit = x0 + vx * dt + 0.5 * ax * dt**2
        y_fit = y0 + vy * dt + 0.5 * ay * dt**2

    err = np.sqrt((xs - x_fit)**2 + (ys - y_fit)**2)
    return float(err.max())

def estimate_tubes_for_traj(traj, max_horizon_s=10.0):
    """
    traj: np.array [N,>=3] with (t, x, y, ...), t in us or s.
    Returns: number of segments (tubes) needed.
    """
    if traj.ndim != 2 or traj.shape[1] < 3 or traj.shape[0] < 2:
        return 0

    # Extract and normalize time
    t = traj[:, 0].astype(float)
    # Heuristic: if looks like microseconds, convert to seconds
    if t.max() > 1e6:
        t = (t - t[0]) / 1e6
    else:
        t = t - t[0]

    x = traj[:, 1].astype(float)
    y = traj[:, 2].astype(float)

    # Restrict to first max_horizon_s seconds
    mask = t <= max_horizon_s
    if not np.any(mask):
        return 0

    t = t[mask]
    x = x[mask]
    y = y[mask]

    if t.size < 3:
        return 1

    # Greedy splitting
    start = 0
    segments = 0
    N = t.size

    while start < N - 1:
        segments += 1
        # at least two points
        end = start + 2

        while end < N:
            dt = t[end] - t[start]
            if dt < MIN_SEG_DURATION:
                end += 1
                continue

            # Compute error if we extend to 'end'
            err = segment_error(t[start:end+1], x[start:end+1], y[start:end+1], use_accel=False)
            if err > MAX_POS_ERR:
                break  # too much deviation; finalize previous segment

            # Heading smoothness check
            h_start = compute_heading_dxdy(x[start], y[start], x[start+1], y[start+1])
            h_end = compute_heading_dxdy(x[end-1], y[end-1], x[end], y[end])
            if abs(h_end - h_start) > MAX_HEADING_JUMP:
                break

            end += 1

        # Next segment starts at last accepted index
        start = max(start + 1, end - 1)

    return segments

def main():
    path = os.path.join(DATA_DIR, NPZ_NAME)
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    data = np.load(path, allow_pickle=True)
    tube_counts = []
    type_counts = Counter()

    print(f"Loaded {len(data.files)} trajectories from {path}")

    for oid in data.files:
        traj = data[oid]
        try:
            traj = np.asarray(traj)
        except Exception:
            continue

        k = estimate_tubes_for_traj(traj, max_horizon_s=10.0)
        if k > 0:
            tube_counts.append(k)

    if not tube_counts:
        print("❌ No valid trajectories found for tube estimation.")
        return

    tube_counts = np.array(tube_counts)
    print("\n=== Tube complexity over 10 s horizon ===")
    print(f"Tracks analyzed: {tube_counts.size}")
    for thr in [1,2,3,4,5,6,8,10]:
        frac = np.mean(tube_counts <= thr) * 100
        print(f"  ≤ {thr} tubes: {frac:5.1f}%")

    print("\nQuantiles (tubes per track):")
    for q in [50, 75, 90, 95, 99]:
        val = int(np.quantile(tube_counts, q/100.0))
        print(f"  {q:2d}th percentile: {val}")

    print(f"\nMax tubes for any track: {tube_counts.max()}")

if __name__ == "__main__":
    main()
