# plot_traj_space_time.py
# 3D visualization of trajectories (X-Y vs time) for Waymo-like trajectory data.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)

# === Load data ===
data = np.load("../data/samples/trajectories_preview.npz", allow_pickle=True)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
shown = 0
auto_scale = True

# --- Helper: frame time spacing ---
FPS = 10.0  # default for Waymo
DEFAULT_DT = 1.0 / FPS

for oid, arr in data.items():
    traj = np.asarray(arr)
    if traj.ndim != 2 or traj.shape[1] < 3 or traj.shape[0] < 2:
        continue

    # Expect [timestamp, x, y, (z?)]
    if traj.shape[1] >= 4:
        ts, x, y = traj[:, 0], traj[:, 1], traj[:, 2]
    else:
        ts = np.arange(len(traj)) * DEFAULT_DT
        x, y = traj[:, 1], traj[:, 2]

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(ts)
    ts, x, y = ts[mask], x[mask], y[mask]
    if x.size < 2:
        continue

    # Convert timestamps to seconds relative to start
    if np.max(ts) > 1e6:  # microseconds
        ts = (ts - np.min(ts)) * 1e-6
    else:
        ts = ts - np.min(ts)

    if auto_scale:
        # Waymo coordinates are already in meters
        unit = "m"
    else:
        unit = "raw"

    ax.plot3D(x, y, ts, linewidth=1.0, alpha=0.7)
    shown += 1
    if shown >= 40:
        break

ax.set_xlabel("X (m, forward)")
ax.set_ylabel("Y (m, left)")
ax.set_zlabel("Time (s)")
ax.set_title(f"3D Trajectories in Space–Time — {shown} tracks (units: meters, seconds)")
ax.grid(True, linestyle='--', alpha=0.5)
ax.view_init(elev=25, azim=-65)
plt.tight_layout()
plt.show()
