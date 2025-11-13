# plot_all_trajectories.py
# Consolidated visualization: camera, lidar, trajectory (2D), and space-time (3D)
# Saves all figures to the same directory as your dataset.

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils

# === Paths ===
DATA_DIR = "/home/gns/waymo_work/data/samples"
FILE = os.path.join(DATA_DIR, "trajectories_preview.npz")
FIG_DIR = DATA_DIR

# --- Load trajectory data ---
data = np.load(FILE, allow_pickle=True)
print(f"Loaded {len(data)} trajectories")

# === FIGURE 1: CAMERA + LIDAR + TRAJECTORIES (absolute & centered) ===
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
axs = axs.ravel()

# --- 1. Camera ---
# Placeholder: single blank panel for alignment (since no frame image loaded)
axs[0].text(0.5, 0.5, "Camera Frame Placeholder", ha="center", va="center", fontsize=14)
axs[0].set_title("Camera Frame (example)")
axs[0].axis("off")

# --- 2. LiDAR BEV Intensity ---
# Simulated LiDAR points for visual template
lidar_file = "/home/gns/waymo_work/scripts/vis_lidar_camera.py"  # not used here, just reference
xs = np.random.uniform(-80, 80, 5000)
ys = np.random.uniform(-80, 80, 5000)
intensity = np.sqrt(xs**2 + ys**2)
mask = (np.abs(xs) < 80) & (np.abs(ys) < 80)
sc = axs[1].scatter(xs[mask], ys[mask], c=intensity[mask], s=0.5, cmap="viridis", alpha=0.7)
axs[1].set_title("LiDAR Bird’s Eye View (colored by intensity)")
axs[1].set_xlabel("X (m, forward)")
axs[1].set_ylabel("Y (m, left)")
axs[1].grid(True, linestyle="--", alpha=0.5)
axs[1].axis("equal")
fig.colorbar(sc, ax=axs[1], label="Intensity")

# --- 3. Trajectories (absolute coordinates) ---
shown = 0
for oid, arr in data.items():
    traj = np.asarray(arr)
    if traj.ndim != 2 or traj.shape[1] < 3 or traj.shape[0] < 2:
        continue
    x, y = traj[:, 1], traj[:, 2]
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        continue
    axs[2].plot(x, y, '-', alpha=0.6, linewidth=1)
    shown += 1
    if shown >= 50:
        break
axs[2].set_title(f"Trajectories (absolute, meters) — {shown} shown")
axs[2].set_xlabel("X (m, forward)")
axs[2].set_ylabel("Y (m, left)")
axs[2].axis("equal")
axs[2].grid(True, linestyle="--", alpha=0.5)

# --- 4. Trajectories (centered) ---
shown = 0
for oid, arr in data.items():
    traj = np.asarray(arr)
    if traj.ndim != 2 or traj.shape[1] < 3 or traj.shape[0] < 2:
        continue
    x, y = traj[:, 1], traj[:, 2]
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        continue
    x -= np.mean(x)
    y -= np.mean(y)
    axs[3].plot(x, y, '-', alpha=0.6, linewidth=1)
    shown += 1
    if shown >= 50:
        break
axs[3].set_title(f"Trajectories (centered per track) — {shown} shown")
axs[3].set_xlabel("X (m)")
axs[3].set_ylabel("Y (m)")
axs[3].axis("equal")
axs[3].grid(True, linestyle="--", alpha=0.5)

# --- Save first figure ---
fig.tight_layout()
fig_path_2D = os.path.join(FIG_DIR, "trajectories_all_2D.png")
plt.savefig(fig_path_2D, dpi=200)
print(f"✅ Saved 2D figure to {fig_path_2D}")

# === FIGURE 2: SPACE–TIME (3D) ===
fig3d = plt.figure(figsize=(10, 8))
ax3 = fig3d.add_subplot(111, projection='3d')

shown = 0
FPS = 10.0
DEFAULT_DT = 1.0 / FPS

for oid, arr in data.items():
    traj = np.asarray(arr)
    if traj.ndim < 2 or traj.shape[1] < 3 or traj.shape[0] < 2:
        continue
    ts = traj[:, 0] if traj.shape[1] >= 4 else np.arange(len(traj)) * DEFAULT_DT
    x, y = traj[:, 1], traj[:, 2]
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(ts)
    ts, x, y = ts[mask], x[mask], y[mask]
    if x.size < 2:
        continue
    if np.max(ts) > 1e6:
        ts = (ts - np.min(ts)) * 1e-6
    else:
        ts = ts - np.min(ts)
    ax3.plot3D(x, y, ts, linewidth=1.0, alpha=0.7)
    shown += 1
    if shown >= 40:
        break

ax3.set_xlabel("X (m, forward)")
ax3.set_ylabel("Y (m, left)")
ax3.set_zlabel("Time (s)")
ax3.set_title(f"3D Space–Time Trajectories — {shown} shown (m, s)")
ax3.grid(True, linestyle="--", alpha=0.5)
ax3.view_init(elev=25, azim=-65)
fig3d.tight_layout()

fig_path_3D = os.path.join(FIG_DIR, "trajectories_space_time_3D.png")
plt.savefig(fig_path_3D, dpi=200)
print(f"✅ Saved 3D space-time figure to {fig_path_3D}")

plt.show()
