# plot_all_trajectories_batch_final.py
# Robust batch visualization for Waymo TFRecords (TF 2.11 / WSL)
# - Handles convert_range_image_to_point_cloud API differences
# - Skips LiDAR failures gracefully
# - Saves all 2D/3D figures automatically

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils

# === CONFIG ===
DATA_DIR = "/mnt/n/waymo_comma/waymo"
SAVE_DIR = "/home/gns/waymo_work/data/samples"
os.makedirs(SAVE_DIR, exist_ok=True)
FPS = 10.0
DEFAULT_DT = 1.0 / FPS

# === Enumerate TFRecord files ===
files = sorted(glob.glob(os.path.join(DATA_DIR, "individual_files_*_with_camera_labels.tfrecord")))
print(f"Found {len(files)} TFRecord files to process.\n")

# ===============================================================
# 🩹 WAYMO SDK PATCH — Robust for 3- or 4-arg convert_range_image_to_point_cloud
# ===============================================================
from waymo_open_dataset.utils import frame_utils as fu
import types
import inspect

_orig_pc = fu.convert_range_image_to_point_cloud

def _safe_convert_pc(frame, range_images, camera_projections, *args, **kwargs):
    """Handle both 3- and 4-argument versions of convert_range_image_to_point_cloud."""
    try:
        sig = inspect.signature(_orig_pc)
        if len(sig.parameters) == 3:
            return _orig_pc(frame, range_images, camera_projections)
        else:
            # Newer API requires 4 args
            if args:
                return _orig_pc(frame, range_images, camera_projections, *args)
            else:
                return _orig_pc(frame, range_images, camera_projections, None)
    except TypeError as e:
        print(f"⚠️ Fallback: convert_range_image_to_point_cloud raised {e}")
        try:
            return _orig_pc(frame, range_images, camera_projections)
        except Exception as e2:
            print(f"⚠️ Secondary failure: {e2}")
            return [np.zeros((0, 3), np.float32)], []

fu.convert_range_image_to_point_cloud = types.FunctionType(
    _safe_convert_pc.__code__, globals()
)
print("✅ Waymo SDK patched: robust convert_range_image_to_point_cloud (handles 3/4 args)\n")

# ===============================================================
# 🚗 Visualization Function
# ===============================================================
def visualize_file(filepath):
    print(f"▶ Processing {os.path.basename(filepath)} ...")
    ds = tf.data.TFRecordDataset(filepath, compression_type="")

    for raw in ds.take(1):  # preview first frame
        frame = open_dataset.Frame()
        frame.ParseFromString(raw.numpy())

        # --- 1️⃣ Camera Image ---
        cam_img = tf.image.decode_jpeg(frame.images[0].image).numpy()

        # --- 2️⃣ LiDAR Conversion (gracefully handled) ---
        try:
            parsed = frame_utils.parse_range_image_and_camera_projection(frame)
            if len(parsed) >= 3:
                range_images, camera_projections, range_image_top_pose = parsed[:3]
            else:
                range_images, camera_projections = parsed[:2]
                range_image_top_pose = None

            points_list, _ = frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections, range_image_top_pose
            )
            pts = np.concatenate(points_list, axis=0)
            xs, ys, intensity = pts[:, 0], pts[:, 1], pts[:, -1]
            mask = (np.abs(xs) < 80) & (np.abs(ys) < 80)
            xs, ys, intensity = xs[mask], ys[mask], intensity[mask]
            lidar_ok = True
        except Exception as e:
            print(f"⚠️ LiDAR step failed for {os.path.basename(filepath)}: {e}")
            lidar_ok = False
            xs, ys, intensity = np.array([]), np.array([]), np.array([])

        # --- 3️⃣ Load Trajectories (optional) ---
        traj_file = os.path.join(SAVE_DIR, "trajectories_preview.npz")
        if os.path.exists(traj_file):
            traj_data = np.load(traj_file, allow_pickle=True)
        else:
            traj_data = {}

        # --- Create Figure 1: Camera + LiDAR + Trajectories ---
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        axs = axs.ravel()

        # Camera
        axs[0].imshow(cam_img)
        axs[0].set_title("Front Camera")
        axs[0].axis("off")

        # LiDAR
        if lidar_ok and xs.size > 0:
            sc = axs[1].scatter(xs, ys, c=intensity, s=0.6, cmap='viridis', alpha=0.7)
            fig.colorbar(sc, ax=axs[1], label="Intensity")
        axs[1].set_title("LiDAR Bird’s Eye View (colored by intensity)")
        axs[1].set_xlabel("X (m, forward)")
        axs[1].set_ylabel("Y (m, left)")
        axs[1].axis("equal")
        axs[1].grid(True, linestyle='--', alpha=0.5)

        # Absolute trajectories
        shown = 0
        for oid, arr in traj_data.items():
            traj = np.asarray(arr)
            if traj.ndim != 2 or traj.shape[1] < 3:
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
        axs[2].set_title(f"Trajectories (absolute) — {shown} shown")
        axs[2].set_xlabel("X (m)")
        axs[2].set_ylabel("Y (m)")
        axs[2].axis("equal")
        axs[2].grid(True, linestyle='--', alpha=0.5)

        # Centered trajectories
        shown = 0
        for oid, arr in traj_data.items():
            traj = np.asarray(arr)
            if traj.ndim != 2 or traj.shape[1] < 3:
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
        axs[3].set_title(f"Trajectories (centered) — {shown} shown")
        axs[3].set_xlabel("X (m)")
        axs[3].set_ylabel("Y (m)")
        axs[3].axis("equal")
        axs[3].grid(True, linestyle='--', alpha=0.5)

        fig.tight_layout()
        base = os.path.splitext(os.path.basename(filepath))[0]
        save_path_2d = os.path.join(SAVE_DIR, f"{base}_2D.png")
        plt.show(block=False); plt.pause(3)
        fig.savefig(save_path_2d, dpi=200)
        plt.close(fig)
        print(f"✅ Saved 2D figure: {save_path_2d}")

        # --- 4️⃣ Space–Time 3D Plot ---
        fig3d = plt.figure(figsize=(10, 8))
        ax3 = fig3d.add_subplot(111, projection='3d')
        shown = 0
        for oid, arr in traj_data.items():
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
        ax3.set_title(f"3D Space–Time Trajectories — {shown} shown")
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.view_init(elev=25, azim=-65)
        fig3d.tight_layout()
        save_path_3d = os.path.join(SAVE_DIR, f"{base}_3D.png")
        plt.show(block=False); plt.pause(3)
        fig3d.savefig(save_path_3d, dpi=200)
        plt.close(fig3d)
        print(f"✅ Saved 3D figure: {save_path_3d}\n")

# ===============================================================
# 🚀 Batch Loop
# ===============================================================
for fp in files:
    try:
        visualize_file(fp)
    except Exception as e:
        print(f"❌ Error with {fp}: {e}\n")
