import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils

# === PATH TO YOUR WAYMO DIRECTORY ===
DATA_DIR = "/mnt/n/waymo_comma/waymo"
OUT_DIR = "./plots"
os.makedirs(OUT_DIR, exist_ok=True)
CAM_INDEX = 0

# === Helper classes ===
class _Dims:
    def __init__(self, dims): self.dims = dims

class FakeTopPose:
    def __init__(self, h, w):
        self._h = int(h); self._w = int(w)
        self.shape = _Dims([self._h, self._w, 6])
        self.data = (np.zeros((self._h * self._w * 6,), dtype=np.float32)).tolist()

def infer_hw_from_range_images(range_images):
    """Return (H,W) from first valid range image."""
    try:
        first_key = next(iter(range_images.keys()))
        ri = range_images[first_key]
        if isinstance(ri, list): ri = ri[0]
        elif isinstance(ri, dict): ri = next(iter(ri.values()))
        if hasattr(ri, "shape") and hasattr(ri.shape, "dims"):
            h, w = int(ri.shape.dims[0]), int(ri.shape.dims[1])
        else:
            h, w = 64, 2650
        return h, w
    except Exception:
        return 64, 2650

# --- Monkey-patch to inject fake top pose ---
import waymo_open_dataset.utils.frame_utils as fu
_orig_convert_pc = fu.convert_range_image_to_point_cloud
def _wrapped_convert_point_cloud(frame, range_images, camera_projections, *maybe_top_pose):
    h, w = infer_hw_from_range_images(range_images)
    fake_top_pose = FakeTopPose(h, w)
    try:
        return _orig_convert_pc(frame, range_images, camera_projections, fake_top_pose)
    except TypeError:
        return _orig_convert_pc(frame, range_images, camera_projections)
fu.convert_range_image_to_point_cloud = _wrapped_convert_point_cloud
print("✅ Waymo SDK patched: using FakeTopPose for TF2.11/WSL")

# === Process all individual_*.tfrecord files ===
files = sorted(f for f in os.listdir(DATA_DIR)
               if f.startswith("individual_") and f.endswith(".tfrecord"))
print(f"Found {len(files)} files to process.")

for fname in files:
    path = os.path.join(DATA_DIR, fname)
    print(f"\n=== Processing {fname} ===")
    ds = tf.data.TFRecordDataset(path, compression_type="")
    for raw in ds.take(1):
        frame = open_dataset.Frame()
        frame.ParseFromString(raw.numpy())

        # --- Camera ---
        img = tf.image.decode_jpeg(frame.images[CAM_INDEX].image).numpy()

        # --- LiDAR ---
        parsed = frame_utils.parse_range_image_and_camera_projection(frame)
        if len(parsed) >= 3:
            range_images, camera_projections = parsed[0], parsed[1]
        else:
            range_images, camera_projections = parsed[:2]
        points_list, _ = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections
        )
        pts = np.concatenate(points_list, axis=0) if points_list else np.zeros((0, 4), np.float32)

        xs, ys = pts[:, 0], pts[:, 1]
        if pts.shape[1] >= 4:
            intensity = pts[:, 3]
        else:
            intensity = np.clip(pts[:, 2], -2, 2)
            intensity = (intensity - intensity.min()) / (intensity.ptp() + 1e-6)
        mask = (np.abs(xs) < 80) & (np.abs(ys) < 80)
        xs, ys, intensity = xs[mask], ys[mask], intensity[mask]

        # --- Combined figure ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(img)
        axes[0].set_title("Front Camera")
        axes[0].axis("off")

        sc = axes[1].scatter(xs, ys, c=intensity, s=0.7, cmap="viridis", alpha=0.7)
        axes[1].set_title("LiDAR Bird’s Eye View (colored by intensity)")
        axes[1].set_xlabel("X (m)")
        axes[1].set_ylabel("Y (m)")
        axes[1].axis("equal")
        cbar = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("Intensity")

        ts = frame.timestamp_micros
        save_path = os.path.join(OUT_DIR, f"frame_{ts}_{fname[:25]}.png")
        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"✅ Saved: {save_path}")

print("🎯 All frames processed and saved.")
