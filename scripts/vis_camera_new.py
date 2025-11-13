# vis_lidar_camera_final.py
# Stable for TF 2.11 + Waymo Open Dataset 1.6.x on WSL
# Fixes range_image_top_pose shape/.data expectations by providing a proto-like shim.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils

# === EDIT THIS PATH ===
FILE = "/mnt/n/waymo_comma/waymo/individual_files_validation_segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord"
CAM_INDEX = 0

# ---------------------------------------------------------------------
# Proto-like shim: has `.data` and `.shape.dims` as Waymo utils expect.
# We synthesize zeros (neutral pose). The SDK will reshape and proceed.
# ---------------------------------------------------------------------
class _Dims:
    def __init__(self, dims): self.dims = dims

class FakeTopPose:
    def __init__(self, h, w):
        # SDK expects [H, W, 6] flattened in `.data`
        self._h = int(h)
        self._w = int(w)
        self.shape = _Dims([self._h, self._w, 6])
        self.data = (np.zeros((self._h * self._w * 6,), dtype=np.float32)).tolist()

# ---------------------------------------------------------------------
# Helper: find a representative range image to get H,W
# range_images is a dict keyed by (laser_name, return_idx) -> MatrixFloat
# MatrixFloat has `.shape.dims` like [H, W, 4] (channels vary).
# ---------------------------------------------------------------------
def infer_hw_from_range_images(range_images):
    # Prefer TOP lidar if present (Waymo's LaserName.TOP == 1), else first entry.
    top_key = None
    for (laser_name, ret_idx) in range_images.keys():
        if laser_name == open_dataset.LaserName.TOP:
            top_key = (laser_name, ret_idx)
            break
    key = top_key if top_key in range_images else next(iter(range_images.keys()))
    ri = range_images[key]
    # ri.shape.dims: e.g., [H, W, C]
    h, w = int(ri.shape.dims[0]), int(ri.shape.dims[1])
    return h, w

# ---------------------------------------------------------------------
# Monkey-patch wrapper: always pass a proto-like top_pose with .data/.shape
# The underlying SDK still calls convert_range_image_to_cartesian, which
# expects those attributes; we provide them.
# ---------------------------------------------------------------------
import waymo_open_dataset.utils.frame_utils as fu
_orig_convert_pc = fu.convert_range_image_to_point_cloud

def _wrapped_convert_point_cloud(frame, range_images, camera_projections, *maybe_top_pose):
    # Build a FakeTopPose from the actual range image H,W
    h, w = infer_hw_from_range_images(range_images)
    fake_top_pose = FakeTopPose(h, w)
    try:
        # Newer signature (expects 4th arg)
        return _orig_convert_pc(frame, range_images, camera_projections, fake_top_pose)
    except TypeError:
        # Older signature (no top pose arg)
        return _orig_convert_pc(frame, range_images, camera_projections)

# Install wrapper
fu.convert_range_image_to_point_cloud = _wrapped_convert_point_cloud

print("✅ Waymo SDK patched: supplying proto-like FakeTopPose(data, shape.dims)")

# ---------------------------------------------------------------------
def main():
    ds = tf.data.TFRecordDataset(FILE, compression_type="")
    for raw in ds.take(1):
        frame = open_dataset.Frame()
        frame.ParseFromString(raw.numpy())

        # --- Camera ---
        img = tf.image.decode_jpeg(frame.images[CAM_INDEX].image).numpy()
        plt.figure(figsize=(8,6))
        plt.imshow(img)
        plt.title("Front Camera")
        plt.axis("off")
        # Show briefly then continue
        plt.show(block=False); plt.pause(1.2); plt.close()

        print("✅ Visualization complete")
        print(f"Timestamp: {frame.timestamp_micros}")
        print(f"Cameras: {len(frame.images)} | LiDARs: {len(frame.lasers)}")
        break

if __name__ == "__main__":
    print("TensorFlow:", tf.__version__)
    main()
