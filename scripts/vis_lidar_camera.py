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
FILE = "/mnt/n/waymo_comma/waymo/individual_files_training_segment-10107710434105775874_760_000_780_000_with_camera_labels.tfrecord"
FILE = "/mnt/n/waymo_comma/waymo/individual_files_training_segment-1022527355599519580_4866_960_4886_960_with_camera_labels.tfrecord"
FILE = "/mnt/n/waymo_comma/waymo/individual_files_training_segment-10526338824408452410_5714_660_5734_660_with_camera_labels.tfrecord"

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
    """Return (H, W) from first valid range image in Waymo dict structure."""
    # Handle both tuple keys (older SDKs) and int keys (newer SDKs)
    try:
        first_key = next(iter(range_images.keys()))
        key_sample = first_key
        print("DEBUG: first key =", key_sample, "type:", type(key_sample))

        # If tuple-style keys → unpack normally
        if isinstance(key_sample, tuple) and len(key_sample) == 2:
            ri = range_images[first_key]
        else:
            # Newer integer-key dict → value may itself be dict or list
            ri = range_images[key_sample]

        # If list of returns, take first one
        if isinstance(ri, list):
            ri = ri[0]
        elif isinstance(ri, dict):
            # If nested dict {return_idx: MatrixFloat}, take first value
            ri = next(iter(ri.values()))

        # Extract dimensions
        if hasattr(ri, "shape") and hasattr(ri.shape, "dims"):
            h, w = int(ri.shape.dims[0]), int(ri.shape.dims[1])
        else:
            print("⚠️ Could not find .shape.dims; using fallback (64×2650)")
            h, w = 64, 2650

        print(f"✅ inferred H={h}, W={w}")
        return h, w

    except Exception as e:
        print("⚠️ infer_hw_from_range_images failed:", repr(e))
        return 64, 2650  # robust default
    
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
        
        #combined plot done later, skip for now
        '''
        plt.figure(figsize=(8,6))
        plt.imshow(img)
        plt.title("Front Camera")
        plt.axis("off")
        # Show briefly then continue
        plt.show(block=False); plt.pause(5); plt.close()
        '''
        # --- Lidar ---
        parsed = frame_utils.parse_range_image_and_camera_projection(frame)

        out = frame_utils.parse_range_image_and_camera_projection(frame)
        print("parse_range_image_and_camera_projection returned:", type(out), "len=", len(out))
        for i, item in enumerate(out):
            print(f"  [{i}] -> type={type(item)}")

            if len(out) >= 3:
                range_images, camera_projections, range_image_top_pose = out[:3]
                print("range_image_top_pose type:", type(range_image_top_pose))
                if isinstance(range_image_top_pose, dict):
                    print("  dict keys sample:", list(range_image_top_pose.keys())[:3])
                elif hasattr(range_image_top_pose, "data"):
                    print("  has .data of len", len(range_image_top_pose.data))
                else:
                    range_images, camera_projections = out[:2]
                    range_image_top_pose = None

            if isinstance(range_image_top_pose, dict):
                print("range_image_top_pose dict length:", len(range_image_top_pose))
                for k, v in range_image_top_pose.items():
                    print("  key:", k, "->", type(v))
                else:
                    print("Not a dict:", type(range_image_top_pose))


            # --- patch if range_image_top_pose is empty dict ---
            if isinstance(range_image_top_pose, dict) and len(range_image_top_pose) == 0:
                # Get shape from the first valid range image
                first_key = next(iter(range_images.keys()))
                ri = range_images[first_key]

                # Handle case where ri is a list (multiple returns)
                if isinstance(ri, list):
                    ri = ri[0]  # use first return

                # Some versions wrap shape as attribute, others as simple list
                if hasattr(ri, "shape"):
                    dims = [int(d) for d in ri.shape.dims]
                else:
                    dims = [int(d) for d in ri["shape"]["dims"]] if isinstance(ri, dict) else [64, 2650, 4]  # fallback
                    
                    h, w = dims[0], dims[1]
                        
                    # Create fake MatrixFloat with zero data
                    fake_pose = open_dataset.MatrixFloat()
                    fake_pose.shape.dims.extend([h, w, 6])
                    fake_pose.data.extend(np.zeros((h * w * 6,), dtype=np.float32))
                    
                    # Assign to dict key (TOP lidar = 1)
                    range_image_top_pose[(open_dataset.LaserName.TOP, 0)] = fake_pose
                    print(f"✅ Injected fake top pose: shape=({h}, {w}, 6)")
            else:
                print("✅ Top pose already populated:", len(range_image_top_pose))
            

        
            if isinstance(parsed, (list, tuple)):
                if len(parsed) == 2:
                    range_images, camera_projections = parsed
                elif len(parsed) >= 3:
                    range_images, camera_projections = parsed[0], parsed[1]
                else:
                    raise RuntimeError(f"Unexpected parse output: len={len(parsed)}")
            else:
                # Extremely old SDK case (unlikely)
                range_images, camera_projections = parsed, {}

        points_list, _ = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections
        )
        pts = np.concatenate(points_list, axis=0) if points_list else np.zeros((0,4), np.float32)

        # --- Plot LiDAR intensity heatmap ---
        xs, ys = pts[:, 0], pts[:, 1]

        # Some TFRecords have only (x, y, z); some have (x, y, z, intensity)
        if pts.shape[1] >= 4:
            intensity = pts[:, 3]
        else:
            # fallback intensity = z height (for visualization contrast)
            intensity = np.clip(pts[:, 2], -2, 2)
            intensity = (intensity - intensity.min()) / (intensity.ptp() + 1e-6)
        
        mask = (np.abs(xs) < 80) & (np.abs(ys) < 80)
        xs, ys, intensity = xs[mask], ys[mask], intensity[mask]

        #combined plot done later, skip for now
        '''
        plt.figure(figsize=(7, 7))
        sc = plt.scatter(xs, ys, c=intensity, s=0.7, cmap='viridis', alpha=0.7)
        plt.colorbar(sc, label="Intensity")
        plt.title("LiDAR Bird’s Eye View (colored by intensity)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.tight_layout()
        plt.show(block=False); plt.pause(5); plt.close()
        '''
        
        # --- Combined Visualization: Camera + LiDAR side by side ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Camera
        axes[0].imshow(img)
        axes[0].set_title("Front Camera")
        axes[0].axis("off")
        
        # Right: LiDAR Bird’s Eye View
        xs, ys = pts[:, 0], pts[:, 1]
        
        # Some TFRecords have (x,y,z) only
        if pts.shape[1] >= 4:
            intensity = pts[:, 3]
        else:
            intensity = np.clip(pts[:, 2], -2, 2)
            intensity = (intensity - intensity.min()) / (intensity.ptp() + 1e-6)

            mask = (np.abs(xs) < 80) & (np.abs(ys) < 80)
            xs, ys, intensity = xs[mask], ys[mask], intensity[mask]
            
            sc = axes[1].scatter(xs, ys, c=intensity, s=0.7, cmap="viridis", alpha=0.7)
            axes[1].set_title("LiDAR Bird’s Eye View (colored by intensity)")
            axes[1].set_xlabel("X (m)")
            axes[1].set_ylabel("Y (m)")
            axes[1].axis("equal")

            # Shared colorbar for LiDAR intensity
            cbar = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)
            cbar.set_label("Intensity")

            plt.tight_layout()
            fig.savefig(f"frame_{frame.timestamp_micros}.png",dpi=150)
            
            plt.show()  # ← keeps both open until manually closed

if __name__ == "__main__":
    print("TensorFlow:", tf.__version__)
    main()
