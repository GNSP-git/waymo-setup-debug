#!/usr/bin/env python

import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

# Directory with your TFRecords
DATA_DIR = "/mnt/n/waymo_comma/waymo"

# Output file (relative to this script or absolute)
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "samples", "trajectories_preview.npz")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def iter_waymo_files():
    """Yield full paths of per-segment TFRecords with 3D labels."""
    for fname in sorted(os.listdir(DATA_DIR)):
        if (
            fname.startswith("individual_files_")
            and fname.endswith("_with_camera_labels.tfrecord")
        ):
            yield os.path.join(DATA_DIR, fname)


def extract_trajectories(max_files=None, max_frames_per_file=None):
    """
    Build trajectories from laser_labels.

    Returns:
        dict[id] -> np.ndarray of shape [T, 8]:
            [timestamp_micros,
             center_x, center_y, center_z,
             length, width, height,
             type_int]
    """
    tracks = defaultdict(list)
    file_count = 0

    for path in iter_waymo_files():
        file_count += 1
        print(f"\n📂 Reading {path}")
        if max_files is not None and file_count > max_files:
            break

        ds = tf.data.TFRecordDataset(path, compression_type="")
        frame_idx = 0

        for raw in ds:
            frame = open_dataset.Frame()
            frame.ParseFromString(raw.numpy())
            t = int(frame.timestamp_micros)

            # Each entry in frame.laser_labels is one 3D box with a stable track id.
            for lab in frame.laser_labels:
                oid = lab.id  # string track id
                box = lab.box

                tracks[oid].append([
                    t,
                    float(box.center_x),
                    float(box.center_y),
                    float(box.center_z),
                    float(box.length),
                    float(box.width),
                    float(box.height),
                    int(lab.type),
                ])

            frame_idx += 1
            if max_frames_per_file is not None and frame_idx >= max_frames_per_file:
                break

        print(f"  ✅ processed {frame_idx} frames")

    # Convert to numpy for compactness
    tracks_np = {oid: np.array(states, dtype=np.float64)
                 for oid, states in tracks.items()
                 if len(states) >= 2}  # keep only trajectories with at least 2 steps

    print(f"\n📊 Built trajectories for {len(tracks_np)} objects")
    return tracks_np


def main():
    # Tune these if you want to go bigger:
    tracks = extract_trajectories(
        max_files=2,          # start small; set to None to use all segments
        max_frames_per_file=200,  # cap per file for quick run; None for full
    )

    if not tracks:
        print("⚠️ No trajectories extracted. Check DATA_DIR or filters.")
        return

    # Peek at a few trajectories
    print("\n🔍 Sample trajectories:")
    for i, (oid, traj) in enumerate(tracks.items()):
        print(f"  id={oid}  steps={traj.shape[0]}")
        print(f"    first: t={int(traj[0,0])}, xyz=({traj[0,1]:.2f},{traj[0,2]:.2f},{traj[0,3]:.2f})")
        if i >= 4:
            break

    # Save
    np.savez(OUT_PATH, **tracks)
    print(f"\n💾 Saved trajectories to: {OUT_PATH}")


if __name__ == "__main__":
    print("TensorFlow:", tf.__version__)
    main()
