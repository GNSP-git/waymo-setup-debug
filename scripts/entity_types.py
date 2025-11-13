#!/usr/bin/env python3
"""
extract_entity_types.py
------------------------------------
Reads Waymo Open Dataset .tfrecord files, extracts entity metadata, and
saves an entities_<basename>.csv for each file.

Supports TF 2.11 + Waymo Open Dataset 1.6.x
Author: G.N. Srinivasa Prasanna (GNSP)
"""

import os, csv, math
import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2  # <-- add this

# === Directory containing your individual_*.tfrecord files ===
DATA_DIR = "/mnt/n/waymo_comma/waymo"
OUT_DIR  = os.path.join(DATA_DIR, "entities")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# Helper: classify entity subtype heuristically
# ---------------------------------------------------------------
def classify_vehicle(box):
    if box.length > 7.0:
        return "truck_or_bus"
    elif box.length < 2.5 and box.height > 1.4:
        return "motorcycle"
    elif box.height < 1.2 and box.width < 1.0:
        return "bicycle"
    else:
        return "car"

# ---------------------------------------------------------------
# Extract one frame’s labeled objects
# ---------------------------------------------------------------
def extract_frame_entities(frame, frame_idx):
    rows = []
    for lbl in frame.laser_labels:
        t = label_pb2.Label.Type.Name(lbl.type)        
        box = lbl.box
        subtype = classify_vehicle(box) if t == "VEHICLE" else t.lower()

        rows.append({
            "frame_idx": frame_idx,
            "object_id": lbl.id,
            "type": t,
            "subtype": subtype,
            "length_m": box.length,
            "width_m": box.width,
            "height_m": box.height,
            "center_x": box.center_x,
            "center_y": box.center_y,
            "center_z": box.center_z,
            "heading_rad": box.heading
        })
    return rows

# ---------------------------------------------------------------
# Process a single TFRecord file → CSV
# ---------------------------------------------------------------
def process_file(path):
    base = os.path.basename(path)
    csv_path = os.path.join(OUT_DIR, f"entities_{os.path.splitext(base)[0]}.csv")

    print(f"▶ Extracting from {base} ...")
    ds = tf.data.TFRecordDataset(path, compression_type="")
    all_rows = []
    for i, raw in enumerate(ds):
        frame = open_dataset.Frame()
        frame.ParseFromString(raw.numpy())
        rows = extract_frame_entities(frame, i)
        all_rows.extend(rows)
        if (i + 1) % 20 == 0:
            print(f"  processed {i+1} frames...")

    if not all_rows:
        print(f"  ⚠️ No labeled objects found in {base}")
        return

    # Write CSV
    keys = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  ✅ Saved {csv_path} ({len(all_rows)} rows)")

# ---------------------------------------------------------------
# Main driver: iterate over all files
# ---------------------------------------------------------------
def main():
    files = [os.path.join(DATA_DIR, f)
             for f in os.listdir(DATA_DIR)
             if f.endswith(".tfrecord") and f.startswith("individual_")]
    if not files:
        print(f"❌ No individual_*.tfrecord found in {DATA_DIR}")
        return

    for path in sorted(files):
        try:
            process_file(path)
        except Exception as e:
            print(f"  ❌ Error processing {path}: {e}")

if __name__ == "__main__":
    print("TensorFlow:", tf.__version__)
    main()
