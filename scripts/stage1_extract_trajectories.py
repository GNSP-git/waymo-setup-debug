import os
import csv
import math
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

# ============================================================
# Verified type map from your own inspection
# ============================================================
TYPE_MAP = {
    0: "Unknown",
    1: "Vehicle",
    2: "Pedestrian",
    3: "Sign",
    4: "Cyclist",
}

# ------------------------------------------------------------
# Extract trajectory info from one TFRecord file
# ------------------------------------------------------------
def extract_from_file(fname, writer):
    print(f"\n▶ Extracting from {os.path.basename(fname)}")

    dataset = tf.data.TFRecordDataset(fname, compression_type="")
    track_buffers = {}   # oid → list

    for raw in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(raw.numpy())
        t = frame.timestamp_micros * 1e-6

        for lab in frame.laser_labels:
            box = lab.box
            oid = lab.id
            typ = TYPE_MAP.get(lab.type, "Unknown")

            # ---------- metadata handling ----------
            if lab.metadata:
                sx = getattr(lab.metadata, "speed_x", 0.0)
                sy = getattr(lab.metadata, "speed_y", 0.0)
                speed = math.hypot(sx, sy)

                is_stationary = 1 if getattr(lab.metadata, "stationary", False) else 0
            else:
                speed = 0.0
                is_stationary = 0

            record = {
                "file": os.path.basename(fname),
                "oid": oid,
                "type": typ,
                "t": t,
                "cx": box.center_x,
                "cy": box.center_y,
                "cz": box.center_z,
                "heading": box.heading,
                "length": box.length,
                "width": box.width,
                "height": box.height,
                "speed": speed,
                "is_stationary": is_stationary,
            }

            track_buffers.setdefault(oid, []).append(record)

    # write sorted tracks
    for oid, seq in track_buffers.items():
        seq.sort(key=lambda r: r["t"])
        for r in seq:
            writer.writerow(r)

    print(f"   ✓ {len(track_buffers)} tracks written")

def load_trajectories_in_memory(fname):
    """
    Wrapper for Stage-1 that returns trajectories in the internal format
    needed by Stage-2 and synthetic scenarios.
    """
    dataset = tf.data.TFRecordDataset(fname, compression_type="")
    trajs = {}

    for raw in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(raw.numpy())
        t = frame.timestamp_micros * 1e-6

        for lab in frame.laser_labels:
            box = lab.box
            oid = lab.id
            typ = TYPE_MAP.get(lab.type, "Unknown")

            if lab.metadata:
                sx = getattr(lab.metadata, "speed_x", 0.0)
                sy = getattr(lab.metadata, "speed_y", 0.0)
                speed = math.hypot(sx, sy)
                is_stationary = 1 if getattr(lab.metadata, "stationary", False) else 0
            else:
                speed = 0.0
                is_stationary = 0

            rec = {
                "t": t,
                "cx": box.center_x,
                "cy": box.center_y,
                "cz": box.center_z,
                "heading": box.heading,
                "length": box.length,
                "width": box.width,
                "height": box.height,
                "speed": speed,
                "is_stationary": is_stationary,
                "type": typ,
                "id": oid,
            }

            trajs.setdefault(oid, []).append(rec)

    # Sort each trajectory by timestamp
    for oid in trajs:
        trajs[oid].sort(key=lambda r: r["t"])

    t0 = min(r["t"] for arr in trajs.values() for r in arr)
    return trajs, t0

    
def main():

    # --------------------------------------------------------
    # FIX: your real TFRecord directory
    # --------------------------------------------------------
    INPUT_DIR = "/mnt/n/waymo_comma/waymo"

    OUT = "/home/gns/waymo_work/data/samples/trajectories_all.csv"

    files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.endswith(".tfrecord")
    )
    print("Found files:", len(files))

    header = [
        "file","oid","type","t",
        "cx","cy","cz","heading",
        "length","width","height",
        "speed","is_stationary"
    ]

    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for f_tf in files:
            extract_from_file(os.path.join(INPUT_DIR, f_tf), writer)

    print("\n=== DONE ===")
    print(f"Trajectories saved → {OUT}")


if __name__ == "__main__":
    main()
