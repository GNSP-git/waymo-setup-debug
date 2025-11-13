# stage1_extract_trajectories.py
# Extracts unified trajectories from a directory of Waymo TFRecords into a single CSV.
# No pandas required. Robust Label detection. Computes speed & stationary flags.
# Output: data/samples/trajectories_all.csv (append-safe)

import os, sys, glob, csv, math, inspect
import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

# ---------- CONFIG ----------
INPUT_GLOB = "/mnt/n/waymo_comma/waymo/individual_files_*_with_camera_labels.tfrecord"
SAVE_DIR   = "/home/gns/waymo_work/data/samples"
OUT_CSV    = os.path.join(SAVE_DIR, "trajectories_all.csv")
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- Label enum (version-agnostic) ----------
Label = None
for name, obj in inspect.getmembers(open_dataset):
    if name.lower().endswith("label"):
        Label = obj
        break
if Label is None:
    raise ImportError("Could not locate Label class in waymo_open_dataset.dataset_pb2")

TYPE_NAMES = {}
for attr in dir(Label):
    if attr.startswith("TYPE_"):
        TYPE_NAMES[getattr(Label, attr)] = attr.replace("TYPE_", "").title()

def norm_type_name(raw):
    name = TYPE_NAMES.get(raw, "Unknown")
    n = name.lower()
    if n.startswith("vehicle"):    return "Vehicle"
    if n.startswith("pedestrian"): return "Pedestrian"
    if n.startswith("cyclist"):    return "Cyclist"
    if n.startswith("sign"):       return "Sign"
    return "Unknown"

# ---------- Vehicle & space-object geometry stubs (used later; exported for consistency) ----------
def get_vehicle_geometry(model_name:str|None):
    # Stub: return (L,W,H) defaults, refined later by model_name
    return (4.5, 1.8, 1.6)

def get_space_object_geometry(object_type:str|None):
    # Stub examples: "Drone", "Crane", "Bridge", "Scaffold"
    geom = {
        "Drone":   (0.7, 0.7, 0.3),
        "Crane":   (12.0, 0.8, 3.0),
        "Bridge":  (30.0, 6.0, 2.0),
        "Scaffold":(6.0,  6.0, 6.0),
    }
    return geom.get(object_type or "", (1.0, 1.0, 1.0))

DEFAULT_DIMS = {
    "Vehicle": (4.5, 1.9, 1.6),
    "Pedestrian": (0.6, 0.6, 1.7),
    "Cyclist": (1.8, 0.6, 1.6),
    "Sign": (1.0, 0.5, 3.0),
    "Unknown": (1.0, 1.0, 1.0),
}

def ensure_dims(obj_type, L, W, H):
    if (L or 0) > 0 and (W or 0) > 0 and (H or 0) > 0:
        return L, W, H
    return DEFAULT_DIMS.get(obj_type, DEFAULT_DIMS["Unknown"])

# ---------- pass 1: write header if new ----------
if not os.path.exists(OUT_CSV):
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "file", "oid", "type", "t",
            "cx","cy","cz","heading","length","width","height",
            "speed","is_stationary"
        ])

# ---------- process ----------
files = sorted(glob.glob(INPUT_GLOB))
if not files:
    print("No TFRecord files matched:", INPUT_GLOB)
    sys.exit(0)

def compute_speeds(records):
    # records sorted by t; add speed (m/s). Simple finite diff on center (cx,cy).
    for i in range(len(records)):
        if i == 0:
            records[i]["speed"] = 0.0
        else:
            dt = records[i]["t"] - records[i-1]["t"]
            if dt <= 0:
                records[i]["speed"] = records[i-1]["speed"]
            else:
                dx = records[i]["cx"] - records[i-1]["cx"]
                dy = records[i]["cy"] - records[i-1]["cy"]
                records[i]["speed"] = math.hypot(dx, dy) / dt
    # stationary flag by small moving-window check
    for i in range(len(records)):
        v = records[i]["speed"]
        records[i]["is_stationary"] = 1 if v < 0.15 else 0  # ~15 cm/s threshold

total_frames = 0
total_labels = 0
for path in files:
    ds = tf.data.TFRecordDataset(path, compression_type="")
    by_id = {}
    for raw in ds:
        frame = open_dataset.Frame()
        frame.ParseFromString(raw.numpy())
        t = frame.timestamp_micros * 1e-6
        total_frames += 1

        for lab in frame.laser_labels:
            oid = lab.id
            typ = norm_type_name(lab.type)
            box = lab.box
            L, W, H = ensure_dims(typ, box.length, box.width, box.height)
            rec = {
                "t": t,
                "cx": box.center_x,
                "cy": box.center_y,
                "cz": box.center_z,
                "heading": box.heading,
                "length": L, "width": W, "height": H,
                "type": typ, "id": oid
            }
            by_id.setdefault(oid, []).append(rec)
            total_labels += 1

    # finalize per-id and append to CSV
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        for oid, arr in by_id.items():
            arr.sort(key=lambda r: r["t"])
            compute_speeds(arr)
            for r in arr:
                w.writerow([
                    os.path.basename(path), oid, r["type"], f"{r['t']:.6f}",
                    f"{r['cx']:.4f}", f"{r['cy']:.4f}", f"{r['cz']:.4f}",
                    f"{r['heading']:.6f}",
                    f"{r['length']:.3f}", f"{r['width']:.3f}", f"{r['height']:.3f}",
                    f"{r['speed']:.3f}", r["is_stationary"]
                ])
    print(f"✔ Extracted {len(by_id)} tracks from {os.path.basename(path)}")

print(f"✅ Stage 1 done. Frames: {total_frames}, labels: {total_labels}")
print(f"Saved: {OUT_CSV}")
