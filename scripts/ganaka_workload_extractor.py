#!/usr/bin/env python3
"""
GANAKA WORKLOAD EXTRACTOR – Single unified script
-------------------------------------------------
Loads ALL Waymo TFRecords in a directory, extracts trajectories,
computes OBB→OBB minimum distances (fast Cython), finds Vehicle–Vehicle
close approaches, saves a unified CSV workload, and provides visualization.

Stages:
  1. Trajectory extraction (all objects)
  2. Pairwise V–V evaluation (OBB)
  3. Ganaka workload accumulation
  4. Visualization + walkaround
  5. Hooks for:
       - full vehicle geometry by model
       - 2D/3D object geometry
       - trajectory supersets + convex hulls (for I-structures)
"""

import os, glob, math, itertools, time, inspect, json
from typing import Dict, List
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -------------------------------
# CONFIG
# -------------------------------

DATA_DIR   = "/mnt/n/waymo_comma/waymo"
SAVE_DIR   = "/home/gns/waymo_work/data/samples"
os.makedirs(SAVE_DIR, exist_ok=True)

DIST_THRESH = 7.5          # meters (for close V–V)
TIME_THRESH = 0.15         # seconds
MAX_PAIRS_PER_FILE = 10
SHOW_SEC = 5.0              # visualization flash

FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*_with_camera_labels.tfrecord")))
print(f"Found {len(FILES)} TFRecord files")

# -------------------------------
# Load OBB distance extension
# -------------------------------
try:
    import obb_distance
    USE_OBB = True
    print("Using optimized OBB module")
except Exception:
    USE_OBB = False
    print("WARNING: No OBB extension – using center distances only")


# -------------------------------
# Waymo label resolution (robust)
# -------------------------------
from waymo_open_dataset import dataset_pb2 as open_dataset

Label = None
for name, obj in inspect.getmembers(open_dataset):
    if name.lower().endswith("label"):
        Label = obj
if Label is None:
    raise ImportError("Cannot find label type in dataset_pb2")

TYPE_NAMES = {}
for attr in dir(Label):
    if attr.startswith("TYPE_"):
        TYPE_NAMES[getattr(Label, attr)] = attr.replace("TYPE_", "").title()

# Normalize → human types
def normalize_type(name: str):
    n = name.lower()
    if "vehicle" in n: return "Vehicle"
    if "pedestrian" in n: return "Pedestrian"
    if "cyclist" in n: return "Cyclist"
    if "sign" in n: return "Sign"
    return "Unknown"


# -------------------------------
# Default object dimensions
# -------------------------------
DEFAULT_DIMS = {
    "Vehicle": (4.5, 1.8, 1.6),
    "Pedestrian": (0.6, 0.6, 1.7),
    "Cyclist": (1.8, 0.6, 1.6),
    "Sign": (1.0, 0.5, 3.0),
    "Unknown": (1.0, 1.0, 1.0),
}

CLASS_COLOR = {
    "Vehicle": "#1f77b4",
    "Pedestrian": "#2ca02c",
    "Cyclist": "#ff7f0e",
    "Sign": "#9467bd",
    "Unknown": "#7f7f7f",
}

def ensure_dims(type_name, L, W, H):
    if L > 0 and W > 0 and H > 0:
        return L, W, H
    return DEFAULT_DIMS.get(type_name, DEFAULT_DIMS["Unknown"])


# -------------------------------
# Vehicle model geometry stub
# -------------------------------
def vehicle_geometry_lookup(model_name: str):
    """
    Stub: Return detailed polygon mesh for exact vehicle model.
    Replace with real data later.
    """
    return None


# -------------------------------
# 2D/3D general object geometry stub
# -------------------------------
def general_object_geometry(obj_type: str, metadata=None):
    """
    Stub: For drones, buildings, construction cones, overhanging signs, etc.
    """
    return None


# -------------------------------
# convex-hull / I-structure stub
# -------------------------------
def trajectory_convex_hull(traj):
    """
    Stub: Return convex hull of trajectory for set-theoretic acceleration.
    """
    return None


# -------------------------------
# OBB helper: corners in XY
# -------------------------------
def obb_rect_xy(cx, cy, L, W, heading):
    hl, hw = 0.5*L, 0.5*W
    local = np.array([
        [-hl, -hw],
        [ hl, -hw],
        [ hl,  hw],
        [-hl,  hw],
    ])
    c, s = math.cos(heading), math.sin(heading)
    R = np.array([[c,-s],[s,c]])
    pts = local @ R.T
    pts[:,0] += cx
    pts[:,1] += cy
    return pts


# -------------------------------
# OBB distance wrapper (FIXED)
# -------------------------------
def obb_dist(a, b):
    """
    Robust distance:
      - always compute center distance
      - if OBB module is present, use min(center, obb)
        and fall back to center on any weird/NaN/inf result.
    """
    # Center distance in XY always
    d_center = math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])

    if USE_OBB:
        try:
            d_obb = obb_distance.obb_min_distance(
                a["cx"], a["cy"], a["length"], a["width"], a["heading"],
                b["cx"], b["cy"], b["length"], b["width"], b["heading"]
            )
            # Guard against NaN/inf or crazy large compared to center
            if not np.isfinite(d_obb):
                return d_center
            # If OBB says something wildly larger than center, prefer center
            return min(d_center, d_obb)
        except Exception:
            # Any error in the extension: fall back to center
            return d_center

    # No OBB extension: just center distance
    return d_center


# -------------------------------
# Stage 1 — Extract trajectories
# -------------------------------
def collect_trajectories(file_path):
    trajs = {}
    t0 = None
    ds = tf.data.TFRecordDataset(file_path, compression_type="")
    for raw in ds:
        frame = open_dataset.Frame()
        frame.ParseFromString(raw.numpy())
        t = frame.timestamp_micros * 1e-6
        if t0 is None:
            t0 = t

        for lab in frame.laser_labels:
            # map numeric type → name via TYPE_NAMES, with fallback
            raw_name = TYPE_NAMES.get(lab.type, "Unknown")       # e.g. "Vehicle"
            typ = normalize_type(raw_name)                       # ensure canonical

            box = lab.box
            L, W, H = ensure_dims(typ, box.length, box.width, box.height)

            rec = {
                "t": t,
                "cx": box.center_x,
                "cy": box.center_y,
                "heading": box.heading,
                "length": L,
                "width": W,
                "height": H,
                "type": typ,
                "id": lab.id,
                "raw_type_code": lab.type,                       # for debugging
                "raw_type_name": raw_name,
            }
            trajs.setdefault(lab.id, []).append(rec)

    for k in trajs:
        trajs[k].sort(key=lambda r: r["t"])

    return trajs, (t0 if t0 else 0.0)


# -------------------------------
# Stage 2 — Find close V–V pairs
# -------------------------------
def find_close_pairs(trajs):
    all_times = sorted({r["t"] for arr in trajs.values() for r in arr})
    best_pairs = {}

    # For debug: track minimum distance seen at all
    global_min_d = float("inf")

    for t in all_times:
        # records within TIME_THRESH/2
        snapshot = []
        half = TIME_THRESH / 2.0
        for oid, arr in trajs.items():
            best = None; best_dt = 1e9
            for rec in arr:
                dt = abs(rec["t"] - t)
                if dt < best_dt and dt <= half:
                    best_d_
