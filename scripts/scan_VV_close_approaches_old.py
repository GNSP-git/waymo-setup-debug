#!/usr/bin/env python
"""
scan_VV_close_approaches.py

Scan ALL Waymo TFRecord files in a directory:
  - Build trajectories from laser OBBs.
  - Find Vehicle–Vehicle close approaches within a space–time window.
  - Compare OBB separation vs center–center distance.
  - Interactive walkthrough (keyboard) with XY and space–time plots.
"""

import os, glob, math, itertools, inspect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as open_dataset

# ============================================================
# CONFIG
# ============================================================

ROOT_DIR    = "/mnt/n/waymo_comma/waymo"
PATTERN     = "individual_files_*_with_camera_labels.tfrecord"

SAVE_DIR    = "/home/gns/waymo_work/data/samples"
os.makedirs(SAVE_DIR, exist_ok=True)

# Try with your “extreme” thresholds to verify:
DIST_THRESH = 50.0      # meters
TIME_THRESH = 10.0      # seconds
MAX_PAIRS_PER_FILE = 50
GLOBAL_MAX_PAIRS    = 200

# For debugging: show min center distance between *vehicles* per frame
DEBUG_MIN_CENTER_PER_FRAME = False

# ============================================================
# Try OBB distance extension
# ============================================================

USE_OBB = True
try:
    import obb_distance
except Exception:
    USE_OBB = False
    print("ℹ️ obb_distance not found; falling back to center-distance only.")

# ============================================================
# Label & type handling
# ============================================================

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

def label_type_name(t):
    """
    Map numeric label type to canonical names:
      Vehicle, Pedestrian, Cyclist, Sign, Unknown
    """
    name = TYPE_NAMES.get(t)
    if not name:
        return "Unknown"
    low = name.lower()
    if "vehicle"   in low: return "Vehicle"
    if "pedestrian" in low: return "Pedestrian"
    if "cyclist"    in low or "bike" in low: return "Cyclist"
    if "sign"       in low: return "Sign"
    return name.title()

DEFAULT_DIMS = {
    "Vehicle":    (4.5, 1.8, 1.6),
    "Pedestrian": (0.6, 0.6, 1.7),
    "Cyclist":    (1.8, 0.6, 1.6),
    "Sign":       (1.0, 0.5, 3.0),
    "Unknown":    (1.0, 1.0, 1.0),
}
CLASS_COLOR = {
    "Vehicle":    "#1f77b4",
    "Pedestrian": "#2ca02c",
    "Cyclist":    "#ff7f0e",
    "Sign":       "#9467bd",
    "Unknown":    "#7f7f7f",
}

def ensure_dims(name, L, W, H):
    if (L or 0) > 0 and (W or 0) > 0 and (H or 0) > 0:
        return L, W, H
    return DEFAULT_DIMS.get(name, DEFAULT_DIMS["Unknown"])

# ============================================================
# Geometry helpers
# ============================================================

def obb_rect_xy(cx, cy, length, width, heading):
    hl, hw = 0.5 * length, 0.5 * width
    local = np.array([
        [-hl, -hw],
        [ hl, -hw],
        [ hl,  hw],
        [-hl,  hw],
    ], dtype=np.float32)
    c, s = math.cos(heading), math.sin(heading)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)
    pts = local @ R.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return pts

def center_distance(a, b):
    dx = a["cx"] - b["cx"]
    dy = a["cy"] - b["cy"]
    return math.hypot(dx, dy)

def obb_distance_safe(a, b):
    if not USE_OBB:
        return center_distance(a, b)
    return obb_distance.obb_min_distance(
        a["cx"], a["cy"], a["length"], a["width"], a["heading"],
        b["cx"], b["cy"], b["length"], b["width"], b["heading"]
    )

# ============================================================
# Trajectory building
# ============================================================

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

        # --- Detect which labels are present ---
        labels = []

        if len(frame.laser_labels) > 0:
            # primary 3D bounding boxes
            for lab in frame.laser_labels:
                labels.append(lab)

        else:
            # fallback: CAMERA labels
            for cam in frame.camera_labels:
                for lab in cam.labels:
                    labels.append(lab)

        # --- Extract OBBs ---
        for lab in labels:
            name = label_type_name(lab.type)
            box = lab.box
            L, W, H = ensure_dims(name, box.length, box.width, box.height)
            rec = {
                "t": t,
                "cx": box.center_x,
                "cy": box.center_y,
                "heading": box.heading,
                "length": L,
                "width": W,
                "height": H,
                "type": name,
                "id": lab.id,
            }
            trajs.setdefault(lab.id, []).append(rec)

    for oid in trajs:
        trajs[oid].sort(key=lambda r: r["t"])
    return trajs, t0 or 0.0


def collect_trajectories_wrong(file_path):
    """
    Returns:
      trajs: dict[obj_id] -> list of recs:
            t, cx, cy, heading, length, width, height, type, id
      t0: first timestamp (s)
    """
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
            name = label_type_name(lab.type)
            box = lab.box
            L, W, H = ensure_dims(name, box.length, box.width, box.height)
            rec = {
                "t": t,
                "cx": box.center_x,
                "cy": box.center_y,
                "heading": box.heading,
                "length": L,
                "width": W,
                "height": H,
                "type": name,
                "id": lab.id,
            }
            trajs.setdefault(lab.id, []).append(rec)

    for oid in trajs:
        trajs[oid].sort(key=lambda r: r["t"])

    if t0 is None:
        t0 = 0.0
    return trajs, t0

def type_histogram(trajs):
    """
    Simple histogram of object types in this file.
    """
    counts = {}
    for arr in trajs.values():
        for r in arr:
            counts[r["type"]] = counts.get(r["type"], 0) + 1
    return counts

# ============================================================
# Close-pair search (Vehicle–Vehicle only)
# ============================================================

def find_close_pairs_vehicle_only(trajs, dist_thresh, time_thresh, max_pairs,
                                  debug_min_center=False):
    """
    Important change vs previous version:
      - snapshot includes ALL types
      - we restrict to Vehicle–Vehicle only when recording pairs

    Returns (ascending by OBB d):
      [
        {
          "oid_i", "oid_j", "d_obb", "d_center",
          "t_at", "rec_i", "rec_j"
        }, ...
      ]
    """
    times = sorted({rec["t"] for arr in trajs.values() for rec in arr})
    best_per_pair = {}

    for t in times:
        # snapshot: ALL objects near this t (no type filter here)
        snapshot = []
        for oid, arr in trajs.items():
            best, best_dt = None, 1e9
            for rec in arr:
                dt = abs(rec["t"] - t)
                if dt < best_dt and dt <= time_thresh / 2.0:
                    best_dt, best = dt, rec
            if best is not None:
                snapshot.append(best)

        if not snapshot:
            continue

        # Optional per-frame debug for Vehicle min center distance
        if debug_min_center:
            vehs = [r for r in snapshot if r["type"] == "Vehicle"]
            if len(vehs) > 1:
                mind = min(
                    center_distance(a, b)
                    for a, b in itertools.combinations(vehs, 2)
                )
                print(f"[t={t:.3f}] vehicles={len(vehs)}, min(center)≈{mind:.2f} m")

        # Pairwise distances, but record only Vehicle–Vehicle
        for a, b in itertools.combinations(snapshot, 2):
            if not (a["type"] == "Vehicle" and b["type"] == "Vehicle"):
                continue
            d_center = center_distance(a, b)
            d_obb    = obb_distance_safe(a, b)
            if d_obb < dist_thresh:
                key = tuple(sorted((a["id"], b["id"])))
                prev = best_per_pair.get(key)
                if (prev is None) or (d_obb < prev[0]):
                    best_per_pair[key] = (d_obb, d_center, t, a, b)

    ranked = sorted(
        (
            {
                "oid_i": k[0],
                "oid_j": k[1],
                "d_obb": v[0],
                "d_center": v[1],
                "t_at": v[2],
                "rec_i": v[3],
                "rec_j": v[4],
            }
            for k, v in best_per_pair.items()
        ),
        key=lambda x: x["d_obb"],
    )
    return ranked[:max_pairs]

# ============================================================
# Plotting helpers
# ============================================================

def draw_xy(ax, trajs, pair):
    for rec, lab in ((pair["rec_i"], "A"), (pair["rec_j"], "B")):
        full = trajs[rec["id"]]
        xs = [r["cx"] for r in full]
        ys = [r["cy"] for r in full]
        color = CLASS_COLOR.get(rec["type"], "#7f7f7f")
        ax.plot(xs, ys, '-', linewidth=2.5, color=color, alpha=0.9,
                label=f'{lab}: {rec["type"]}')

        # heading arrow
        ax.quiver(
            rec["cx"], rec["cy"],
            math.cos(rec["heading"]), math.sin(rec["heading"]),
            angles='xy', scale_units='xy', scale=1.5,
            width=0.004, color=color
        )

        # footprint
        poly = obb_rect_xy(rec["cx"], rec["cy"],
                           rec["length"], rec["width"], rec["heading"])
        ax.add_patch(
            Polygon(poly, closed=True, facecolor=color,
                    alpha=0.25, edgecolor=color, linewidth=1.5)
        )
        ax.text(rec["cx"], rec["cy"], lab,
                fontsize=10, weight='bold', color=color)

    # Red circle around mid-point
    midx = 0.5 * (pair["rec_i"]["cx"] + pair["rec_j"]["cx"])
    midy = 0.5 * (pair["rec_i"]["cy"] + pair["rec_j"]["cy"])
    radius = max(pair["d_center"], pair["d_obb"]) + 2.0
    circ = plt.Circle((midx, midy), radius,
                      fill=False, color='red', linewidth=2.0, alpha=0.8)
    ax.add_patch(circ)

    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"XY Vehicle–Vehicle close approach "
        f"(d_OBB={pair['d_obb']:.2f} m, d_center={pair['d_center']:.2f} m)"
    )
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc='best')

def draw_spacetime(ax3d, trajs, pair, t0):
    for rec, lab in ((pair["rec_i"], "A"), (pair["rec_j"], "B")):
        full = trajs[rec["id"]]
        color = CLASS_COLOR.get(rec["type"], "#7f7f7f")
        X = np.array([r["cx"] for r in full])
        Y = np.array([r["cy"] for r in full])
        T = np.array([r["t"] - t0 for r in full])
        ax3d.plot3D(X, Y, T, color=color, linewidth=2.0,
                    label=f'{lab}: {rec["type"]}')

        N = max(1, len(full) // 30)
        blocks = []
        for r in full[::N]:
            poly = obb_rect_xy(r["cx"], r["cy"],
                               r["length"], r["width"], r["heading"])
            z0 = (r["t"] - t0) - 0.03
            z1 = (r["t"] - t0) + 0.03
            bottom = [(poly[k, 0], poly[k, 1], z0) for k in range(4)]
            top    = [(poly[k, 0], poly[k, 1], z1) for k in range(4)]
            faces = [
                [bottom[0], bottom[1], bottom[2], bottom[3]],
                [top[0],    top[1],    top[2],    top[3]],
                [bottom[0], bottom[1], top[1],    top[0]],
                [bottom[1], bottom[2], top[2],    top[1]],
                [bottom[2], bottom[3], top[3],    top[2]],
                [bottom[3], bottom[0], top[0],    top[3]],
            ]
            blocks.extend(faces)
        if blocks:
            pc = Poly3DCollection(blocks, facecolor=color, edgecolor=color,
                                  alpha=0.10, linewidths=0.3)
            ax3d.add_collection3d(pc)

    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("t (s from first frame)")
    ax3d.set_title("Space–time tubes (two vehicles)")
    ax3d.grid(True)

def on_key_2d_zoom(fig, axes):
    def handler(event):
        if event.key in ('z', 'Z'):
            for ax in axes:
                if not hasattr(ax, 'get_xlim'):
                    continue
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                cx = 0.5 * (x0 + x1)
                cy = 0.5 * (y0 + y1)
                sx = (x1 - x0)
                sy = (y1 - y0)
                if event.key == 'z':
                    sx *= 0.8; sy *= 0.8
                else:
                    sx *= 1.25; sy *= 1.25
                ax.set_xlim(cx - sx/2, cx + sx/2)
                ax.set_ylim(cy - sy/2, cy + sy/2)
            fig.canvas.draw_idle()
    return handler

def on_key_3d_walk(fig, ax3d):
    def handler(event):
        az, el = ax3d.azim, ax3d.elev
        changed = False
        if event.key in ('left', 'a'):
            az -= 5; changed = True
        elif event.key in ('right', 'd'):
            az += 5; changed = True
        elif event.key in ('up', 'w'):
            el += 3; changed = True
        elif event.key in ('down', 's'):
            el -= 3; changed = True
        if changed:
            ax3d.view_init(elev=el, azim=az)
            fig.canvas.draw_idle()

        if event.key in ('z', 'Z'):
            x0, x1 = ax3d.get_xlim()
            y0, y1 = ax3d.get_ylim()
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            sx = (x1 - x0)
            sy = (y1 - y0)
            if event.key == 'z':
                sx *= 0.8; sy *= 0.8
            else:
                sx *= 1.25; sy *= 1.25
            ax3d.set_xlim(cx - sx/2, cx + sx/2)
            ax3d.set_ylim(cy - sy/2, cy + sy/2)
            fig.canvas.draw_idle()
    return handler

def visualize_pair(file_path, trajs, t0, pair, index_in_global):
    base = os.path.basename(file_path)
    rel_t = pair["t_at"] - t0

    print("\n=== VISUALIZING PAIR #{idx} ===".format(idx=index_in_global))
    print(f"  File: {base}")
    print(f"  t_rel = {rel_t:.3f} s from first frame in file")
    print(f"  Vehicles: {pair['oid_i']} – {pair['oid_j']}")
    print(f"  d_OBB = {pair['d_obb']:.3f} m, d_center = {pair['d_center']:.3f} m")

    fig_xy, ax_xy = plt.subplots(figsize=(7.5, 7.0))
    draw_xy(ax_xy, trajs, pair)
    fig_xy.canvas.mpl_connect('key_press_event', on_key_2d_zoom(fig_xy, [ax_xy]))
    out_xy = os.path.join(
        SAVE_DIR,
        f"vv_close_xy_{index_in_global:03d}_{base.replace('.tfrecord','')}.png"
    )
    plt.tight_layout()
    fig_xy.savefig(out_xy, dpi=140, bbox_inches="tight")
    print(f"  Saved XY view: {out_xy}")

    fig_st = plt.figure(figsize=(8.5, 6.8))
    ax3d = fig_st.add_subplot(111, projection='3d')
    draw_spacetime(ax3d, trajs, pair, t0)
    fig_st.canvas.mpl_connect('key_press_event', on_key_3d_walk(fig_st, ax3d))
    out_st = os.path.join(
        SAVE_DIR,
        f"vv_close_spacetime_{index_in_global:03d}_{base.replace('.tfrecord','')}.png"
    )
    plt.tight_layout()
    fig_st.savefig(out_st, dpi=140, bbox_inches="tight")
    print(f"  Saved space–time view: {out_st}")

    print("  Controls: z/Z zoom, arrows or WASD in 3D, close both figures to continue.")
    plt.show()

# ============================================================
# MAIN
# ============================================================

def main():
    print("TensorFlow:", tf.__version__)
    pattern = os.path.join(ROOT_DIR, PATTERN)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files matching {pattern}")
        return

    print(f"Found {len(files)} files under {ROOT_DIR}")
    all_pairs = []

    for fpath in files:
        print(f"\n▶ Processing file: {os.path.basename(fpath)}")
        trajs, t0 = collect_trajectories(fpath)
        n_tracks = len(trajs)
        n_samples = sum(len(v) for v in trajs.values())
        print(f"  Tracks: {n_tracks}, samples: {n_samples}")

        # type histogram per file
        hist = type_histogram(trajs)
        print("  Type histogram:", hist)

        pairs = find_close_pairs_vehicle_only(
            trajs, DIST_THRESH, TIME_THRESH, MAX_PAIRS_PER_FILE,
            debug_min_center=DEBUG_MIN_CENTER_PER_FRAME
        )
        print(f"  Vehicle–Vehicle pairs within {DIST_THRESH} m & Δt={TIME_THRESH}s: {len(pairs)}")

        for p in pairs:
            p["file"] = fpath
            p["t0"]   = t0
        all_pairs.extend(pairs)

        if len(all_pairs) >= GLOBAL_MAX_PAIRS:
            break

    if not all_pairs:
        print("\n❌ No Vehicle–Vehicle pairs found under these thresholds.")
        return

    all_pairs.sort(key=lambda x: x["d_obb"])
    print(f"\n✅ Total V–V close approaches collected: {len(all_pairs)}")
    print("Top 10 by OBB distance:")
    for i, p in enumerate(all_pairs[:10]):
        base = os.path.basename(p["file"])
        print(f"  #{i}: d_OBB={p['d_obb']:.3f} m, d_center={p['d_center']:.3f} m "
              f"file={base}, t_rel≈{p['t_at']-p['t0']:.3f} s")

    idx = 0
    N = len(all_pairs)
    while True:
        print(f"\n--- Pair index {idx}/{N-1} ---")
        p = all_pairs[idx]
        base = os.path.basename(p["file"])
        print(f"  From file {base}, t_rel≈{p['t_at']-p['t0']:.3f}s, "
              f"d_OBB={p['d_obb']:.3f} m, d_center={p['d_center']:.3f} m")
        cmd = input("Visualize this pair? [Enter=yes, n=next, p=prev, j=jump, q=quit]: ").strip().lower()
        if cmd == "q":
            break
        if cmd == "n":
            idx = (idx + 1) % N
            continue
        if cmd == "p":
            idx = (idx - 1) % N
            continue
        if cmd == "j":
            try:
                j = int(input(f"  Jump to index (0..{N-1}): ").strip())
                if 0 <= j < N:
                    idx = j
                else:
                    print("  Out of range.")
            except Exception:
                print("  Invalid index.")
            continue

        trajs, t0 = collect_trajectories(p["file"])
        visualize_pair(p["file"], trajs, t0, p, idx)
        idx = (idx + 1) % N

if __name__ == "__main__":
    main()
