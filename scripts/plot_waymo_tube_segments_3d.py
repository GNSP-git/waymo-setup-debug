# plot_waymo_tube_segments_3d.py
# Uses only Waymo trajectories saved earlier (no synthetic paths).
# Shows piecewise tube segments for a curved track + a real crossing track that
# intersects in XY but clears in space-time. Saves figures and supports z/Z zoom.

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

DATA_DIR = "/home/gns/waymo_work/data/samples"
NPZ_PATH = os.path.join(DATA_DIR, "trajectories_preview.npz")
OUT_2D = os.path.join(DATA_DIR, "tube_segments_xy.png")
OUT_3D = os.path.join(DATA_DIR, "tube_segments_space_time.png")

# ---------- helpers to load and normalize ----------
def load_trajectories(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "trajectories" in data.files:
        trajs = data["trajectories"].item()
    else:
        trajs = {k: data[k] for k in data.files}
    # Ensure arrays are Nx3: [t, x, y] (and drop extras if present)
    clean = {}
    for k, arr in trajs.items():
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] < 3:
            continue
        clean[k] = arr[:, :3]
    return clean

def to_seconds(ts):
    ts = np.asarray(ts)
    t0 = ts.min()
    span = ts.max() - ts.min()
    # Heuristic: microseconds if spans huge
    scale = 1e6 if ts.max() > 1e12 else 1.0
    return (ts - t0) / scale

# ---------- trajectory metrics & segmentation ----------
def heading_series(x, y):
    vx = np.gradient(x)
    vy = np.gradient(y)
    ang = np.arctan2(vy, vx)
    # unwrap to avoid jumps
    return np.unwrap(ang)

def cumulative_turn(x, y):
    ang = heading_series(x, y)
    return np.sum(np.abs(np.diff(ang)))

def piecewise_segments(t, x, y, max_turn=np.deg2rad(8), max_dv=2.0, max_dt=0.25):
    """
    Split trajectory when:
      - heading change since last cut > max_turn (radians),
      - speed change > max_dv (m/s) across a window,
      - time gap > max_dt (s).
    Returns list of (start_idx, end_idx) inclusive of start, exclusive of end.
    """
    n = len(t)
    if n < 3:
        return [(0, n)]
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    v = np.hypot(vx, vy)
    ang = np.unwrap(np.arctan2(vy, vx))
    segs = []
    s = 0
    last_ang = ang[0]
    last_v = v[0]
    last_t = t[0]

    for i in range(1, n):
        turn = abs(ang[i] - last_ang)
        dv = abs(v[i] - last_v)
        dt = t[i] - last_t
        if turn > max_turn or dv > max_dv or dt > max_dt:
            if i - s >= 2:
                segs.append((s, i))
            s = i
            last_ang = ang[i]
            last_v = v[i]
            last_t = t[i]
    if n - s >= 2:
        segs.append((s, n))
    return segs if segs else [(0, n)]

# ---------- tube (XY footprint + space-time prism) ----------
def segment_bounds(t, x, y, pad=1.0):
    """Axis-aligned bounds with padding (meters) and time span."""
    return (x.min()-pad, x.max()+pad, y.min()-pad, y.max()+pad, t.min(), t.max())

def prism_vertices(xmin, xmax, ymin, ymax, tmin, tmax):
    # bottom @ tmin, top @ tmax
    return np.array([
        [xmin, ymin, tmin], [xmax, ymin, tmin], [xmax, ymax, tmin], [xmin, ymax, tmin],
        [xmin, ymin, tmax], [xmax, ymin, tmax], [xmax, ymax, tmax], [xmin, ymax, tmax],
    ])

def prism_faces(verts):
    # return 6 faces as idx lists
    return [
        [verts[0], verts[1], verts[2], verts[3]],  # bottom
        [verts[4], verts[5], verts[6], verts[7]],  # top
        [verts[0], verts[1], verts[5], verts[4]],  # side
        [verts[1], verts[2], verts[6], verts[5]],
        [verts[2], verts[3], verts[7], verts[6]],
        [verts[3], verts[0], verts[4], verts[7]],
    ]

# ---------- pick tracks ----------
def pick_curved_track(trajs):
    # choose by max cumulative turn; prefer longer tracks
    best = None
    best_score = -1
    for k, arr in trajs.items():
        t, x, y = arr[:,0], arr[:,1], arr[:,2]
        if len(t) < 8:
            continue
        score = cumulative_turn(x, y) * np.log1p(len(t))
        if score > best_score:
            best_score = score
            best = k
    return best

def find_xy_cross_clear_spacetime(trajs, key_a, xy_thresh=2.0, min_time_sep=0.8):
    """
    Find another track whose XY approaches within xy_thresh (m),
    but whose minimum |tA - tB| when near is >= min_time_sep (s).
    """
    A = trajs[key_a]
    tA, xA, yA = A[:,0], A[:,1], A[:,2]
    for k, B in trajs.items():
        if k == key_a or len(B) < 4:
            continue
        tB, xB, yB = B[:,0], B[:,1], B[:,2]
        # downsample for speed
        idxA = np.arange(0, len(A), max(1, len(A)//200+1))
        idxB = np.arange(0, len(B), max(1, len(B)//200+1))
        near = []
        for i in idxA:
            dx = xB[idxB] - xA[i]
            dy = yB[idxB] - yA[i]
            d = np.hypot(dx, dy)
            jmin = np.argmin(d)
            if d[jmin] < xy_thresh:
                near.append((i, idxB[jmin]))
        if not near:
            continue
        # time separation at near points
        dt_min = np.inf
        for i, j in near:
            dt_min = min(dt_min, abs(tA[i] - tB[j]))
        if dt_min >= min_time_sep:
            return k
    return None

# ---------- main ----------
def main():
    trajs_raw = load_trajectories(NPZ_PATH)
    if not trajs_raw:
        print("No trajectories found.")
        return

    # normalize time to seconds (individually per track so each starts near 0)
    trajs = {}
    for k, arr in trajs_raw.items():
        t_sec = to_seconds(arr[:,0])
        trajs[k] = np.c_[t_sec, arr[:,1], arr[:,2]]

    key_main = pick_curved_track(trajs)
    if key_main is None:
        print("Could not find a curved track; picking arbitrary.")
        key_main = next(iter(trajs.keys()))

    key_cross = find_xy_cross_clear_spacetime(trajs, key_main, xy_thresh=2.0, min_time_sep=0.8)

    A = trajs[key_main]
    tA, xA, yA = A[:,0], A[:,1], A[:,2]
    segsA = piecewise_segments(tA, xA, yA, max_turn=np.deg2rad(8), max_dv=1.5, max_dt=0.25)

    B = trajs[key_cross] if key_cross else None
    if B is not None:
        tB, xB, yB = B[:,0], B[:,1], B[:,2]
        segsB = piecewise_segments(tB, xB, yB, max_turn=np.deg2rad(8), max_dv=1.5, max_dt=0.25)

    # ---------- 2D projection with footprints ----------
    fig2, ax2 = plt.subplots(figsize=(7.5, 7.5))
    ax2.set_title("Tube segments (XY projection) — real Waymo tracks")
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)"); ax2.grid(True, ls="--", alpha=0.4)

    ax2.plot(xA, yA, lw=2.5, label=f"Track A ({key_main[:6]})")
    for (s, e) in segsA:
        xmin, xmax, ymin, ymax, *_ = segment_bounds(tA[s:e], xA[s:e], yA[s:e], pad=1.0)
        ax2.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                    fill=True, alpha=0.15, edgecolor="none", facecolor="C0"))
        ax2.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                    fill=False, lw=1.0, edgecolor="C0"))

    if B is not None:
        ax2.plot(xB, yB, lw=2.5, label=f"Track B ({key_cross[:6]})", color="C3")
        for (s, e) in segsB:
            xmin, xmax, ymin, ymax, *_ = segment_bounds(tB[s:e], xB[s:e], yB[s:e], pad=1.0)
            ax2.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                        fill=True, alpha=0.15, edgecolor="none", facecolor="C3"))
            ax2.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                        fill=False, lw=1.0, edgecolor="C3"))

    ax2.legend(loc="best")
    ax2.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.pause(0.5)
    fig2.savefig(OUT_2D, dpi=140)

    # ---------- 3D space-time with prisms ----------
    fig3 = plt.figure(figsize=(9, 7))
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.set_title("Space–time tubes (real Waymo tracks)")
    ax3.set_xlabel("X (m)"); ax3.set_ylabel("Y (m)"); ax3.set_zlabel("Time (s)")
    ax3.grid(True)

    # thick trajectories
    ax3.plot(xA, yA, tA, lw=3.0, color="C0", label=f"Track A ({key_main[:6]})")
    if B is not None:
        ax3.plot(xB, yB, tB, lw=3.0, color="C3", label=f"Track B ({key_cross[:6]})")

    # draw prisms
    def add_tubes(ax, t, x, y, segs, color="C0"):
        for (s, e) in segs:
            xmin, xmax, ymin, ymax, tmin, tmax = segment_bounds(t[s:e], x[s:e], y[s:e], pad=1.0)
            V = prism_vertices(xmin, xmax, ymin, ymax, tmin, tmax)
            faces = prism_faces(V)
            coll = Poly3DCollection(faces, alpha=0.18, facecolor=color, edgecolor=color, linewidths=0.8)
            ax.add_collection3d(coll)

    add_tubes(ax3, tA, xA, yA, segsA, color="C0")
    if B is not None:
        add_tubes(ax3, tB, xB, yB, segsB, color="C3")

    ax3.legend(loc="upper left")

    # keyboard zoom handler
    def on_key(event):
        ax = event.canvas.figure.axes[0]
        if event.key == 'z':   # zoom in
            for setter, getter in ((ax.set_xlim3d, ax.get_xlim3d),
                                   (ax.set_ylim3d, ax.get_ylim3d),
                                   (ax.set_zlim3d, ax.get_zlim3d)):
                lo, hi = getter()
                ctr = 0.5*(lo+hi); rng = (hi-lo)*0.9*0.5
                setter(ctr-rng, ctr+rng)
        elif event.key == 'Z': # zoom out
            for setter, getter in ((ax.set_xlim3d, ax.get_xlim3d),
                                   (ax.set_ylim3d, ax.get_ylim3d),
                                   (ax.set_zlim3d, ax.get_zlim3d)):
                lo, hi = getter()
                ctr = 0.5*(lo+hi); rng = (hi-lo)/0.9*0.5
                setter(ctr-rng, ctr+rng)
        event.canvas.draw_idle()

    fig3.canvas.mpl_connect('key_press_event', on_key)

    plt.tight_layout()
    plt.pause(0.5)
    fig3.savefig(OUT_3D, dpi=140)

    # keep windows open for inspection
    plt.show()

if __name__ == "__main__":
    main()
