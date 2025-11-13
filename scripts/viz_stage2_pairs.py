# viz_stage2_pairs.py
# Fixed red-circle overlay for the closest pair and verified coordinate scaling.

import os, csv, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SAVE_DIR = "/home/gns/waymo_work/data/samples"
TRAJ_CSV = os.path.join(SAVE_DIR, "trajectories_all.csv")
PAIRS_CSV = os.path.join(SAVE_DIR, "workload_pairs_min.csv")

DIST_SHOW = 10.0
CIRCLE_RADIUS = 5.0

colors = {
    "Vehicle": "tab:blue",
    "Pedestrian": "tab:orange",
    "Cyclist": "tab:green",
    "Sign": "tab:purple",
    "Unknown": "gray",
}

def load_traj_csv(path):
    data = {}
    min_time = 1e18
    with open(path, "r") as f:
        r = csv.reader(f)
        hdr = next(r)
        idx = {h:i for i,h in enumerate(hdr)}
        for row in r:
            fid = row[idx["file"]]; oid = row[idx["oid"]]
            t = float(row[idx["t"]])
            entry = {
                "t": t,
                "cx": float(row[idx["cx"]]),
                "cy": float(row[idx["cy"]]),
                "cz": float(row[idx["cz"]]),
                "heading": float(row[idx["heading"]]),
                "L": float(row[idx["length"]]),
                "W": float(row[idx["width"]]),
                "type": row[idx["type"]],
            }
            data.setdefault(fid, {}).setdefault(oid, []).append(entry)
            if t < min_time: min_time = t
    for f in data:
        for oid in data[f]:
            for r in data[f][oid]:
                r["t"] -= min_time
            data[f][oid].sort(key=lambda r: r["t"])
    print(f"⏱ Subtracted base time = {min_time:.0f}")
    return data

def load_pairs_csv(path):
    pairs = []
    with open(path, "r") as f:
        r = csv.reader(f)
        hdr = next(r)
        idx = {h:i for i,h in enumerate(hdr)}
        for row in r:
            dmin = float(row[idx["d_obb_min"]])
            if dmin <= DIST_SHOW:
                pairs.append({
                    "file": row[idx["file"]],
                    "oid_i": row[idx["oid_i"]],
                    "oid_j": row[idx["oid_j"]],
                    "t": float(row[idx["t_at"]]),
                    "type_i": row[idx["type_i"]],
                    "type_j": row[idx["type_j"]],
                    "dmin": dmin,
                })
    return pairs

def obb_polygon(cx, cy, L, W, heading):
    c, s = math.cos(heading), math.sin(heading)
    dx, dy = L/2, W/2
    corners = np.array([[ dx,dy],[ dx,-dy],[-dx,-dy],[-dx,dy]])
    rot = np.array([[c,-s],[s,c]])
    return (rot @ corners.T).T + np.array([cx,cy])

def draw_frame(ax, trajs, t, highlight_pair=None):
    ax.clear()
    ax.set_title(f"t = {t:.2f} s")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")

    all_x, all_y = [], []
    for oid, arr in trajs.items():
        rec = min(arr, key=lambda r: abs(r["t"]-t))
        pts = obb_polygon(rec["cx"], rec["cy"], rec["L"], rec["W"], rec["heading"])
        all_x.extend(pts[:,0]); all_y.extend(pts[:,1])
        poly = Polygon(pts, closed=True, facecolor=colors.get(rec["type"],"gray"), alpha=0.4)
        ax.add_patch(poly)
        ax.text(rec["cx"], rec["cy"], rec["type"][:1], fontsize=7, ha='center')

    if highlight_pair:
        (x1,y1),(x2,y2)=highlight_pair
        midx, midy = (x1+x2)/2, (y1+y2)/2
        circ = Circle((midx, midy), radius=CIRCLE_RADIUS, edgecolor="red", facecolor="none", lw=2)
        ax.add_patch(circ)
        ax.plot([x1,x2],[y1,y2],"r--",lw=1)
        print(f"🔴 Highlight circle at ({midx:.2f},{midy:.2f}) radius={CIRCLE_RADIUS}")

        all_x += [x1,x2,midx]; all_y += [y1,y2,midy]

    # autoscale dynamically
    if all_x and all_y:
        xmin, xmax = min(all_x)-10, max(all_x)+10
        ymin, ymax = min(all_y)-10, max(all_y)+10
        ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
    else:
        ax.set_xlim(-50,50); ax.set_ylim(-50,50)
    ax.grid(True)

def plot_spacetime(ax, trajs, oids):
    for oid in oids:
        arr = trajs.get(oid, [])
        if len(arr)<2: continue
        xs=[r["cx"] for r in arr]; ys=[r["cy"] for r in arr]; ts=[r["t"] for r in arr]
        ax.plot(xs, ys, ts, label=f"{oid[:6]}", linewidth=2)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("t (s)")
    ax.legend()

def main():
    trajs_all = load_traj_csv(TRAJ_CSV)
    pairs = load_pairs_csv(PAIRS_CSV)
    if not pairs:
        print("No close pairs below threshold.")
        return

    p = pairs[0]
    fid, t = p["file"], p["t"]
    trajs = trajs_all[fid]
    base_t0 = min(r["t"] for arr in trajs.values() for r in arr)
    t -= base_t0

    a = min(trajs[p["oid_i"]], key=lambda r: abs(r["t"]-t))
    b = min(trajs[p["oid_j"]], key=lambda r: abs(r["t"]-t))
    print(f"Plotting {p['oid_i']}–{p['oid_j']} at {t:.3f}s (d={p['dmin']:.2f} m)")

    fig, ax = plt.subplots(figsize=(8,8))
    draw_frame(ax, trajs, t, highlight_pair=((a["cx"],a["cy"]),(b["cx"],b["cy"])))
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR,"xy_view.png"), dpi=150)
    print("✅ Saved XY view → xy_view.png")

    fig = plt.figure(figsize=(8,6))
    ax3 = fig.add_subplot(111, projection='3d')
    plot_spacetime(ax3, trajs, [p["oid_i"], p["oid_j"]])
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR,"spacetime_view.png"), dpi=150)
    print("✅ Saved space–time view → spacetime_view.png")
    plt.show()

if __name__ == "__main__":
    main()
