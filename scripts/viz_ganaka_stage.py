# viz_ganaka_stage.py
# Unified close-approach visualization with multiple-pair browsing and walkaround mode.

import os, math, itertools, time, inspect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as open_dataset

# ---------------- CONFIG ----------------
FILE = "/mnt/n/waymo_comma/waymo/individual_files_training_segment-10107710434105775874_760_000_780_000_with_camera_labels.tfrecord"
SAVE_DIR = "/home/gns/waymo_work/data/samples"
os.makedirs(SAVE_DIR, exist_ok=True)

DIST_THRESH = 6.0
TIME_THRESH = 0.1
MAX_PAIRS = 20

# OBB distance availability
USE_OBB = True
try:
    import obb_distance
except Exception:
    USE_OBB = False
    print("⚠️ Using fallback center distance.")

# -------- Label enum discovery --------
Label = None
for name, obj in inspect.getmembers(open_dataset):
    if name.lower().endswith("label"):
        Label = obj
        break
if Label is None:
    raise ImportError("Cannot locate Label class")

TYPE_NAMES = {getattr(Label, a): a.replace("TYPE_", "").title()
              for a in dir(Label) if a.startswith("TYPE_")}

DEFAULT_DIMS = {
    "Vehicle": (4.5,1.8,1.6),
    "Pedestrian": (0.6,0.6,1.7),
    "Cyclist": (1.8,0.6,1.6),
    "Sign": (1.0,0.5,3.0),
    "Unknown": (1.0,1.0,1.0),
}

CLASS_COLOR = {
    "Vehicle": "#1f77b4",
    "Pedestrian": "#2ca02c",
    "Cyclist": "#ff7f0e",
    "Sign": "#9467bd",
    "Unknown": "#7f7f7f",
}

# ------------- helpers --------------

def label_type_name(t):
    nm = TYPE_NAMES.get(t, "Unknown")
    if nm.lower().startswith("vehicle"): return "Vehicle"
    if nm.lower().startswith("pedestrian"): return "Pedestrian"
    if nm.lower().startswith("cyclist"): return "Cyclist"
    if nm.lower().startswith("sign"): return "Sign"
    return nm

def ensure_dims(name, L, W, H):
    if L>0 and W>0 and H>0:
        return L,W,H
    return DEFAULT_DIMS.get(name, DEFAULT_DIMS["Unknown"])

def obb_rect_xy(cx,cy,L,W,heading):
    hl, hw = L/2, W/2
    base = np.array([
        [-hl,-hw], [hl,-hw], [hl,hw], [-hl,hw]
    ],dtype=np.float32)
    c,s = math.cos(heading), math.sin(heading)
    R = np.array([[c,-s],[s,c]])
    pts = base @ R.T
    pts[:,0]+=cx; pts[:,1]+=cy
    return pts

def pair_distance(a,b):
    if USE_OBB:
        return obb_distance.obb_min_distance(
            a["cx"], a["cy"], a["length"], a["width"], a["heading"],
            b["cx"], b["cy"], b["length"], b["width"], b["heading"])
    return math.hypot(a["cx"]-b["cx"], a["cy"]-b["cy"])

# -------- trajectories ----------
def collect_trajectories(fp):
    trajs={}
    t0=None
    for raw in tf.data.TFRecordDataset(fp):
        fr = open_dataset.Frame()
        fr.ParseFromString(raw.numpy())
        t = fr.timestamp_micros *1e-6
        if t0 is None: t0=t
        for lab in fr.laser_labels:
            oid = lab.id
            nm = label_type_name(lab.type)
            box = lab.box
            L,W,H = ensure_dims(nm, box.length, box.width, box.height)
            rec = dict(
                t=t, cx=box.center_x, cy=box.center_y,
                heading=box.heading,
                length=L, width=W, height=H,
                type=nm, id=oid
            )
            trajs.setdefault(oid,[]).append(rec)
    for oid in trajs:
        trajs[oid].sort(key=lambda r:r["t"])
    return trajs,t0

# -------- close pairs -----------
def find_pairs(trajs, dthresh, tthresh, maxp):
    times = sorted({r["t"] for arr in trajs.values() for r in arr})
    indexed=[]
    half = tthresh/2
    for t in times:
        snap=[]
        for oid,arr in trajs.items():
            best=None;bdt=1e9
            for r in arr:
                dt=abs(r["t"]-t)
                if dt<bdt and dt<=tthresh:#changed from half
                    bdt=dt; best=r
            if best: snap.append(best)
        indexed.append((t,snap))

    best={}
    for t,arr in indexed:
        # only Vehicle–Vehicle
        vehs=[r for r in arr if r["type"].lower().startswith("Vehicle")]
        for a,b in itertools.combinations(vehs,2):
            d = pair_distance(a,b)
            if d<dthresh:
                key=tuple(sorted((a["id"],b["id"])))
                old=best.get(key)
                if old is None or d<old[0]:
                    best[key]=(d,t,a,b)

    ranked=sorted(
        (dict(oid_i=k[0],oid_j=k[1],d_min=v[0],t_at=v[1],
              rec_i=v[2],rec_j=v[3]) for k,v in best.items()),
        key=lambda x:x["d_min"]
    )
    return ranked[:maxp]

# ------------- plotting -------------------

def draw_xy(ax,trajs,pair,t0):
    rec_i,rec_j = pair["rec_i"], pair["rec_j"]
    for rec,lab in ((rec_i,"A"),(rec_j,"B")):
        full=trajs[rec["id"]]
        X=[r["cx"] for r in full]
        Y=[r["cy"] for r in full]
        col = CLASS_COLOR.get(rec["type"],"#888")
        ax.plot(X,Y,'-',lw=2.2,color=col,label=f"{lab}: {rec['type']}")

        poly=obb_rect_xy(rec["cx"],rec["cy"],rec["length"],rec["width"],rec["heading"])
        ax.add_patch(Polygon(poly,closed=True,fc=col,ec=col,alpha=0.25,lw=1.4))

        ax.quiver(rec["cx"],rec["cy"],math.cos(rec["heading"]),math.sin(rec["heading"]),
                  angles='xy',scale_units='xy',scale=1.5,color=col,width=0.004)

        ax.text(rec["cx"],rec["cy"],lab,color=col,fontsize=11,weight="bold")

    # red circle around closest pair
    mx = 0.5*(rec_i["cx"]+rec_j["cx"])
    my = 0.5*(rec_i["cy"]+rec_j["cy"])
    circ = Circle((mx,my), 3.0, fc='none', ec='red', lw=2.0)
    ax.add_patch(circ)

    ax.set_aspect("equal")
    ax.set_title(f"XY — d_min={pair['d_min']:.2f} m,  t={pair['t_at']-t0:.2f}s")
    ax.grid(True,ls='--',alpha=0.4)
    ax.legend()

def draw_spacetime(ax3d,trajs,pair,t0):
    for rec,lab in ((pair["rec_i"],"A"),(pair["rec_j"],"B")):
        full=trajs[rec["id"]]
        col = CLASS_COLOR.get(rec["type"],"#888")
        X=np.array([r["cx"] for r in full])
        Y=np.array([r["cy"] for r in full])
        T=np.array([r["t"]-t0 for r in full])
        ax3d.plot3D(X,Y,T,color=col,lw=2,label=f"{lab}: {rec['type']}")

        # small spacetime blocks
        N=max(1,len(full)//40)
        blocks=[]
        for r in full[::N]:
            poly=obb_rect_xy(r["cx"],r["cy"],r["length"],r["width"],r["heading"])
            z0=(r["t"]-t0)-0.03
            z1=(r["t"]-t0)+0.03
            bottom=[(poly[k,0],poly[k,1],z0) for k in range(4)]
            top=[(poly[k,0],poly[k,1],z1) for k in range(4)]
            faces = [
                [bottom[0],bottom[1],bottom[2],bottom[3]],
                [top[0],top[1],top[2],top[3]],
                [bottom[0],bottom[1],top[1],top[0]],
                [bottom[1],bottom[2],top[2],top[1]],
                [bottom[2],bottom[3],top[3],top[2]],
                [bottom[3],bottom[0],top[0],top[3]]
            ]
            blocks.extend(faces)
        if blocks:
            ax3d.add_collection3d(
                Poly3DCollection(blocks,fc=col,ec=col,alpha=0.08,lw=0.2)
            )

    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("t (s)")
    ax3d.set_title("Spacetime Tubes")
    ax3d.legend()
    ax3d.grid(True)

# ------------- walkthrough (keyboard) ---------------

class Viewer:
    def __init__(self,trajs,t0,pairs):
        self.trajs=trajs
        self.t0=t0
        self.pairs=pairs
        self.k=0
        self.fig_xy = None
        self.fig_3d = None
        self.rotating=False

    def show_pair(self):
        pair=self.pairs[self.k]
        # 2D
        if self.fig_xy: plt.close(self.fig_xy)
        self.fig_xy,ax=plt.subplots(figsize=(8,7))
        draw_xy(ax,self.trajs,pair,self.t0)
        self.fig_xy.canvas.mpl_connect('key_press_event', self.on_key)

        # 3D
        if self.fig_3d: plt.close(self.fig_3d)
        self.fig_3d=plt.figure(figsize=(9,7))
        ax3d=self.fig_3d.add_subplot(111,projection='3d')
        draw_spacetime(ax3d,self.trajs,pair,self.t0)
        self.fig_3d.canvas.mpl_connect('key_press_event', self.on_key)

        plt.show(block=False)

    def rotate_step(self):
        if not self.rotating: return
        ax = self.fig_3d.axes[0]
        az = ax.azim + 1
        el = ax.elev + 0.2
        ax.view_init(elev=el, azim=az)
        self.fig_3d.canvas.draw_idle()
        self.fig_3d.canvas.flush_events()
        self.fig_3d.canvas.new_timer(interval=40,
              callbacks=[(self.rotate_step,(),{})]).start()

    def on_key(self,event):
        if event.key=='n':
            self.k = (self.k+1)%len(self.pairs)
            self.show_pair()
        elif event.key=='p':
            self.k = (self.k-1)%len(self.pairs)
            self.show_pair()
        elif event.key=='w': # walkaround
            self.rotating=True
            self.rotate_step()
        elif event.key=='s':
            self.rotating=False
        elif event.key=='q':
            plt.close('all')

def main():
    print("Loading trajectories...")
    trajs,t0 = collect_trajectories(FILE)
    pairs = find_pairs(trajs,DIST_THRESH,TIME_THRESH,MAX_PAIRS)
    if not pairs:
        print("No pairs found.")
        return

    print(f"Found {len(pairs)} close Vehicle–Vehicle encounters.")
    V = Viewer(trajs,t0,pairs)
    V.show_pair()
    plt.show()

if __name__=="__main__":
    print("TF =", tf.__version__)
    main()
