import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib import cm

# === CONFIG ===
SAVE_PATH = "/home/gns/waymo_work/data/samples/trajectory_tubes_3d.png"
TUBE_RADIUS = 0.9
VEH_LENGTH, VEH_WIDTH = 4.5, 1.9
TYPE1, TYPE2 = "Vehicle A", "Vehicle B"

# ---------------------------------------------------------------------
def make_obb(cx, cy, heading, length, width):
    """Return (4,2) array of OBB corners in XY plane."""
    hl, hw = length / 2, width / 2
    c, s = np.cos(heading), np.sin(heading)
    corners = np.array([
        [ hl,  hw],
        [ hl, -hw],
        [-hl, -hw],
        [-hl,  hw]
    ])
    rot = np.array([[c, -s], [s, c]])
    return corners @ rot.T + np.array([cx, cy])

def make_space_time_box(cx, cy, heading, length, width, t1, t2, dz=0.05):
    base = make_obb(cx, cy, heading, length, width)
    bottom = np.c_[base, np.full(4, t1 - dz)]
    top = np.c_[base, np.full(4, t2 + dz)]
    return np.vstack((bottom, top))

def make_vehicle_trajectory(offset_y=0.0, delay_t=0.0, invert=False):
    """Generate a smooth trajectory that curves and optionally intersects in XY."""
    t = np.linspace(0, 10, 100)
    x = 20 * np.sin(0.2 * t)
    y = 10 * np.cos(0.2 * t) + offset_y
    if invert:
        x = -x
    t = t + delay_t
    return np.stack([t, x, y], axis=1)

# ---------------------------------------------------------------------
def plot_vehicle(ax, traj, cmap, label, alpha_vol=0.2):
    """Plot 3D trajectory with velocity-based color coding and arrows."""
    t, x, y = traj[:, 0], traj[:, 1], traj[:, 2]
    vx, vy = np.gradient(x, t), np.gradient(y, t)
    speed = np.hypot(vx, vy)
    norm = plt.Normalize(speed.min(), speed.max())
    colors = cmap(norm(speed))

    # Centerline
    for i in range(len(x) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], [t[i], t[i + 1]],
                color=colors[i], lw=3.0)

    # Velocity arrows
    step = max(1, len(x) // 20)
    for i in range(0, len(x), step):
        ax.quiver(x[i], y[i], t[i],
                  vx[i], vy[i], 0.1, color=colors[i],
                  length=1.0, normalize=True, linewidth=0.6)

    # Space-time OBB volumes
    for i in range(len(x) - 1):
        heading = np.arctan2(y[i + 1] - y[i], x[i + 1] - x[i])
        box = make_space_time_box(x[i], y[i], heading,
                                  VEH_LENGTH, VEH_WIDTH, t[i], t[i + 1])
        faces = [
            [box[j] for j in [0,1,2,3]],
            [box[j] for j in [4,5,6,7]],
            [box[j] for j in [0,1,5,4]],
            [box[j] for j in [1,2,6,5]],
            [box[j] for j in [2,3,7,6]],
            [box[j] for j in [3,0,4,7]],
        ]
        poly = Poly3DCollection(faces, facecolor=colors[i], alpha=alpha_vol, edgecolor='k', linewidths=0.3)
        ax.add_collection3d(poly)

    # XY projection
    ax.plot(x, y, np.zeros_like(t), color='gray', lw=2.0, alpha=0.6)
    ax.text(x[0], y[0], t[0] + 0.5, f"{label}", color='blue', fontsize=10, weight='bold')

# ---------------------------------------------------------------------
def main():
    traj1 = make_vehicle_trajectory(offset_y=0.0, delay_t=0.0)
    traj2 = make_vehicle_trajectory(offset_y=3.0, delay_t=1.0, invert=True)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    cmap = cm.viridis
    plot_vehicle(ax, traj1, cmap, TYPE1)
    plot_vehicle(ax, traj2, cm.plasma, TYPE2)

    # Colorbar legend for speed
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 25))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.05, shrink=0.6)
    cbar.set_label("Speed [m/s]", fontsize=10)

    # Labels and appearance
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Time [s]")
    ax.set_title("3D Space-Time OBB Tubes with Velocity Arrows & XY Projection")
    ax.view_init(elev=28, azim=-60)
    ax.grid(True)
    plt.tight_layout()

    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot to {SAVE_PATH}")

    def on_key(event):
        ax = event.canvas.figure.axes[0]
        if event.key == 'z':       # zoom in
            ax.set_xlim3d(ax.get_xlim3d()[0]*0.9, ax.get_xlim3d()[1]*0.9)
            ax.set_ylim3d(ax.get_ylim3d()[0]*0.9, ax.get_ylim3d()[1]*0.9)
            ax.set_zlim3d(ax.get_zlim3d()[0]*0.9, ax.get_zlim3d()[1]*0.9)
        elif event.key == 'Z':     # zoom out
            ax.set_xlim3d(ax.get_xlim3d()[0]*1.1, ax.get_xlim3d()[1]*1.1)
            ax.set_ylim3d(ax.get_ylim3d()[0]*1.1, ax.get_ylim3d()[1]*1.1)
            ax.set_zlim3d(ax.get_zlim3d()[0]*1.1, ax.get_zlim3d()[1]*1.1)
            event.canvas.draw_idle()
            
            fig.canvas.mpl_connect('key_press_event', on_key)
            
    plt.show(block=False)
    plt.pause(20)
    plt.close()

if __name__ == "__main__":
    main()
