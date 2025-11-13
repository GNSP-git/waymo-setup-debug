# plot_traj_colored.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

data = np.load("../data/samples/trajectories_preview.npz", allow_pickle=True)

plt.figure(figsize=(8, 8))
shown = 0
auto_scale = True

# Collect lengths for color normalization
lengths = {}
for oid, arr in data.items():
    traj = np.asarray(arr)
    if traj.ndim == 2 and traj.shape[1] >= 3:
        mask = np.isfinite(traj[:, 1]) & np.isfinite(traj[:, 2])
        lengths[oid] = np.count_nonzero(mask)
max_len = max(lengths.values()) if lengths else 1
norm = plt.Normalize(vmin=0, vmax=max_len)
cmap = plt.get_cmap("plasma")

for oid, arr in data.items():
    traj = np.asarray(arr)
    if traj.ndim != 2 or traj.shape[1] < 3 or traj.shape[0] < 2:
        continue

    x, y = traj[:, 1], traj[:, 2]
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        continue

    if auto_scale:
        max_abs = max(np.max(np.abs(x)), np.max(np.abs(y)))
        if max_abs > 1e9:
            x, y = x * 1e-9, y * 1e-9
            unit = "km"
        elif max_abs > 1e6:
            x, y = x * 1e-6, y * 1e-6
            unit = "m"
        else:
            unit = "raw"
    else:
        unit = "raw"

    color = cmap(norm(lengths.get(oid, 0)))
    plt.plot(x, y, '-', color=color, alpha=0.8, linewidth=1)
    shown += 1
    if shown >= 50:
        break

sm = ScalarMappable(norm=norm, cmap=cmap)
cbar = plt.colorbar(sm, label="Trajectory length (#points)")

plt.xlabel(f"X ({unit})")
plt.ylabel(f"Y ({unit})")
plt.title(f"Trajectories — {shown} shown (colored by length)")
plt.axis("equal")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
