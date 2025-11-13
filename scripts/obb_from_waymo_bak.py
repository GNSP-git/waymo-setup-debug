import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
import obb_distance
import itertools, time, inspect

# --- Detect where Label enum lives (SDK version-independent) ---
Label = None
for name, obj in inspect.getmembers(open_dataset):
    if name.lower().endswith("label"):
        Label = obj
        print(f"✅ Found label class as open_dataset.{name}")
        break
if Label is None:
    raise ImportError("Could not locate Label class in waymo_open_dataset.dataset_pb2")

# === Input file ===
FILE = "/mnt/n/waymo_comma/waymo/individual_files_validation_segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord"

# --- Safely build type mapping ---
TYPE_NAMES = {}
for attr in dir(Label):
    if attr.startswith("TYPE_"):
        TYPE_NAMES[getattr(Label, attr)] = attr.replace("TYPE_", "").title()

print("✅ Label type map:", TYPE_NAMES)


def extract_obbs(frame):
    obbs = []
    type_map = {1: "Vehicle", 2: "Pedestrian", 3: "Sign", 4: "Cyclist", 5: "Unknown"}
    for label in frame.laser_labels:
        box = label.box
        obbs.append({
            "cx": box.center_x,
            "cy": box.center_y,
            "cz": box.center_z,
            "length": box.length,
            "width": box.width,
            "height": box.height,
            "heading": box.heading,
            "type": type_map.get(label.type, f"Type{label.type}")
        })
    return obbs

def compute_pairwise_distances(obbs):
    n = len(obbs)
    results = []
    for i, j in itertools.combinations(range(n), 2):
        a, b = obbs[i], obbs[j]
        d = obb_distance.obb_min_distance(
            a["cx"], a["cy"], a["length"], a["width"], a["heading"],
            b["cx"], b["cy"], b["length"], b["width"], b["heading"]
        )
        results.append((i, j, d))
    return results

def main():
    ds = tf.data.TFRecordDataset(FILE, compression_type="")
    for raw in ds.take(1):
        frame = open_dataset.Frame()
        frame.ParseFromString(raw.numpy())
        obbs = extract_obbs(frame)
        print(f"Frame timestamp: {frame.timestamp_micros}")
        print(f"Extracted {len(obbs)} OBBs")

        # Summary by type
        counts = {}
        for o in obbs:
            counts[o["type"]] = counts.get(o["type"], 0) + 1
        print("Object type counts:", counts)

        # Pairwise OBB distances
        start = time.perf_counter()
        pairs = compute_pairwise_distances(obbs)
        elapsed = time.perf_counter() - start

        dists = np.array([p[2] for p in pairs])
        print(f"Pairwise distances computed: {len(pairs)} in {elapsed:.4f} s")
        print(f"Min={dists.min():.2f} m, Mean={dists.mean():.2f} m, Max={dists.max():.2f} m")

        # Closest pairs
        print("\nTop 10 closest pairs:")
        for idx in np.argsort(dists)[:10]:
            i, j, d = pairs[idx]
            print(f"  {obbs[i]['type']}–{obbs[j]['type']}: {d:.2f} m")

if __name__ == "__main__":
    main()
