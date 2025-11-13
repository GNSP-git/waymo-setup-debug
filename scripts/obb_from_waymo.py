import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
import obb_distance
import itertools, time, inspect, os, glob, sys

# --- Detect where Label enum lives (SDK version-independent) ---
Label = None
for name, obj in inspect.getmembers(open_dataset):
    if name.lower().endswith("label"):
        Label = obj
        print(f"✅ Found label class as open_dataset.{name}")
        break
if Label is None:
    raise ImportError("Could not locate Label class in waymo_open_dataset.dataset_pb2")

# --- Build type map (manual override for safety) ---
TYPE_MAP = {1: "Vehicle", 2: "Pedestrian", 3: "Sign", 4: "Cyclist", 5: "Unknown"}

# --- Input directory (same as FILE before) ---
DATA_DIR = "/mnt/n/waymo_comma/waymo/"
tfrecord_files = sorted(glob.glob(os.path.join(DATA_DIR, "individual_files_*_with_camera_labels.tfrecord")))

if not tfrecord_files:
    print(f"❌ No Waymo TFRecord files found in {DATA_DIR}")
    sys.exit(1)

print(f"▶ Found {len(tfrecord_files)} TFRecord files to process.\n")


def extract_obbs(frame):
    """Extract Oriented Bounding Boxes (OBBs) from a Waymo Frame."""
    obbs = []
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
            "type": TYPE_MAP.get(label.type, f"Type{label.type}")
        })
    return obbs


def compute_pairwise_distances(obbs):
    """Compute pairwise OBB distances using compiled obb_distance extension."""
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


def analyze_file(filepath):
    """Process a single Waymo TFRecord file."""
    try:
        ds = tf.data.TFRecordDataset(filepath, compression_type="")
        raw = next(iter(ds.take(1)))  # process first frame only
        frame = open_dataset.Frame()
        frame.ParseFromString(raw.numpy())
        obbs = extract_obbs(frame)

        print(f"\n▶ Processing {os.path.basename(filepath)} ...")
        print(f"  Frame timestamp: {frame.timestamp_micros}")
        print(f"  Extracted {len(obbs)} OBBs")

        if not obbs:
            print("  ⚠️ No laser_labels found — skipping.")
            return

        counts = {}
        for o in obbs:
            counts[o["type"]] = counts.get(o["type"], 0) + 1
        print("  Object type counts:", counts)

        start = time.perf_counter()
        pairs = compute_pairwise_distances(obbs)
        elapsed = time.perf_counter() - start

        if not pairs:
            print("  ⚠️ No pairs to compute distances.")
            return

        dists = np.array([p[2] for p in pairs])
        print(f"  Pairwise distances: {len(pairs)} pairs in {elapsed:.4f} s")
        print(f"  Min={dists.min():.2f} m, Mean={dists.mean():.2f} m, Max={dists.max():.2f} m")

        # Top 5 closest pairs
        print("  Closest pairs:")
        for idx in np.argsort(dists)[:5]:
            i, j, d = pairs[idx]
            print(f"    {obbs[i]['type']}–{obbs[j]['type']}: {d:.2f} m")

    except Exception as e:
        print(f"  ❌ Error processing {filepath}: {e}")


def main():
    print(f"TensorFlow {tf.__version__} | Processing {len(tfrecord_files)} files")
    for f in tfrecord_files:
        analyze_file(f)


if __name__ == "__main__":
    main()
