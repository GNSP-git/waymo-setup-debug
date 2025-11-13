# Waymo Setup & Debug Environment

This repository tracks a stable, verified environment and helper scripts for
loading, validating, and visualizing data from the Waymo Open Dataset (WOD).

## ✅ Current Status

- ✔️ Working conda env for TensorFlow 2.11 + Waymo 1.6.x
- ✔️ Successful parsing of `.tfrecord` frames
- ✔️ Camera + LiDAR data decode verified
- ✔️ Correct label counts (19–29 objects per frame typical)
- ✔️ Scripts saved for reproducibility

This repo intentionally **does not contain dataset files**.  
Download data via the official Waymo portal.

## 🧪 Verified Environment

Environment saved at:

envs/environment_waymo_tf211.yml

Recreate it via:

```bash
conda env create -f envs/environment_waymo_tf211.yml
conda activate waymo_tf211

waymo_work/
 ├── envs/
 ├── scripts/
 ├── wod_frame_quickcheck.py     # Frame + label sanity check
 ├── wod_vis.py                  # Basic lidar & camera visualization
 ├── waymo_env.yml               # Earlier environment freeze
 └── README.md

⛔ Not Included

TFRecord dataset files

Conda env directories

Waymo SDK local clones (redundant)

Large binaries

📍 Next Steps

Add trajectory extraction

Visualize 2D + 3D bounding boxes

Build export to numpy / parquet

Prepare pipeline for Ganaka accelerator comparisons

