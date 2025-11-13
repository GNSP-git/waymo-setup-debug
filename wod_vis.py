import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
import matplotlib.pyplot as plt
import numpy as np

def safe_parse_range_image_and_camera_projection(frame):
    """Patch Waymo SDK bytearray issue for WSL + TF2.13"""
    range_images = {}
    camera_projections = {}
    segmentation_labels = {}

    for laser in frame.lasers:
        if len(laser.ri_return1.range_image_compressed) > 0:
            ri = open_dataset.MatrixFloat()
            ri.ParseFromString(laser.ri_return1.range_image_compressed)
            range_images[(laser.name, 0)] = ri

        if len(laser.ri_return2.range_image_compressed) > 0:
            ri = open_dataset.MatrixFloat()
            ri.ParseFromString(laser.ri_return2.range_image_compressed)
            range_images[(laser.name, 1)] = ri

    # Convert to numpy points (standard Waymo util)
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame, range_images, {})
    points_all = np.concatenate(points, axis=0)

    return points_all


FILE = "/mnt/n/waymo_comma/waymo/individual_files_validation_segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord"
dataset = tf.data.TFRecordDataset(FILE, compression_type="")

for raw in dataset.take(1):
    frame = open_dataset.Frame()
    frame.ParseFromString(raw.numpy())

    # --- Camera image ---
    img = tf.image.decode_jpeg(frame.images[0].image).numpy()
    plt.figure(figsize=(8,6))
    plt.title("Front Camera")
    plt.imshow(img)
    plt.axis('off')

    # --- LiDAR BEV (patched for WSL + TF2.13) ---
    points = safe_parse_range_image_and_camera_projection(frame)
    pts = np.concatenate(points, axis=0)
    xs, ys, zs  = pts[:,0],pts[:,1],pts[:,2]

    plt.figure(figsize=(6,6))
    plt.scatter(xs, ys, s=1)
    plt.title("LiDAR BEV")
    plt.xlabel("X meters")
    plt.ylabel("Y meters")
    plt.show()
