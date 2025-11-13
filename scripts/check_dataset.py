from waymo_open_dataset import dataset_pb2 as open_dataset
import tensorflow as tf

FILE = "/mnt/n/waymo_comma/waymo/individual_files_validation_segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord"

dataset = tf.data.TFRecordDataset(FILE, compression_type="")

for raw in dataset.take(1):
    frame = open_dataset.Frame()
    frame.ParseFromString(raw.numpy())

    print("Timestamp:", frame.timestamp_micros)
    print("Cameras:", len(frame.images))
    print("LiDARs:", len(frame.lasers))

    # ✅ labels fix
    print("3D labels:", len(frame.laser_labels))
