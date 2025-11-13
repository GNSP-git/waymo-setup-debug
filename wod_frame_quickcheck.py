from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import scenario_pb2
import tensorflow as tf

fname = "/mnt/n/waymo_comma/waymo/test_202504211836-202504220845.tfrecord-00020-of-00266"
ds = tf.data.TFRecordDataset(fname)

for i, r in enumerate(ds.take(20)):
    b = bytes(r.numpy())
    
    # Try Frame
    f = open_dataset.Frame()
    try:
        f.ParseFromString(b)
        if len(f.images) > 0 or len(f.lasers) > 0:
            print(f"[{i}] ✅ FRAME | cams={len(f.images)}, lidars={len(f.lasers)}")
            continue
    except:
        pass

    # Try Scenario
    s = scenario_pb2.Scenario()
    try:
        s.ParseFromString(b)
        if len(s.tracks) > 0:
            print(f"[{i}] ✅ SCENARIO | agents={len(s.tracks)}, steps={len(s.timestamps_seconds)}")
        else:
            print(f"[{i}] ⚠ Scenario with 0 agents, steps={len(s.timestamps_seconds)} (skipped)")
        continue
    except:
        pass

    print(f"[{i}] ❓ Unknown record")
