#!/usr/bin/env python3
"""
Synthetic stress scenarios for Ganaka workload testing.

Generates trajectories in the SAME format as collect_trajectories(...)
from ganaka_workload_extractor.py:

  trajs: Dict[object_id, List[record]]
  record: {
      "t": float,          # seconds
      "cx": float, "cy": float,
      "heading": float,    # radians
      "length": float, "width": float, "height": float,
      "type": "Vehicle",
      "id": object_id,
      "raw_type_code": 1,
      "raw_type_name": "Vehicle"
  }

We implement:

  Scenario 1: Dense highway jam (N² FLOP explosion)
  Scenario 2: Multi-lane merge cascade (many near-misses)
"""

import math
import numpy as np
from typing import Dict, List, Tuple


# Common vehicle dims (approx)
DEFAULT_VEHICLE_L = 4.5
DEFAULT_VEHICLE_W = 1.8
DEFAULT_VEHICLE_H = 1.6


def _make_rec(t, x, y, heading, obj_id):
    """Helper to make one record with your standard fields."""
    return {
        "t": t,
        "cx": x,
        "cy": y,
        "heading": heading,
        "length": DEFAULT_VEHICLE_L,
        "width": DEFAULT_VEHICLE_W,
        "height": DEFAULT_VEHICLE_H,
        "type": "Vehicle",
        "id": obj_id,
        "raw_type_code": 1,        # arbitrary but consistent: "Vehicle"
        "raw_type_name": "Vehicle",
    }


# ============================================================
#  Scenario 1: Dense highway jam (FLOP explosion)
# ============================================================

def synth_dense_highway_jam(
    num_vehicles: int = 300,
    num_lanes: int = 4,
    lane_width: float = 3.7,
    duration: float = 20.0,
    dt: float = 0.1,
    mean_speed: float = 5.0,
    speed_jitter: float = 1.0,
    lateral_jitter: float = 0.4,
    seed: int = 42,
) -> Tuple[Dict[str, List[dict]], float]:
    """
    Scenario 1: High-density multi-lane highway segment.

    - Vehicles distributed across num_lanes lanes.
    - Longitudinal positions packed reasonably tightly.
    - Small speed & lateral jitter so trajectories evolve.
    - All headings are ~0 (along +X), with small noise.
    """

    rng = np.random.default_rng(seed)
    times = np.arange(0.0, duration, dt)
    trajs: Dict[str, List[dict]] = {}

    # Layout: initial x positions in a dense band, staggered by ~vehicle length
    # We make blocks of num_lanes vehicles repeatedly.
    base_spacing = DEFAULT_VEHICLE_L * 1.3  # ~ safe-ish distance
    num_blocks = math.ceil(num_vehicles / num_lanes)

    veh_index = 0
    for block in range(num_blocks):
        for lane in range(num_lanes):
            if veh_index >= num_vehicles:
                break

            obj_id = f"veh_{veh_index}"
            lane_center_y = lane * lane_width

            # Initial longitudinal position: packed blocks, with small noise
            x0 = -50.0 + block * base_spacing + rng.normal(0.0, 0.5)
            # Lateral small jitter around lane center
            y0 = lane_center_y + rng.normal(0.0, lateral_jitter)

            # Speeds: near mean_speed, small jitter
            v = max(0.5, rng.normal(mean_speed, speed_jitter))

            # Heading: mostly 0, slight noise
            heading0 = rng.normal(0.0, 0.02)

            recs: List[dict] = []
            for t in times:
                # Simple model: x(t) = x0 + v * t
                x = x0 + v * t
                # y(t): keep near lane center with small wander
                y = y0 + rng.normal(0.0, lateral_jitter * 0.05)

                recs.append(_make_rec(t, x, y, heading0, obj_id))

            trajs[obj_id] = recs
            veh_index += 1

    t0 = 0.0
    return trajs, t0


# ============================================================
#  Scenario 2: Multi-lane merge cascade (near-miss generator)
# ============================================================

def synth_multi_lane_merge(
    num_vehicles: int = 80,
    upstream_lanes: int = 5,
    downstream_lanes: int = 2,
    lane_width: float = 3.7,
    duration: float = 20.0,
    dt: float = 0.1,
    mean_speed: float = 10.0,
    speed_jitter: float = 1.5,
    merge_start_x: float = 0.0,
    merge_end_x: float = 120.0,
    lane_change_noise: float = 0.3,
    seed: int = 7,
) -> Tuple[Dict[str, List[dict]], float]:
    """
    Scenario 2: Upstream 5 lanes -> downstream 2 lanes over [merge_start_x, merge_end_x].

    - Vehicles start in 5 lanes, end in 2 lanes.
    - Each vehicle is assigned a downstream "target lane".
    - Between merge_start_x and merge_end_x, lateral position interpolates
      towards the target lane, creating many lane-change interactions.
    """

    assert downstream_lanes <= upstream_lanes

    rng = np.random.default_rng(seed)
    times = np.arange(0.0, duration, dt)
    trajs: Dict[str, List[dict]] = {}

    base_spacing = DEFAULT_VEHICLE_L * 2.0  # more generous than scenario 1; we want merges, not huge jam
    num_blocks = math.ceil(num_vehicles / upstream_lanes)

    # Downstream lane centers: we put them roughly in the middle of the upstream set
    offset = (upstream_lanes - downstream_lanes) * lane_width / 2.0
    downstream_lane_centers = [
        offset + i * lane_width for i in range(downstream_lanes)
    ]

    veh_index = 0
    for block in range(num_blocks):
        for lane in range(upstream_lanes):
            if veh_index >= num_vehicles:
                break

            obj_id = f"veh_merge_{veh_index}"

            # Upstream lane center
            y_up = lane * lane_width

            # Assign a downstream target lane (spread traffic)
            target_lane_idx = rng.integers(0, downstream_lanes)
            y_down = downstream_lane_centers[target_lane_idx]

            # Initial longitudinal position
            x0 = -40.0 + block * base_spacing + rng.normal(0.0, 0.5)
            v = max(2.0, rng.normal(mean_speed, speed_jitter))

            # Precompute lane-change shape as function of x
            def lane_y(x):
                """
                Piecewise:
                  x < merge_start_x   -> stay in upstream lane
                  x > merge_end_x     -> be at downstream lane
                  otherwise           -> smooth blend between them
                """
                if x <= merge_start_x:
                    return y_up
                if x >= merge_end_x:
                    return y_down
                # normalized progress in [0,1]
                u = (x - merge_start_x) / (merge_end_x - merge_start_x)
                # Smooth polynomial (ease-in-out)
                u_smooth = 3*u**2 - 2*u**3
                return (1 - u_smooth) * y_up + u_smooth * y_down

            recs: List[dict] = []
            for t in times:
                x = x0 + v * t
                y_nom = lane_y(x)
                # Add a bit of noise so we don’t get razor-straight diagonals
                y = y_nom + rng.normal(0.0, lane_change_noise)

                # Heading: roughly along +X, with small yaw depending on lane slope
                # Approx derivative dy/dx from finite difference
                dx = 0.1
                y_before = lane_y(x - dx)
                y_after  = lane_y(x + dx)
                dy = (y_after - y_before) / (2*dx)
                heading = math.atan2(dy, 1.0)  # slope dy/dx

                recs.append(_make_rec(t, x, y, heading, obj_id))

            trajs[obj_id] = recs
            veh_index += 1

    t0 = 0.0
    return trajs, t0


# ============================================================
#  Quick visual sanity check (optional)
# ============================================================

if __name__ == "__main__":
    # Simple smoke test: plot XY for a few synthetic vehicles
    import matplotlib.pyplot as plt

    trajs1, t01 = synth_dense_highway_jam(num_vehicles=50, num_lanes=4)
    trajs2, t02 = synth_multi_lane_merge(num_vehicles=40)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Scenario 1
    for i, (oid, recs) in enumerate(trajs1.items()):
        if i > 20: break
        xs = [r["cx"] for r in recs]
        ys = [r["cy"] for r in recs]
        ax[0].plot(xs, ys, alpha=0.7)
    ax[0].set_title("Scenario 1: Dense Highway Jam")
    ax[0].set_xlabel("X (m)")
    ax[0].set_ylabel("Y (m)")
    ax[0].set_aspect("equal")
    ax[0].grid(True)

    # Scenario 2
    for i, (oid, recs) in enumerate(trajs2.items()):
        if i > 40: break
        xs = [r["cx"] for r in recs]
        ys = [r["cy"] for r in recs]
        ax[1].plot(xs, ys, alpha=0.7)
    ax[1].set_title("Scenario 2: Multi-lane Merge")
    ax[1].set_xlabel("X (m)")
    ax[1].set_ylabel("Y (m)")
    ax[1].set_aspect("equal")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()
