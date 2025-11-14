#!/usr/bin/env python3
"""
Unified driver for:
    - Real Waymo workloads (Stage1 → Stage2 → Stage3)
    - Synthetic stress scenarios (Scenario 1, Scenario 2, ...)

This file orchestrates the full pipeline:
    * Stage 1  : Extract trajectories from Waymo TFRecords
    * Stage 2  : Pairwise VV evaluation (OBB or fallback)
    * Stage 3a : Fine-time refinement (dt = 0.001)
    * Stage 3b : Workload analysis + FLOP spikes + histograms

It also supports bypassing Stage 1 for synthetic scenarios.
"""

import os
import glob
import sys

# === Import your modular pipeline ===
from stage1_extract_trajectories import load_trajectories_in_memory
from stage2_pairwise_eval import evaluate_pairs
from stage3_fine_time_refine import refine_fine_time
from stage3_workload_analysis import accumulate_workload

# === Synthetic scenarios ===
from ganaka_synth_scenarios import (
    synth_dense_highway_jam,
    synth_multi_lane_merge
)

# === Output directories ===
BASE_DIR = "/mnt/n/waymo_comma"
PERCEPTION_DIR = os.path.join(BASE_DIR, "waymo")
MOTION_DIR = os.path.join(BASE_DIR, "waymo_motion")
SYNTH_OUTPUT_DIR = "/home/gns/waymo_work/synthetic_out"
os.makedirs(SYNTH_OUTPUT_DIR, exist_ok=True)


# ============================================================
# Real Waymo Pipeline
# ============================================================
def process_real_waymo():
    tfrecords = sorted(glob.glob(os.path.join(PERCEPTION_DIR, "*_with_camera_labels.tfrecord")))
    if not tfrecords:
        print("No TFRecord files found in", PERCEPTION_DIR)
        return

    print("\n========== Processing REAL Waymo TFRecords ==========")
    for f in tfrecords:
        print(f"\n=== File: {os.path.basename(f)} ===")

        # Stage 1
        trajs, t0 = load_trajectories_in_memory(f)
        print(f" Stage1: {len(trajs)} tracks extracted")

        # Stage 2
        coarse_pairs = evaluate_pairs(trajs)
        print(f" Stage2: Found {len(coarse_pairs)} coarse V–V pairs")

        # Stage 3a
        fine_pairs = refine_fine_time(trajs, coarse_pairs)
        print(f" Stage3a: Fine-time pairs: {len(fine_pairs)}")

        # Stage 3b
        workload = accumulate_workload(fine_pairs, f)
        print(f" Stage3b: Workload saved for {os.path.basename(f)}")


# ============================================================
# Synthetic Scenario Pipeline
# ============================================================
def process_synthetic_scenario(label, trajs, t0=0.0):
    print(f"\n========== Synthetic Scenario: {label} ==========")
    print(f"Tracks = {len(trajs)}, Samples = {sum(len(v) for v in trajs.values())}")

    # Stage 2 (bypass Stage 1)
    coarse_pairs = evaluate_pairs(trajs)
    print(f" Stage2: Found {len(coarse_pairs)} coarse pairs")

    # Stage 3a
    fine_pairs = refine_fine_time(trajs, coarse_pairs)
    print(f" Stage3a: Fine-time pairs: {len(fine_pairs)}")

    # Stage 3b
    out_file = f"synthetic_{label}"
    workload = accumulate_workload(fine_pairs, out_file)
    print(f" Stage3b: Synthetic workload saved as {out_file}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("Ganaka VV Close-Approach Scanner")
    print("=================================")
    print("1. Process REAL Waymo TFRecords")
    print("2. Run Synthetic Scenario 1 (Dense Highway Jam)")
    print("3. Run Synthetic Scenario 2 (Multi-lane Merge)")
    print("4. Run BOTH synthetic scenarios")
    print("5. Exit")

    while True:
        choice = input("Select option [1-5]: ").strip()

        if choice == "1":
            process_real_waymo()

        elif choice == "2":
            trajs, t0 = synth_dense_highway_jam(num_vehicles=300)
            process_synthetic_scenario("dense_highway_jam", trajs, t0)

        elif choice == "3":
            trajs, t0 = synth_multi_lane_merge(num_vehicles=80)
            process_synthetic_scenario("multi_lane_merge", trajs, t0)

        elif choice == "4":
            trajs1, t01 = synth_dense_highway_jam(num_vehicles=300)
            process_synthetic_scenario("dense_highway_jam", trajs1, t01)

            trajs2, t02 = synth_multi_lane_merge(num_vehicles=80)
            process_synthetic_scenario("multi_lane_merge", trajs2, t02)

        elif choice == "5":
            print("Exiting.")
            break

        else:
            print("Invalid choice. Try again.")
