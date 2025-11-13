#!/usr/bin/env python
# stage3_fine_time_refine.py
#
# Stage 3: Fine-time refinement around close approaches (Stage 2 output),
#          and workload counting for OBB checks – globally and for EGO.
#
# Uses:
#   - /home/gns/waymo_work/data/samples/trajectories_all.csv
#   - /home/gns/waymo_work/data/samples/workload_pairs_min.csv
#
# Outputs:
#   - Console summary of workload
#   - /home/gns/waymo_work/data/samples/workload_fine_summary.csv
#
# Assumes:
#   - stage1_extract_trajectories.py has produced trajectories_all.csv
#   - stage2_pairwise_eval.py has produced workload_pairs_min.csv
#   - obb_distance extension is available (falls back to center distance)

import os
import csv
import math
from collections import defaultdict

import numpy as np

# ---------- CONFIG ----------

BASE_DIR    = "/home/gns/waymo_work"
SAVE_DIR    = os.path.join(BASE_DIR, "data", "samples")
TRAJ_CSV    = os.path.join(SAVE_DIR, "trajectories_all.csv")
PAIRS_CSV   = os.path.join(SAVE_DIR, "workload_pairs_min.csv")
SUMMARY_CSV = os.path.join(SAVE_DIR, "workload_fine_summary.csv")

# fine-time refinement window centered at Stage2 t_at (seconds)
REFINE_WINDOW = 1.0     # total window size (e.g. 1.0 → ±0.5 s around t_at)
FINE_DT       = 0.01    # fine timestep (seconds) – 10 ms by default

# how close we still care about at fine resolution
THRESH_OBB_FINE = 2.5   # meters (same as Stage2 OBB threshold typically)

# optional: limit pairs per file to keep runtime bounded (None = use all)
MAX_PAIRS_PER_FILE = None

# ---------- Try OBB distance extension ----------

USE_OBB = True
try:
    import obb_distance
except Exception:
    USE_OBB = False
    print("ℹ️ obb_distance not found; falling back to center-distance only.")

# ---------- Helpers: distance & interpolation ----------

def obb_min_dist(rec_a, rec_b):
    """
    Compute OBB min distance between two records (with cx, cy, length, width, heading).
    Falls back to center distance if obb_distance is unavailable.
    """
    if USE_OBB:
        return obb_distance.obb_min_distance(
            rec_a["cx"], rec_a["cy"], rec_a["length"], rec_a["width"], rec_a["heading"],
            rec_b["cx"], rec_b["cy"], rec_b["length"], rec_b["width"], rec_b["heading"]
        )
    # center distance fallback
    dx = rec_a["cx"] - rec_b["cx"]
    dy = rec_a["cy"] - rec_b["cy"]
    return math.hypot(dx, dy)


def interp_angle(a0, a1, alpha):
    """
    Interpolate angles a0→a1 (radians) taking the shortest wrap-around path.
    alpha in [0,1].
    """
    # wrap difference into [-pi, pi]
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    return a0 + alpha * da


def interp_record(track, t):
    """
    Given a track = list of dicts sorted by r["t"], interpolate an approximate
    record at time t (clamped to [t_min, t_max]).
    Fields interpolated: t, cx, cy, cz, heading, length, width, height, speed.
    type and is_stationary are copied from nearest.
    """
    if not track:
        return None

    # clamp boundaries
    if t <= track[0]["t"]:
        return track[0]
    if t >= track[-1]["t"]:
        return track[-1]

    # binary search for t in [t_i, t_{i+1}]
    # we assume times are sorted and relatively dense
    lo, hi = 0, len(track) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if track[mid]["t"] < t:
            lo = mid + 1
        else:
            hi = mid
    j = lo
    i = j - 1
    t0 = track[i]["t"]
    t1 = track[j]["t"]
    if t1 <= t0:
        alpha = 0.0
    else:
        alpha = (t - t0) / (t1 - t0)

    # linear interpolation for positions & dims
    def lerp(k):
        return track[i][k] + alpha * (track[j][k] - track[i][k])

    cx = lerp("cx")
    cy = lerp("cy")
    cz = lerp("cz")
    heading = interp_angle(track[i]["heading"], track[j]["heading"], alpha)
    length = lerp("length")
    width  = lerp("width")
    height = lerp("height")
    speed  = lerp("speed")

    # type & stationary flag from nearest sample
    nearest = track[i] if abs(t - t0) <= abs(t - t1) else track[j]

    return {
        "t": t,
        "cx": cx,
        "cy": cy,
        "cz": cz,
        "heading": heading,
        "length": length,
        "width": width,
        "height": height,
        "speed": speed,
        "type": nearest["type"],
        "is_stationary": nearest["is_stationary"],
        "oid": nearest["oid"],
    }

# ---------- Load trajectories (Stage 1) ----------

def load_trajectories(path):
    """
    Load trajectories_all.csv into nested dict:
      trajs[file][oid] = list of records sorted by time

    CSV header assumed:
      file,oid,type,t,cx,cy,cz,heading,length,width,height,speed,is_stationary
    """
    trajs = defaultdict(lambda: defaultdict(list))
    with open(path, "r") as f:
        r = csv.reader(f)
        hdr = next(r)
        idx = {h: i for i, h in enumerate(hdr)}

        for row in r:
            file_id = row[idx["file"]]
            oid     = row[idx["oid"]]
            rec = {
                "file": file_id,
                "oid": oid,
                "type": row[idx["type"]],
                "t": float(row[idx["t"]]),
                "cx": float(row[idx["cx"]]),
                "cy": float(row[idx["cy"]]),
                "cz": float(row[idx["cz"]]),
                "heading": float(row[idx["heading"]]),
                "length": float(row[idx["length"]]),
                "width":  float(row[idx["width"]]),
                "height": float(row[idx["height"]]),
                "speed": float(row[idx["speed"]]),
                "is_stationary": int(row[idx["is_stationary"]]),
            }
            trajs[file_id][oid].append(rec)

    # sort by time
    for f in trajs:
        for oid in trajs[f]:
            trajs[f][oid].sort(key=lambda r: r["t"])

    print(f"▶ Loaded trajectories from {path}")
    print(f"  Files: {len(trajs)}")
    return trajs

# ---------- Load close pairs (Stage 2) ----------

def load_pairs(path):
    """
    Load workload_pairs_min.csv from Stage 2.

    Expected columns include at least:
      file,oid_i,oid_j,t_at,type_i,type_j,d_obb_min
    """
    pairs_by_file = defaultdict(list)
    with open(path, "r") as f:
        r = csv.reader(f)
        hdr = next(r)
        idx = {h: i for i, h in enumerate(hdr)}
        for row in r:
            file_id = row[idx["file"]]
            p = {
                "file": file_id,
                "oid_i": row[idx["oid_i"]],
                "oid_j": row[idx["oid_j"]],
                "t_at": float(row[idx["t_at"]]),
                "type_i": row[idx["type_i"]],
                "type_j": row[idx["type_j"]],
                "d_obb_min": float(row[idx["d_obb_min"]]),
            }
            pairs_by_file[file_id].append(p)
    print(f"▶ Loaded pairs from {path}")
    total_pairs = sum(len(v) for v in pairs_by_file.values())
    print(f"  Files with pairs: {len(pairs_by_file)}, total pairs: {total_pairs}")
    return pairs_by_file

# ---------- Ego detection ----------

def pick_ego_for_file(trajs_for_file):
    """
    Heuristic: among Vehicle tracks, pick the one closest to origin on average.
    If tie, pick the one with longest track.

    Returns oid or None if no suitable Vehicle found.
    """
    best_oid = None
    best_score = None

    for oid, arr in trajs_for_file.items():
        if not arr:
            continue
        if arr[0]["type"] != "Vehicle":
            continue

        # median radial distance
        rs = [math.hypot(r["cx"], r["cy"]) for r in arr]
        if not rs:
            continue
        med_r = float(np.median(rs))
        length = len(arr)
        # score: prioritize small radius, then long track
        score = (med_r, -length)

        if best_score is None or score < best_score:
            best_score = score
            best_oid = oid

    return best_oid

# ---------- Core refinement ----------

def refine_for_file(file_id, trajs_for_file, pairs_for_file):
    """
    For a single Waymo file:
      - Detect EGO
      - For each close pair from Stage 2, refine in time around t_at with fine Δt
      - Count OBB checks globally and for EGO
      - Also count how many of those checks have d_obb <= THRESH_OBB_FINE

    Returns a summary dict for this file.
    """
    if not pairs_for_file:
        return {
            "file": file_id,
            "n_pairs": 0,
            "ego_oid": "",
            "n_checks_total": 0,
            "n_checks_ego": 0,
            "n_near_total": 0,
            "n_near_ego": 0,
        }

    ego_oid = pick_ego_for_file(trajs_for_file)
    if ego_oid is None:
        print(f"  ⚠ No EGO Vehicle found for file {file_id}; EGO counts will be zero.")
    else:
        print(f"  EGO candidate for {file_id}: {ego_oid}")

    if MAX_PAIRS_PER_FILE is not None and len(pairs_for_file) > MAX_PAIRS_PER_FILE:
        pairs = pairs_for_file[:MAX_PAIRS_PER_FILE]
        print(f"  Limiting pairs for {file_id}: {len(pairs_for_file)} → {len(pairs)}")
    else:
        pairs = pairs_for_file

    checks_total = 0
    checks_ego   = 0
    near_total   = 0
    near_ego     = 0

    for p in pairs:
        oid_i = p["oid_i"]
        oid_j = p["oid_j"]
        t_center = p["t_at"]

        track_i = trajs_for_file.get(oid_i)
        track_j = trajs_for_file.get(oid_j)
        if not track_i or not track_j:
            continue

        t_start = t_center - REFINE_WINDOW / 2.0
        t_end   = t_center + REFINE_WINDOW / 2.0
        if t_end <= t_start:
            continue

        n_steps = int(round((t_end - t_start) / FINE_DT)) + 1

        for k in range(n_steps):
            t = t_start + k * FINE_DT

            rec_i = interp_record(track_i, t)
            rec_j = interp_record(track_j, t)
            if rec_i is None or rec_j is None:
                continue

            d_obb = obb_min_dist(rec_i, rec_j)
            checks_total += 1
            if d_obb <= THRESH_OBB_FINE:
                near_total += 1

            if ego_oid is not None and (oid_i == ego_oid or oid_j == ego_oid):
                checks_ego += 1
                if d_obb <= THRESH_OBB_FINE:
                    near_ego += 1

    return {
        "file": file_id,
        "n_pairs": len(pairs),
        "ego_oid": ego_oid or "",
        "n_checks_total": checks_total,
        "n_checks_ego": checks_ego,
        "n_near_total": near_total,
        "n_near_ego": near_ego,
    }

# ---------- Main ----------

def main():
    if not os.path.exists(TRAJ_CSV):
        raise FileNotFoundError(f"Trajectories CSV not found: {TRAJ_CSV}")
    if not os.path.exists(PAIRS_CSV):
        raise FileNotFoundError(f"Pairs CSV not found: {PAIRS_CSV}")

    trajs_all = load_trajectories(TRAJ_CSV)
    pairs_all = load_pairs(PAIRS_CSV)

    summaries = []
    global_checks_total = 0
    global_checks_ego   = 0
    global_near_total   = 0
    global_near_ego     = 0

    for file_id, trajs_for_file in trajs_all.items():
        pairs_for_file = pairs_all.get(file_id, [])
        if not pairs_for_file:
            print(f"\n▶ File {file_id}: no close pairs from Stage 2, skipping fine refinement.")
            continue

        print(f"\n▶ Fine refine for file: {file_id}")
        print(f"  Stage2 pairs: {len(pairs_for_file)}")

        summary = refine_for_file(file_id, trajs_for_file, pairs_for_file)
        summaries.append(summary)

        global_checks_total += summary["n_checks_total"]
        global_checks_ego   += summary["n_checks_ego"]
        global_near_total   += summary["n_near_total"]
        global_near_ego     += summary["n_near_ego"]

        print(f"  → Pairs refined: {summary['n_pairs']}")
        print(f"    Checks total: {summary['n_checks_total']}, near (≤ {THRESH_OBB_FINE:.2f} m): {summary['n_near_total']}")
        if summary["ego_oid"]:
            print(f"    EGO {summary['ego_oid']}: checks={summary['n_checks_ego']}, near={summary['n_near_ego']}")

    # write per-file summary CSV
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "file",
            "ego_oid",
            "n_pairs",
            "n_checks_total",
            "n_checks_ego",
            "n_near_total",
            "n_near_ego",
            "refine_window_s",
            "fine_dt_s",
            "thresh_obb_fine_m",
        ])
        for s in summaries:
            w.writerow([
                s["file"],
                s["ego_oid"],
                s["n_pairs"],
                s["n_checks_total"],
                s["n_checks_ego"],
                s["n_near_total"],
                s["n_near_ego"],
                REFINE_WINDOW,
                FINE_DT,
                THRESH_OBB_FINE,
            ])

    print("\n=== GLOBAL WORKLOAD SUMMARY (Stage 3) ===")
    print(f"  Total fine-time OBB checks (all pairs): {global_checks_total}")
    print(f"  Total fine-time OBB checks (EGO only):  {global_checks_ego}")
    print(f"  Near checks (≤ {THRESH_OBB_FINE:.2f} m) – all pairs: {global_near_total}")
    print(f"  Near checks (≤ {THRESH_OBB_FINE:.2f} m) – EGO only:  {global_near_ego}")
    print(f"\n✅ Per-file summary written → {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
