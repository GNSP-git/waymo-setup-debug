# stage2_pairwise_eval.py
# Evaluates pairwise separations using OBBs with adaptive refinement.
# Outputs two CSVs in data/samples/.
# Optional: obb_distance extension for fast, accurate OBB min distance.

import os, csv, math, collections
from typing import Dict, List, Tuple

SAVE_DIR = "/home/gns/waymo_work/data/samples"
IN_CSV   = os.path.join(SAVE_DIR, "trajectories_all.csv")
SAMPLES_CSV = os.path.join(SAVE_DIR, "workload_samples.csv")
PAIRS_MIN_CSV = os.path.join(SAVE_DIR, "workload_pairs_min.csv")

# thresholds
DIST_THRESH  = 10.0     # report if within this (m)
REFINE_PROX  = 15.0     # trigger interpolation if last center distance under this (m)
SUBSTEPS     = 4        # K pieces between frames when refining
SKIP_STATIC_STATIC = True

# Try OBB distance
USE_OBB = True
try:
    import obb_distance
except Exception:
    USE_OBB = False
    print("ℹ️ obb_distance not found; falling back to center-distance.")

def read_csv_grouped_by_file(path:str) -> Dict[str, Dict[str, List[dict]]]:
    """
    Returns: data[file][oid] = list of records, sorted by time
    record fields: type,t,cx,cy,cz,heading,length,width,height,speed,is_stationary
    """
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(path, "r") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {name:i for i,name in enumerate(header)}
        for row in r:
            rec = {
                "file":  row[idx["file"]],
                "oid":   row[idx["oid"]],
                "type":  row[idx["type"]],
                "t":     float(row[idx["t"]]),
                "cx":    float(row[idx["cx"]]),
                "cy":    float(row[idx["cy"]]),
                "cz":    float(row[idx["cz"]]),
                "heading": float(row[idx["heading"]]),
                "length": float(row[idx["length"]]),
                "width":  float(row[idx["width"]]),
                "height": float(row[idx["height"]]),
                "speed":  float(row[idx["speed"]]),
                "is_stationary": int(row[idx["is_stationary"]]),
            }
            data[rec["file"]][rec["oid"]].append(rec)
    # sort
    for f in data:
        for oid in data[f]:
            data[f][oid].sort(key=lambda r: r["t"])
    return data

def obb_min_distance(a, b) -> float:
    if USE_OBB:
        return obb_distance.obb_min_distance(
            a["cx"], a["cy"], a["length"], a["width"], a["heading"],
            b["cx"], b["cy"], b["length"], b["width"], b["heading"]
        )
    # fallback
    dx = a["cx"] - b["cx"]; dy = a["cy"] - b["cy"]
    return math.hypot(dx, dy)

def center_distance(a, b) -> float:
    dx = a["cx"] - b["cx"]; dy = a["cy"] - b["cy"]
    return math.hypot(dx, dy)

def lerp(a:float,b:float,u:float)->float: return a + u*(b-a)

def slerp_heading(h0:float, h1:float, u:float)->float:
    # simple unwrap towards smallest delta
    dh = (h1 - h0 + math.pi) % (2*math.pi) - math.pi
    return h0 + u*dh

def interp_pose(r0:dict, r1:dict, u:float)->dict:
    return {
        "cx": lerp(r0["cx"], r1["cx"], u),
        "cy": lerp(r0["cy"], r1["cy"], u),
        "cz": lerp(r0["cz"], r1["cz"], u),
        "heading": slerp_heading(r0["heading"], r1["heading"], u),
        "length": r0["length"], "width": r0["width"], "height": r0["height"],
        "type": r0["type"], "oid": r0["oid"], "t": lerp(r0["t"], r1["t"], u),
        "speed": lerp(r0["speed"], r1["speed"], u),
        "is_stationary": 1 if (r0["is_stationary"] and r1["is_stationary"]) else 0
    }

def pairwise_eval_for_file(file_id:str, tracks:Dict[str,List[dict]],
                           samples_writer, minima_writer):
    # Build aligned timeline
    times = sorted({r["t"] for arr in tracks.values() for r in arr})
    # quickly map oid -> index over time (two-pointer while scanning)
    cursors = {oid:0 for oid in tracks}

    # minima across whole file per pair
    best = {}  # (oid_i,oid_j) -> (d_min, t_at, rec_i, rec_j, center_d_min, margin_long)

    def record_sample(a:dict, b:dict, d_center:float, d_obb:float):
        margin_long = d_obb - 0.5*(a["length"] + b["length"])
        samples_writer.writerow([
            file_id, a["oid"], b["oid"], f"{a['t']:.6f}",
            a["type"], b["type"],
            f"{d_center:.3f}", f"{d_obb:.3f}", f"{margin_long:.3f}",
            a["is_stationary"], b["is_stationary"]
        ])
        key = tuple(sorted((a["oid"], b["oid"])))
        cur = best.get(key)
        if (cur is None) or (d_obb < cur[0]):
            best[key] = (d_obb, a["t"], a, b, d_center, margin_long)

    # Iterate over consecutive frames; refine if near
    oids = sorted(tracks.keys())
    for k in range(len(times)-1):
        t0, t1 = times[k], times[k+1]
        # snapshot near t0 and t1
        snap = {0:[], 1:[]}
        for which, tt in enumerate((t0, t1)):
            for oid, arr in tracks.items():
                # advance cursor to closest time <= tt
                c = cursors[oid]
                while c+1 < len(arr) and abs(arr[c+1]["t"]-tt) <= abs(arr[c]["t"]-tt):
                    c += 1
                cursors[oid] = c
                snap[which].append(arr[c])

        # base check at t0
        recs0 = snap[0]
        # optionally skip fully stationary pairs
        for i in range(len(recs0)):
            for j in range(i+1, len(recs0)):
                a, b = recs0[i], recs0[j]
                if SKIP_STATIC_STATIC and a["is_stationary"] and b["is_stationary"]:
                    continue
                dC = center_distance(a,b)
                dO = obb_min_distance(a,b)
                record_sample(a,b,dC,dO)

        # decide which pairs to refine
        # We refine between t0->t1 if pair center distance < REFINE_PROX at either endpoint
        need_refine = []
        recs1 = snap[1]
        for i in range(len(recs0)):
            for j in range(i+1, len(recs0)):
                a0, b0 = recs0[i], recs0[j]
                a1, b1 = recs1[i], recs1[j]
                if SKIP_STATIC_STATIC and a0["is_stationary"] and b0["is_stationary"]:
                    continue
                near0 = center_distance(a0,b0) < REFINE_PROX
                near1 = center_distance(a1,b1) < REFINE_PROX
                if near0 or near1:
                    need_refine.append((a0,b0,a1,b1))

        if not need_refine: 
            continue

        # interpolate K substeps
        for a0,b0,a1,b1 in need_refine:
            for s in range(1, SUBSTEPS):
                u = s/float(SUBSTEPS)
                ai = interp_pose(a0,a1,u)
                bi = interp_pose(b0,b1,u)
                dC = center_distance(ai,bi)
                dO = obb_min_distance(ai,bi)
                record_sample(ai,bi,dC,dO)

    # write minima per pair
    for (oid_i, oid_j), (dmin, t_at, ra, rb, dCmin, margin_long) in best.items():
        if dmin <= DIST_THRESH:
            minima_writer.writerow([
                file_id, oid_i, oid_j, f"{t_at:.6f}",
                ra["type"], rb["type"],
                f"{dCmin:.3f}", f"{dmin:.3f}", f"{margin_long:.3f}"
            ])

def main():
    if not os.path.exists(IN_CSV):
        print("Missing input:", IN_CSV)
        return
    data = read_csv_grouped_by_file(IN_CSV)

    # open writers
    new_samples = not os.path.exists(SAMPLES_CSV)
    new_pairs   = not os.path.exists(PAIRS_MIN_CSV)
    fs = open(SAMPLES_CSV, "a", newline="")
    fp = open(PAIRS_MIN_CSV, "a", newline="")
    sw = csv.writer(fs); pw = csv.writer(fp)
    if new_samples:
        sw.writerow([
            "file","oid_i","oid_j","t","type_i","type_j",
            "d_center","d_obb","margin_longitudinal",
            "stationary_i","stationary_j"
        ])
    if new_pairs:
        pw.writerow([
            "file","oid_i","oid_j","t_at",
            "type_i","type_j","d_center_min","d_obb_min","margin_longitudinal_min"
        ])

    files = sorted(data.keys())
    print(f"Files: {len(files)}")
    for f in files:
        print(f"▶ Evaluating {f} ...")
        pairwise_eval_for_file(f, data[f], sw, pw)

    fs.close(); fp.close()
    print("✅ Stage 2 done.")
    print("Samples  ->", SAMPLES_CSV)
    print("Pair mins->", PAIRS_MIN_CSV)

if __name__ == "__main__":
    main()
