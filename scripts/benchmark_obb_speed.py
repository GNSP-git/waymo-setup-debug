import time, math, random
import numpy as np
import obb_distance  # your compiled module

def random_obb():
    """Generate random (cx, cy, length, width, heading) tuple."""
    return (
        random.uniform(-50, 50),  # x center (m)
        random.uniform(-50, 50),  # y center (m)
        random.uniform(2, 5),     # length (m)
        random.uniform(1, 2.5),   # width (m)
        random.uniform(-math.pi, math.pi)  # heading (rad)
    )

def benchmark(num_pairs=1_000_000, warmup=1000):
    print(f"Benchmarking {num_pairs:,} OBB distance computations...")
    pairs = [(random_obb(), random_obb()) for _ in range(num_pairs)]

    # Warmup
    for i in range(warmup):
        a, b = pairs[i]
        _ = obb_distance.obb_min_distance(*a, *b)

    start = time.perf_counter()
    for (a, b) in pairs:
        _ = obb_distance.obb_min_distance(*a, *b)
    end = time.perf_counter()

    elapsed = end - start
    per_call = elapsed / num_pairs * 1e6  # µs per call
    print(f"Total time: {elapsed:.3f} s for {num_pairs:,} calls")
    print(f"Average time per OBB comparison: {per_call:.3f} µs ({1e6/per_call:.1f} ops/sec)")

if __name__ == "__main__":
    benchmark(500_000)
