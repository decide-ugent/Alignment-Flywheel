"""Benchmark LSH-accelerated evaluate vs brute-force evaluate."""
import time
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_flywheel_kernel import load_kernel, evaluate, evaluate_lsh

ENVS = [
    ("Ant",         "adaptive", "regression"),
    ("Ant",         "medium",   "regression"),
    ("HalfCheetah", "medium",   "regression"),
    ("Hopper",      "medium",   "regression"),
    ("Swimmer",     "medium",   "regression"),
    ("Walker2d",    "medium",   "regression"),
    ("Suction",     "adaptive", "regression"),
]

BATCH_SIZES = [1, 10, 100, 1000, 10000]


def run():
    for env_name, strictness, decider in ENVS:
        try:
            k = load_kernel(env_name, strictness=strictness, decider=decider)
        except Exception as e:
            print(f"  Skip {env_name}/{strictness}/{decider}: {e}")
            continue

        n_patches = sum(p["n_patches"] for p in k["patches"])
        lsh = k["_lsh"]
        D = k["_fast"]["enc_WT"].shape[1]  # input dim

        print(f"\n{'='*70}")
        print(f"  {env_name} — {strictness} ({decider})")
        print(f"  {n_patches} patches, {k['bottleneck_dim']}B × {k['num_experts']}E")
        print(f"  SimHash: {lsh.get('n_bits', '?')} bits × {lsh.get('n_tables', '?')} tables, "
              f"{lsh['total_cells']} buckets, "
              f"median_bw={lsh.get('median_bw', 0):.3f}")
        print(f"{'='*70}")
        print(f"  {'N':>6}  {'Brute(ms)':>10}  {'LSH(ms)':>10}  "
              f"{'Speedup':>8}  {'MaxErr':>8}")
        print(f"  {'-'*52}")

        rng = np.random.default_rng(42)

        for N in BATCH_SIZES:
            obs = rng.standard_normal((N, D)).astype(np.float32) * 2

            # Warmup
            _ = evaluate(k, obs[:1])
            _ = evaluate_lsh(k, obs[:1])

            # Brute-force
            reps = max(1, min(100, 50000 // N))
            t0 = time.perf_counter()
            for _ in range(reps):
                s_bf = evaluate(k, obs)
            t_bf = (time.perf_counter() - t0) / reps * 1000

            # LSH
            t0 = time.perf_counter()
            for _ in range(reps):
                s_lsh = evaluate_lsh(k, obs)
            t_lsh = (time.perf_counter() - t0) / reps * 1000

            max_err = float(np.max(np.abs(s_bf - s_lsh)))
            speedup = t_bf / t_lsh if t_lsh > 0 else float('inf')

            print(f"  {N:>6}  {t_bf:>10.3f}  {t_lsh:>10.3f}  "
                  f"{speedup:>7.2f}×  {max_err:>8.6f}")


if __name__ == "__main__":
    run()
