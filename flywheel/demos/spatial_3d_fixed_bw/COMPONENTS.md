# Spatial 3D Fixed-Bandwidth Demo — Component Mechanics

This demo is a **baseline comparison** for the main spatial demo.
It uses the same architecture but swaps three OODA steps via YAML
to demonstrate what happens without adaptive bandwidth and
without the cumulative regression test.

For everything that's identical to the main spatial demo, see
`spatial_3d/COMPONENTS.md`. This document only covers the
differences.

---

## What changes

| Role         | Step    | Adaptive demo                  | This demo                      |
|--------------|---------|--------------------------------|--------------------------------|
| Red Team     | observe | `FineSamplingObserver`         | `GridObserver`                 |
| Refinement   | orient  | `AdaptiveBandwidthOrienter`    | `FixedBandwidthOrienter`       |
| Refinement   | decide  | `CumulativeRegressionDecider`  | `NoCumulativeDecider`          |

Three component swaps, no other changes. The point: the
architecture supports rigorous baseline comparisons through
nothing but YAML edits.

---

## `GridObserver`

Always queries the full 8000-cell grid, never adapts. On every
iteration it sends the same set of points to the oracle. No
fine-grained sampling around prior flaws.

Consequence: discovers fewer near-boundary flaws as iterations
progress, because the grid is too coarse to resolve them. That
slows convergence further on top of the bandwidth issues below.

---

## `FixedBandwidthOrienter`

For every flaw, regardless of distance:

```python
proposed_bw = config.fixed_bw   # = 0.05
```

A flaw at d = 0.95 gets the same bandwidth as a flaw at d = 0.4.
Distant flaws no longer cover their neighbours through a wide
kernel — every patch only suppresses a tiny region around its
exact point.

---

## `NoCumulativeDecider`

Accepts all proposed patches up to `max_patches` (= 60 in this
demo's config) without any regression check:

```python
def decide(self, oriented):
    accepted = oriented["planned"][:max_patches]
    return {"accepted": accepted, "rejected": 0, "shrinks": 0, ...}
```

Why is this safe? Because `fixed_bw = 0.05` is small enough that
a single kernel can't damage the basin: the closest basin point
is typically at distance > 0.07 (boundary 0.34 minus epsilon)
from any flaw, and `exp(−(0.07)² / (2 × 0.05²)) ≈ 0.37` — high
enough to *partially* affect a basin point if very close, but
the design assumes flaws are far enough that no significant
suppression reaches the basin.

In practice the basin stays at 837 throughout, so the assumption
holds for this dataset. But the architecture has not *verified*
that it holds — it just assumed.

Compare with the adaptive demo's `CumulativeRegressionDecider`
which explicitly checks per-patch and cumulative basin impact
before accepting any patch.

---

## `FIFOTriage`

Switches from `PriorityTriage` to first-in-first-out. Verified
violations are processed in arrival order, not by distance.
Consequence: with fixed narrow bandwidths, far flaws don't get
prioritised — but also don't matter much since they wouldn't
help neighbours either way.

---

## What you see

```
It   Found  Patch  Collat  Reject  Basin  Flaws  Oracle
1     4927    60     34       0    837   4833   oracle:v1
2     4833    60     35       0    837   4738   oracle:v2
…
25    2945    60      6       0    837   2879   oracle:v25
```

After 25 iterations, **2,879 flaws remain** (compared with 0
flaws after 14 iterations in the adaptive demo).

Why so slow? Per-iteration:
- Patches placed: 60 (config-capped — note the adaptive demo
  caps at 200)
- Collateral gain: ~30 (vs ~50–1500 in adaptive)
- Total reduction: ~90 flaws/iteration.

To clear 4927 flaws at ~90 per iteration → ~55 iterations needed.
We stop at 25 to keep the demo runtime reasonable; the residual
2879 is the headline baseline number.

The basin is preserved (837 → 837, 100%) — the assumption that
narrow kernels don't reach the basin happens to hold for this
dataset, even without the cumulative check. But the demo cannot
*prove* it would hold for a different dataset, while the
adaptive demo's regression decider can.

---

## What this comparison shows

The adaptive demo and the fixed-BW demo expose the trade-off:

|                          | Fixed BW (this) | Adaptive (main)   |
|--------------------------|-----------------|-------------------|
| Iterations to converge   | ~55 (truncated to 25) | 14         |
| Patches per iteration    | 60              | up to 200         |
| Collateral per iteration | ~30             | up to 1557        |
| Regression test          | none (assumed)  | per-patch + batch |
| Basin preserved          | yes (luck)      | yes (proven)      |

The flywheel architecture admits both strategies as YAML
choices. A deployment can start with the fast adaptive strategy
in development and fall back to fixed-narrow patches in
production where regression evidence is unavailable.
