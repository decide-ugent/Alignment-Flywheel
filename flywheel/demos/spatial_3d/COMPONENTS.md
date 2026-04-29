# Spatial 3D Demo — Component Mechanics

This document explains exactly what every component does in this
demo: what the oracle computes, what governance batches change,
how the Red Team finds flaws, how Refinement decides which
patches are safe to apply.

---

## Setup

The 3D space is the cube `[-1, 1]³` discretised on a 20×20×20
grid (8000 cells). An "expert path" is 30 sampled points on a
synthetic curve through the cube. An IIRL training run produced
a loss value at each grid cell — `loss_values.npy`.

The flywheel's job: discover the cells where the model assigns
high reward but is far from the expert path (false positives in
the learned reward function), then patch them without damaging
the legitimate basin around the path.

---

## Oracle

**Class:** `PrecomputedGridOracle` (the adapter), wrapped by `SpatialOracle`.

### What it stores

```python
self._loss        # (8000,) precomputed loss per grid cell
self._rewards     # 1 - clip(loss / 0.3, 0, 1)   → reward per cell
self._supp_centers  # (N, 3) suppression kernel centres
self._supp_bw       # (N,)    suppression kernel bandwidths
self._version       # increments each time a batch is applied
```

### `query_points(points)`

For a list of 3D points, the oracle:

1. Snaps each point to the nearest grid cell index.
2. Looks up that cell's reward.
3. Subtracts the **cumulative suppression** at that point:
   ```
   suppression(p) = min(Σᵢ exp(-‖p - cᵢ‖² / (2 bwᵢ²)), 1)
   ```
   Each previously-applied governance patch contributes one
   Gaussian centred at `cᵢ` with bandwidth `bwᵢ`.
4. Returns `safety = max(0, reward − suppression)` plus a
   confidence score derived from `|2·reward − 1|`.

So the oracle's output is *not* the raw IIRL reward — it's the
reward attenuated by every suppression kernel previously added
through governance batches.

### What `apply_batch` changes

A `GovernanceBatch` may carry many `LocalCorrection` artifacts.
The oracle filters for `correction_type == SPATIAL_FLAW_PATCH`
and for each one appends:

```python
self._supp_centers ← center from payload["flaw_point"]
self._supp_bw      ← bandwidth from payload["support_radius"]
```

Then bumps `self._version` to `oracle:v{N+1}`. After this point,
all future `query_points` calls return suppressed values at
those locations.

This is the patching mechanism in concrete form: the oracle's
public behaviour changes because its private state grew, with
zero retraining.

---

## Proposer

**Class:** `SpatialProposer`.

Given a `Context` containing `{"point": [x, y, z]}`, returns a
`Trajectory` of kind `SPATIAL` with one step whose payload is
the same point. There's no generation — this is the boundary
between an external query and the governance pipeline's
artifact protocol.

---

## Flywheel Overlay

**Class:** `SpatialOverlay`.

Holds one norm:

```python
Norm(id="SPATIAL_SUPPORT_REQUIRED",
     kind=SPATIAL_BOUNDARY,
     spec={"require_support": True})
```

This norm declares: a point is in violation if its distance to
the expert path exceeds the boundary `B = 0.34`. The Verifier
reads this norm and applies the check.

The overlay also tracks audit coverage. Each
`AUDIT_COVERAGE_UPDATE` correction in a batch adds an entry to
`self._covered`. Once any region has coverage, the overlay
returns `c_a = 0.2` (low — well-covered); otherwise `c_a = 0.8`
(high — needs audit). This signal feeds into the Enforcement
layer.

---

## Enforcement

**Class:** `DefaultEnforcement(safety_margin=0.4)`.

Three checks in order:

1. **Uncertainty:** if `u ≥ u_thresh` → `ESCALATE`.
2. **Safety:** if `s < safety_margin (=0.4)` → `BLOCK`.
3. **Coverage:** if `c_a ≥ c_a_thresh` (= 0.6, "low coverage")
   AND `evidence_status` is weak AND `s < 0.7` → `ESCALATE`,
   else `ALLOW` with audit flag.

For spatial points, the score `s` is high inside the basin and
near zero in suppressed regions, so suppressed flaws yield
`BLOCK`.

---

## Red Team

**OODA composition:**

| Step    | Class                        |
|---------|------------------------------|
| Observe | `FineSamplingObserver`       |
| Orient  | `DistanceOrienter`           |
| Decide  | `FarthestFirstDecider`       |
| Act     | `CandidateSubmitter`         |

### `FineSamplingObserver`

The observer always queries the full 8000-cell grid. On
iteration 2+, it ALSO samples extra dense points around
previously-discovered flaws — the radius and sample count
adapt to how many flaws remained from the previous iteration:

```python
n_fine = clip(8000 / max(1, prev_remaining // 50), 500, 2000)
fine_radius = clip(prev_remaining / 5000, 0.12, 0.35)
```

Many flaws → wide search; few flaws → tight search around the
remaining boundary region. The observer hits `query_points` for
all sampled points in one HTTP call to the Oracle.

### `DistanceOrienter`

Computes the minimum Euclidean distance from each queried point
to the expert path. Marks a point as a flaw candidate if:

```
value > safety_floor (=0.01)   AND   dist > boundary (=0.34)
```

i.e. the oracle still considers it rewarding AND it's outside
the basin.

### `FarthestFirstDecider`

Sorts the flaw indices by distance descending. Flaws far from
the path get processed first — they're the ones that cover the
most area when patched (a kernel at distance 0.6 needs a wide
bandwidth, which suppresses many neighbouring flaws too).

### `CandidateSubmitter`

For each sorted index, emits a `CandidateFlaw` artifact:

```python
CandidateFlaw(
    context={"point": [x,y,z], "dist_to_path": d},
    trajectory={"kind": "spatial", "steps": [{"payload": {"point": [...]}}]},
    s=value, u=0.2, u_thresh=0.5,
    v_O=oracle_version)
```

These artifacts are pushed to the Verifier.

---

## Verifier

**OODA composition:**

| Step    | Class                        |
|---------|------------------------------|
| Observe | `NormLoader`                 |
| Orient  | `SpatialNormMatcher`         |
| Decide  | `SpatialViolationDecider`    |
| Act     | `VerificationEmitter`        |

### `SpatialNormMatcher`

Pairs the candidate with each loaded norm. For
`SPATIAL_BOUNDARY` norms, attaches the candidate's distance to
the path.

### `SpatialViolationDecider`

For each matched `SPATIAL_BOUNDARY` norm: if `dist > boundary`,
return `VerificationOutcome.VIOLATION`. The norm spec is the
formal expression of the rule; the decider is the executable
check.

### `VerificationEmitter`

Wraps the result in a `VerificationResult` artifact:

```python
VerificationResult(
    candidate_ref=candidate.id,
    outcome=VIOLATION,
    violated_norm_id="SPATIAL_SUPPORT_REQUIRED",
    evidence_ref="dist=0.612>boundary=0.34")
```

Verified violations get pushed to the Triage queue.

---

## Triage

**Class:** `PriorityTriage`.

Each verified candidate is scored as
`max(dist_to_path, category_weight)`. For spatial flaws, only
distance applies (no category), so larger distances rise to the
top of the queue. The Refinement step pops all items at once.

---

## Refinement

**OODA composition:**

| Step    | Class                          |
|---------|--------------------------------|
| Observe | `QueueObserver`                |
| Orient  | `AdaptiveBandwidthOrienter`    |
| Decide  | `CumulativeRegressionDecider`  |
| Act     | `BatchDeployer`                |

### `QueueObserver`

Reads the queue and forwards parameters from the demo's YAML:
`max_patches=200`, `max_basin_loss=0.10`, `min_bw=0.03`,
`max_bw=0.30`.

### `AdaptiveBandwidthOrienter`

For each verified flaw at distance `d`, proposes:

```
proposed_bw = clip((d − boundary) × 0.5, min_bw, max_bw)
        = clip((d − 0.34) × 0.5, 0.03, 0.30)
```

A flaw at d = 0.4 (just past the boundary) gets bw ≈ 0.03 —
narrow, won't damage anything nearby. A flaw at d = 0.95 (corner
of the cube) gets bw ≈ 0.30 — wide, suppresses a large region.

This is the wide-far / narrow-near rule: distant flaws get
generous kernels because there's nothing they can hurt.

### `CumulativeRegressionDecider`

This is the key safety mechanism. The basin (cells inside the
boundary with reward > floor) must be preserved.

For each proposed patch in priority order:

1. **Per-patch cap.** Compute the analytical safe maximum
   bandwidth so a single patch can't suppress the nearest basin
   point by more than ~10%:
   ```
   max_safe_bw = nearest_basin_dist / sqrt(-2 ln(0.10))
   ```
   Shrink `bw` to `0.9 × max_safe_bw` if larger. (Track this as
   a "shrink" event.)

2. **Cumulative cap.** Track the running suppression sum on
   every basin point from all previously-accepted patches in
   THIS batch. If adding the new patch pushes any basin point
   over `max_basin_loss = 0.10`, do a binary search for the
   largest bandwidth that stays under the cap. If even
   `min_bw = 0.03` doesn't fit → reject the patch.

3. **Accept** the patch with its final (possibly shrunk)
   bandwidth.

Stop when `max_patches = 200` is reached.

### `BatchDeployer`

For each accepted patch, builds two `LocalCorrection`s:

1. `SPATIAL_FLAW_PATCH` with `{flaw_point: [x,y,z], support_radius: bw}`
2. `AUDIT_COVERAGE_UPDATE` with `{case_class: "spatial|d=0.6"}`

Wraps them in a `GovernanceBatch`:

```python
GovernanceBatch(
    from_oracle_version="oracle:v3",
    to_oracle_version="oracle:v4",
    local_corrections=[ ... ],
    regression_evidence={"patched": 187, "rejected": 24, "shrinks": 14},
    signature="regression-verified")
```

The governance engine ships this batch to:
- `POST /oracle/apply_batch` → installs suppression kernels
- `POST /flywheel/apply_batch` → records audit coverage

Two HTTP calls because oracle and flywheel are independent
services.

---

## Blue Team

**Class:** `CollateralMonitor`.

After deployment, the engine identifies the candidate flaws that
were *not* directly patched (Refinement only takes the top 200,
the rest are unpatched). Blue Team re-queries those points
through the oracle. Many will now read as suppressed because a
neighbouring patch's wide kernel happened to cover them — that's
**collateral coverage**, the main reason the demo converges in
~14 iterations rather than 4900/200 = 25.

The number of points that drop below the safety floor without
being directly patched is logged as the iteration's "collateral"
count.

---

## What you see per iteration

```
It   Found  Patch  Collat  Reject  Basin  Flaws  Oracle
1     4927   200    1557     0     837   3170   oracle:v1
…
14    211    187     5      24     837    0     oracle:v14
```

- **Found**: candidates the Red Team produced.
- **Patch**: corrections accepted by Refinement.
- **Collat**: flaws the Blue Team confirmed are now suppressed
  through neighbouring patches (no patch placed there directly).
- **Reject**: patches rejected because no safe bandwidth existed.
- **Basin**: count of cells inside the boundary still above the
  safety floor — must remain constant.
- **Flaws**: cells outside the boundary still above the floor.

Convergence: `Found → Patch + Collat ≈ Found`, so per-iteration
flaw reduction roughly equals the patch budget plus collateral
gains. The basin row should never decrease: that's
`CumulativeRegressionDecider`'s only job.
