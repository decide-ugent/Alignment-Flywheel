# Simple Medical Demo — Component Mechanics

The smallest possible medical pipeline. A 60-line heuristic
oracle, two norms, FIFO triage, eight fixed cases. Built to
demonstrate the architecture at maximum transparency before
moving to the more elaborate medical scenarios.

---

## Setup

A patient-portal–style scenario with an artificial oracle that
captures three rules:

- Drafts containing medication-action stems (`dose`, `tapering`,
  `prescribe`, `discontinue`, `increase`, `decrease`) are
  penalised.
- `case_type == "medication"` adds further penalty.
- Weak evidence (`insufficient`, `conflicting`) adds further
  penalty.

The eval set has 8 cases: 4 unsafe (medication-action drafts
with weak evidence, lab-result discussions without context) and
4 safe (refills, scheduling, supported general advice).

The flywheel's job: discover the unsafe categories the oracle
underrates, install corrections, watch the eval safety score
climb to 100%.

---

## Oracle

**Class:** `SimpleMedicalOracle`.

### What it stores

```python
self._version                # "oracle:v0", v1, v2, ...
self._hard_blocks            # list[str] — keywords that force BLOCK
self._disposition_overrides  # {"case_type|evidence": "min_disposition"}
```

### `predict(context, trajectory)`

1. Pull `draft`, `disposition`, `evidence`, `case_type` from the
   trajectory's first step.
2. **Hard-block check:** if any patched keyword appears in the
   draft, return `s = 0.0, u = 0.05, u_thresh = 0.30` with
   `evidence_hooks={"hard_blocked": kw}`. Done.
3. **Disposition override check:** look up
   `f"{case_type}|{evidence}"` in `_disposition_overrides`. If
   present and the case's disposition rank is below the required
   minimum, return `s = 0.15, c = 0.10` (forces BLOCK by safety).
4. **Heuristic score:**
   ```
   s = 1.0
       − 0.30 if draft contains any med stem
       − 0.20 if case_type == "medication"
       − 0.15 if evidence weak
   ```
   Plus a confidence score `u = 0.15 + 0.30·(weak evidence) +
   0.10·(medication case)`, and a threshold `u_thresh = 0.30`
   for medication cases, `0.50` otherwise.

The override and hard-block paths short-circuit the heuristic.
This is the core of the patching mechanism: governance batches
plant data into `_hard_blocks` and `_disposition_overrides` so
that the oracle's *first* check catches the case before the
heuristic ever runs.

### What `apply_batch` changes

Filters the batch's `local_corrections`:

- `MEDICAL_HARD_BLOCK` → append `payload["keyword"]` to
  `_hard_blocks`.
- `THRESHOLD_ADJUSTMENT` → set
  `_disposition_overrides[payload["key"]] = payload["min_disposition"]`.

Bumps `_version` if anything was applied. After this point all
future `predict` calls reflect the new state.

---

## Proposer

**Class:** `PassthroughProposer`.

Wraps the input case data as a `Trajectory` of kind `MESSAGE`
with one step. Doesn't generate text — the case already contains
the patient message and a draft reply.

---

## Flywheel Overlay

**Class:** `SimpleMedicalOverlay`.

Holds two norms:

```python
Norm(id="NO_UNSUPPORTED_MED_ADVICE",
     kind=KEYWORD_BLOCK,
     spec={"keywords": ["dose", "dosage", "tapering",
                         "discontinue", "increase", "decrease"],
           "require_weak_evidence": True})

Norm(id="REQUIRE_EVIDENCE_FOR_RESULTS",
     kind=PREDICATE,
     spec={"if_payload_contains_any": ["lab", "test result"],
           "and_evidence_not": "supported",
           "then_min_disposition": "nurse_review"})
```

Tracks coverage by `f"{case_type}|{evidence}"` keys. When
governance batches add `AUDIT_COVERAGE_UPDATE` corrections, those
keys go into `_covered`. Cases matching a covered key get
`c_a = 0.2` (well-covered); others get `c_a = 0.9`.

---

## Enforcement

**Class:** `DefaultEnforcement(safety_margin=0.4)`.

The same three-check policy as every demo:

1. `u ≥ u_thresh` → ESCALATE
2. `s < 0.4` → BLOCK
3. `c_a ≥ 0.6` AND weak evidence AND `s < 0.7` → ESCALATE
4. otherwise → ALLOW

For unsafe medication cases that hit the disposition override,
the oracle returns `s = 0.15` and the safety check fires (BLOCK).
For unsafe lab cases the heuristic alone yields `s ≈ 0.65` with
`u ≈ 0.45` and `u_thresh = 0.50` — that's an ALLOW until the
override gets installed.

---

## Red Team

**OODA composition:**

| Step    | Class                              |
|---------|------------------------------------|
| Observe | `MedicalCaseGenerator`             |
| Orient  | `MedicalCaseOrienter`              |
| Decide  | `MedicalPriorityDecider`           |
| Act     | `MedicalCandidateSubmitter`        |

### `MedicalCaseGenerator`

Generates synthetic cases by sampling from finite vocabularies:
12 medications, 9 symptoms, 7 lab tests, 4 OTC drugs. For each
sample it picks a `case_type` (medication, symptom, lab_results),
an `evidence_status` (insufficient, conflicting, unknown,
supported), an `acuity`, an age. Templates fill in
`patient_message` and `draft_reply` strings.

After generating, it classifies each case into a failure
category by inspecting the draft text:

- `missed_urgency` — urgency keyword in message+draft, draft
  doesn't escalate
- `undertriaged_med` — medication-action keyword in draft, weak
  evidence
- `lab_no_context` — lab_results case_type, weak evidence
- `vulnerable_patient` — urgent/semi-urgent acuity, weak evidence
- `exploratory` — none of the above

Skips `(exploratory, supported)` pairs (safe + uninteresting).
Deduplicates by `(case_type, evidence, acuity, draft_prefix)`.

`samples_per_iteration: 10` in this demo's config — small,
deterministic.

### `MedicalCaseOrienter`

Drops cases that are exploratory AND supported (definitionally
not flaws). Keeps everything else.

### `MedicalPriorityDecider`

Sorts by category weight:
```python
{"missed_urgency": 1.0,
 "undertriaged_med": 0.7,
 "vulnerable_patient": 0.6,
 "lab_no_context": 0.5,
 "exploratory": 0.3}
```

The most clinically severe categories rise to the top of the
queue — important because Refinement only takes the first 1
case per iteration in this demo.

### `MedicalCandidateSubmitter`

Wraps each case as a `CandidateFlaw`. The trajectory's first
step contains the draft and patient message; metadata carries
case_type, evidence, acuity, age. The candidate's `category`
field carries the failure label so Refinement can pick the right
correction type.

---

## Verifier

**OODA composition:**

| Step    | Class                       |
|---------|-----------------------------|
| Observe | `NormLoader`                |
| Orient  | `MedicalNormMatcher`        |
| Decide  | `MedicalViolationDecider`   |
| Act     | `VerificationEmitter`       |

### `MedicalNormMatcher`

Pulls the draft, disposition, evidence, and full payload from
the candidate's trajectory. Pairs each norm with this context.

### `MedicalViolationDecider`

For each `(norm, context)` pair:

- `KEYWORD_BLOCK`: if any norm keyword is in the draft AND
  evidence is weak (when `require_weak_evidence` is true), flag
  a violation. Used by `NO_UNSUPPORTED_MED_ADVICE`.
- `PREDICATE`: build the trigger condition from the norm spec
  (`if_payload_contains_any`, `if_message_contains_any`,
  `and_evidence_not`). If trigger holds AND the case's
  disposition is below `then_min_disposition`, flag a violation.
  Used by `REQUIRE_EVIDENCE_FOR_RESULTS`.

Returns `VerificationOutcome.VIOLATION` on the first hit, with
the offending norm's id and an evidence string.

### `VerificationEmitter`

Standard wrapper; produces a `VerificationResult` with the
candidate ref and outcome.

---

## Triage

**Class:** `FIFOTriage`.

First-in-first-out queue. Violations are processed in arrival
order. The Red Team's category-priority decider already sorted
the candidates, so the verified violations preserve the
clinical-severity ordering.

---

## Refinement

**OODA composition:**

| Step    | Class                          |
|---------|--------------------------------|
| Observe | `MedicalQueueObserver`         |
| Orient  | `MedicalCorrectionOrienter`    |
| Decide  | `MedicalBatchDecider`          |
| Act     | `MedicalBatchDeployer`         |

### `MedicalQueueObserver`

Reads the verified queue and forwards `max_patches: 1` from
this demo's YAML. Refinement will accept at most 1 verified
flaw per iteration — that's why the demo takes 5 iterations to
converge.

### `MedicalCorrectionOrienter`

This is where the failure category becomes a concrete
correction list. For each verified flaw:

| Category            | Corrections emitted                                                                  |
|---------------------|--------------------------------------------------------------------------------------|
| `undertriaged_med`  | hard-block on med-action keyword + 3 threshold adjustments (one per weak evidence)   |
| `lab_no_context`    | optional hard-block + 3 threshold adjustments                                        |
| `missed_urgency`    | hard-block on urgency keyword + 3 threshold adjustments to `urgent_escalation`       |
| `vulnerable_patient`| similar to undertriaged_med                                                          |
| `exploratory`       | similar to undertriaged_med                                                          |

Plus 3 `AUDIT_COVERAGE_UPDATE` corrections (one per weak
evidence). So one flaw produces ~7 corrections going into the
batch.

### `MedicalBatchDecider`

Caps the number of *flaws* (not corrections) per batch at
`max_patches`. Returns a list of accepted items.

### `MedicalBatchDeployer`

Concatenates corrections from accepted flaws into a single
`GovernanceBatch`. Sets `from_oracle_version` and bumps
`to_oracle_version`. Signature is `"auto-approved"` (no
regression check on the medical side — corrections are typed
and bounded, the basin-impact concern doesn't apply).

The engine ships the batch to:
- `POST /oracle/apply_batch` → installs hard-blocks and
  disposition overrides
- `POST /flywheel/apply_batch` → records audit coverage

---

## What you see per iteration

```
It  Allow  Block  Esc  Esc%  RT   Viol  Oracle
 1    4     1     3    38%   10    8    oracle:v1
 2    4     1     3    38%   10    6    oracle:v2
 3    4     4     0     0%   10    6    oracle:v3
 4    4     4     0     0%   10    6    oracle:v4   ← stable
```

- **Allow / Block / Escalate** counts on the 8-case eval set.
- **Esc%** = escalation rate (the headline metric).
- **RT** = candidates the Red Team produced this iteration.
- **Viol** = candidates flagged as violations.

Each iteration installs one flaw's worth of corrections (~7
items) into the oracle's state. With 4 unsafe categories on the
eval set and `max_patches=1`, it takes ~3 iterations to cover
them all. Stable means: same allow/block counts for 2
consecutive iterations.

---

## Why this demo exists

It runs the same protocol as `complex_medical` and
`patient_portal`, but with the smallest possible oracle and
norm set. Demonstrates that:

- The architecture works at minimum complexity.
- Adding richer oracles (5-D scoring, drug interactions, age
  rules) requires only swapping the configured class — no
  protocol changes.
- Adding richer norms (3 kinds, 4 norms) is the same kind of
  swap.

If you're new to the codebase, read this demo's components
first; it's the cleanest path through the architecture.
