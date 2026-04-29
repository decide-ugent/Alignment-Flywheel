# Patient Portal Governance Demo

The paper's main medical scenario.  A heuristic proxy oracle
evaluates patient-message draft replies; an overlay declares 4
norms (medication safety, urgency escalation, lab-result context,
unsupported-keyword block); a generative Red Team probes the case
space; refinement installs disposition overrides + hard-blocks
into the oracle's patchable state.

Expected: 33% escalation → 0% in 5 iterations on the 15-case eval set.

## Run

```bash
python -m flywheel.demos.patient_portal.run --port 5002 \
    --output outputs/patient_portal
```

## Architecture

- **Server side (Flask API at :5002)**
  - `PatientPortalOracle` (heuristic with patchable hard-blocks
    and disposition overrides)
  - `PassthroughProposer`
  - `PatientPortalOverlay` (4 norms across PREDICATE and
    KEYWORD_BLOCK kinds)
  - `DefaultEnforcement`
- **Governance side (OODA roles, in-process, HTTP clients)**
  - Red Team: `MedicalCaseGenerator` → `MedicalCaseOrienter` →
    `MedicalPriorityDecider` → `MedicalCandidateSubmitter`
  - Verifier: `NormLoader` → `MedicalNormMatcher` →
    `MedicalViolationDecider` → `VerificationEmitter`
  - Refinement: `MedicalQueueObserver` →
    `MedicalCorrectionOrienter` → `MedicalBatchDecider` →
    `MedicalBatchDeployer`
  - `PriorityTriage`

## Key claim demonstrated

The oracle is patched.  Each governance batch installs typed
corrections (`MEDICAL_HARD_BLOCK`, `THRESHOLD_ADJUSTMENT`) into
the oracle's internal state. On the next query, those corrections
short-circuit the heuristic — the same case that scored
`s=0.65, allow` before now returns `s=0.15, block` because the
governance layer added a constraint to the oracle's patchable
state. No retraining, no proposer change.
