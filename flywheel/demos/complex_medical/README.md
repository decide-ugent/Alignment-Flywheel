# Complex Medical Demo

Richer medical scenario.  5-dimensional oracle (medication risk ×
interaction × action severity × patient vulnerability × evidence
quality), 4 norms across 3 `NormKind`s, priority triage, 18 fixed
cases including drug interactions, vulnerable patients, and
multi-specialty content.

## Run

```bash
python -m flywheel.demos.complex_medical.run --port 5004 \
    --output outputs/complex_medical
```

## Components

- **Oracle:** `ComplexMedicalOracle` — 5-dimensional scoring with
  drug-interaction tables, age/comorbidity vulnerability factors,
  patchable hard-blocks and per-specialty thresholds
- **Overlay:** `ComplexMedicalOverlay` — 4 norms covering
  high-risk meds, interactions, vulnerable patients, severe
  actions
- **Triage:** `PriorityTriage`
- **Refinement:** caps batch at 5 corrections per iteration

## Why include this

Same governance protocol as `simple_medical` and
`patient_portal`, but with substantially more complex oracle
internals.  Demonstrates that the architecture scales with oracle
complexity — only the configured class names change.
