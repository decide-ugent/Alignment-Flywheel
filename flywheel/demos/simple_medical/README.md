# Simple Medical Demo — Minimal Viable Pipeline

Smallest possible medical scenario.  Trivial heuristic oracle (60
lines), 2 norms, FIFO triage, 8 fixed cases.  The architecture
runs end-to-end through the API with the minimum interesting set
of components.

## Run

```bash
python -m flywheel.demos.simple_medical.run --port 5003 \
    --output outputs/simple_medical
```

## Components

- **Oracle:** `SimpleMedicalOracle` — three rules, patchable hard-blocks
- **Overlay:** `SimpleMedicalOverlay` — 2 norms (KEYWORD_BLOCK, PREDICATE)
- **Triage:** `FIFOTriage`
- **Refinement:** caps batch at 2 corrections per iteration

## Why include this

Demonstrates that the architecture works at the lowest possible
complexity. Adding the patient-portal complexity (5 norms,
disposition overrides, generative Red Team) requires nothing but
a different YAML.
