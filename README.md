# CIG-AMF

CIG-AMF is a research codebase for two linked MARL studies:

1. **Paper A — Causal Influence Spectrum**: interventional response recovery,
   structural capacity `C`, behavioural direction `D`, optional gated latency,
   and regime-wise structural tracking.
2. **Paper B — CIG-AMF**: capacity-aware allocation of explicit pair modelling
   and direction-aware peripheral compression.

The confirmatory code path is deliberately fail-closed.  A green unit suite is
not sufficient by itself: H1 requires oracle-only threshold/support artifacts,
H2 requires a provenance-bound Page-CUSUM calibration, and Paper-B claim
validation requires all allocation/representation/scaling panels.

## First checks

```bash
python -m pip install -r requirements.txt
python -m compileall -q .
python -m pytest -q
python smoke_test.py
python test_bc_loss_control.py
```

For the full protocol and artifact-generation order, read `RUN_GUIDE.md`.

## External benchmark status

External adapters are capability-gated.  A benchmark is not counted as causal
validation unless its adapter exposes the operations required by that claim
(e.g. clone/restore plus the relevant intervention oracle).  Do not reinterpret
an unavailable capability as a negative experimental result.
