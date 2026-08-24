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

External repositories are centralized under `external_envs/repos/` and are
resolved through `envs.external.registry`.  Use `scripts/setup_external_envs.sh`
and `scripts/external_env_manager.py status` rather than adding ad-hoc
`sys.path` entries or cloning benchmark copies elsewhere in the project.

### Python environments for external benchmarks

Pinned external benchmarks are intentionally isolated from the main CIG-AMF
environment. Run `scripts/setup_external_envs.sh --install` with Python 3.12;
the managed runtime lives at `external_envs/runtime/` and external runners
switch to it automatically. See `RUN_GUIDE.md` for macOS commands.
