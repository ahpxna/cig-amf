# CIG-AMF Run Guide

## 1. Validate the code path

```bash
python -m compileall -q .
python -m pytest -q
python smoke_test.py
python test_bc_loss_control.py
bash scripts/run_all.sh --quick
```

`--quick` is a protocol-path smoke check only.  Scientific gates that fail due
to deliberately tiny budgets are not confirmatory evidence.

## 2. H1 oracle support and thresholds

Generate oracle-only pair rows with `scripts/collect_h1_oracle_support.py`.
Then certify support and freeze thresholds from development seeds only:

```bash
python scripts/validate_h1_oracle_support.py ... --out h1_support.json
python scripts/calibrate_h1_oracle_thresholds.py ... --out h1_thresholds.json
```

The confirmatory H1 seeds must be disjoint from both development seed sets.
The loaders verify artifact protocol, source SHA-256 digests, support metadata,
and target-policy metadata.

## 3. H2 Page-CUSUM calibration

Use at least 40 independent no-change development trajectories.  Collection
pretrains the same H2 common-policy protocol, restores its monitoring-ready
witness, freezes policy/representation learning, and records a canonical null
contract hash.

```bash
python scripts/collect_cusum_no_change.py \
  --seeds <development seeds> \
  --pretrain-episodes 60 --episodes 200 --max-steps 30 \
  --out cusum_null.json

python scripts/calibrate_cusum_threshold.py \
  --no-change-z-json cusum_null.json \
  --allowance 0.5 --false-alarm-target 0.05 \
  --development-seeds <same development seeds> \
  --out cusum_calibration.json
```

The H2 launcher rejects malformed ranges, protocol mismatches, source hash
mismatches, contract mismatches, insufficient trajectories, and seed overlap.
A separate held-out no-change validation bank is recommended for reporting the
final empirical false-alarm rate.

## 4. Paper B panels

The Paper-B allocation runner now uses two distinct branch points:

* selector isolation: selector-neutral **full-explicit pretraining**;
* end-to-end: a **common untrained initialization** for every selector.

Decision fidelity in selector isolation is measured only after the selector is
committed and pair allocation has been reconciled.

Run the additional adaptive-budget panel:

```bash
python scripts/run_paper_b_adaptive_budget.py \
  --seeds 0 1 2 3 4 --episodes 200 --k-min 2 --k-max 5
```

The scaling panel reports reward, throughput, representation memory,
Full-Explicit policy/value/action fidelity, and mean/p50/p95 inference latency.

## 5. Confirmatory orchestration

Set the frozen artifacts used by `scripts/run_all.sh`:

```bash
export CIG_H1_THRESHOLD_CALIBRATION=/abs/path/h1_thresholds.json
export CIG_H1_ORACLE_SUPPORT=/abs/path/h1_support.json
export CIG_CUSUM_CALIBRATION=/abs/path/cusum_calibration.json
bash scripts/run_all.sh --run-id <unique-id>
```

Do not reuse stale result directories.  Regenerate `SOURCE_MANIFEST.json`
after any source change:

```bash
python scripts/build_source_manifest.py --out SOURCE_MANIFEST.json
```

## 6. External benchmark repository and adapter workflow

All pinned third-party repositories are managed under one root:

```text
external_envs/
  manifest.json
  repos/
    flatland-rl/
    robotic-warehouse/
    CybORG/
    CityFlow/
```

Existing clones from the old `external_envs/<repo>` layout are migrated in
place the next time setup is run.

Clone/verify the pinned sources:

```bash
bash scripts/setup_external_envs.sh
python scripts/external_env_manager.py status
```

Install each pinned repository into the active Python environment when the
native/system prerequisites are available:

```bash
bash scripts/setup_external_envs.sh --install
```

Before a long external experiment, run the executable adapter contract check:

```bash
python scripts/run_external_suite.py \
  --environment flatland \
  --panels training h1 h2 latency \
  --out results/external_flatland_contract.json
```

Repeat with `rware`, `cyborg`, and `cityflow`.  A blocked H2 or latency panel is
intentional unless that environment has a scientifically defined structural /
behavioural intervention or a ground-truth latency oracle.  Do not enable a
capability flag merely to make the manifest green.

Actual Final-CIGAMF architecture-generalization training is separate from the
capability manifest:

```bash
python scripts/run_external_training.py \
  --environment rware --seeds 0 1 2 3 4 \
  --episodes 100 --agent-count 6 --max-steps 60 \
  --out results/external_rware_training
```

For CityFlow, pass `--config-path /abs/path/to/config.json` when automatic
example-config discovery is not appropriate.  External training evidence is
an architecture/generalization result; it is not automatically an H1/H2 or
latency claim.

## External benchmark runtime isolation

Do **not** install the pinned external repositories directly into the main CIG
virtual environment.  The pinned Flatland revision requires `numpy<2`, while
the confirmatory CIG environment is validated separately.  External packages
are installed into `external_envs/runtime/` and external commands automatically
re-exec there once the runtime is marked ready.

On macOS, use Python 3.12 for the external runtime:

```bash
brew install python@3.12
CIG_EXTERNAL_PYTHON="$(brew --prefix python@3.12)/bin/python3.12" \
  bash scripts/setup_external_envs.sh --install --recreate-runtime
python scripts/external_env_manager.py status
```

If the main CIG environment already runs the project but only lacks pytest,
install the test-only dependency set without perturbing NumPy/Torch:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

Legacy external-suite outputs such as `results/external_flatland` may be JSON
files from older code.  If a new command needs the same path as a directory,
the runner preserves the old file as `*.legacy-file` and creates the directory
rather than raising `FileExistsError`.
