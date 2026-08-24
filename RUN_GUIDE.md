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
