"""
env_audit_staged.py — audit Omni-Arena through three cumulative stages
instead of enabling P1-P4 together as env_audit.py does.

RATIONALE:
P1-P4 can silently cancel one another when built and tested together. P3
(congestion) can swamp P1 (conditional influence) and collapse T3; P3 queue
noise combined with P2's designed latency can make T4 look favorable for the
wrong reason; P2's H=8 oracle window can average across two structures if P4
occurs mid-window; and P4 moves the bottleneck, changing the same state
variables read by P1's delta_ij(s).

Cumulative structure (P1 and P2 share a stage because neither changes the
environment dynamics; they only affect the influence table and do not
interact with one another):

  BASELINE  P0 only (all P1-P4 flags disabled)        -- Block A reference
  Block A   P0 + P1 + P2  (conditional_gates=True, latency_ladder=True,
                            congestion=False, structural_shift=False)
  Block B   Block A + P3  (congestion=True)           -- MANDATORY; isolate it
  Block C   Block B + P4  (structural_shift=True)

This script does not replace env_audit.py. The latter enables every flag at
once and remains useful as the final all-flags check. All metric functions
(T1-T6 and corr(Phi,W*)) are imported from env_audit.py without duplicating
their logic.
"""
import argparse
import json
import os
import sys
import numpy as np

from omni_arena import OmniArena
import env_audit as ea  # reuse metric-computation functions, no forking


# ============================================================
# Configuration matches env_audit.py so results remain comparable.
# ============================================================
BASE_ENV_KWARGS = dict(
    n_agents=ea.N_AGENTS,
    grid_size=ea.GRID_SIZE,
    n_zones=ea.N_ZONES,
    max_steps=ea.MAX_STEPS,
    phase_length=ea.PHASE_LENGTH,
    causal_horizon=ea.HORIZON,
)
SEED = ea.SEED

# ------------------------------------------------------------------
# P1-P4 flags for each stage. BASELINE is not one of the three formal stages
# (A/B/C); it is only the P0-only reference used to measure Block A's delta,
# consistent with the specification that T3 should improve from a weak,
# DIG-like baseline to PASS.
# ------------------------------------------------------------------
BLOCKS = [
    (
        "BASELINE",
        "P0 only (mọi cờ P1-P4 tắt) -- điểm mốc, KHÔNG phải 1 trong 3 khối chính thức",
        dict(
            enable_conditional_gates=False,
            enable_latency_ladder=False,
            enable_congestion=False,
            enable_structural_shift=False,
        ),
    ),
    (
        "A",
        "P0 + P1 + P2 (conditional_gates, latency_ladder) -- rủi ro THẤP, chỉ đụng bảng ảnh hưởng",
        dict(
            enable_conditional_gates=True,
            enable_latency_ladder=True,
            enable_congestion=False,
            enable_structural_shift=False,
        ),
    ),
    (
        "B",
        "A + P3 (congestion) -- rủi ro CAO, có thể phá T1 -- MANDATORY, phải cô lập",
        dict(
            enable_conditional_gates=True,
            enable_latency_ladder=True,
            enable_congestion=True,
            enable_structural_shift=False,
        ),
    ),
    (
        "C",
        "B + P4 (structural_shift) -- rủi ro TRUNG BÌNH, tương tác với H",
        dict(
            enable_conditional_gates=True,
            enable_latency_ladder=True,
            enable_congestion=True,
            enable_structural_shift=True,
        ),
    ),
]


def run_block(flags):
    """
    Run the complete T1-T6 and corr(Phi,W*) metric set for one stage using
    env_audit.py's functions without duplicating their logic. The primary
    environment for T1/T2/T3/T4/T5/correlation/sign checks always uses
    mode="behavioral_drift", matching env_audit.py.main(). Structural shifts
    are probed separately by the dedicated environments inside ea.run_t6().
    """
    env_kwargs = dict(BASE_ENV_KWARGS, **flags)
    main_env = OmniArena(mode="behavioral_drift", seed=SEED, **env_kwargs)
    main_env.reset()

    m1 = ea.run_oracle_based_metrics(main_env)
    m3 = ea.run_t3(main_env)
    m4 = ea.run_t4(main_env)
    m6 = ea.run_t6(env_kwargs=dict(env_kwargs), seed=SEED)
    msign = ea.run_oracle_sign_check(main_env)
    mabs = ea.oracle_no_abs_direct_check(main_env)

    return {
        "t1_gini": m1["t1_gini"],
        "t2_neg_frac": m1["t2_neg_frac"],
        "t2_pos_frac": m1["t2_pos_frac"],
        "t3_cv_mean": m3["t3_cv_mean"],
        "t4_spread": m4["t4_spread"],
        "t5_snr": m1["t5_snr"],
        "t6_ratio": m6["t6_ratio"],
        # RC-2 reverses the former semantics. This field previously asked
        # whether behavioural ‖dPhi‖ was exactly zero. That condition was
        # always true because the value was a hard-coded literal, which also
        # pinned T6 at 1e12. It now measures Φ̃ = E_s[phi*delta], and PASS
        # requires a nonzero value: behavioural drift must leave a measurable
        # trace, although one much smaller than structural drift.
        "t6_behav_exact": m6["t6_delta_phi_behavioural"] > 0.0,
        "t6_static_phi_invariant": m6["t6_static_phi_invariant"],
        "corr_phi_w": m1["corr_phi_w"],
        "corr_phi_w_static": m1["corr_phi_w_static"],
        "corr_phi_w_mobile": m1["corr_phi_w_mobile"],
        "state_unique_fraction": m1["state_bank_diagnostics"]["unique_state_fraction"],
        "control_nonzero_fraction": m1["control_diagnostics"]["state_nonzero_fraction"],
        "_raw": {"m1": m1, "m3": m3, "m4": m4, "m6": m6, "msign": msign, "mabs": mabs},
    }


METRIC_ORDER = [
    # [docs/CIG-AMF_training_debug_master.md section 5.5] t5_snr, t6_ratio,
    # and corr_phi_w now have upper as well as lower bounds. The old audit
    # only asked whether a signal existed (v > threshold), so an SNR of 1e9
    # passed easily. Such a value merely indicates that a lookup table is
    # hard-coded and then measured against its own assumption (control pairs
    # have W*=0 exactly); it does not establish detectable but nontrivial
    # structure. The [3,20] SNR/T6 and [0.70,0.95] correlation intervals turn
    # the audit into a nontriviality test, the debug document's central gate
    # change.
    ("t1_gini", "T1 Gini(|W*|)", lambda v: v > 0.30),
    ("t2_neg_frac", "T2 neg frac", lambda v: v > 0.15),
    ("t2_pos_frac", "T2 pos frac", lambda v: v > 0.15),
    ("t3_cv_mean", "T3 mean CV", lambda v: v > 0.30),
    ("t4_spread", "T4 sign-flip frac (NEW methodology)", lambda v: v > 0.3),
    ("t5_snr", "T5 SNR", lambda v: 3.0 <= v <= 20.0),
    ("t6_ratio", "T6 ratio", lambda v: 3.0 <= v <= 20.0),
    ("corr_phi_w_static", "corr(Phi,W*) static", lambda v: 0.65 <= v <= 0.95),
    ("state_unique_fraction", "state-bank unique frac", lambda v: v >= 0.80),
    ("control_nonzero_fraction", "control nonzero frac", lambda v: v >= 0.25),
]


def fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if abs(v) >= 1e6:
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


def print_block_result(name, desc, flags, metrics, prev_metrics, prev_name):
    print("\n" + "=" * 78)
    print(f"BLOCK {name}: {desc}")
    print("=" * 78)
    print(
        f"flags: enable_conditional_gates={flags['enable_conditional_gates']}  "
        f"enable_latency_ladder={flags['enable_latency_ladder']}  "
        f"enable_congestion={flags['enable_congestion']}  "
        f"enable_structural_shift={flags['enable_structural_shift']}"
    )

    name_w = max(len(label) for _, label, _ in METRIC_ORDER)
    for key, label, passfn in METRIC_ORDER:
        val = metrics[key]
        status = "PASS" if passfn(val) else "FAIL"
        line = f"  [{status}] {label:{name_w}s}  value = {fmt(val)}"
        if prev_metrics is not None:
            prev_val = prev_metrics[key]
            delta = val - prev_val
            prev_status = "PASS" if passfn(prev_val) else "FAIL"
            arrow = f"{fmt(prev_val)} -> {fmt(val)} (delta{delta:+.4f}, was {prev_status} now {status})"
            line += f"    [{prev_name}->{name}: {arrow}]"
        print(line)

    print(
        f"  [T6 ||dPhi~||(behavioural) > 0 (đo trên Φ̃)] = "
        f"{metrics['t6_behav_exact']}   "
        f"[phi tĩnh bất biến = {metrics['t6_static_phi_invariant']}]"
    )
    print(
        f"  [oracle no-abs() has_negative_delta] = "
        f"{metrics['_raw']['mabs']['has_negative_delta']}"
    )
    print(
        f"  [corr diagnostic] mobile={fmt(metrics['corr_phi_w_mobile'])}  "
        f"global={fmt(metrics['corr_phi_w'])}"
    )


def main(json_out=None):
    print("#" * 78)
    print("# OMNI-ARENA env_audit_staged.py -- 3 cumulative blocks (A, B, C)")
    print("# Reuses T1-T6/corr metric functions from env_audit.py verbatim.")
    print("#" * 78)
    print(
        f"\nshared config: n_agents={BASE_ENV_KWARGS['n_agents']} "
        f"grid={BASE_ENV_KWARGS['grid_size']} n_zones={BASE_ENV_KWARGS['n_zones']} "
        f"H={BASE_ENV_KWARGS['causal_horizon']} n_states_T1={ea.N_STATES_T1} "
        f"n_states_T3={ea.N_STATES_T3} n_control_pairs={ea.N_CONTROL_PAIRS} "
        f"n_trials={ea.N_TRIALS} seed={SEED}"
    )

    results = {}
    prev_name = None
    prev_metrics = None
    order = []

    for name, desc, flags in BLOCKS:
        print(f"\n>>> running block {name} ...")
        metrics = run_block(flags)
        results[name] = (desc, flags, metrics)
        order.append(name)
        print_block_result(name, desc, flags, metrics, prev_metrics, prev_name)
        prev_name, prev_metrics = name, metrics

    # ------------------------------------------------------------
    # Diagnostic call-outs defined by the staged-audit acceptance criteria.
    # ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("DIAGNOSTIC CALL-OUTS (per staged-audit plan)")
    print("=" * 78)

    base_m = results["BASELINE"][2]
    a_m = results["A"][2]
    b_m = results["B"][2]
    c_m = results["C"][2]

    print(
        f"\n[Block A] T3 mean CV: BASELINE={fmt(base_m['t3_cv_mean'])} -> "
        f"A={fmt(a_m['t3_cv_mean'])}  "
        f"({'PASS' if a_m['t3_cv_mean'] > 0.30 else 'FAIL'}, expect BASELINE weak -> A PASS)"
    )
    print(
        f"[Block A] T4 sign-flip role frac (NEW methodology -- fixed "
        f"forced_step=0, H-sweep {ea.T4_H_SWEEP}, see env_audit.py's run_t4() "
        f"docstring; key 't4_spread' kept for compat, meaning changed from the "
        f"old sliding-forced_step 'peak spread'): "
        f"BASELINE={fmt(base_m['t4_spread'])} -> A={fmt(a_m['t4_spread'])}  "
        f"({'PASS' if a_m['t4_spread'] > 0.3 else 'FAIL'}, expect BASELINE weak -> A PASS)"
    )
    print(
        f"[Block A] T1/T2/T5 should already be fine at baseline (P1/P2 don't "
        f"touch congestion): "
        f"T1 BASELINE={fmt(base_m['t1_gini'])} A={fmt(a_m['t1_gini'])}; "
        f"T5 BASELINE={fmt(base_m['t5_snr'])} A={fmt(a_m['t5_snr'])}"
    )

    t1_b_verdict = (
        "STILL PASS -- OK" if b_m["t1_gini"] > 0.30 else
        "DROPPED BELOW 0.30 -- per blueprint Sec 1.2, HALVE the P3 congestion "
        "amplitude cap (0.15x -> 0.075x) and re-run Block B"
    )
    print(
        f"\n[Block B -- CRITICAL] T1 Gini MUST STAY PASS (>0.30): "
        f"A={fmt(a_m['t1_gini'])} -> B={fmt(b_m['t1_gini'])}  ({t1_b_verdict})"
    )
    print(
        f"[Block B] T3 mean CV (checking the exact P3<->P1 interaction risk -- "
        f"does congestion swamp the blocker's conditional gate?): "
        f"A={fmt(a_m['t3_cv_mean'])} -> B={fmt(b_m['t3_cv_mean'])}  "
        f"({'held up' if b_m['t3_cv_mean'] > 0.30 else 'DEGRADED below 0.30 threshold'})"
    )

    print(
        f"\n[Block C] T6 ratio: B={fmt(b_m['t6_ratio'])} -> C={fmt(c_m['t6_ratio'])}  "
        f"({'PASS (>3.0)' if c_m['t6_ratio'] > 3.0 else 'FAIL (<=3.0)'})"
    )
    print(
        f"[Block C] ||dPhi~||(behavioural) > 0: {c_m['t6_behav_exact']}  "
        f"({'PASS' if c_m['t6_behav_exact'] else 'FAIL -- Φ̃ không phản ứng với behavioural drift: state bank quá nhỏ hoặc gate không phụ thuộc hành vi'})"
    )

    # ------------------------------------------------------------
    # Final side-by-side summary table
    # ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("FINAL SUMMARY -- all blocks side by side")
    print("=" * 78)

    col_w = 14
    header = f"  {'metric':22s}" + "".join(f"{n:>{col_w}s}" for n in order)
    print(header)
    print("  " + "-" * (22 + col_w * len(order)))
    for key, label, passfn in METRIC_ORDER:
        row = f"  {label:22s}"
        for n in order:
            v = results[n][2][key]
            status = "P" if passfn(v) else "F"
            row += f"{fmt(v) + '(' + status + ')':>{col_w}s}"
        print(row)

    row = f"  {'T6 dPhi~behav > 0':22s}"
    for n in order:
        v = results[n][2]["t6_behav_exact"]
        row += f"{str(v):>{col_w}s}"
    print(row)

    print(
        "\nNote: BASELINE column is the P0-only reference point used to compute "
        "Block A's delta; it is not one of the 3 official audited blocks (A/B/C)."
    )

    # Gate only the interactions each cumulative block is designed to expose.
    # BASELINE is a reference, and T4 remains diagnostic rather than required.
    gate_checks = [
        ("A conditional variability", a_m["t3_cv_mean"], a_m["t3_cv_mean"] > 0.30),
        ("B preserves influence inequality", b_m["t1_gini"], b_m["t1_gini"] > 0.30),
        ("B preserves conditional variability", b_m["t3_cv_mean"], b_m["t3_cv_mean"] > 0.30),
        ("C structural/behavioural separation", c_m["t6_ratio"], 3.0 <= c_m["t6_ratio"] <= 20.0),
        ("C behavioural effective-structure response", c_m["t6_behav_exact"], bool(c_m["t6_behav_exact"])),
        ("C static Phi invariant under behavioural drift", c_m["t6_static_phi_invariant"], bool(c_m["t6_static_phi_invariant"])),
    ]
    required_pass = all(bool(ok) for _, _, ok in gate_checks)
    print(
        "\nRequired staged environment gate: "
        + ("PASS" if required_pass else "FAIL")
    )
    machine_checks = [
        {
            "name": name,
            "value": (
                float(value)
                if isinstance(value, (int, float, np.integer, np.floating))
                else bool(value)
            ),
            "required": True,
            "passed": bool(ok),
        }
        for name, value, ok in gate_checks
    ]
    if json_out:
        output_path = os.path.abspath(json_out)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "audit": "cumulative_staged",
                    "required_gate_pass": required_pass,
                    "checks": machine_checks,
                },
                handle,
                indent=2,
            )
            handle.write("\n")

    return {
        "blocks": results,
        "required_gate_pass": required_pass,
        "machine_checks": machine_checks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default=None)
    cli_args = parser.parse_args()
    audit_result = main(json_out=cli_args.json_out)
    raise SystemExit(0 if audit_result["required_gate_pass"] else 2)
