"""
smoke_test.py — run this first after obtaining the repository.

    python smoke_test.py

Verify every corrected mechanism before connecting it to the production
runner. NumPy checks run immediately; Torch checks skip themselves if Torch is
not installed.

Every test prints PASS/FAIL with measured values, exposing the mechanism's
behaviour rather than only returning "ok".
"""

import sys
import traceback

import numpy as np

PASS, FAIL, SKIP = [], [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  |  {detail}" if detail else ""))
    return cond


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =====================================================================
section("1. SEMANTIC SLOTS — four-role assignment")
# =====================================================================
try:
    from models.influence_signature import (
        ROLE_UNCERTAIN, ROLE_BENEFICIAL, ROLE_HARMFUL, ROLE_NAMES,
        ROLE_NEUTRAL, soft_role_assignment,
    )

    mu = np.array([0.30, -0.30, 0.00, 0.02, -0.02, 0.25])
    sg = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.90])
    P = soft_role_assignment(mu, sg, tau_role=0.05, sigma_hi=0.5)

    expect = [ROLE_BENEFICIAL, ROLE_HARMFUL, ROLE_NEUTRAL,
              ROLE_NEUTRAL, ROLE_NEUTRAL, ROLE_UNCERTAIN]
    got = P.argmax(1)

    for i, (m, s_, e, g) in enumerate(zip(mu, sg, expect, got)):
        print(f"       mu={m:+.2f} sigma={s_:.2f} -> {ROLE_NAMES[g]}")

    check("all six examples map to the expected role", bool(np.all(got == expect)))
    check("each probability row sums to one", bool(np.allclose(P.sum(1), 1.0)))
except Exception:
    traceback.print_exc(); FAIL.append("semantic slots")


# =====================================================================
section("2. C/D SIGNATURE — capacity remains independent of direction")
# =====================================================================
try:
    from models.influence_signature import InfluenceSignatureTracker

    tr = InfluenceSignatureTracker(n_agents=4, window=40)
    r = np.random.RandomState(0)

    for t in range(40):
        tr.update(
            0, 1, capacity=0.5 + r.randn() * 0.05, direction=0.5,
            sigma_capacity=0.1, sigma_direction=0.1, context_key=t % 3,
        )
        tr.update(
            0, 2, capacity=0.5, direction=0.5 * (1 if t % 2 else -1),
            sigma_capacity=0.1, sigma_direction=0.1, context_key=t % 3,
        )

    s1, s2 = tr.get_signature(0, 1), tr.get_signature(0, 2)
    print(f"       stable direction : C={s1[0]:.3f} D={s1[1]:+.3f} sigma_D={s1[3]:.3f}")
    print(f"       reversing D      : C={s2[0]:.3f} D={s2[1]:+.3f} sigma_D={s2[3]:.3f}")

    # The revised tracker deliberately uses an odd, short direction window
    # so D reacts faster than C. An alternating signal therefore retains at
    # most one unmatched sample in its current window.
    check("reversing direction remains near D=0", abs(s2[1]) <= 0.11)
    check("capacity remains high despite direction cancellation", s2[0] > 0.4)
except Exception:
    traceback.print_exc(); FAIL.append("signature")


# =====================================================================
section("3. BELIEF — core size follows structural capacity")
# =====================================================================
try:
    from models.belief_layer import BayesLightBeliefState as B

    def run(n_true, seeds=5):
        sizes, f1s = [], []
        for sd in range(seeds):
            rr = np.random.RandomState(sd)
            truth = {j: (0.6 if j <= n_true else 0.01)
                     for j in range(1, 13)}
            b = B(0, list(range(1, 13)), max_core_size=8, min_core_size=1,
                  core_rule="lcb", kappa=1.0, tau=0.10, alpha_decay=0.7)
            for t in range(60):
                sgm = max(0.02, 0.6 * np.exp(-t / 12))
                b.update_batch({j: (truth[j] + rr.randn() * 0.08,
                                    sgm + abs(rr.randn()) * 0.02) for j in truth})
            c = b.get_core_set(); gt = set(range(1, n_true + 1))
            sizes.append(len(c))
            f1s.append(2 * len(c & gt) / (len(c) + len(gt)))
        return float(np.mean(sizes)), float(np.mean(f1s))

    rows = [(n,) + run(n) for n in (1, 3, 6)]
    for n, sz, f1 in rows:
        print(f"       n_true={n} -> core_size={sz:.1f}  F1={f1:.2f}")

    check("core size increases with the true core", rows[0][1] < rows[1][1] < rows[2][1])
    check("F1 is high at each cardinality", all(f1 > 0.9 for _, _, f1 in rows))
except Exception:
    traceback.print_exc(); FAIL.append("belief")


# =====================================================================
section("4. UNCERTAINTY INFLATION — recovery after a structural shift")
# =====================================================================
try:
    import copy
    from models.belief_layer import BayesLightBeliefState as B

    old = {1: 0.6, 2: 0.6, 3: 0.4}
    for j in range(4, 13):
        old[j] = 0.01
    new = {j: 0.01 for j in range(1, 13)}
    new[7], new[8] = 0.7, 0.7

    def feed(b, tr, rng, n=60):
        for t in range(n):
            sgm = max(0.02, 0.6 * np.exp(-t / 12))
            b.update_batch({j: (tr[j] + rng.randn() * 0.08,
                                sgm + abs(rng.randn()) * 0.02) for j in tr})

    b = B(0, list(range(1, 13)), max_core_size=8, min_core_size=1,
          core_rule="lcb", kappa=1.0, tau=0.10, alpha_decay=0.7)
    feed(b, old, np.random.RandomState(0))
    print(f"       before shift       : core={sorted(b.get_core_set())}")

    b_no = copy.deepcopy(b)
    feed(b_no, new, np.random.RandomState(1), 40)
    core_no = sorted(b_no.get_core_set())

    b_yes = copy.deepcopy(b)

    # T4 [GPU_OPTIMIZATION_CONTRACT.md section 1.3]. Reset Pi only after
    # reading debiased_mu/sigma. Resetting first during vectorization makes
    # debiased_mu return the old anchor instead of the current estimate, so
    # inflation erases the estimate rather than lowering confidence.
    mu_before_1 = b_yes.debiased_mu(1)
    sig_before_1 = b_yes.debiased_sigma(1)

    st = b_yes.inflate_uncertainty(factor=2.5, t_reset=1)
    core_mid = sorted(b_yes.get_core_set())

    check("T4: inflation preserves the estimate while re-anchoring",
          abs(b_yes.debiased_mu(1) - mu_before_1) < 1e-9,
          f"mu before={mu_before_1:.4f} mu after={b_yes.debiased_mu(1):.4f}")
    check("T4: inflation materially increases uncertainty",
          b_yes.debiased_sigma(1) > 2.0 * sig_before_1,
          f"sigma before={sig_before_1:.4f} sigma after={b_yes.debiased_sigma(1):.4f}")

    feed(b_yes, new, np.random.RandomState(1), 80)
    core_yes = sorted(b_yes.get_core_set())

    print(f"       no inflation, 40   : core={core_no}")
    print(f"       after inflation    : core={core_mid}  sigma {st['sigma_before']:.3f}->{st['sigma_after']:.3f}")
    print(f"       inflation, 80      : core={core_yes}")

    check(
        "inflation permits recovery to new structural neighbours",
        {7, 8}.issubset(set(core_yes)) and not ({1, 2, 3} & set(core_yes)),
    )
    check("without inflation the old core remains sticky", set(core_no) != {7, 8})
except Exception:
    traceback.print_exc(); FAIL.append("inflation")


# =====================================================================
section("5. TARGETED EPSILON FORCING — budget and positivity")
# =====================================================================
try:
    from models.intervention import EpsilonForcedActionController as C

    c = C(n_agents=6, action_dim=4, eps=0.03,
          max_forced_per_step=None, rng=np.random.RandomState(0))
    c.set_priority(np.array([0.02, 0.02, 0.9, 0.8, 0.02, 0.02]))
    e = c.get_eps_per_agent()

    print(f"       epsilon by agent : {np.round(e, 4)}")
    print(f"       mean             : {e.mean():.4f} (target 0.0300)")

    check("total budget is preserved", abs(e.mean() - 0.03) < 1e-6)
    check("positivity: every agent has epsilon > 0", e.min() > 0, f"min={e.min():.4f}")
    check("priority allocation is nonuniform", e.max() / e.min() > 3, f"{e.max()/e.min():.1f}x")

    acts = [0] * 6
    probs = np.full((6, 4), 0.25, dtype=np.float32)
    mask, eff = c.apply(acts, probs)
    check("logged propensities are normalized",
          bool(np.allclose(eff.sum(1), 1.0, atol=1e-5)))
except Exception:
    traceback.print_exc(); FAIL.append("targeted forcing")


# =====================================================================
section("6. SCHEDULER — trigger and refractory period")
# =====================================================================
try:
    from training.scheduler import TwoTimescaleScheduler

    class FakeBelief:
        def __init__(self):
            self.n = 0
        def inflate_uncertainty(self, **kw):
            self.n += 1
            return {"n_pairs_inflated": 5}

    sch = TwoTimescaleScheduler(k0_warmup=3, refractory=5, z_threshold=3.0)
    bel = {0: FakeBelief()}

    for _ in range(3):
        sch.step_episode()

    r1 = sch.evaluate_drift(probe_z=5.0, matrix_z=0.0, belief_modules=bel)
    sch.step_episode()
    r2 = sch.evaluate_drift(probe_z=5.0, matrix_z=0.0, belief_modules=bel)

    for _ in range(6):
        sch.step_episode()
    r3 = sch.evaluate_drift(probe_z=5.0, matrix_z=0.0, belief_modules=bel)

    print(f"       first  (z=5): fired={r1['fired']} reason={r1['reason']}")
    print(f"       immediate retry: fired={r2['fired']} reason={r2['reason']}")
    print(f"       after refractory: fired={r3['fired']} reason={r3['reason']}")

    check("triggers when z exceeds the threshold", r1["fired"])
    check("refractory period blocks repeated triggers", not r2["fired"])
    check("triggers again after the refractory period", r3["fired"])
    check("belief uncertainty is inflated", bel[0].n == 2)
except Exception:
    traceback.print_exc(); FAIL.append("scheduler")


# =====================================================================
section("7. TORCH — proxy, memory, and ego latents")
# =====================================================================
try:
    import torch
    print(f"       torch {torch.__version__}")

    from models.peripheral_memory import PeripheralMultiMemory
    from models.structural_proxy import LocalCounterfactualProxyEnsemble
    from models.ego_conditioned_latent import EgoConditionedHeads

    A, OD, CD, PD, BD, H = 5, 12, 16, 24, 20, 3

    # ---- proxy ----
    px = LocalCounterfactualProxyEnsemble(
        obs_dim=OD, action_dim=A, core_dim=CD, periph_dim=PD, belief_dim=BD,
        n_ensemble=3, n_horizons=H, effect_mode="signed_aristocrat",
        use_doubly_robust=True, use_belief_input=False, seed=0,
    )
    rr = np.random.RandomState(0)
    for _ in range(400):
        px.add_sample(
            ego_id=0, neighbor_id=1,
            obs_i=rr.randn(OD), action_i=rr.randint(A),
            observed_action_j=rr.randint(A),
            z_core_excl_j=rr.randn(CD), m_periph_excl_j=rr.randn(PD),
            belief_summary=rr.randn(BD), target_return_h=float(rr.randn()),
            target_returns_multi=rr.randn(H),
            behaviour_prob_j=float(rr.uniform(0.1, 0.9)),
            was_forced=bool(rr.rand() < 0.1), state_key=int(rr.randint(3)),
        )
    px.train_step(n_steps=3, batch_size=64, holdout_size=64)
    out = px.score_batch_full(
        obs_i_batch=rr.randn(8, OD),
        action_i_batch=rr.randint(0, A, 8),
        observed_action_j_batch=rr.randint(0, A, 8),
        z_core_excl_j_batch=rr.randn(8, CD),
        m_periph_excl_j_batch=rr.randn(8, PD),
        belief_summary_batch=rr.randn(8, BD),
        policy_probs_j_batch=np.full((8, A), 1.0 / A, dtype=np.float32),
        observed_returns_batch=rr.randn(8),
        behaviour_probs_obs_batch=np.full(8, 0.3),
    )
    for k in ("mu", "sigma", "mu_per_h", "mu_range"):
        print(f"       {k:<10} shape={np.shape(out[k])}")

    check("directional output remains signed", bool(np.any(out["mu"] < 0)))
    check("mu_per_h shape [B,H]", out["mu_per_h"].shape == (8, H))
    check("proxy output has no ungated latency field (5D signature)", "latency" not in out)
    check("capacity range is nonnegative", bool(np.all(out["mu_range"] >= -1e-6)))

    dis = px.get_diagnostics()["ensemble_disagreement"]
    check("ensemble members are non-identical", dis > 1e-6,
          f"disagreement={dis:.5f} (v1 = 0.000)")

    # T1 [BB4]: vmap must match the Python-loop reference. Both paths receive
    # the same input, so any difference must come from a layout error such as
    # inverted repeat_interleave/repeat/view order, not different data.
    t1_obs = rr.randn(6, OD); t1_ai = rr.randint(0, A, 6)
    t1_z = rr.randn(6, CD); t1_m = rr.randn(6, PD); t1_b = rr.randn(6, BD)
    fast = px._predict_all_actions(t1_obs, t1_ai, t1_z, t1_m, t1_b)
    ref = px._predict_all_actions_reference(t1_obs, t1_ai, t1_z, t1_m, t1_b)
    check("T1: vmap matches the reference implementation (BB4, layout B*A)",
          bool(torch.allclose(fast, ref, atol=1e-5)),
          f"max diff={float((fast - ref).abs().max()):.2e}")

    # T3 [BB3]: member 0's gradient scale must not depend on n_ensemble. With
    # the same seed, member 0 at E=1 and E=4 uses the same bootstrap mask
    # because it depends on seed*7919+k rather than n_ensemble. Identically
    # populated buffers therefore provide identical training data.
    def _make_and_train(n_ens):
        p = LocalCounterfactualProxyEnsemble(
            obs_dim=OD, action_dim=A, core_dim=CD, periph_dim=PD, belief_dim=BD,
            n_ensemble=n_ens, n_horizons=H, effect_mode="signed_aristocrat",
            use_doubly_robust=False, use_belief_input=False, seed=42,
        )
        rr2 = np.random.RandomState(7)
        for _ in range(300):
            p.add_sample(
                ego_id=0, neighbor_id=1,
                obs_i=rr2.randn(OD), action_i=rr2.randint(A),
                observed_action_j=rr2.randint(A),
                z_core_excl_j=rr2.randn(CD), m_periph_excl_j=rr2.randn(PD),
                belief_summary=rr2.randn(BD), target_return_h=float(rr2.randn()),
                target_returns_multi=rr2.randn(H),
            )
        p.train_step(n_steps=50, batch_size=32, holdout_size=0)
        return p

    px1 = _make_and_train(1)
    px4 = _make_and_train(4)
    loss_e1 = float(px1.latest_loss_per_member[0])
    loss_e4_m0 = float(px4.latest_loss_per_member[0])
    rel_diff = abs(loss_e1 - loss_e4_m0) / max(abs(loss_e1), 1e-8)
    check("T3: member-0 gradient scale matches E=1 and E=4 (BB3)",
          rel_diff < 0.15,
          f"loss_e1={loss_e1:.4f} loss_e4_m0={loss_e4_m0:.4f} rel_diff={rel_diff:.3f}")

    # ---- peripheral memory ----
    pm = PeripheralMultiMemory(action_dim=A, n_free_slots=2,
                                 lb_coeff=1e-2, orth_coeff=1e-2)
    from models.peripheral_memory import FULL_ITEM_DIM
    items = np.zeros((14, FULL_ITEM_DIM), dtype=np.float32)
    items[:, 0] = rr.randint(0, A, 14)
    items[:, 1] = rr.uniform(0.1, 0.5, 14)  # C
    items[:, 2] = np.concatenate([
        rr.uniform(0.2, 0.5, 4), rr.uniform(-0.5, -0.2, 4),
        rr.uniform(-0.02, 0.02, 3), rr.uniform(0.1, 0.3, 3)])  # D
    items[:, 3] = 0.1  # sigma_C
    items[:, 4] = np.concatenate([np.full(11, 0.1), np.full(3, 0.9)])  # sigma_D
    items[:, 5] = rr.uniform(0, 1, 14)  # v_ctx
    items[:, 6] = 1.0                   # context-validity mask
    items[:, 7] = rr.uniform(0, 1, 14)  # normalized latency
    items[:, 8] = 1.0                   # latency-valid mask
    items[:, 9:] = rr.randn(14, 4)      # opaque relation features

    res = pm.forward_full(items)
    print(f"       memory shape={tuple(res['memory'].shape)} "
          f"slot_probs={tuple(res['slot_probs'].shape)}")
    usage = res["slot_usage"].detach().numpy()
    print(f"       slot usage = {np.round(usage, 3)}")

    check("memory shape [out_dim]", res["memory"].shape == (pm.out_dim,))
    check("six slots (four semantic plus two free)", res["slot_probs"].shape[1] == 6)
    check("aux_loss > 0", float(res["aux_loss"].detach()) > 0)
    check("no slot is dead", float(usage.min()) > 1e-4,
          f"min usage={usage.min():.4f}")

    ent = pm.get_slot_diagnostics()["usage_entropy_ratio"]
    check("usage entropy is adequate", ent > 0.5, f"entropy ratio={ent:.3f}")

    # ---- ego-conditioned heads ----
    hd = EgoConditionedHeads(latent_dim=32)
    z = torch.randn(24, 32)
    ego = torch.tensor(np.repeat([0, 1, 2], 8))
    nbr = torch.tensor(np.tile([1, 1, 2, 2, 3, 3, 4, 4], 3))
    # Keep repeated observations of a pair profile-stable while assigning
    # different profiles to the same neighbour under different egos. This is
    # the signal-aware positive/negative contract used by the revised loss.
    cd_by_pair = {}
    cd_rows = []
    for ego_id, neighbour_id in zip(ego.tolist(), nbr.tolist()):
        pair = (int(ego_id), int(neighbour_id))
        if pair not in cd_by_pair:
            cd_by_pair[pair] = torch.tensor(
                [0.2 + 0.25 * ego_id, -0.3 + 0.2 * ego_id],
                dtype=torch.float32,
            )
        cd_rows.append(cd_by_pair[pair])
    cd_target = torch.stack(cd_rows)
    lo = hd.compute_loss(z, ego, nbr, cd_target=cd_target)
    print(f"       L_influence={float(lo['influence'].detach()):.4f} "
          f"L_contrastive={float(lo['contrastive'].detach()):.4f}")
    check("influence loss is finite", np.isfinite(float(lo["influence"].detach())))
    check("contrastive loss > 0", float(lo["contrastive"].detach()) > 0,
          "requires same-neighbour, different-ego samples in the batch")

    # ---- drift probe ----
    from models.drift_probe import DriftDetector, MatrixDriftDetector
    det = DriftDetector(obs_dim=OD, action_dim=A, n_horizons=H,
                        warmup_batches=5, batch_size=32)
    for ep in range(4):
        st = det.step(ep, px.buffer, n_train_batches=3)
    print(f"       probe phase={st['phase']} frozen={det.frozen is not None}")
    check("probe can be frozen", det.frozen is not None)
    check("frozen probe has no trainable parameters",
          all(not p.requires_grad for p in det.frozen.parameters()))

    md = MatrixDriftDetector()
    W = np.random.RandomState(0).randn(6, 6) * 0.1
    md.update(W); md.update(W * 1.01)
    for _ in range(6):
        md.update(W * (1 + 0.01 * np.random.rand()))
    z_small = md.z_score()
    md.update(W * 5.0)
    z_big = md.z_score()
    print(f"       matrix z: small change={z_small:.2f}  large change={z_big:.2f}")
    check("matrix distinguishes small and large changes", z_big > z_small + 1)

except ImportError as e:
    print(f"       SKIP torch section: {e}")
    SKIP.append("torch tests")
except Exception:
    traceback.print_exc(); FAIL.append("torch")


# =====================================================================
print("\n" + "=" * 70)
print(f"RESULT: {len(PASS)} PASS | {len(FAIL)} FAIL | {len(SKIP)} SKIP")
print("=" * 70)
if FAIL:
    print("Failures:", ", ".join(FAIL))
    sys.exit(1)
print("All mechanism checks passed. Next: read RUN_GUIDE.md.")
