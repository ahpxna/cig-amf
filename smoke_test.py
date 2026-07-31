"""
smoke_test.py — Chạy cái này ĐẦU TIÊN sau khi tải về.

    python smoke_test.py

Kiểm tra từng cơ chế đã vá có hoạt động không, TRƯỚC khi cắm vào runner
thật. Phần numpy chạy được ngay; phần torch tự bỏ qua nếu chưa cài.

Mỗi test in ra PASS/FAIL kèm con số, để bạn thấy cơ chế làm gì chứ không
chỉ thấy "ok".
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
section("1. SLOT NGỮ NGHĨA — gán đúng 4 vai trò chưa?")
# =====================================================================
try:
    from models.influence_signature import (
        ROLE_ANOMALOUS, ROLE_BENEFICIAL, ROLE_HARMFUL, ROLE_NAMES,
        ROLE_NEUTRAL, soft_role_assignment,
    )

    mu = np.array([0.30, -0.30, 0.00, 0.02, -0.02, 0.25])
    sg = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.90])
    P = soft_role_assignment(mu, sg, tau_role=0.05, sigma_hi=0.5)

    expect = [ROLE_BENEFICIAL, ROLE_HARMFUL, ROLE_NEUTRAL,
              ROLE_NEUTRAL, ROLE_NEUTRAL, ROLE_ANOMALOUS]
    got = P.argmax(1)

    for i, (m, s_, e, g) in enumerate(zip(mu, sg, expect, got)):
        print(f"       mu={m:+.2f} sigma={s_:.2f} -> {ROLE_NAMES[g]}")

    check("gán đúng cả 6 trường hợp", bool(np.all(got == expect)))
    check("mỗi hàng tổng = 1", bool(np.allclose(P.sum(1), 1.0)))
except Exception:
    traceback.print_exc(); FAIL.append("slot ngữ nghĩa")


# =====================================================================
section("2. SIGNATURE — kênh độ lớn có độc lập với dấu không?")
# =====================================================================
try:
    from models.influence_signature import InfluenceSignatureTracker

    tr = InfluenceSignatureTracker(n_agents=4, window=40)
    r = np.random.RandomState(0)

    for t in range(40):
        tr.update(0, 1, +0.5 + r.randn() * 0.05, 0.1, 0.2, context_key=t % 3)
        tr.update(0, 2, 0.5 * (1 if t % 2 else -1), 0.1, 1.0, context_key=t % 3)

    s1, s2 = tr.get_signature(0, 1), tr.get_signature(0, 2)
    print(f"       nhất quán  : signed={s1[0]:+.3f} abs={s1[1]:.3f} tstd={s1[3]:.3f}")
    print(f"       đảo chiều  : signed={s2[0]:+.3f} abs={s2[1]:.3f} tstd={s2[3]:.3f}")

    check("đảo chiều: signed≈0 nhưng abs lớn", abs(s2[0]) < 0.1 and s2[1] > 0.4,
          "nếu abs=|mean| thì cả hai đều ≈0 -> mất thông tin")
    check("phân biệt được qua temporal_std", s2[3] > 5 * s1[3])
except Exception:
    traceback.print_exc(); FAIL.append("signature")


# =====================================================================
section("3. BELIEF — core size có bám cấu trúc thật không?")
# =====================================================================
try:
    from models.belief_layer import BayesLightBeliefState as B

    def run(n_true, seeds=5):
        sizes, f1s = [], []
        for sd in range(seeds):
            rr = np.random.RandomState(sd)
            truth = {j: ((0.6 if j % 2 else -0.6) if j <= n_true else 0.01)
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

    check("core size tăng theo n_true", rows[0][1] < rows[1][1] < rows[2][1])
    check("F1 cao ở mọi mức", all(f1 > 0.9 for _, _, f1 in rows))
except Exception:
    traceback.print_exc(); FAIL.append("belief")


# =====================================================================
section("4. BƠM PHỒNG BẤT ĐỊNH — có thoát được core cũ sau shift không?")
# =====================================================================
try:
    import copy
    from models.belief_layer import BayesLightBeliefState as B

    old = {1: +0.6, 2: -0.6, 3: +0.4}
    for j in range(4, 13):
        old[j] = 0.01
    new = {j: 0.01 for j in range(1, 13)}
    new[7], new[8] = +0.6, -0.5

    def feed(b, tr, rng, n=60):
        for t in range(n):
            sgm = max(0.02, 0.6 * np.exp(-t / 12))
            b.update_batch({j: (tr[j] + rng.randn() * 0.08,
                                sgm + abs(rng.randn()) * 0.02) for j in tr})

    b = B(0, list(range(1, 13)), max_core_size=8, min_core_size=1,
          core_rule="lcb", kappa=1.0, tau=0.10, alpha_decay=0.7)
    feed(b, old, np.random.RandomState(0))
    print(f"       trước shift        : core={sorted(b.get_core_set())}")

    b_no = copy.deepcopy(b)
    feed(b_no, new, np.random.RandomState(1), 40)
    core_no = sorted(b_no.get_core_set())

    b_yes = copy.deepcopy(b)

    # ---- T4 [GPU_OPTIMIZATION_CONTRACT.md mục 1.3] ----
    # Reset Pi phải nằm SAU khi đã đọc debiased_mu/sigma. Nếu ai đó vector
    # hoá và lỡ reset trước, debiased_mu trả điểm neo CŨ thay vì ước lượng
    # hiện tại -> bơm phồng biến thành xoá trắng thay vì hạ độ tin cậy.
    mu_before_1 = b_yes.debiased_mu(1)
    sig_before_1 = b_yes.debiased_sigma(1)

    st = b_yes.inflate_uncertainty(factor=2.5, t_reset=1)
    core_mid = sorted(b_yes.get_core_set())

    check("T4: inflate KHÔNG xoá mu (re-anchor, không reset)",
          abs(b_yes.debiased_mu(1) - mu_before_1) < 1e-9,
          f"mu trước={mu_before_1:.4f} mu sau={b_yes.debiased_mu(1):.4f}")
    check("T4: inflate làm sigma tăng rõ rệt",
          b_yes.debiased_sigma(1) > 2.0 * sig_before_1,
          f"sigma trước={sig_before_1:.4f} sigma sau={b_yes.debiased_sigma(1):.4f}")

    feed(b_yes, new, np.random.RandomState(1), 40)
    core_yes = sorted(b_yes.get_core_set())

    print(f"       KHÔNG bơm, sau 40  : core={core_no}")
    print(f"       CÓ bơm, ngay sau   : core={core_mid}  sigma {st['sigma_before']:.3f}->{st['sigma_after']:.3f}")
    print(f"       CÓ bơm, sau 40     : core={core_yes}")

    check("có bơm -> chuyển sang core mới [7,8]", set(core_yes) == {7, 8})
    check("không bơm -> kẹt ở core cũ", set(core_no) != {7, 8},
          "đây chính là nghịch lý Robbins-Monro")
except Exception:
    traceback.print_exc(); FAIL.append("inflation")


# =====================================================================
section("5. TARGETED ε-FORCING — ngân sách giữ nguyên, positivity còn?")
# =====================================================================
try:
    from models.intervention import EpsilonForcedActionController as C

    c = C(n_agents=6, action_dim=4, eps=0.03,
          max_forced_per_step=None, rng=np.random.RandomState(0))
    c.set_priority(np.array([0.02, 0.02, 0.9, 0.8, 0.02, 0.02]))
    e = c.get_eps_per_agent()

    print(f"       eps mỗi agent : {np.round(e, 4)}")
    print(f"       trung bình    : {e.mean():.4f} (mục tiêu 0.0300)")

    check("ngân sách tổng giữ nguyên", abs(e.mean() - 0.03) < 1e-6)
    check("positivity: mọi agent > 0", e.min() > 0, f"min={e.min():.4f}")
    check("có tập trung thật", e.max() / e.min() > 3, f"{e.max()/e.min():.1f}x")

    acts = [0] * 6
    probs = np.full((6, 4), 0.25, dtype=np.float32)
    mask, eff = c.apply(acts, probs)
    check("propensity khớp eps từng agent",
          bool(np.allclose(eff.sum(1), 1.0, atol=1e-5)))
except Exception:
    traceback.print_exc(); FAIL.append("targeted forcing")


# =====================================================================
section("6. SCHEDULER — cò súng và thời gian trơ")
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

    print(f"       lần 1 (z=5): fired={r1['fired']} reason={r1['reason']}")
    print(f"       lần 2 ngay sau: fired={r2['fired']} reason={r2['reason']}")
    print(f"       lần 3 sau trơ: fired={r3['fired']} reason={r3['reason']}")

    check("bắn khi z vượt ngưỡng", r1["fired"])
    check("thời gian trơ chặn bắn liên hồi", not r2["fired"])
    check("bắn lại sau khi hết trơ", r3["fired"])
    check("belief được bơm phồng", bel[0].n == 2)
except Exception:
    traceback.print_exc(); FAIL.append("scheduler")


# =====================================================================
section("7. TORCH — proxy, memory, ego-latent (bỏ qua nếu chưa cài torch)")
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
    for k in ("mu", "sigma", "mu_per_h", "latency", "mu_range"):
        print(f"       {k:<10} shape={np.shape(out[k])}")

    check("mu có dấu (không phải luôn ≥0)", bool(np.any(out["mu"] < 0)))
    check("mu_per_h shape [B,H]", out["mu_per_h"].shape == (8, H))
    check("latency trong [0,H-1]",
          bool(out["latency"].min() >= -1e-6 and out["latency"].max() <= H - 1 + 1e-6))
    check("mu_range luôn ≥ 0", bool(np.all(out["mu_range"] >= -1e-6)))

    dis = px.get_diagnostics()["ensemble_disagreement"]
    check("ensemble THẬT SỰ bất đồng", dis > 1e-6,
          f"disagreement={dis:.5f} (v1 = 0.000)")

    # ---- T1 [BB4]: vmap khớp bản tham chiếu (vòng lặp Python) ----
    # Cùng MỘT input cho cả hai đường -> chênh lệch chỉ có thể đến từ lỗi
    # layout (repeat_interleave/repeat/view bị đảo), không phải từ dữ liệu
    # khác nhau.
    t1_obs = rr.randn(6, OD); t1_ai = rr.randint(0, A, 6)
    t1_z = rr.randn(6, CD); t1_m = rr.randn(6, PD); t1_b = rr.randn(6, BD)
    fast = px._predict_all_actions(t1_obs, t1_ai, t1_z, t1_m, t1_b)
    ref = px._predict_all_actions_reference(t1_obs, t1_ai, t1_z, t1_m, t1_b)
    check("T1: vmap khớp bản tham chiếu (BB4 — layout B*A)",
          bool(torch.allclose(fast, ref, atol=1e-5)),
          f"max diff={float((fast - ref).abs().max()):.2e}")

    # ---- T3 [BB3]: thang gradient member-0 không đổi theo n_ensemble ----
    # Cùng seed -> member 0 của E=1 và E=4 dùng CHUNG bootstrap mask (mask
    # chỉ phụ thuộc seed*7919+k, không phụ thuộc n_ensemble) -> cùng dữ
    # liệu train nếu buffer được nạp giống hệt nhau.
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
    check("T3: thang gradient member-0 khớp E=1 vs E=4 (BB3)",
          rel_diff < 0.15,
          f"loss_e1={loss_e1:.4f} loss_e4_m0={loss_e4_m0:.4f} rel_diff={rel_diff:.3f}")

except ImportError as e:
    print(f"       BỎ QUA phần torch: {e}")
    SKIP.append("torch tests")
except Exception:
    traceback.print_exc(); FAIL.append("torch")

    # ---- peripheral memory ----
    pm = PeripheralMultiMemory(action_dim=A, n_free_slots=2,
                                 lb_coeff=1e-2, orth_coeff=1e-2)
    items = np.zeros((14, 9), dtype=np.float32)
    items[:, 0] = rr.randint(0, A, 14)
    items[:, 1] = np.concatenate([
        rr.uniform(0.2, 0.5, 4), rr.uniform(-0.5, -0.2, 4),
        rr.uniform(-0.02, 0.02, 3), rr.uniform(0.1, 0.3, 3)])
    items[:, 2] = np.concatenate([np.full(11, 0.1), np.full(3, 0.9)])
    items[:, 3] = rr.uniform(0, 1, 14)
    items[:, 5:] = rr.randn(14, 4)

    res = pm.forward_full(items)
    print(f"       memory shape={tuple(res['memory'].shape)} "
          f"slot_probs={tuple(res['slot_probs'].shape)}")
    usage = res["slot_usage"].detach().numpy()
    print(f"       slot usage = {np.round(usage, 3)}")

    check("memory shape [out_dim]", res["memory"].shape == (pm.out_dim,))
    check("6 slot (4 ngữ nghĩa + 2 tự do)", res["slot_probs"].shape[1] == 6)
    check("aux_loss > 0", float(res["aux_loss"]) > 0)
    check("không slot nào chết", float(usage.min()) > 1e-4,
          f"min usage={usage.min():.4f}")

    ent = pm.get_slot_diagnostics()["usage_entropy_ratio"]
    check("usage entropy hợp lý", ent > 0.5, f"entropy ratio={ent:.3f}")

    # ---- ego-conditioned heads ----
    hd = EgoConditionedHeads(latent_dim=32, n_horizons=H)
    z = torch.randn(24, 32)
    ego = torch.tensor(np.repeat([0, 1, 2], 8))
    nbr = torch.tensor(np.tile([1, 1, 2, 2, 3, 3, 4, 4], 3))
    lo = hd.compute_loss(z, ego, nbr, w_target=torch.randn(24, H))
    print(f"       L_influence={float(lo['influence']):.4f} "
          f"L_contrastive={float(lo['contrastive']):.4f}")
    check("influence loss hữu hạn", np.isfinite(float(lo["influence"])))
    check("contrastive loss > 0", float(lo["contrastive"]) > 0,
          "cần cùng-j-khác-ego trong batch")

    # ---- drift probe ----
    from models.drift_probe import DriftDetector, MatrixDriftDetector
    det = DriftDetector(obs_dim=OD, action_dim=A, n_horizons=H,
                        warmup_batches=5, batch_size=32)
    for ep in range(4):
        st = det.step(ep, px.buffer, n_train_batches=3)
    print(f"       probe phase={st['phase']} frozen={det.frozen is not None}")
    check("probe đóng băng được", det.frozen is not None)
    check("probe không có tham số học được sau khi băng",
          all(not p.requires_grad for p in det.frozen.parameters()))

    md = MatrixDriftDetector()
    W = np.random.RandomState(0).randn(6, 6) * 0.1
    md.update(W); md.update(W * 1.01)
    for _ in range(6):
        md.update(W * (1 + 0.01 * np.random.rand()))
    z_small = md.z_score()
    md.update(W * 5.0)
    z_big = md.z_score()
    print(f"       matrix z: đổi nhỏ={z_small:.2f}  đổi lớn={z_big:.2f}")
    check("ma trận phân biệt đổi nhỏ vs lớn", z_big > z_small + 1)

except ImportError as e:
    print(f"       BỎ QUA phần torch: {e}")
    SKIP.append("torch tests")
except Exception:
    traceback.print_exc(); FAIL.append("torch")


# =====================================================================
print("\n" + "=" * 70)
print(f"KẾT QUẢ: {len(PASS)} PASS | {len(FAIL)} FAIL | {len(SKIP)} SKIP")
print("=" * 70)
if FAIL:
    print("Thất bại:", ", ".join(FAIL))
    sys.exit(1)
print("Mọi cơ chế hoạt động. Bước tiếp: đọc RUN_GUIDE.md.")
