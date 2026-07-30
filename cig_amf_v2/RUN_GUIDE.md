# CIG-AMF v2 — Hướng dẫn chạy

## 0. Chạy ngay sau khi tải về (2 phút)

```bash
cd cig_amf_v2
pip install torch numpy          # nếu chưa có
python smoke_test.py
```

Kỳ vọng: `KẾT QUẢ: 25 PASS | 0 FAIL | 0 SKIP`.
Nếu chưa cài torch thì được 16 PASS, 1 SKIP — vẫn ổn để đọc tiếp, nhưng
phải cài torch trước khi cắm vào runner.

**Nếu có FAIL: dừng lại, đừng cắm vào runner.** Mỗi test in ra con số cụ
thể nên nhìn là biết hỏng chỗ nào.

---

## 1. Danh sách file

| File | Vai trò | Thay cho file cũ |
|---|---|---|
| `intervention.py` | ε-forcing + oracle sampler | *(mới)* |
| `structural_proxy_v2.py` | proxy có dấu + DR + ensemble thật | `structural_proxy.py` |
| `belief_layer_v2.py` | LCB + bias correction + bơm phồng | `belief_layer.py` |
| `influence_signature.py` | chữ ký 6 chiều + vai trò | *(mới)* |
| `peripheral_memory_v2.py` | slot ngữ nghĩa + chống collapse | `peripheral_memory.py` |
| `ego_conditioned_latent.py` | L_influence + L_contrastive | bổ sung cho `core_behavior.py` |
| `drift_probe.py` | probe đóng băng + matrix detector | *(mới)* |
| `scheduler_v2.py` | điều phối + bơm phồng khi bắn | `scheduler.py` |
| `reciprocity.py` | chiều ngược i→j (chẩn đoán) | *(mới)* |
| `diagnostics.py` | selectivity, bootstrap, calibration | *(mới)* |
| `smoke_test.py` | kiểm tra mọi cơ chế | *(mới)* |

Giữ nguyên: `adaptive_resource_flow_arena_v3.py`, `tiny_oracle_*.py`,
`policy_value.py`, `belief_summary.py`, `replay_builder.py`, `core_behavior.py`.

---

## 2. Thứ tự chạy thí nghiệm

### Bước 0 — CỬA CHẶN, chạy trước tiên (nửa buổi)

```python
from diagnostics import structure_sensitivity_test

def run_one(condition):
    # condition ∈ {"pure_mean_field", "oracle_core", "full_explicit"}
    # "oracle_core" = biến thể ĐƯỢC CHO SẴN core đúng, không phải học
    ...
    return mean_reward

res = structure_sensitivity_test(run_one, n_seeds=3)
print(res["structure_value"], res["verdict"])
```

**Ngưỡng quyết định:**

| `structure_value` | Nghĩa là | Làm gì |
|---|---|---|
| < 0.02 | biết hết cấu trúc chỉ đáng 2% reward | **DỪNG.** Sửa môi trường: thắt lane, tăng phạt tắc nghẽn, giảm đường vòng, tăng chênh lệch vai trò |
| 0.02 – 0.05 | biên mỏng | chạy tiếp nhưng đừng kỳ vọng reward tách bạch; bán bài bằng Core F1 + selectivity |
| > 0.05 | môi trường nhạy cấu trúc | chạy tiếp bình thường |

Bảng cũ của bạn cho Pure MF −0.211 vs Full Explicit −0.196, tức **0.015**.
Nếu oracle-core cũng quanh đó thì bạn đang ở hàng đầu tiên của bảng này.
Biết sớm tốt hơn biết sau khi đã chạy 200 episode.

### Bước 1 — Chạy 20 episode với đủ diagnostic

Không cần kết quả đẹp, chỉ cần **4 con số này đổi đúng hướng**:

```python
print(proxy.get_diagnostics()["ensemble_disagreement"])   # phải > 0
print(belief.get_saturation_stats()["hit_max_rate"])      # phải < 1.0
print(periph.get_slot_diagnostics()["usage_entropy_ratio"])# phải > 0.5
print(pair_specificity_score(pair_rel, n_agents)["specificity_ratio"])  # phải < 1
```

| Chỉ số | v1 | Kỳ vọng v2 | Nếu sai thì |
|---|---|---|---|
| `ensemble_disagreement` | 0.000 | > 0 | ensemble vẫn giả → kiểm tra `bootstrap_ratio` |
| `hit_max_rate` | ~1.0 | < 1.0 | core vẫn chạm trần → tăng `kappa` hoặc `tau` |
| `usage_entropy_ratio` | thấp | > 0.5 | slot vẫn collapse → tăng `lb_coeff` |
| `specificity_ratio` | ~1.0 | < 1.0 | z vẫn toàn cục → tăng `lambda_c` |

### Bước 2–6 — Các thí nghiệm chính

Theo đúng thứ tự trong paper: Exp 1 (calibration) → Exp 2 (selectivity)
→ Exp 3 (recovery) → Exp 4 (slot) → Exp 5 (scale) → Exp 6 (reciprocity).

---

## 3. Cắm vào runner

### 3.1 Khởi tạo

```python
from structural_proxy_v2  import LocalCounterfactualProxyEnsembleV2
from belief_layer_v2      import BayesLightBeliefStateV2
from peripheral_memory_v2 import PeripheralMultiMemoryV2
from influence_signature  import InfluenceSignatureTracker
from intervention         import EpsilonForcedActionController
from ego_conditioned_latent import EgoConditionedHeads
from drift_probe          import DriftDetector, MatrixDriftDetector
from scheduler_v2         import TwoTimescaleSchedulerV2
from reciprocity          import ReciprocityTracker

proxy = LocalCounterfactualProxyEnsembleV2(
    obs_dim=OD, action_dim=A, core_dim=CD, periph_dim=PD, belief_dim=BD,
    n_ensemble=4, n_horizons=3,
    effect_mode="signed_aristocrat",   # ← CÓ DẤU
    use_doubly_robust=True,
    use_belief_input=False,            # ← cắt vòng lặp phản hồi
    bootstrap_ratio=0.8, seed=SEED,
)

belief = {i: BayesLightBeliefStateV2(
    ego_id=i, neighbor_ids=[j for j in range(N) if j != i],
    core_rule="lcb", kappa=1.0, tau=0.10,
    alpha_decay=0.7,                   # ← Robbins-Monro
    adaptive_k=True, max_core_size=6, min_core_size=1,
) for i in range(N)}

periph = PeripheralMultiMemoryV2(
    action_dim=A, n_free_slots=2, lb_coeff=1e-2, orth_coeff=1e-2,
)

sig     = InfluenceSignatureTracker(n_agents=N, window=30)
forcer  = EpsilonForcedActionController(
    n_agents=N, action_dim=A, eps=0.03, max_forced_per_step=2,
    anneal_to=0.01, anneal_episodes=60, rng=np.random.RandomState(SEED),
)
heads   = EgoConditionedHeads(latent_dim=pair_rel.hidden_dim, n_horizons=3)
drift   = DriftDetector(obs_dim=OD, action_dim=A, n_horizons=3,
                        warmup_batches=200, recalibrate_after=15, seed=SEED)
matdet  = MatrixDriftDetector(window=20)
sched   = TwoTimescaleSchedulerV2(k0_warmup=20, z_threshold=3.0,
                                  refractory=10, inflation_factor=2.5)
recip   = ReciprocityTracker(n_agents=N, min_causal_samples=20)
```

### 3.2 Trong vòng thu thập

```python
logits, values = policy(...)
probs = torch.softmax(logits, -1).detach().cpu().numpy()   # [N, A]
actions = [int(np.random.choice(A, p=probs[i])) for i in range(N)]

forced_mask, behaviour_probs = forcer.apply(actions, probs)

step["forced_mask"]          = forced_mask                       # [N] bool
step["behaviour_probs"]      = behaviour_probs                   # [N, A]
step["policy_probs"]         = probs                             # [N, A]
step["behaviour_prob_taken"] = behaviour_probs[np.arange(N), actions]  # [N]
```

⚠️ **Cache context TẠI THỜI ĐIỂM CHỌN HÀNH ĐỘNG**, không tính lại sau:
`step["core_context_excluding"]`, `step["periph_context_excluding"]`.

### 3.3 Đẩy replay — cần multi-horizon

Sửa `replay_builder.build_h_step_returns` thành tích luỹ từng bước:

```python
vals, run = [], 0.0
for h in range(self.horizon):
    if t + h < T:
        run += (self.discount ** h) * float(trajectory[t+h]["rewards"][ego])
    vals.append(run)          # → [R^(1), R^(2), ..., R^(H)]
```

```python
proxy.add_sample(
    ...,                                        # như cũ
    target_returns_multi=vals,                  # ← mới
    behaviour_prob_j=step["behaviour_prob_taken"][j],
    was_forced=bool(step["forced_mask"][j]),
    state_key=env.agent_zone[ego],
)
```

### 3.4 Cuối episode

```python
sched.step_episode()
forcer.step_episode()

# probe học/đo (bản đóng băng không bao giờ bị đụng)
probe_state = drift.step(sched.episode, proxy.buffer, n_train_batches=5)

if sched.should_update_graph():
    proxy.train_step(n_steps=4, batch_size=256, holdout_size=256)

    for i in range(N):
        nbrs = belief[i].neighbor_ids
        out = proxy.score_batch_full(
            obs_i_batch=..., action_i_batch=..., observed_action_j_batch=...,
            z_core_excl_j_batch=..., m_periph_excl_j_batch=...,
            belief_summary_batch=...,
            policy_probs_j_batch=...,          # [B, A]  cho DR
            observed_returns_batch=...,        # [B]
            behaviour_probs_obs_batch=...,     # [B]
        )
        sig.update_from_proxy_output(i, nbrs, out,
                                     context_keys=[env.agent_zone[j] for j in nbrs])
        belief[i].update_batch({j: (out["mu"][b], out["sigma"][b])
                                for b, j in enumerate(nbrs)})

# ma trận ảnh hưởng cho detector thứ hai
from diagnostics import influence_matrix_from_beliefs
W = influence_matrix_from_beliefs(belief, N)
matdet.update(W)

# cò súng + bơm phồng
ev = sched.evaluate_drift(
    probe_z=probe_state["z"], matrix_z=matdet.z_score(),
    belief_modules=belief, drift_detector=drift,
)
if ev["fired"]:
    print(f"[ep {sched.episode}] SHIFT ({ev['reason']}), "
          f"bơm phồng {ev['n_inflated']} cặp")

# hiệu chỉnh ngưỡng vai trò — CHỈ MỘT LẦN, sau warm-up
if sched.episode == 20:
    cal = sig.auto_calibrate()
    periph.set_role_thresholds(cal["tau_role"], cal["sigma_hi"])

# nhắm can thiệp theo bất định (từ episode 25 trở đi)
if sched.episode > 25:
    unc = np.zeros(N)
    for i in range(N):
        for j in belief[i].neighbor_ids:
            unc[j] = max(unc[j], belief[i].debiased_sigma(j))
    forcer.set_priority(unc)
```

### 3.5 Cộng aux loss vào policy loss

```python
p_out = periph.forward_full(periph_items)
M_i   = p_out["memory"]
loss  = policy_loss + value_loss + p_out["aux_loss"]
```

### 3.6 Ego-conditioned latent

```python
lo = heads.compute_loss(
    z=z_batch, ego_ids=ego_ids, neighbor_ids=nbr_ids,
    w_target=w_batch,          # [B, H] từ out["mu_per_h"]
    w_influence=1.0, w_contrastive=0.3,
)
total_z_loss = bc_loss + lo["total"]
```

⚠️ Batch phải chứa **cùng một j dưới nhiều ego khác nhau**, nếu không
contrastive loss = 0. Sample theo neighbour, không theo ego.

---

## 4. Hyperparameter — bắt đầu từ đây

| Tham số | Giá trị | Nguồn |
|---|---|---|
| `eps` | 0.03 → 0.01 | đủ mẫu, đủ rẻ |
| `iw_clip` | 10.0 | chặn phương sai |
| `n_ensemble` | 4 | 3 quá ít để ước lượng std |
| `bootstrap_ratio` | 0.8 | đa dạng vừa phải |
| `kappa` (LCB) | 1.0 | ~84% một phía |
| `alpha_decay` (d) | 0.7 | Pieroth dùng 0.726 |
| `lb_coeff` | **0.01** | Fedus et al. quét 1e-1…1e-5, chọn 1e-2 |
| `orth_coeff` | 0.01 | cùng thang |
| `role_sharpness` | 3.0 | đã chuẩn hoá theo τ |
| `z_threshold` | 3.0 | 3 độ lệch chuẩn |
| `inflation_factor` | 2.5 | đủ để core co, không xoá trắng |
| `refractory` | 10 | chặn bắn liên hồi |

**Thứ tự tune nếu phải tune:** `tau`/`kappa` (ảnh hưởng core size nhiều
nhất) → `lb_coeff` → `inflation_factor` → phần còn lại.

---

## 5. Bảng ablation (mọi flag đã có sẵn)

| Tên | Cấu hình |
|---|---|
| v1-replica | `effect_mode="mean_abs"`, `use_doubly_robust=False`, `n_free_slots=0`, `use_uniform_mix=True`, `core_rule="p_core"`, `alpha_decay=0`, `eps=0` |
| +signed | `effect_mode="signed_aristocrat"` |
| +DR | `use_doubly_robust=True` |
| +ε-forcing | `eps=0.03` |
| +targeted ε | gọi `forcer.set_priority(...)` |
| +bias correction | `alpha_decay=0.7` |
| +LCB | `core_rule="lcb"` |
| +inflation | `inflation_factor=2.5` |
| +semantic slots | `n_free_slots=2` |
| +aux losses | `lb_coeff=1e-2, orth_coeff=1e-2` |
| Pieroth baseline | `effect_mode="range"` |

---

## 6. Lỗi thường gặp

| Triệu chứng | Nguyên nhân | Sửa |
|---|---|---|
| `contrastive loss = 0` | batch không có cùng-j-khác-ego | sample theo neighbour |
| `ensemble_disagreement ≈ 0` | buffer quá nhỏ | tăng warm-up trước khi train |
| core luôn = min | τ quá cao / chưa calibrate | giảm `tau`, kiểm tra `debiased_sigma` |
| slot usage lệch hẳn | `lb_coeff` quá nhỏ | tăng lên 0.05 |
| trigger bắn liên tục | probe chưa re-snapshot | kiểm tra `notify_trigger` có được gọi |
| `nan` trong reciprocity | thiếu mẫu forced-ego | đúng như thiết kế; tăng `eps` hoặc giảm `min_causal_samples` |
| DR làm ước lượng nổ | `behaviour_prob_j` quá nhỏ | giảm `iw_clip` xuống 5 |

---

## 7. Con số phải báo cáo trong paper

Sáu chỉ số khiến lỗi thầm lặng thành nhìn thấy được:

1. `ensemble_disagreement` — σ có thật không
2. `hit_max_rate` — core có phải hằng số cứng không
3. `usage_entropy_ratio` + slot cosine — slot có collapse không
4. `specificity_ratio` — z_ij có ego-centric không
5. `dr_correction_magnitude` — outcome model lệch bao nhiêu
6. `realised_forcing_rate` + chi phí reward — ε tốn bao nhiêu

Cộng thêm: `selectivity_ratio` (Exp 2), `sign_agreement` (Exp 1),
`n_inflations` + trigger log (Exp 3), quadrant coverage (Exp 6).

Mọi so sánh reward dùng `bootstrap_ci` / `compare_two_methods`, **không**
dùng mean±std trần như bảng cũ — chênh 0.012 với std 0.018 trên 5 seed
là không kết luận được gì.
