# CIG-AMF v2 — Hướng dẫn tích hợp

Bản vá cho các lỗi phát hiện khi đọc lại source v1. Mọi module giữ **nguyên
chữ ký hàm** mà runner đang gọi, nên cắm vào được từng phần một.

---

## 0. Ba lỗi mới phát hiện KHI ĐỌC CODE (chưa từng nêu trong các phân tích trước)

### 0.1 ⚠️ Proxy và Oracle đo HAI ĐẠI LƯỢNG KHÁC NHAU — Exp3 bất khả thi

```
oracle (adaptive_resource_flow_arena_v3.py:1246)
    deltas.append(int_returns[ego] - base_returns[ego])   →  CÓ DẤU

proxy  (structural_proxy.py:630)
    abs_effect = torch.abs(alt_preds - base_preds)        →  TRỊ TUYỆT ĐỐI
```

Comment trong proxy ghi *"khớp hơn với tiny oracle vì tiny oracle cũng lấy mean
absolute"* — **sai, oracle không hề lấy abs**.

Hệ quả: một neighbour có ảnh hưởng đối xứng (giúp ở action này, hại ở action
kia) cho oracle ≈ 0 nhưng proxy ≈ lớn. **Exp3 không thể pass dù thuật toán
đúng.** Đây có thể là lý do Exp3 không bao giờ có số liệu trong paper.

### 0.2 Ensemble là GIẢ — σ = 0.000 ± 0.000 có nguyên nhân cơ học

```python
# structural_proxy.py:378  — sample MỘT batch
batch = random.sample(list(self.buffer), n_total)
...
# dòng 406 — rồi cho CẢ 3 member train trên đúng batch đó, cùng thứ tự
for model, optim in zip(self.models, self.optims):
```

Cùng dữ liệu → hội tụ về gần cùng một hàm → không bất đồng → σ vô nghĩa.
Con số `0.676 ± 0.000` giống hệt nhau ở Final và NoMultiMemory, ở cả hai task,
là hệ quả trực tiếp.

### 0.3 Core size = 6.0 là CHẠM TRẦN, config đã xác nhận

`run_experiment.py:151` → `"max_core_size": 6`. Kết hợp với `abs()` trong
estimator (μ không bao giờ triệt tiêu → thổi phồng có hệ thống, đúng như
Pieroth quan sát: *"non-zero approximation error leads to overestimation"*)
→ mọi neighbour vượt τ_in → cap ở 6. Không có gì được "học" ở đây.

---

## 1. Bản đồ lỗi → file vá

| # | Lỗi | File v1 | Vá ở | Loại |
|---|-----|---------|------|------|
| L1 | `abs()` huỷ dấu trong estimator | structural_proxy.py:630 | `structural_proxy_v2.py` | novelty |
| L2 | Plug-in estimator, thừa hưởng bias | structural_proxy.py:684 | DR trong `structural_proxy_v2.py` | mượn+cải tiến |
| L3 | Ensemble giả, cùng batch | structural_proxy.py:406 | `_sample_for_member()` | bê nguyên xi |
| L4 | Một horizon → không có latency | structural_proxy.py:65 | multi-horizon head | cải tiến |
| B1 | p_core bão hoà (σ ở mẫu số) | belief_layer.py:281 | LCB trong `belief_layer_v2.py` | bê nguyên xi (LCB) |
| B2 | Core size là hằng số cứng | belief_layer.py:138 | `adaptive_k` + `get_saturation_stats()` | cải tiến |
| B3 | α không thoả Robbins-Monro | belief_layer.py:274 | lịch trình `1/t^d` | mượn (Pieroth Thm 5.6) |
| B4 | Belief không có dấu | belief_layer.py:277 | giữ dấu xuyên suốt | novelty |
| **B5** | **Bias khởi tạo** (phát hiện khi test) | — | `debiased_mu/sigma()` | mượn (Adam) |
| T1 | Slot không được giao việc | peripheral_memory.py:239 | slot ngữ nghĩa | **novelty** |
| T2 | Sụp độc quyền | — | load-balancing loss | bê nguyên xi (Switch) |
| T3 | Sụp đồng phục | `uniform_mix=0.25` làm tệ thêm | orthogonality loss | bê nguyên xi |
| E1 | `L_z` không ép pair-specific | core_behavior.py:928 | `EgoConditionedHeads` | mượn+cải tiến |
| H6 | Vòng lặp belief→proxy→belief | structural_proxy.py:84 | `use_belief_input=False` | novelty |
| H7 | Residual nhiễm bẩn | structural_proxy.py:437 | holdout tách riêng | bê nguyên xi |
| D1 | Chưa đo "tách 2 tầng" | — | `SelectiveResponsivenessTracker` | **novelty** |
| D2 | Chưa biết env có nhạy cấu trúc | — | `structure_sensitivity_test()` | bê nguyên xi |
| D3 | Chưa validate "Causal" | — | `proxy_calibration_report()` | bê nguyên xi |
| — | Không có kiểm định thống kê | — | `bootstrap_ci`, `compare_two_methods` | bê nguyên xi |

---

## 2. Kết quả unit test (chạy được, không cần torch)

```
TEST 1  p_core bão hoà                σ=0.05 → p nhảy 0.27→1.00 | σ=0.663 → mọi μ đều ~0.5
TEST 2  soft_role_assignment          6/6 trường hợp gán đúng vai trò ✓
TEST 3  k-means trên signature        purity 0.767  (không tách được consumer vs inert)
TEST 5  KIẾN TRÚC LAI                 purity 0.967  ← tốt hơn hẳn
        └ tách blocker/consumer trong nhóm "Ác": purity 1.000
TEST 6  belief chọn core              core = [1,2,3] khớp đúng ground truth ✓
TEST 7  kênh độ lớn độc lập           đảo chiều: signed≈0, abs=0.50, temporal_std=0.50 ✓
TEST 8  core size bám cấu trúc        n_true=1,2,3,5,8 → core=1,2,3,5,8, F1=1.00 ✓
```

**Ba lỗi trong code v2 do chính test bắt được** (đã sửa, ghi trong comment):

1. `sharpness` không chuẩn hoá theo `tau` → μ=0 bị gán nhầm vào "Thiện"
   (g_neu=0.244 < g_pos=0.371). Sửa: `k = sharpness/tau`.
2. `abs_mu = |mean|` dư thừa với `signed_mu`. Sửa: `abs_mu = mean(|·|)` →
   tỷ số hai kênh phân biệt được "nhất quán" vs "đảo chiều".
3. Bias correction dùng `x/(1-prod)` — chỉ đúng khi khởi tạo = 0. σ khởi tạo
   1.0 nên bị **thổi phồng 6 lần** (1.15 thay vì 0.19) → LCB luôn âm → core
   rỗng. Sửa: `(x - prod*init)/(1-prod)`.

> ⚠️ **Đính chính quan trọng:** sau khi vá bias correction, **cả hai** luật
> chọn core (`lcb` và `p_core`) đều đạt F1 = 1.00. Nghĩa là **bias correction
> mới là thứ quyết định**, không phải LCB. Đừng viết trong paper rằng LCB là
> nguyên nhân cải thiện — số liệu không ủng hộ. (Lưu ý nhánh `p_core` trong
> v2 đã dùng công thức mới, nên đây là so *hai luật chọn* trên cùng nền
> belief đã vá, không phải so v1 với v2.)

---

## 3. Cách cắm vào runner

### 3.1 Thay import

```python
# final_runner.py
from cig_amf_v2.structural_proxy_v2  import LocalCounterfactualProxyEnsembleV2
from cig_amf_v2.belief_layer_v2      import BayesLightBeliefStateV2
from cig_amf_v2.peripheral_memory_v2 import PeripheralMultiMemoryV2
from cig_amf_v2.influence_signature  import InfluenceSignatureTracker
from cig_amf_v2.intervention         import EpsilonForcedActionController
from cig_amf_v2.ego_conditioned_latent import EgoConditionedHeads
```

### 3.2 Khởi tạo

```python
self.proxy = LocalCounterfactualProxyEnsembleV2(
    obs_dim=..., action_dim=..., core_dim=..., periph_dim=..., belief_dim=...,
    n_ensemble=4,
    effect_mode="signed_aristocrat",   # ← có dấu
    use_doubly_robust=True,
    use_belief_input=False,            # ← cắt vòng lặp phản hồi
    n_horizons=3,
    seed=cfg["seed"],
)

self.belief_modules[ego] = BayesLightBeliefStateV2(
    ego_id=ego, neighbor_ids=...,
    core_rule="lcb", kappa=1.0, alpha_decay=0.7,
    adaptive_k=True,                   # ← core size thật sự thích nghi
    max_core_size=cfg["max_core_size"],
)

self.periph_module = PeripheralMultiMemoryV2(
    action_dim=..., n_free_slots=2,
    lb_coeff=1e-2,                     # Fedus et al. khuyến nghị
    orth_coeff=1e-2,
)

self.sig_tracker = InfluenceSignatureTracker(n_agents=n, window=30)

self.forcer = EpsilonForcedActionController(
    n_agents=n, action_dim=a, eps=0.03,
    max_forced_per_step=2, anneal_to=0.01, anneal_episodes=60,
    rng=np.random.RandomState(cfg["seed"]),
)
```

### 3.3 Trong vòng thu thập (chỗ đang sample action)

```python
logits, values = policy_net(...)
probs   = torch.softmax(logits, dim=-1).detach().cpu().numpy()   # [n_agents, A]
actions = [int(np.random.choice(A, p=probs[i])) for i in range(n)]

forced_mask, behaviour_probs = self.forcer.apply(actions, probs)

step["forced_mask"]     = forced_mask                                  # [n]
step["behaviour_probs"] = behaviour_probs                              # [n, A]
step["policy_probs"]    = probs                                        # [n, A]
step["behaviour_prob_taken"] = behaviour_probs[np.arange(n), actions]  # [n]
```

### 3.4 Khi push replay (`replay_builder.py`)

```python
proxy_ensemble.add_sample(
    ...,                                        # như cũ
    target_returns_multi=[R1, R2, R3],          # ← multi-horizon
    behaviour_prob_j=step["behaviour_prob_taken"][j],   # ← cho DR
    was_forced=bool(step["forced_mask"][j]),           # ← oversample
    state_key=env.agent_zone[ego],                     # ← cho context_std
)
```

`build_h_step_returns()` cần trả list các horizon thay vì một số — sửa vòng
trong thành tích luỹ từng bước:

```python
vals, run = [], 0.0
for h in range(self.horizon):
    if t + h < T:
        run += (self.discount ** h) * float(trajectory[t+h]["rewards"][ego])
    vals.append(run)                 # vals = [R^(1), R^(2), ..., R^(H)]
```

### 3.5 Khi update belief

```python
out = self.proxy.score_batch_full(
    ..., policy_probs_j_batch=..., observed_returns_batch=...,
    behaviour_probs_obs_batch=...,
)

self.sig_tracker.update_from_proxy_output(
    ego_id=ego, neighbor_ids=nbr_ids, proxy_out=out,
    context_keys=[env.agent_zone[j] for j in nbr_ids],
)

belief.update_batch({j: (out["mu"][b], out["sigma"][b])
                     for b, j in enumerate(nbr_ids)})
```

### 3.6 Hiệu chỉnh ngưỡng vai trò — SAU Stage 0, ĐỪNG QUÊN

```python
if scheduler.episode == cfg["k0_warmup"]:
    cal = self.sig_tracker.auto_calibrate()
    self.periph_module.set_role_thresholds(cal["tau_role"], cal["sigma_hi"])
```

Thang của μ phụ thuộc hoàn toàn vào thang reward của môi trường. Cắm cứng
`tau_role` có thể khiến **tất cả** neighbour rơi vào "Trung tính" (reward nhỏ)
hoặc **không ai** vào (reward lớn) — cả hai làm slot ngữ nghĩa vô dụng.

### 3.7 Cộng aux loss vào policy loss

```python
p_out = self.periph_module.forward_full(periph_items)
M_i   = p_out["memory"]
total_loss = policy_loss + value_loss + p_out["aux_loss"]
```

---

## 4. Thứ tự chạy thí nghiệm

**Vòng 0 — chẩn đoán (làm TRƯỚC mọi thứ, 1–2 ngày)**

1. `structure_sensitivity_test()` → env có nhạy cấu trúc không?
   Pure MF −0.211 vs Full Explicit −0.196 = chênh **0.015**. Nếu oracle-core
   cũng chỉ ~−0.196 thì **trần lợi ích của mọi phương pháp structural** chỉ là
   1.5% → **sửa môi trường trước, đừng tối ưu thuật toán.**
2. `proxy.get_diagnostics()["ensemble_disagreement"]` → nếu ≈ 0, ensemble vẫn giả.
3. `belief.get_saturation_stats()["hit_max_rate"]` → nếu ≈ 1.0, core vẫn chạm trần.
4. `pair_specificity_score()` → nếu `cross_ego_similarity` ≈ 1.0 thì z_ij chỉ là
   global opponent model, đóng góp số 2 của paper chưa tồn tại.

**Vòng 1 — vá cơ chế**: bật lần lượt từng module v2, kiểm tra 4 chỉ số trên đổi đúng hướng.

**Vòng 2 — nâng cấp estimator**: ε-forcing + DR, chạy `proxy_calibration_report()`
với `effect_mode="signed_oracle_matched"`.

**Vòng 3 — novelty**: slot ngữ nghĩa; vẽ `get_cluster_centroids()` thành heatmap;
chạy `SelectiveResponsivenessTracker` để có hình headline.

**Vòng 4 — trình bày**: bootstrap CI, baseline mới (CMFQ, GAT-MF, QMIX,
Pieroth TIM/SIM), Discussion, cắt gọn.

---

## 5. Bảng ablation nên chạy (mọi flag đã có sẵn)

| Tên | Cấu hình | Kiểm chứng điều gì |
|---|---|---|
| v1-replica | `effect_mode="mean_abs"`, `use_doubly_robust=False`, `n_free_slots=0`, `use_uniform_mix=True`, `core_rule="p_core"` | tái lập bản cũ |
| +signed | `effect_mode="signed_aristocrat"` | dấu có giúp không |
| +DR | `use_doubly_robust=True` | debias có giúp không |
| +ε-forcing | `eps=0.03` | can thiệp thật có giúp không |
| +LCB | `core_rule="lcb"` | luật chọn core |
| +semantic slots | `n_free_slots=2` | **novelty chính** |
| +load-balance | `lb_coeff=1e-2` | chống collapse |
| Pieroth baseline | `effect_mode="range"` | có dấu > không dấu? |

---

## 6. Điều PHẢI nói thẳng trong Discussion

- Tỷ lệ chạm trần core (`hit_max_rate`) — nếu cao thì "adaptive allocation"
  chưa có căn cứ.
- `ensemble_disagreement` — chứng minh σ là bất định thật, không phải lịch trình.
- Chi phí của ε-forcing lên reward (`get_stats()["realised_forcing_rate"]`).
- Throughput: v1 chậm hơn Pure MF **45×**. Slot ngữ nghĩa rẻ hơn softmax học,
  nhưng vẫn phải báo cáo thẳng.
- Giới hạn lý thuyết: trích Pieroth Theorem 5.3
  `|ŵ − w| ≤ 2‖r̂ − r‖∞` — sai số proxy bị chặn bởi 2× sai số reward model.
- **Không mượn `noise monotonicity` của cf-ASE.** Trong môi trường resource-flow
  nhiều agent, giả định này rất khó biện minh. ε-forcing cho nhân quả **mà không
  cần giả định gì cả**, vì randomization là thật.

## 7. Chưa chạy được trong môi trường này

Không có mạng để cài PyTorch, nên **phần torch chưa được smoke-test**. Phần
toán đã verify bằng numpy mirror test (mục 2). Khi chạy lần đầu, kiểm tra:
shape của `_predict_all_actions` (`[E, B, A, H]`), và `einsum("ebah,ba->ebh")`.
