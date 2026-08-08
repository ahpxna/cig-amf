"""
env_audit.py — kiểm định Omni-Arena theo Phần 7 (tiêu chí nghiệm thu) của
docs/OMNI_ARENA_BLUEPRINT.md.

Instantiate OmniArena (P1-P4) + oracle can thiệp đã sửa (P0), chạy đủ số
episode/oracle rollout để tính:

  T1  Gini(|W*|) trên tập cặp agent
  T2  cân bằng dấu (tỷ lệ cặp W* âm/dương "đáng kể")
  T3  CV của w_ij(s) qua nhiều trạng thái, cho các cặp có điều kiện
  T4  độ trải (spread) của latency-to-peak-effect qua 4 vai trò
  T5  SNR (biên độ tín hiệu core / biên độ nhiễu control-pairs)
  T6  tier_separation_ratio (structural vs behavioural ||dPhi||_F)
  corr(Phi, W*)

In bảng PASS/FAIL theo Phần 7, kèm toàn bộ số thô.

GHI CHÚ VỀ QUY MÔ MẪU (đọc trước khi diễn giải số liệu):
Chi phí oracle đầy đủ O(N^2 * S * T * H) là ~1 triệu bước env cho N=24 (Phần
4.2). Trong phiên làm việc này, để chạy được trong vài phút, T1/T2/T5/corr
dùng một tập mẫu CÓ CHỦ ĐÍCH gồm:
  - toàn bộ 20 cặp "declared" (5 role -> {collector hoặc gatekeeper} x 4 zone)
  - một tập ~20 cặp "control" lấy ngẫu nhiên (ego, j) KHÔNG nằm trong declared
    pairs, dùng làm ước lượng noise-floor cho T5 và làm đối chứng cho Gini.
Đây KHÔNG phải cắt tỉa theo bán kính (không rơi vào bẫy tự-xác-nhận A1 của
Phần 4.2) — chỉ là giảm số state/trial để chạy trong ngân sách thời gian của
phiên làm việc. Con số S (states), T (trials), forced_step quét được ghi rõ
trong phần cấu hình bên dưới và trong output.
"""
import sys
import numpy as np

from omni_arena import OmniArena


# ============================================================
# Config (xem ghi chú quy mô mẫu ở đầu file)
# ============================================================
N_AGENTS = 24
GRID_SIZE = 24
N_ZONES = 4
MAX_STEPS = 60
PHASE_LENGTH = 6          # nhỏ để structural shift + behavioural drift xảy ra nhanh trong audit
HORIZON = 8
N_STATES_T1 = 10           # số state lấy mẫu cho T1/T2/T5/corr
N_STATES_T3 = 24           # số state lấy mẫu cho T3 (rẻ -- không cần oracle)
N_CONTROL_PAIRS = 20
N_TRIALS = 1                # CRN cho phép n_trials nhỏ (P0d)
SEED = 123

SIGNIFICANT_W = 0.01        # ngưỡng "đáng kể" cho T2 (đơn vị reward/step)

# RC-2: tham số phép đo Φ̃ = E_s[phi * delta] cho T6.
# T6_N_STATES là đánh đổi nhiễu/thời gian: ‖dΦ̃‖_behavioural là hiệu của hai
# trung bình Monte-Carlo nên sai số ~ 1/sqrt(N). Dưới ~24 state thì nhiễu lấy
# mẫu át tín hiệu drift và tỉ số T6 mất ổn định giữa các seed.
T6_N_STATES = 48
T6_BURN_IN = 3
# Hai cực của lịch drift trong _behaviour_mode(): "cooperative" (gatekeeper mở
# cổng mọi bước) vs "selfish" (bỏ nhiệm vụ). Chọn hai cực để ‖dΦ̃‖_behavioural
# là CẬN TRÊN của drift — nếu ngay cả cận trên còn nhỏ hơn structural 3-20 lần
# thì kết luận tách tầng mới có giá trị.
T6_BEHAVIOUR_PAIR = ("cooperative", "selfish")
T6_INVARIANCE_PHASES = 2    # was 5 -- chỉ để kiểm bất biến phi tĩnh, 2 là đủ


def gini(values):
    x = np.sort(np.abs(np.asarray(values, dtype=np.float64)))
    n = len(x)
    if n == 0 or np.sum(x) == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def coefficient_of_variation(values):
    x = np.asarray(values, dtype=np.float64)
    mean = np.mean(x)
    if abs(mean) < 1e-9:
        return 0.0
    return float(np.std(x) / abs(mean))


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def oracle_w_star(env, ego, j, horizon=HORIZON, n_trials=N_TRIALS, forced_step=0):
    """
    W*_ij (dùng cho T1/T2/T5/corr): dùng đúng oracle đã sửa theo P0
    (compute_oracle_influence_from_current_state, CRN, không abs), CAN THIỆP
    MỘT BƯỚC tại forced_step, action = STAY.

    LƯU Ý QUAN TRỌNG (phát hiện trong quá trình audit, xem báo cáo triển
    khai): với can thiệp một-bước tại step 0, một vài cặp có độ trễ cao
    (relay 4-5 bước, controller 6+ bước) gần như không đổi được rollout kịp
    trong horizon còn lại -> W* đo được gần 0, không phản ánh đúng dấu Phi.
    Đây là hạn chế đã biết của phép đo một-bước trên một môi trường có độ
    trễ dài, không phải lỗi oracle. Dùng oracle_w_star_sustained() bên dưới
    cho phép đo "vô hiệu hoá vai trò j suốt horizon" (theo hướng cùng gốc
    khái niệm can thiệp, chỉ khác ở việc can thiệp được duy trì thay vì một
    bước) để làm chỉ số corr(Phi, W*) chính -- đây là phép đo dùng cho
    T1/T2/T5/corr trong report này.
    """
    profile = env.compute_oracle_influence_from_current_state(
        ego_id=ego,
        agent_j=j,
        intervention_action=env.STAY,
        horizon=horizon,
        n_trials=n_trials,
        forced_step=forced_step,
        candidate_actions=[env.STAY],
        crn_seed=SEED,
    )
    return -float(profile["signed"])


def _disengage_action(env, ego, j):
    """
    Hành động đơn bước làm TĂNG khoảng cách agent j <-> ego nhiều nhất -- đại
    diện tốt hơn cho "j không tham gia tương tác" so với STAY cố định (STAY
    có thể vô tình đứng lại đúng ngay trên đường đi/vùng nguy hiểm của ego,
    xem báo cáo triển khai, mục "deviations").
    """
    # [P-2 FINAL DEBUG] Rời VỊ TRÍ NHIỆM VỤ (duty anchor) của j, không phải
    # rời ego. Bản cũ maximize dist(j, ego) => kênh khai báo của CHÍNH j
    # (vd. gate_gk_collector cần gatekeeper cách gate <= 1) thường KHÔNG đổi
    # trạng thái giữa base/alt (đo được d_w = 0.000 ở mọi state), trong khi
    # thân j vẫn quẩn quanh làm lệch queue/crowding của các gate KHÁC.
    # "Vai trò j không tham gia" = j rời anchor mà delta_ij(s) của nó neo vào.
    z = env.agent_zone[j]
    role = env.agent_role[j]
    anchor = {
        env.ROLE_GATEKEEPER: env.zone_gate[z],
        env.ROLE_RELAY: env.zone_lane_a[z],
        env.ROLE_BLOCKER: env.zone_checkpoint[z],
        env.ROLE_CONTROLLER: env.zone_panel[z],
        env.ROLE_COLLECTOR: env.zone_resource[z],
    }.get(role, env.positions[ego])
    src = env.positions[j]
    best_act, best_d = env.STAY, env._dist(src, anchor)
    for act in (env.UP, env.DOWN, env.LEFT, env.RIGHT):
        nxt = env._move(src, act)
        d = env._dist(nxt, anchor)
        if d > best_d:
            best_d = d
            best_act = act
    return best_act


PAIR_LATENCY_CAP = {
    "gatekeeper->collector": 3,
    "relay->collector": 5,
    "blocker->collector": 1,
    "controller->collector": 8,
    "collector->gatekeeper": 2,
    "control": 8,
}


def oracle_w_star_sustained(env, ego, j, horizon=HORIZON, crn_seed=SEED,
                            n_force_steps=None, candidate_actions=None):
    """
    W*_ij theo ĐÚNG Eq (4): contrast giữa hành động THỰC của j và BASELINE
    TRUNG BÌNH THEO PHÂN BỐ HÀNH ĐỘNG, ép suốt horizon.

        W*_ij = R_i(a_j thực) - E_{a'~b_j}[ R_i(do a') ]

    ==================================================================
    [ORACLE-EQ4] SỬA MỘT SAI LỆCH KHỎI SPEC, KHÔNG PHẢI TINH CHỈNH.
    ==================================================================
    Bản cũ dùng _disengage_action — một baseline DO NGƯỜI CHỌN:
        W*_cũ = R_i(a_j) - R_i(do a_disengage)
    trong khi Eq (4) định nghĩa baseline là trung bình theo phân bố hành
    động. Paper §II.B nói thẳng lý do: "rather than an arbitrarily chosen
    alternative action, follows [12] and reduces variance because it
    averages over the full action distribution." Bản cũ chạy đúng cái
    "arbitrarily chosen alternative action" mà đoạn đó bác bỏ.

    Hệ quả đã đo được: oracle và estimator nhắm HAI ĐẠI LƯỢNG KHÁC NHAU —
    đúng thứ §V.A cảnh báo ("oracle and estimator must target the same
    quantity"). Cụ thể với cặp collector->gatekeeper:
        dist(collector, gate)   base 6.5 -> alt 1.5  (mọi zone)
        dd(collector, gk)       -> 0.000 (mọi zone)
    _disengage_action đẩy collector RA XA duty anchor (zone_resource), mà
    gate nằm phía đối diện resource trên cùng polyline, nên nó đẩy thẳng
    collector VÀO MẶT gatekeeper. Obstruct bắn mạnh hơn ở nhánh alt =>
    W* = base - alt DƯƠNG THEO CẤU TẠO, và 4/4 cặp lệch dấu so với Phi.
    W* khi đó không đo "collector ảnh hưởng gatekeeper thế nào" mà đo
    "chuyện gì xảy ra khi ta đẩy collector vào mặt gatekeeper".

    TẠI SAO BASELINE LÀ UNIFORM, KHÔNG PHẢI pi_j:
    Eq (4) viết E_{a'~pi_j}. Nhưng scripted_policy TẤT ĐỊNH, nên
    E_{a'~pi_j} = chính hành động đó = base => contrast ≡ 0, phép đo suy
    biến. Phân bố đúng để lấy trung bình ở đây là phân bố mà can thiệp
    THỰC SỰ bốc từ: eps-forcing bốc UNIFORM trên A (xem
    EpsilonForcedActionController.apply). A2 do đó thoả theo cấu tạo
    (b_j >= eps/|A| > 0) — baseline không bao giờ nằm ngoài support, khác
    hẳn một hành động hand-designed vốn không có bảo đảm nào về b_j.
    Uniform cũng là baseline mà _compute_tiny_oracle_scores (H1) đang
    dùng, nên sau sửa này env_audit và H1 nhắm CÙNG một đại lượng.

    Tính bằng |A| rollout tất định (mỗi action ép suốt horizon) rồi lấy
    trung bình — chính xác hơn là bốc mẫu, và không thêm nhiễu.

    GIỮ tính "sustained" (ép suốt n_force_steps): T4 H-sweep cho thấy
    blocker đi -0.14 -> -1.73 từ H=1 tới H=8 chưa bão hoà, nên contrast
    một-bước sẽ bóp mọi cặp có độ trễ dài.
    """
    if n_force_steps is None:
        n_force_steps = horizon
    if candidate_actions is None:
        candidate_actions = list(range(env.N_ACTIONS))

    snapshot = env.clone_state()
    crn_rng = np.random.RandomState(int(crn_seed) * 9973 + 1)
    buffer = env._make_crn_buffer(horizon, crn_rng)
    gamma = 0.95

    def _rollout(forced_action):
        env.restore_state(snapshot)
        env.set_noise_buffer(buffer)
        total = 0.0
        for t in range(horizon):
            acts = [env.scripted_policy(i) for i in range(env.n_agents)]
            if forced_action is not None and t < n_force_steps:
                acts[j] = int(forced_action)
            _, rew, done, _ = env.step(acts)
            total += (gamma ** t) * float(rew[ego])
            if done:
                break
        return total

    base_total = _rollout(None)
    alt_totals = [_rollout(a) for a in candidate_actions]

    env.clear_noise_buffer()
    env.restore_state(snapshot)

    # Eq (4): hành động thực TRỪ trung bình baseline.
    # Dấu: >0 nghĩa là hành động thực của j GIÚP ego so với một hành động
    # trung bình. Kiểm chứng bắt buộc: blocker->collector phải ÂM,
    # relay->collector phải DƯƠNG.
    return float(base_total - float(np.mean(alt_totals)))


def build_declared_pair_list(env):
    pairs = []
    for z in range(env.n_zones):
        ra = env.zone_role_agents[z]
        collector = ra[env.ROLE_COLLECTOR]
        gatekeeper = ra[env.ROLE_GATEKEEPER]
        relay = ra[env.ROLE_RELAY]
        blocker = ra[env.ROLE_BLOCKER]
        controller = ra[env.ROLE_CONTROLLER]
        pairs.append((gatekeeper, collector, "gatekeeper->collector"))
        pairs.append((relay, collector, "relay->collector"))
        pairs.append((blocker, collector, "blocker->collector"))
        pairs.append((controller, collector, "controller->collector"))
        pairs.append((collector, gatekeeper, "collector->gatekeeper"))
    return pairs


def build_control_pair_list(env, declared_set, rng, n=N_CONTROL_PAIRS):
    """
    RC-3(b). Noise floor phải là "cặp CÓ THỂ tương tác nhưng KHÔNG được khai
    báo" — tức CÙNG ZONE và không thuộc declared_set.

    Bản cũ bốc ngẫu nhiên hai chỉ số trên toàn quần thể: 24 agent / 4 zone ⇒
    ~75% cặp khác zone. Mọi kênh tương tác của env (declared, collision, lane,
    queue, crowding) đều yêu cầu đồng vị trí vật lý, nên cặp khác zone cho
    W* = 0 CHÍNH XÁC ⇒ np.std(np.abs(w_control)) = 0 ⇒ T5 SNR = ∞. Con số
    3.8e9 là artefact của phép lấy mẫu, không nói gì về env.

    Liệt kê vét cạn rồi shuffle thay vì rejection sampling: không gian cặp
    cùng-zone chỉ cỡ n_zones * k^2 (k = agent/zone ≈ 6) ~ 144 phần tử, vét
    cạn là O(n_agents^2) một lần và loại hẳn khả năng vòng while quay đủ
    n*20 lượt mà vẫn trả về thiếu cặp.
    """
    candidates = [
        (i, j, "control_same_zone")
        for z in range(env.n_zones)
        for i in [a for a in range(env.n_agents) if env.agent_zone[a] == z]
        for j in [a for a in range(env.n_agents) if env.agent_zone[a] == z]
        if i != j and (i, j) not in declared_set
    ]
    if not candidates:
        return []

    order = rng.permutation(len(candidates))
    return [candidates[k] for k in order[:n]]


def sample_states(env, n_states):
    return env.sample_state_bank(n_states=n_states, burn_in=3)


# ============================================================
# T1, T2, T5, corr(Phi, W*)  -- sử dụng oracle
# ============================================================

def run_oracle_based_metrics(env):
    rng = np.random.RandomState(SEED)
    declared_pairs = build_declared_pair_list(env)
    declared_set = {(i, j) for (i, j, _) in declared_pairs}
    control_pairs = build_control_pair_list(env, declared_set, rng, n=N_CONTROL_PAIRS)

    states = sample_states(env, N_STATES_T1)

    declared_records = []  # (i, j, label, phi, mean_w_star)
    control_records = []

    # LƯU Ý: đã thử ép disengage CHỈ trong n_force_steps = PAIR_LATENCY_CAP
    # (khớp băng trễ thiết kế mỗi cặp) thay vì suốt horizon, với kỳ vọng
    # tránh hiệu ứng phóng đại của agent trung tâm (collector). Kết quả:
    # corr(Phi,W*) TỆ HƠN (-0.11 so với +0.615) -- việc ngắt ép giữa chừng
    # tạo động lực "đuổi-kịp" (catch-up) kỳ lạ sau khi thả agent về kịch
    # bản, nhiễu hơn là ép suốt horizon. Giữ lại full-horizon (n_force_steps
    # mặc định = horizon) làm phép đo chính thức -- xem báo cáo triển khai,
    # mục "deviations" để biết chi tiết và khuyến nghị hướng khắc phục.
    for (i, j, label) in declared_pairs:
        phi = env.gt_influence_by_ego[j].get(i, 0.0)
        vals = []
        for st in states:
            env.restore_state(st)
            vals.append(oracle_w_star_sustained(env, ego=j, j=i))
        declared_records.append((i, j, label, phi, float(np.mean(vals)), vals))

    for (i, j, label) in control_pairs:
        vals = []
        for st in states:
            env.restore_state(st)
            vals.append(oracle_w_star_sustained(env, ego=j, j=i))
        control_records.append((i, j, label, 0.0, float(np.mean(vals)), vals))

    all_records = declared_records + control_records
    w_all = [r[4] for r in all_records]
    w_declared = [r[4] for r in declared_records]
    w_control = [r[4] for r in control_records]

    # T1
    t1_gini = gini(w_all)

    # T2 sign balance: tỷ lệ tính trên tập cặp CÓ HIỆU ỨNG ĐÁNG KỂ (|W*| >
    # ngưỡng), KHÔNG lấy toàn bộ mẫu (declared+control) làm mẫu số -- vì
    # phần lớn control pairs có W*=0 by design (chúng ta CHỦ Ý lấy mẫu
    # control để đo noise-floor cho T5, không phải để pha loãng T2). Lấy
    # toàn bộ mẫu làm mẫu số sẽ đánh giá sai "cân bằng dấu của đồ thị nhân
    # quả" bằng cách pha loãng nó với các cặp vốn không có quan hệ nhân quả.
    n_neg = sum(1 for w in w_all if w < -SIGNIFICANT_W)
    n_pos = sum(1 for w in w_all if w > SIGNIFICANT_W)
    n_significant = n_neg + n_pos
    t2_neg_frac = n_neg / n_significant if n_significant else 0.0
    t2_pos_frac = n_pos / n_significant if n_significant else 0.0

    # ----------------------------------------------------------------------
    # T5 SNR — TÍNH TRONG TỪNG ZONE RỒI TRUNG BÌNH.
    #
    # [T5-WITHIN-ZONE] Bản cũ gộp toàn cục:
    #     T5 = mean|W*|(declared, tất cả zone) / std|W*|(control, tất cả zone)
    # Sau khi phá đối xứng zone (zone_path_len/zone_scale khác nhau mỗi zone),
    # std|W*|(control) gộp toàn cục CHỨA CẢ PHƯƠNG SAI GIỮA-ZONE — tức chính
    # thứ ta CỐ Ý tạo ra. Đo thật: control std 0.172 -> 0.9500, std/mean = 2.2,
    # trong khi declared blocker theo zone là −0.13 / −1.23 / −1.60 / −0.18.
    # Mẫu số phình vì THIẾT KẾ, không phải vì nhiễu => T5 tụt một cách giả
    # tạo, và "sửa" nó bằng cách bóp zone_scale về U(0.9,1.1) sẽ phá luôn
    # gate zone-asymmetry vừa đạt (std 0.03 -> 0.5507).
    #
    # Pooling qua các zone khác thang là lỗi thống kê, không phải lựa chọn.
    # Định nghĩa đúng: SNR nội-zone rồi lấy trung bình qua zone.
    # ----------------------------------------------------------------------
    def _zone_of(rec):
        try:
            return int(env.agent_zone[rec[1]])   # zone của ego (bên nhận)
        except Exception:
            return -1

    zone_snrs = {}
    for z in sorted({_zone_of(r) for r in all_records}):
        dz = [abs(r[4]) for r in declared_records if _zone_of(r) == z]
        cz = [abs(r[4]) for r in control_records if _zone_of(r) == z]
        if len(dz) == 0 or len(cz) < 2:
            continue
        zone_snrs[z] = float(np.mean(dz)) / max(float(np.std(cz)), 1e-9)

    t5_snr = float(np.mean(list(zone_snrs.values()))) if zone_snrs else 0.0

    # giữ bản gộp toàn cục để so sánh/chẩn đoán, KHÔNG dùng làm gate nữa
    core_amp = float(np.mean(np.abs(w_declared))) if w_declared else 0.0
    noise_amp = max(float(np.std(np.abs(w_control))) if w_control else 1e-9, 1e-9)
    t5_snr_pooled = core_amp / noise_amp

    # corr(Phi, W*) -- chỉ trên declared pairs (control có phi=0 by definition,
    # không thuộc "corr(A,B)" của Phần 0.2)
    phis = [r[3] for r in declared_records]
    w_star_declared = [r[4] for r in declared_records]
    corr_phi_w = pearson_corr(phis, w_star_declared)

    return {
        "t1_gini": t1_gini,
        "t2_neg_frac": t2_neg_frac,
        "t2_pos_frac": t2_pos_frac,
        "t5_snr": t5_snr,                    # within-zone mean (GATE)
        "t5_snr_pooled": t5_snr_pooled,      # bản gộp cũ, chỉ để chẩn đoán
        "t5_snr_per_zone": zone_snrs,
        "corr_phi_w": corr_phi_w,
        "declared_records": declared_records,
        "control_records": control_records,
        "n_declared": len(declared_records),
        "n_control": len(control_records),
        "n_states": len(states),
    }


# ============================================================
# T3 -- CV of w_ij(s) across states (rẻ, không cần oracle)
# ============================================================

def run_t3(env):
    declared_pairs = build_declared_pair_list(env)
    per_pair_values = {label: [] for (_, _, label) in declared_pairs}

    env.reset()
    for _ in range(N_STATES_T3):
        for _ in range(3):
            acts = [env.scripted_policy(k) for k in range(env.n_agents)]
            _, _, done, info = env.step(acts, return_obs=False)
            if done:
                env.reset()
        w_by_pair = info["w_by_pair"] if "w_by_pair" in info else {}
        for (i, j, label) in declared_pairs:
            key = f"{i}->{j}"
            per_pair_values[label].append(float(w_by_pair.get(key, 0.0)))

    cvs = {}
    for label, vals in per_pair_values.items():
        cvs[label] = coefficient_of_variation(vals)

    t3_cv_mean = float(np.mean(list(cvs.values())))
    return {"t3_cv_by_pair_type": cvs, "t3_cv_mean": t3_cv_mean}


# ============================================================
# T4 -- latency profile across the 4 non-collector roles
#
# PHƯƠNG PHÁP ĐÃ ĐỔI (xem review kết quả trước): bản CŨ ép can thiệp tại một
# forced_step TRƯỢT từ 0 đến H-1 với H=HORIZON cố định, rồi nhìn |W*| biến
# thiên theo forced_step. Điều này CHẮC CHẮN giảm dần khi forced_step tăng,
# BẤT KỂ latency thật của vai trò -- vì can thiệp càng muộn thì horizon còn
# lại để tích luỹ hiệu ứng càng ngắn, đó là hệ quả cơ học của "horizon còn
# lại co hẹp", không phải phép đo latency thật. Mọi vai trò vì vậy luôn "có
# vẻ" đạt đỉnh ở forced_step=0 rồi giảm dần -- không có tính chẩn đoán.
#
# Bản MỚI (theo code_test.py's measure_at_horizon() -- tái dùng đúng pattern,
# không viết lại logic oracle): CAN THIỆP CỐ ĐỊNH tại forced_step/step 0 (dùng
# oracle_w_star_sustained(), disengage j suốt horizon kể từ bước 0), rồi QUÉT
# horizon H qua {1,2,3,5,8} và xem W*(H) tăng/đổi dấu ra sao khi horizon dài
# ra -- đây mới thật sự chẩn đoán latency (ví dụ mẫu của người dùng:
# gatekeeper W* đi từ -0.23 tại H=1 lên +6.80 tại H=8, đổi dấu thật; blocker
# giữ nguyên âm suốt, không đổi dấu).
#
# Hai đặc trưng MỚI thay cho "spread của peak forced_step" cũ:
#   sign-flip step:   H nhỏ nhất trong sweep mà sign(W*(H)) khác
#                      sign(W*(H_min)) (H_min = sweep[0]); None nếu không có
#                      đổi dấu trong sweep.
#   saturation step:  H nhỏ nhất mà mức tăng rời rạc |W*(H)-W*(H_prev)| tụt
#                      xuống dưới 20% mức tăng rời rạc lớn nhất quan sát được
#                      trong sweep của vai trò đó; None nếu không có.
# ============================================================

T4_H_SWEEP = [1, 2, 3, 5, 8]


def run_t4(env, h_sweep=None, n_states=2):
    """
    LƯU Ý TƯƠNG THÍCH NGƯỢC: chữ ký run_t4(env) (không tham số bắt buộc
    khác) và các khoá "t4_peak_steps"/"t4_profiles"/"t4_spread" trong dict
    trả về được GIỮ NGUYÊN để env_audit_staged.py (gọi ea.run_t4(main_env))
    không phải sửa call site -- nhưng Ý NGHĨA của "t4_spread" đã đổi hoàn
    toàn (xem docstring module phía trên): KHÔNG còn là "độ trải của peak
    forced_step trượt", mà là "tỷ lệ vai trò (trên 4) có sign-flip trong
    H-sweep cố định forced_step=0". "t4_peak_steps" cũng đổi ý nghĩa thành
    "sign_flip_step" per role (None = không đổi dấu trong sweep). Đọc nhãn
    in ra ở main()/env_audit_staged.py, không suy diễn từ tên khoá cũ.
    """
    if h_sweep is None:
        h_sweep = T4_H_SWEEP
    states = sample_states(env, n_states)
    z = 0
    ra = env.zone_role_agents[z]
    collector = ra[env.ROLE_COLLECTOR]

    profiles = {}          # role -> [mean W*(H) for H in h_sweep] (forced_step=0 cố định)
    sign_flip_step = {}    # role -> H nhỏ nhất có sign khác H_min, hoặc None
    saturation_step = {}   # role -> H nhỏ nhất có increment < 20% peak increment, hoặc None

    for role_name in ["blocker", "gatekeeper", "relay", "controller"]:
        j = ra[role_name]
        means = []
        for H in h_sweep:
            vals = []
            for st in states:
                env.restore_state(st)
                vals.append(oracle_w_star_sustained(env, ego=collector, j=j, horizon=H))
            means.append(float(np.mean(vals)))
        profiles[role_name] = means

        sign0 = np.sign(means[0])
        flip = None
        if sign0 != 0:
            for idx in range(1, len(h_sweep)):
                s = np.sign(means[idx])
                if s != 0 and s != sign0:
                    flip = h_sweep[idx]
                    break
        sign_flip_step[role_name] = flip

        incs = [abs(means[idx] - means[idx - 1]) for idx in range(1, len(h_sweep))]
        peak_inc = max(incs) if incs else 0.0
        sat = None
        if peak_inc > 0:
            for idx, inc in enumerate(incs, start=1):
                if inc < 0.2 * peak_inc:
                    sat = h_sweep[idx]
                    break
        saturation_step[role_name] = sat

    n_flip_roles = sum(1 for v in sign_flip_step.values() if v is not None)
    # NEW meaning (xem docstring trên): tỷ lệ vai trò (0..1) có sign-flip
    # trong H-sweep cố định forced_step=0. KHÔNG phải "spread của peak
    # forced_step trượt" như bản cũ -- tên khoá giữ nguyên chỉ để tương
    # thích env_audit_staged.py, ý nghĩa đã đổi, xem nhãn in ở main().
    t4_spread = n_flip_roles / 4.0

    return {
        "t4_h_sweep": list(h_sweep),
        "t4_profiles": profiles,               # role -> W*(H) list, forced_step=0 cố định (bản mới)
        "t4_sign_flip_step": sign_flip_step,    # role -> H đổi dấu đầu tiên, hoặc None
        "t4_saturation_step": saturation_step,  # role -> H bão hoà đầu tiên, hoặc None
        "t4_peak_steps": sign_flip_step,        # ALIAS tương thích ngược -- Ý NGHĨA ĐÃ ĐỔI, xem docstring
        "t4_spread": t4_spread,                 # ALIAS tương thích ngược -- Ý NGHĨA ĐÃ ĐỔI, xem docstring
    }


# ============================================================
# T6 -- tier_separation_ratio
# ============================================================

def run_t6(env_kwargs=None, seed=SEED):
    """
    env_kwargs: dict of OmniArena constructor kwargs shared by BOTH the
    structural-shift probe env and the behavioural-drift probe env (mode is
    always overridden per-probe below, so don't pass mode= in env_kwargs).
    Defaults to the original hardcoded audit config for backward
    compatibility with the existing no-arg env_audit.py call site.

    Used by env_audit_staged.py to build T6 probe envs with a given block's
    enable_* flag combination (e.g. Block C's enable_structural_shift=True),
    without forking this function's logic.
    """
    if env_kwargs is None:
        env_kwargs = dict(
            n_agents=N_AGENTS, grid_size=GRID_SIZE, n_zones=N_ZONES,
            max_steps=MAX_STEPS, phase_length=PHASE_LENGTH,
            causal_horizon=HORIZON,
        )

    # ------------------------------------------------------------------
    # RC-2. Cả TỬ và MẪU giờ đều đo trên Φ̃ = E_s[phi * delta], KHÔNG phải
    # trên bảng phi tĩnh.
    #
    # Bản cũ: tử = ‖dPhi‖ tĩnh tại ranh giới shift, mẫu = literal
    # `0.0 if behavioural_invariant else 1.0`. Mẫu số đó không phải phép đo —
    # nó là hằng số suy ra từ một assert về chính giả định thiết kế. T6 =
    # structural / 1e-12 = 9.9e11 là hệ quả số học, audit không hề chạm vào
    # dữ liệu nào.
    #
    # Trên Φ̃ thì behavioural drift CÓ tín hiệu thật: phi bất biến, nhưng
    # agent đi khác đi ⇒ delta_ij(s) đổi ⇒ Φ̃ đổi. Đây là thứ đáng gọi là
    # "tách tầng": cùng một đại lượng, đo hai loại can thiệp khác nhau.
    # ------------------------------------------------------------------
    env_t6 = OmniArena(mode="behavioral_drift", seed=seed, **env_kwargs)
    structural, behavioural = env_t6.measure_realized_phi_tiers(
        n_states=T6_N_STATES,
        burn_in=T6_BURN_IN,
        bank_seed=seed,
        behaviour_pair=T6_BEHAVIOUR_PAIR,
    )
    t6_ratio = env_t6.tier_separation_ratio()   # inf nếu behavioural == 0

    # Bất biến thiết kế P4 vẫn phải giữ: bảng phi TĨNH không được đổi trong
    # behavioural_drift. Kiểm riêng, và KHÔNG dùng nó làm mẫu số nữa.
    env_b = OmniArena(mode="behavioral_drift", seed=seed, **env_kwargs)
    static_phi_invariant = True
    try:
        env_b.assert_behavioural_phi_invariance(n_phases=T6_INVARIANCE_PHASES)
    except AssertionError as e:
        static_phi_invariant = False
        print(f"  !! static Phi invariance FAILED: {e}")

    return {
        "t6_delta_phi_structural_max": float(structural),
        "t6_delta_phi_behavioural": float(behavioural),
        "t6_ratio": float(t6_ratio),
        # Giữ nguyên key cũ cho env_audit_staged.py; ngữ nghĩa đổi thành
        # "bảng phi TĨNH bất biến" (vẫn phải True), không còn là mẫu số của T6.
        "t6_behavioural_invariant_exact": static_phi_invariant,
        "t6_static_phi_invariant": static_phi_invariant,
    }


# ============================================================
# Oracle no-abs() sanity check
# ============================================================

def run_oracle_sign_check(env):
    z = 0
    ra = env.zone_role_agents[z]
    collector = ra[env.ROLE_COLLECTOR]
    blocker = ra[env.ROLE_BLOCKER]  # phi = -0.50 (const, không phụ thuộc lane)

    states = sample_states(env, 4)
    vals = []
    for st in states:
        env.restore_state(st)
        vals.append(oracle_w_star_sustained(env, ego=collector, j=blocker))
    mean_w = float(np.mean(vals))
    phi = env.gt_influence_by_ego[collector].get(blocker, 0.0)
    return {
        "phi_negative_pair_phi": phi,
        "phi_negative_pair_mean_w_star": mean_w,
        "no_abs_ok": (phi < 0),  # phi luôn âm theo thiết kế; kiểm tra thật là
                                  # profile['per_action'] có thể âm (không bị
                                  # abs() ép dương) -- xem raw per_action dưới.
        "raw_per_action_signed_example": None,
    }


def oracle_no_abs_direct_check(env):
    """ Gọi trực tiếp oracle, kiểm tra per_action KHÔNG bị abs() ép dương. """
    z = 0
    ra = env.zone_role_agents[z]
    collector = ra[env.ROLE_COLLECTOR]
    blocker = ra[env.ROLE_BLOCKER]

    env.reset()
    for _ in range(3):
        acts = [env.scripted_policy(i) for i in range(env.n_agents)]
        env.step(acts, return_obs=False, return_info=False)

    profile = env.compute_oracle_influence_from_current_state(
        ego_id=collector, agent_j=blocker, intervention_action=env.STAY,
        horizon=HORIZON, n_trials=1, forced_step=0,
    )
    per_action = profile["per_action"]
    has_negative = any(v < 0 for v in per_action.values())
    has_positive = any(v > 0 for v in per_action.values())
    return {
        "per_action": per_action,
        "has_negative_delta": has_negative,
        "has_positive_delta": has_positive,
        "no_abs_verified": has_negative,  # abs() cũ sẽ KHÔNG BAO GIỜ cho âm
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 78)
    print("OMNI-ARENA env_audit.py -- P0-P4 acceptance check")
    print("=" * 78)

    env = OmniArena(
        n_agents=N_AGENTS, grid_size=GRID_SIZE, n_zones=N_ZONES,
        max_steps=MAX_STEPS, phase_length=PHASE_LENGTH,
        causal_horizon=HORIZON, mode="behavioral_drift", seed=SEED,
    )
    env.reset()

    print(f"\nconfig: n_agents={N_AGENTS} grid={GRID_SIZE} n_zones={N_ZONES} "
          f"H={HORIZON} n_states_T1={N_STATES_T1} n_states_T3={N_STATES_T3} "
          f"n_control_pairs={N_CONTROL_PAIRS} n_trials={N_TRIALS} seed={SEED}")

    print("\n[1/6] T1/T2/T5/corr(Phi,W*) via oracle rollouts ...")
    m1 = run_oracle_based_metrics(env)

    print("[2/6] T3 (CV of w_ij(s)) ...")
    m3 = run_t3(env)

    print("[3/6] T4 (latency spread) ...")
    m4 = run_t4(env)

    print("[4/6] T6 (tier_separation_ratio) ...")
    m6 = run_t6()

    print("[5/6] oracle no-abs() sign check ...")
    msign = run_oracle_sign_check(env)
    mabs = oracle_no_abs_direct_check(env)

    print("[6/6] done.\n")

    # ------------------------------------------------------------
    # Raw numbers
    # ------------------------------------------------------------
    print("=" * 78)
    print("RAW NUMBERS")
    print("=" * 78)

    print(f"\n-- T1: Gini(|W*|) across sampled pairs "
          f"(n_declared={m1['n_declared']}, n_control={m1['n_control']}, "
          f"n_states={m1['n_states']}) --")
    print(f"T1 Gini = {m1['t1_gini']:.4f}")

    print("\ndeclared pairs (i->j, phi, mean W* across states):")
    for (i, j, label, phi, mean_w, vals) in m1["declared_records"]:
        print(f"  {label:28s} agent{i:2d}->agent{j:2d}  phi={phi:+.3f}  "
              f"mean_W*={mean_w:+.4f}  raw={['%.3f' % v for v in vals]}")

    print(f"\ncontrol pairs: mean|W*| = "
          f"{np.mean(np.abs([r[4] for r in m1['control_records']])):.4f}, "
          f"std|W*| = {np.std(np.abs([r[4] for r in m1['control_records']])):.4f}")

    print(f"\n-- T2: sign balance --")
    print(f"T2 fraction significantly negative (< -{SIGNIFICANT_W}) = {m1['t2_neg_frac']:.4f}")
    print(f"T2 fraction significantly positive (> +{SIGNIFICANT_W}) = {m1['t2_pos_frac']:.4f}")

    print(f"\n-- T3: CV of w_ij(s) across {N_STATES_T3} states, per declared pair type --")
    for label, cv in m3["t3_cv_by_pair_type"].items():
        print(f"  {label:28s} CV = {cv:.4f}")
    print(f"T3 mean CV = {m3['t3_cv_mean']:.4f}")

    print(f"\n-- T4: latency profile — NEW methodology (forced_step CỐ ĐỊNH tại 0, "
          f"quét H qua {m4['t4_h_sweep']}; xem env_audit.py's run_t4() docstring "
          f"cho lý do đổi khỏi bản cũ 'trượt forced_step, H cố định') --")
    for role in ["blocker", "gatekeeper", "relay", "controller"]:
        H_sweep = m4["t4_h_sweep"]
        profile = m4["t4_profiles"][role]
        flip = m4["t4_sign_flip_step"][role]
        sat = m4["t4_saturation_step"][role]
        flip_str = f"H={flip}" if flip is not None else "no flip in sweep"
        sat_str = f"H={sat}" if sat is not None else "no saturation in sweep"
        print(f"  {role:12s} W*(H) by H={H_sweep} = "
              f"{['%+.4f' % v for v in profile]}")
        print(f"               sign-flip step (vs H={H_sweep[0]}) = {flip_str}   "
              f"saturation step (<20% peak increment) = {sat_str}")
    print(f"T4 sign-flip role fraction (0..1, NEW metric, key 't4_spread' kept "
          f"for backward-compat with env_audit_staged.py -- NOT the old "
          f"'peak forced_step spread') = {m4['t4_spread']:.4f}")

    print(f"\n-- T5: SNR --")
    print(f"T5 SNR (WITHIN-ZONE mean, GATE) = {m1['t5_snr']:.4f}")
    print(f"   per-zone: { {z: round(v, 3) for z, v in m1['t5_snr_per_zone'].items()} }")
    print(f"   [chẩn đoán] bản gộp toàn cục (KHÔNG dùng làm gate, phình vì "
          f"phương sai giữa-zone do zone asymmetry) = {m1['t5_snr_pooled']:.4f}")

    print(f"\n-- T6: tier_separation_ratio (RC-2: đo trên Φ̃ = E_s[phi*delta], "
          f"n_states={T6_N_STATES}, behaviour_pair={T6_BEHAVIOUR_PAIR}) --")
    print(f"  ||dPhi~||_F structural  (lật active_lane)           = "
          f"{m6['t6_delta_phi_structural_max']:.6f}")
    print(f"  ||dPhi~||_F behavioural (đổi behaviour mode, PHẢI > 0) = "
          f"{m6['t6_delta_phi_behavioural']:.6f}")
    print(f"  bảng phi TĨNH bất biến trong drift (P4 design)      = "
          f"{m6['t6_static_phi_invariant']}")
    print(f"T6 tier_separation_ratio = {m6['t6_ratio']:.4f}")

    print(f"\n-- corr(Phi, W*) -- (design intent vs oracle-measured ground truth) --")
    print(f"corr(Phi, W*) = {m1['corr_phi_w']:.4f}")

    print(f"\n-- oracle no-abs() check --")
    print(f"phi(blocker->collector) = {msign['phi_negative_pair_phi']:.3f} "
          f"(declared negative pair)")
    print(f"mean W*(blocker->collector) across states = "
          f"{msign['phi_negative_pair_mean_w_star']:+.4f}")
    print(f"per_action deltas (single call, no abs applied by oracle) = "
          f"{mabs['per_action']}")
    print(f"has_negative_delta = {mabs['has_negative_delta']}  "
          f"has_positive_delta = {mabs['has_positive_delta']}")

    # ------------------------------------------------------------
    # PASS/FAIL table (Phan 7)
    # ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PASS/FAIL — Phan 7 (P0-P4 relevant acceptance criteria)")
    print("=" * 78)

    # [docs/CIG-AMF_training_debug_master.md mục 5.5] T5/T6/corr giờ có
    # CHẶN TRÊN -- SNR=1e9 hay T6=9.9e11 không còn PASS: chúng chứng minh
    # control pairs suy biến (W*=0 tuyệt đối) / Phi là hằng số cấu hình,
    # tức audit đang đo lại chính giả định thiết kế, không đo tín hiệu
    # thật. Khoảng [3,20] là "cấu trúc phát hiện được nhưng không tầm
    # thường" -- vùng duy nhất kết quả có giá trị (mục 5.2/5.3).
    checks = [
        ("T1 Gini(|W*|) > 0.30", m1["t1_gini"], m1["t1_gini"] > 0.30),
        ("T2 sign balance: neg frac > 0.15", m1["t2_neg_frac"], m1["t2_neg_frac"] > 0.15),
        ("T2 sign balance: pos frac > 0.15", m1["t2_pos_frac"], m1["t2_pos_frac"] > 0.15),
        ("T3 CV (mean over declared pair types) > 0.30", m3["t3_cv_mean"], m3["t3_cv_mean"] > 0.30),
        ("T4 (NEW: fixed forced_step=0, H-sweep) sign-flip role frac > 0.3 (nice-to-have)",
         m4["t4_spread"], m4["t4_spread"] > 0.3),
        ("T5 SNR (within-zone) in [3, 20]", m1["t5_snr"], 3.0 <= m1["t5_snr"] <= 20.0),
        ("T6 ratio in [3, 20]", m6["t6_ratio"], 3.0 <= m6["t6_ratio"] <= 20.0),
        # RC-2: check này giờ đo THẬT. Trước đây mẫu số là literal 0.0/1.0
        # nên nó chỉ phản chiếu lại giả định thiết kế. Đại lượng đúng là Φ̃ =
        # E_s[phi*delta]: phi bất biến nhưng delta_ij(s) phụ thuộc vị trí
        # thật của cả hai agent, nên behavioural drift PHẢI để lại dấu vết
        # khác 0. Nếu vẫn == 0 thì hoặc state bank quá nhỏ, hoặc các gate
        # không thực sự phụ thuộc hành vi -- cả hai đều là lỗi cần sửa,
        # không được nới điều kiện để cho pass.
        ("T6 ||dPhi~||(behavioural) > 0 (đo trên Φ̃, không phải hằng số hardcode)",
         m6["t6_delta_phi_behavioural"], m6["t6_delta_phi_behavioural"] > 0.0),
        ("P4 design: bảng phi TĨNH bất biến trong behavioural_drift",
         m6["t6_static_phi_invariant"], m6["t6_static_phi_invariant"]),
        ("corr(Phi, W*) in [0.65, 0.95]  # [A5] ha nguong: Phi la dai luong 1 buoc, W* la rollout H buoc -> hap thu ca hieu ung gian tiep", m1["corr_phi_w"], 0.65 <= m1["corr_phi_w"] <= 0.95),
        ("oracle no abs() (per_action has negative deltas on a real pair)", mabs["has_negative_delta"], mabs["has_negative_delta"]),
    ]

    name_w = max(len(c[0]) for c in checks)
    for name, val, ok in checks:
        status = "PASS" if ok else "FAIL"
        val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
        print(f"  [{status}] {name:{name_w}s}  value = {val_str}")

    n_pass = sum(1 for _, _, ok in checks if ok)
    print(f"\n{n_pass}/{len(checks)} checks passed.")

    return {
        "t1": m1, "t3": m3, "t4": m4, "t6": m6,
        "sign_check": msign, "abs_check": mabs, "checks": checks,
    }


if __name__ == "__main__":
    main()
