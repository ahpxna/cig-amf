"""
OMNI-ARENA — hợp nhất DIG (Phi khai báo sạch) + arena_v3 (tắc nghẽn/hàng đợi)
theo docs/OMNI_ARENA_BLUEPRINT.md.

File MỚI, độc lập — envs/population_main_dig.py và
envs/adaptive_resource_flow_arena_v3.py không bị đụng tới (Phần 6, "Chỉ đạo
về quản lý mã nguồn").

Triển khai P1-P4:

P1 (Phần 2.1) — ma trận ảnh hưởng có dấu, có điều kiện:
    w_ij(s) = phi_ij * delta_ij(s)
    phi_ij:   hằng số có dấu, CHỈ đổi tại ranh giới structural shift.
    delta_ij(s): cổng kích hoạt in [0,1], tính lại MỖI BƯỚC từ state, ghi
                 vào info["delta_by_pair"].

P2 (Phần 2.2) — 5 vai trò / zone (collector, gatekeeper, relay, blocker,
    controller) với 4 tầm trễ latency: blocker h=1, gatekeeper h=2-3,
    relay h=4-5, controller h=6+. H_causal = 8 (>= max_latency(6) + 2).

P3 (Phần 1.1, 1.2, 2.3) — kênh reward tách bạch:
    r_i = r_solo + sum_j delta_ij(s)*phi_ij*psi(a_j)  [2] khai báo
        + r_emergent(i,s,a)                            [3] nổi lên (tắc nghẽn)
    Ràng buộc: max|r_emergent| <= 0.15 * min|phi_ij| (core pairs).

P4 (Phần 3.2, 3.3) — structural shift bằng DỜI BOTTLENECK (lane A -> lane B),
    KHÔNG hoán đổi vai trò -> không có cửa sổ chuyển tiếp vô định.
    behavioural_drift giữ Phi bất biến (assert được).
    tier_separation_ratio = ||dPhi||_F(structural) / ||dPhi||_F(behavioural)
    xuất trong info mỗi episode.
"""
import copy
import numpy as np

try:
    from tiny_oracle_dig import OracleInfluenceProfile
except ModuleNotFoundError:
    from envs.tiny_oracle_dig import OracleInfluenceProfile

class OmniArena:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    OPEN = 5   # hành động "kích hoạt tại chỗ" — gatekeeper mở cổng / controller
               # bấm panel, ngữ nghĩa phụ thuộc vai trò + ô đang đứng.
    N_ACTIONS = 6

    ROLE_COLLECTOR = "collector"
    ROLE_GATEKEEPER = "gatekeeper"
    ROLE_RELAY = "relay"
    ROLE_BLOCKER = "blocker"
    ROLE_CONTROLLER = "controller"
    ROLE_DRIFTER = "drifter"

    ROLE_ORDER = [
        ROLE_COLLECTOR, ROLE_GATEKEEPER, ROLE_RELAY,
        ROLE_BLOCKER, ROLE_CONTROLLER, ROLE_DRIFTER,
    ]

    # ------------------------------------------------------------------
    # P1: bảng Phi thiết kế (tier A) — hằng số có dấu, xem Phần 2.1.
    # Các cặp không liệt kê => phi = 0 (không có kênh khai báo).
    # ------------------------------------------------------------------
    # RC-1: thang phi GIỮ NGUYÊN. Không nâng |phi|, vì MAX_EMERGENT_MAGNITUDE
    # = 0.15 * MIN_CORE_PHI = 0.0375 đã đủ: kênh [3] nổ bão hoà qua H=8 cho
    # |W*| ~ 0.3, đưa T5 SNR về ~13 (trong gate [3,20]) mà không phải nâng
    # trần. Bài toán của T5 là TẦN SUẤT NỔ, không phải biên độ.
    PHI_GATEKEEPER_TO_COLLECTOR = -0.60
    PHI_RELAY_TO_COLLECTOR = 0.35
    PHI_BLOCKER_TO_COLLECTOR = -0.50
    PHI_CONTROLLER_TO_COLLECTOR = 0.25
    PHI_COLLECTOR_TO_GATEKEEPER = 0.30
    # P4: bonus lane-B do dời bottleneck — cộng thêm vào cặp (blocker, collector)
    # khi active_lane == 'B' (blocker tình cờ đã đứng gần lane B từ đầu, không
    # phải di chuyển gì để "trở thành" ảnh hưởng cao).
    PHI_LANE_B_BONUS = 0.35

    CORE_PHI_MAGNITUDES = [
        abs(PHI_GATEKEEPER_TO_COLLECTOR),
        abs(PHI_RELAY_TO_COLLECTOR),
        abs(PHI_BLOCKER_TO_COLLECTOR),
        abs(PHI_CONTROLLER_TO_COLLECTOR),
        abs(PHI_COLLECTOR_TO_GATEKEEPER),
    ]
    MIN_CORE_PHI = min(CORE_PHI_MAGNITUDES)  # 0.25
    # P3 / Phần 1.2: max|r_emergent| <= 0.15 * min|phi_ij|
    #
    # RC-3(a): trước khi xoá hai khối hand-coded, bất biến này vô nghĩa — nó
    # neo vào bảng phi DANH NGHĨA (min 0.25) trong khi dòng reward pairwise
    # THẬT là 2.5, tức tỉ lệ thực 1.5% chứ không phải 15%. Sau RC-1, phi LÀ
    # dòng reward pairwise duy nhất nên bất biến này lại đúng theo nghĩa đen.
    # [P-3 FINAL DEBUG] Trần nới 0.0375 -> 0.045 để đi kèm ZONE_CROWDING_COEF
    # 0.012 -> 0.020: crowding điển hình ~0.030 + queue ~0.010 = ~0.040 phải
    # nằm DƯỚI trần, nếu không r_emergent clip liên tục và đạo hàm theo vị trí
    # biến mất (đúng cảnh báo RC-3 bên dưới).
    MAX_EMERGENT_MAGNITUDE = 0.18 * MIN_CORE_PHI / 3.33   # [KNOB-2] cùng nhịp

    # ------------------------------------------------------------------
    # RC-5: tham số kênh [3]. Bản cũ dùng radius=1 + ngưỡng `> 2` trên lưới
    # 24x24 với 6 agent/zone ⇒ mật độ quá thấp, đo thật max|r_emergent| = 0.0
    # cả episode. Kênh arena_v3 khi đó KHÔNG TỒN TẠI: "DIG + arena_v3" thực
    # chất là "DIG + 0".
    #
    # HIỆU CHỈNH (RC-3): biên độ cộng dồn điển hình phải nằm TRONG khoảng
    # (0, MAX_EMERGENT_MAGNITUDE) chứ không dán vào trần. Nếu r_emergent bị
    # clip liên tục thì đạo hàm theo vị trí agent khác biến mất ⇒ W*(control)
    # về 0 lần nữa và SNR lại nổ. Nói cách khác: "bão hoà" ở đây nghĩa là NỔ
    # MỖI BƯỚC, không phải CHẠM TRẦN mỗi bước.
    # Tổng điển hình: crowding ~0.018 + queue ~0.010 = ~0.028 < 0.0375.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # [KNOB-2] KÊNH NHIỄU NỀN (congestion) hạ ĐÚNG 3.33x cùng nhịp với
    # GAME_COST_WEIGHT 0.05 -> 0.015.
    #
    # SỬA MỘT NHẬN ĐỊNH SAI CỦA LƯỢT TRƯỚC: tôi đã nói mọi gate audit là tỉ
    # số nên bất biến theo thang. ĐÚNG với T1 Gini và T6 ratio, SAI với T5 và
    # corr. Lý do: W* đo trên RETURN, mà return có BA kênh
    #       task reward  +  declared w_ij  +  r_emergent (congestion)
    # GAME_COST_WEIGHT chỉ nhân kênh GIỮA. Hạ nó 3.3x làm tỉ trọng declared
    # trong W* co lại trong khi hai kênh kia giữ nguyên => corr(Phi,W*) tụt
    # (Phi chỉ mô hình hoá kênh declared) và mẫu số T5 phình tương đối.
    # Bằng chứng: corr 0.7246 -> 0.5519 trong ĐÚNG lần sửa đó.
    #
    # Đây là bài toán HAI bậc tự do:
    #     núm 1 GAME_COST_WEIGHT   -> declared / task
    #     núm 2 congestion coeffs  -> noise / declared
    # Sửa núm 1 mà không sửa núm 2 = đổi tỉ số thứ hai một cách vô ý.
    # Hạ cả cụm congestion cùng hệ số giữ nguyên declared/noise.
    # ------------------------------------------------------------------
    CONGESTION_SCALE = 1.0 / 3.33

    LANE_CONGESTION_RADIUS = 2
    LANE_CAPACITY = 2                 # `>= 2` thay vì `> 2`
    LANE_CONGESTION_PENALTY = 0.010 * CONGESTION_SCALE
    STATION_QUEUE_RADIUS = 2
    STATION_QUEUE_PENALTY = 0.020 * CONGESTION_SCALE
    COLLISION_PENALTY = 0.015 * CONGESTION_SCALE
    # Coupling nền TRƠN theo khoảng cách, phạm vi cùng-zone. Đây là thứ RC-3(b)
    # bắt buộc phải có: đo thật cho thấy các cặp control cùng zone KHÔNG chứa
    # blocker (relay->controller, relay->gatekeeper, ...) trả W* = 0 tuyệt đối
    # vì giữa chúng không tồn tại BẤT KỲ dòng code tương tác nào. Sampling
    # cùng zone là cần nhưng KHÔNG đủ — phải có kênh vật lý nối chúng.
    # [P-3 FINAL DEBUG] 0.012 -> 0.020: T5 SNR đo được 33.38 hơi vượt band
    # chấp nhận [3, 20] vì std|W*|(control) = 0.0197 hơi nhỏ. Tăng ~1.67x kênh
    # nền trơn kéo std control lên ~0.033 => SNR ~ 20. Nếu sau khi chạy lại
    # env_audit mà T1 Gini < 0.30 hoặc r_emergent clip thường xuyên, hạ về 0.016.
    ZONE_CROWDING_COEF = 0.020 * CONGESTION_SCALE   # [KNOB-2]

    # ------------------------------------------------------------------
    # [C3] Thang trọng số khối tương tác — neo theo SGTP game_cost_config.yaml
    #      safety 50 > block 10 > long 2 > contest 1   (spread 50x)
    # SGTP dùng s tính bằng mét trên track ~100m với contest_s_gap = 8m ≈ 8%
    # chiều dài. Ở đây s ĐÃ chuẩn hoá về [0,1] trên chuỗi gate->resource->sink,
    # nên các ngưỡng được quy đổi GIỮ NGUYÊN TỈ LỆ: s_role/s_contest = 1/8.
    #
    # Chỉ chỉnh MỘT số duy nhất — GAME_COST_WEIGHT — khi cân khối tương tác
    # với task reward (delivery 0.7, pickup 0.3, r_solo 0.05). Không chỉnh
    # từng hằng số rời như bản cũ (BLOCKER_PENALTY 2.5 vs baseline 0.01 =
    # lệch 250x).
    # ------------------------------------------------------------------
    # Đã quét trên gate S4 (T5 SNR ở BLOCK A, tức P3 congestion TẮT):
    #   50/10/2/1.0  -> SNR 2.910  FAIL
    #   20/10/2/0.5  -> SNR 2.999  FAIL
    #   10/10/2/0.3  -> SNR 3.030  PASS   <-- chọn
    #    2/10/2/0.2  -> SNR 3.044  PASS nhưng bỏ hẳn số hạng an toàn
    # Hạ W_SAFETY khỏi 50: tỉ lệ đó của SGTP là cho va chạm VẬT LÝ ở tốc độ
    # cao; ở đây va chạm đã được xử lý bởi grid physics + COLLISION_PENALTY,
    # nên safety=50 chỉ bơm spike vào các cặp control (đẩy std|W*| lên).
    W_SAFETY = 10.0
    W_BLOCK = 10.0
    # [RELAY-BUFF] kênh hỗ trợ-từ-phía-sau, cùng bậc với block (x4 so với
    # W_LONG) để relay/controller nổi lên trên nền coupling.
    # [BUFF-3] x3 sau khi số hạng obstruction/support đã THỰC SỰ bắn
    # (chỉ đúng sau [UNIT-FIX]; trước đó nâng W_SUPPORT là vô nghĩa vì
    # gatekeeper/relay nằm ngoài mọi band).
    # [SUPPORT-FIX] 24 -> 10, và nhân thêm alpha trong công thức.
    # 24 làm gatekeeper thành cặp mạnh nhất và lộ lại mâu thuẫn dấu ở biên
    # độ 1.2 (phi +0.05..+0.23 vs W* -1.28..+0.49).
    W_SUPPORT = 10.0
    # [OBSTRUCTION] kênh đồng-vị-trí. Đây là kênh chính của blocker/relay
    # trong env này (Δs~0), nên nó phải MẠNH tương đương W_BLOCK.
    # Quét trên gate T5 within-zone (sau [UNIT-FIX]):
    #   Wobs=12 Dlat=.25 Scon=.25 -> T5 13.23
    #   Wobs=12 Dlat=.45 Scon=.65 -> T5 12.32
    #   Wobs= 6 Dlat=.45 Scon=.65 -> T5  3.96   <-- chọn (giữa band, không sát trần)
    W_OBSTRUCT = 6.0
    # [ENABLE] hệ số họ tiền-điều-kiện. Cùng bậc với W_OBSTRUCT để hai họ
    # cạnh tranh được với nhau (gatekeeper vừa mở cổng vừa bám đuôi).
    # Quét trên (corr, sign agreement):
    #    0 -> corr +0.336 sign 12/20      3 -> +0.344 12/20
    #    6 -> corr +0.398 sign 14/20     10 -> +0.500 16/20   <-- chọn
    # corr vẫn dưới sàn 0.65 nhưng ĐƠN ĐIỆU tăng theo W_ENABLE và không
    # chạm trần 0.95 => chưa phải fit. Không nâng tiếp quá 10 khi chưa xử
    # xong cặp collector->gatekeeper (xem báo cáo).
    W_ENABLE = 10.0
    S_NEAR = 0.06      # ngưỡng "đồng vị trí" theo trục s
    # D_LAT_OBS nới 0.25 -> 0.45 để phủ relay ở zone có lane xa (đo được
    # z1 relay dd=0.400 vẫn NGOÀI band ở ngưỡng 0.25).
    # [PARTITION] Ngưỡng ngang DUY NHẤT phân hoạch vùng "phía sau" thành
    # bám-đuôi (obstruct) vs nhường-đường (support). Quét trên hai điều kiện
    # đồng thời (Phi(blocker)<0 và Phi(relay)>0 ở CẢ 4 zone):
    #     0.15 / 0.18 -> blocker OK nhưng z1 yếu hẳn (-0.019 vs -0.101)
    #     0.21 / 0.25 -> CẢ HAI OK ở 4/4        <-- cửa sổ hợp lệ
    #     0.45        -> relay LẬT DẤU 4/4 (bị xếp nhầm thành bám đuôi)
    # Chọn 0.23 = giữa cửa sổ, biên an toàn rộng nhất về cả hai phía.
    # Đo thật: blocker dd ~ 0.11-0.20, relay dd ~ 0.22-0.40 — ngưỡng nằm
    # đúng khe giữa hai vai trò, đó là lý do phân hoạch một-ngưỡng hoạt động.
    D_LAT_OBS = 0.23
    W_LONG = 2.0
    W_CONTEST = 0.3
    # 0.25 -> 0.65: controller đo được ds = +0.33..+0.60 nên NGOÀI band ở
    # ngưỡng cũ; nó chưa bao giờ có kênh declared nào.
    S_CONTEST = 0.65
    S_ROLE = 0.03125          # = S_CONTEST / 8, giữ đúng tỉ lệ SGTP
    D_SAFE = 0.08
    D_LAT = 0.05              # ngưỡng lệch ngang cho CSD (C4)
    # [GCW-CAL] 0.05 -> 0.015. Lần đầu chạy runner trên ĐÚNG OmniArena (sau
    # khi sửa [ENV-RESOLVE]) lộ ra kênh tương tác đang ÁP ĐẢO task reward.
    # Đo dưới scripted policy, mỗi agent mỗi bước:
    #     tổng w_ij      = -0.0438   (p5 -0.504, p95 +0.346)
    #     task reward    = +0.0179   (đã trừ kênh w)
    #     => kênh tương tác chiếm 169% độ lớn reward
    # Learner vì thế không thể cải thiện task reward: mọi tiến bộ về giao
    # hàng đều bị phạt tương tác nuốt. Log: reward -2.1 -> -3.4 và f1 = 0.000
    # suốt run, TỆ HƠN cả scripted (-0.778/episode/agent).
    # 0.015 đưa kênh tương tác về ~50% độ lớn task reward — đủ để cấu trúc
    # đáng học, không đủ để nó là toàn bộ bài toán.
    #
    # AN TOÀN VỚI AUDIT: mọi gate (T1 Gini, T5 SNR, T6 ratio, corr) đều là
    # TỈ SỐ hoặc thống kê bất biến theo thang, nên đổi số này KHÔNG làm lệch
    # chúng. Đó chính là lý do thiết kế "một núm duy nhất".
    GAME_COST_WEIGHT = 0.015

    # P0(d): noise purposes có trong step() — dùng cho CRN buffer của oracle.
    NOISE_PURPOSES = ["resource_respawn"]

    def __init__(
        self,
        n_agents=24,
        grid_size=24,
        n_zones=4,
        obs_radius=5,
        max_steps=60,
        phase_length=40,
        causal_horizon=8,
        mode="behavioral_drift",
        seed=42,
        enable_conditional_gates=True,   # P1 — delta_ij(s), off => delta collapses to 1 (unconditional phi, DIG-like)
        enable_latency_ladder=True,      # P2 — 4-tier latency, off => single flat h=1-like window for all gates
        enable_congestion=True,          # P3 — collision/lane/queue r_emergent, off => r_emergent EXACTLY 0
        enable_structural_shift=True,    # P4 — bottleneck relocation, off => only behavioural_drift ever runs
        use_sgtp_phi=True,               # [C2] Phi liên tục kiểu SGTP; False = bảng tra cũ (ablation)
    ):
        assert n_agents >= 5 * n_zones, "Cần ít nhất 5 agents / zone (P2: 5 vai trò)."
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.n_zones = n_zones
        self.obs_radius = obs_radius
        self.max_steps = max_steps
        self.phase_length = phase_length
        self.causal_horizon = causal_horizon
        self.mode = mode
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.gate_cycle_length = 8
        self.gate_open_duration = 2
        self.gatekeeper_sight = 2

        # ------------------------------------------------------------
        # Diagnostic on/off flags (staged audit — see env_audit_staged.py).
        # Real branching everywhere these are read: OFF means the branch
        # bypasses the P1-P4 logic entirely, it is never "computed then
        # zeroed" -- that would defeat the point of isolating each phase.
        # ------------------------------------------------------------
        self.enable_conditional_gates = bool(enable_conditional_gates)
        self.enable_latency_ladder = bool(enable_latency_ladder)
        self.enable_congestion = bool(enable_congestion)
        self.enable_structural_shift = bool(enable_structural_shift)
        # [C2] True: w_ij = -GAME_COST_WEIGHT * sgtp_pair_cost(Δs, Δd), liên
        # tục trên mọi cặp cùng zone. False: bảng tra phi x delta cũ.
        self.use_sgtp_phi = bool(use_sgtp_phi)
        self._in_phi_measurement = False

        # [C2d] P1 (conditional_gates) và P2 (latency_ladder) là NO-OP khi
        # use_sgtp_phi bật: cả hai chỉ tác động qua _compute_deltas(), mà
        # nhánh SGTP không dùng delta nữa. Bằng chứng: trong staged audit,
        # block A trừ BASELINE = 0.0000 ở MỌI metric (trừ T6, vốn đang hỏng
        # riêng). Latency ladder đã chết — đó cũng là lý do T4 = 0.25.
        # Không xoá tham số (giữ tương thích ablation use_sgtp_phi=False),
        # nhưng nói thẳng ra thay vì để người đọc tưởng chúng còn tác dụng.
        if self.use_sgtp_phi and not (
            self.enable_conditional_gates and self.enable_latency_ladder
        ):
            print("[OmniArena][NOTE] use_sgtp_phi=True => "
                  "enable_conditional_gates / enable_latency_ladder là NO-OP "
                  "(kênh ảnh hưởng không đi qua delta_ij nữa).")

        self.supported_egos = list(range(self.n_agents))
        self.current_phase = 0
        self.episode_count = 0
        self.episode_deliveries = 0

        # RC-2: ép _behaviour_mode() trả về một mode cố định. CHỈ dùng cho phép
        # đo Φ̃ (xem measure_realized_phi_tiers) — nó cần cô lập biến "hành vi"
        # khỏi biến "pha episode", nếu không hai lần đo sẽ khác nhau ở cả hai
        # trục và không quy được chênh lệch cho trục nào.
        self._behaviour_override = None

        # [Guard A -- P2<->P4 trap] True CHỈ trong lúc reset() đang chạy.
        # Đây là tín hiệu ĐÚNG để _do_structural_shift() tự bảo vệ mình
        # (xem docstring hàm đó): không phải self.t/self.done của episode
        # TRƯỚC (không nói lên gì về lệnh gọi hiện tại), mà là "lệnh gọi
        # này có thực sự bắt nguồn từ reset() hay không".
        self._in_reset = False

        # noise buffer infra P0(d)
        self.noise_buffer = None
        self._noise_call_counter = {}

        # T6 bookkeeping
        self.delta_phi_frobenius_structural_last = 0.0
        self.delta_phi_frobenius_behavioural_last = 0.0
        self._prev_gt_influence_by_ego = None

        self._init_zone_heterogeneity()
        self._init_zone_layout()
        self._init_population_roles()
        self.reset()
        # [C2b] lần đo Φ đầu tiên: phải sau reset() vì cần positions.
        if self.use_sgtp_phi:
            self._measure_phi_from_sgtp()

    # ============================================================
    # Layout
    # ============================================================

    def _init_zone_heterogeneity(self):
        """
        [ZONE-ASYM] Phá đối xứng zone.

        Bằng chứng buộc phải làm: trong hình W*_ij(t), đường 1->0 TRÙNG KHÍT
        TỪNG ĐIỂM với 6->5, và 3->0 với 8->5. blocker W* = -1.2548/-1.2571/
        -1.1915/-1.2548 qua 4 zone. Mọi zone dùng CÙNG một công thức offset
        cố định từ tâm (gate=c-2, resource=c+1, sink=c+3), nên sau khi
        _frenet_sd chuẩn hoá thì hình học tương đối GIỐNG HỆT NHAU tuyệt đối.
        N=24 thực chất là N=6 nhân 4 => Experiment 5 (sweep N) vô nghĩa.

        BẪY RNG (quan trọng): self.rng đang nuôi noise_buffer/CRN, mà toàn bộ
        phép so sánh oracle-paired (kết quả có ý nghĩa thống kê duy nhất hiện
        có: dr_eps005 + xi_ij) dựa vào CRN để hai nhánh can thiệp/control
        dùng CHUNG số ngẫu nhiên. Rút draw từ self.rng ở đây sẽ lệch offset
        mọi draw sau đó. => RNG RIÊNG, seed dẫn xuất từ self.seed, sinh ĐÚNG
        MỘT LẦN trong __init__, KHÔNG BAO GIỜ trong reset().

        Không gian tham số: path_len 4 x lane_capacity 3 x respawn_delay 3 =
        36 tổ hợp rời rạc — đủ tới n_zones=16 (tức N=96 với ~6 agent/zone).
        Từ n_zones=32 trở lên sẽ có zone trùng cả ba; khi đó zone_scale
        (liên tục, nhân thẳng vào w_ij) là thứ duy nhất đảm bảo không hai
        zone giống hệt.
        """
        zone_param_rng = np.random.RandomState(
            (int(self.seed) * 7919 + 104729) % (2 ** 31 - 1)
        )

        # Dải path_len PHẢI co theo ô lưới của zone, nếu không _clip sẽ dán
        # gate/sink của nhiều zone vào cùng biên và tạo ra một kiểu trùng lặp
        # KHÁC (đúng cái ta đang đi sửa). cell_h tính lại đúng công thức của
        # _init_zone_layout. Với grid=24/n_zones=4 -> cell_h=12 -> pl <= 9.
        _rows = max(2, int(np.sqrt(self.n_zones)))
        _cell_h = self.grid_size // _rows
        _max_pl = max(3, _cell_h - 3)
        _cands = sorted({
            int(round(v)) for v in np.linspace(3, _max_pl, 4)
        })
        self._zone_path_len_choices = _cands

        self.zone_path_len = {}
        self.zone_lane_capacity = {}
        self.zone_respawn_delay = {}
        self.zone_scale = {}
        for z in range(self.n_zones):
            self.zone_path_len[z] = int(zone_param_rng.choice(_cands))
            self.zone_lane_capacity[z] = int(zone_param_rng.choice([1, 2, 3]))
            self.zone_respawn_delay[z] = int(zone_param_rng.choice([1, 2, 4]))
            self.zone_scale[z] = float(zone_param_rng.uniform(0.7, 1.4))

    def _init_zone_layout(self):
        self.zone_gate = {}
        self.zone_resource = {}
        self.zone_sink = {}
        self.zone_lane_a = {}      # relay's home lane ("lane A")
        self.zone_lane_b = {}      # alternate lane ("lane B") — near blocker
        self.zone_panel = {}       # controller's target tile
        self.zone_checkpoint = {}  # RC-4: blocker's own duty tile
        self.zone_centers = {}

        rows = max(2, int(np.sqrt(self.n_zones)))
        cols = int(np.ceil(self.n_zones / rows))
        cell_h = self.grid_size // rows
        cell_w = self.grid_size // cols

        zone = 0
        for rr in range(rows):
            for cc in range(cols):
                if zone >= self.n_zones:
                    break
                r0 = rr * cell_h
                c0 = cc * cell_w
                r1 = min(self.grid_size - 1, r0 + cell_h - 1)
                c1 = min(self.grid_size - 1, c0 + cell_w - 1)

                cr, cc_ = (r0 + r1) // 2, (c0 + c1) // 2
                self.zone_centers[zone] = (cr, cc_)

                # [ZONE-ASYM] path_len[z] đổi trục s => đổi mọi Δs_ij =>
                # đổi cả alpha_ij lẫn điều kiện role. Đây là biến quan trọng
                # nhất để phá trùng khít. Tỉ lệ resource ~60% quãng đường giữ
                # đúng như bản cũ (3/5 khi path_len=5).
                pl = int(self.zone_path_len[zone])
                assert pl < cell_h - 2, (
                    f"path_len={pl} vượt cell_h={cell_h} của zone {zone}; "
                    f"_clip sẽ dán nhiều zone vào cùng biên và tạo trùng lặp "
                    f"kiểu khác. Tăng grid_size hoặc giảm n_zones."
                )
                res_off = max(1, int(round(pl * 0.6)))
                self.zone_gate[zone] = (self._clip(cr - 2), self._clip(cc_))
                self.zone_resource[zone] = (
                    self._clip(cr - 2 + res_off), self._clip(cc_))
                self.zone_sink[zone] = (self._clip(cr - 2 + pl), self._clip(cc_))
                self.zone_lane_a[zone] = (self._clip(cr), self._clip(cc_ - 3))
                self.zone_lane_b[zone] = (self._clip(cr), self._clip(cc_ + 3))
                self.zone_panel[zone] = (self._clip(cr - 3), self._clip(cc_ - 3))
                # RC-4: ô trực của blocker, nằm GIỮA resource (cr+1) và
                # sink (cr+3) — tức trên đúng tuyến vận chuyển của collector.
                # Chọn vị trí này có chủ đích: blocker chỉ theo đuổi nhiệm vụ
                # tĩnh của nó (đứng đúng ô), nhưng vì ô đó nằm trên tuyến nên
                # dist(blocker, collector) <= 2 vẫn xảy ra thường xuyên ⇒ cạnh
                # khai báo blocker->collector vẫn sống. Ảnh hưởng là HỆ QUẢ
                # PHỤ của nhiệm vụ, không phải thứ blocker được trả tiền để
                # tối đa hoá. Nếu đặt ô trực ra xa tuyến (vd lane_b) thì
                # delta của gate_blocker_collector về 0 và cạnh âm duy nhất
                # trong đồ thị chết theo — T2 sign balance sập.
                self.zone_checkpoint[zone] = (self._clip(cr + 2), self._clip(cc_))
                zone += 1

    def _init_population_roles(self):
        self.agent_role = {}
        self.agent_zone = {}
        self.zone_role_agents = {}
        self._spawn_offsets = {}

        idx = 0
        for z in range(self.n_zones):
            collector = idx
            gatekeeper = idx + 1
            relay = idx + 2
            blocker = idx + 3
            controller = idx + 4
            idx += 5

            self.zone_role_agents[z] = {
                self.ROLE_COLLECTOR: collector,
                self.ROLE_GATEKEEPER: gatekeeper,
                self.ROLE_RELAY: relay,
                self.ROLE_BLOCKER: blocker,
                self.ROLE_CONTROLLER: controller,
            }
            for role, aid in self.zone_role_agents[z].items():
                self.agent_role[aid] = role
                self.agent_zone[aid] = z

        for a in range(idx, self.n_agents):
            self.agent_role[a] = self.ROLE_DRIFTER
            self.agent_zone[a] = self.rng.randint(0, self.n_zones)

        # active bottleneck lane per zone -- P4
        self.active_lane = {z: "A" for z in range(self.n_zones)}

        self._refresh_gt_graph()

    # ============================================================
    # P1 + P4: bảng Phi (tier A, hằng số có dấu, chỉ đổi tại shift boundary)
    # ============================================================

    def _refresh_gt_graph(self):
        """
        Tính lại Phi (tier A). CHỈ được gọi:
          - lúc __init__
          - tại ranh giới structural shift (P4)
        KHÔNG BAO GIỜ được gọi trong behavioural_drift -> Phi bất biến,
        kiểm chứng bằng assert_behavioural_invariance() / test_phi_invariance.py.
        """
        self.gt_core_by_ego = {}
        self.gt_influence_by_ego = {}
        self.declared_pairs = []  # list of (i, j, phi, gate_fn_name) cho P1/P3
        self._declared_set = None  # [SCOPE-DECL] dựng lại cùng declared_pairs

        for ego in range(self.n_agents):
            self.gt_core_by_ego[ego] = set()
            self.gt_influence_by_ego[ego] = {j: 0.0 for j in range(self.n_agents) if j != ego}

        for z in range(self.n_zones):
            ra = self.zone_role_agents[z]
            collector = ra[self.ROLE_COLLECTOR]
            gatekeeper = ra[self.ROLE_GATEKEEPER]
            relay = ra[self.ROLE_RELAY]
            blocker = ra[self.ROLE_BLOCKER]
            controller = ra[self.ROLE_CONTROLLER]

            # gatekeeper -> collector (phi bất đổi theo lane)
            self._set_phi(gatekeeper, collector, self.PHI_GATEKEEPER_TO_COLLECTOR)
            self.declared_pairs.append((gatekeeper, collector, "gate_gk_collector", z))

            # relay -> collector: chỉ có "trọng số thiết kế" khi lane A đang
            # là bottleneck đang hoạt động (P4 — dời bottleneck).
            relay_phi = self.PHI_RELAY_TO_COLLECTOR if self.active_lane[z] == "A" else 0.0
            self._set_phi(relay, collector, relay_phi)
            self.declared_pairs.append((relay, collector, "gate_relay_collector", z))

            # blocker -> collector: phi "nguy hiểm" cố định + bonus lane-B khi
            # lane B đang là bottleneck (blocker đã đứng sẵn gần đó, KHÔNG di
            # chuyển gì cả -- đúng tinh thần "dời bottleneck" của Phần 3.2).
            blocker_phi = self.PHI_BLOCKER_TO_COLLECTOR
            if self.active_lane[z] == "B":
                blocker_phi += self.PHI_LANE_B_BONUS
            self._set_phi(blocker, collector, blocker_phi)
            self.declared_pairs.append((blocker, collector, "gate_blocker_collector", z))

            # RC-4 (đã cân nhắc và LOẠI BỎ): KHÔNG thêm cạnh mirror zero-sum
            # (collector -> blocker). Runner tối ưu return PER-AGENT, không có
            # mixer/QMIX/team reward, nên "tổng âm" không phải cơ chế gây hại.
            # Zero-sum hoá còn phản tác dụng: blocker sẽ ăn ĐÚNG phần collector
            # mất ⇒ động cơ đuổi bắt MẠNH HƠN, và framing hợp tác của env bị
            # phá. Cách đúng là cắt động cơ đuổi bắt ở gốc (xem r_solo trong
            # step()): blocker theo nhiệm vụ riêng, ảnh hưởng lên collector là
            # HỆ QUẢ PHỤ. Về mặt nghiên cứu đây mới là thứ cần đo — ảnh hưởng
            # cấu trúc do môi trường sinh ra, không phải thứ agent cố ý tạo.

            # controller -> collector
            self._set_phi(controller, collector, self.PHI_CONTROLLER_TO_COLLECTOR)
            self.declared_pairs.append((controller, collector, "gate_controller_collector", z))

            # collector -> gatekeeper
            self._set_phi(collector, gatekeeper, self.PHI_COLLECTOR_TO_GATEKEEPER)
            self.declared_pairs.append((collector, gatekeeper, "gate_collector_gatekeeper", z))

            core = {gatekeeper, relay, blocker, controller}
            self.gt_core_by_ego[collector] |= core
            self.gt_core_by_ego[gatekeeper] |= {collector}

        # ------------------------------------------------------------------
        # [C2b] ĐO LẠI Φ TỪ CHÍNH CÔNG THỨC LIÊN TỤC, không dùng bảng tra.
        #
        # Bằng chứng buộc phải làm việc này:
        #   - corr(Φ,W*) = −0.0283: Φ vẫn là bảng tra cũ trong khi W* đã do
        #     công thức liên tục sinh ra => hai đại lượng nói về hai env khác
        #     nhau, hệ số tương quan giữa chúng vô nghĩa.
        #   - gatekeeper: phi=−0.600 vs W*=+1.5151 (ngược dấu, khuếch đại 13×)
        #   - relay: +0.9899 → −0.0721 (lật dấu do refactor)
        #   - T6 = 5.9004 GIỐNG TỪNG CHỮ SỐ trước và sau đại tu kênh ảnh
        #     hưởng => T6 đang đo bảng phi tĩnh, đã tách rời khỏi cơ chế đang
        #     chạy. Cùng loại bệnh với hardcode `= 0.0` ngày xưa.
        #
        # Φ_ij := E_s[ w_ij(s) ] trên một state bank cố định seed. Sau bước
        # này, Φ là ĐẠI LƯỢNG ĐO ĐƯỢC của env chứ không phải hằng số khai
        # báo — nên corr(Φ,W*), dấu, và T6 đều bám vào cơ chế thật.
        # gt_core_by_ego (nhãn vai trò) GIỮ NGUYÊN: nó là ground truth để
        # chấm Core F1, không tham gia tính reward.
        # ------------------------------------------------------------------
        # positions chưa tồn tại ở lần gọi đầu (trong _init_population_roles,
        # trước reset()) -> đo sau, ở cuối __init__.
        if (getattr(self, "use_sgtp_phi", False)
                and not getattr(self, "_in_phi_measurement", False)
                and hasattr(self, "positions")):
            self._measure_phi_from_sgtp()

    def _measure_phi_from_sgtp(self, n_states=48, burn_in=3, bank_seed=None):
        """Φ_ij = E_s[w_ij(s)] — xem giải thích ở cuối _refresh_gt_graph.

        [PHI-SAMPLING] bank_seed 777 CỐ ĐỊNH -> None (xoay theo lần đo), và
        n_states 12 -> 48.

        Bằng chứng buộc phải sửa — phân rã W*(collector->gatekeeper) theo
        kênh, 4 zone:
            d_total  +0.270 | -0.006 | -0.173 | +0.771
            d_w(c->gk) +0.263 | -0.015 | -0.181 | +0.764
            d_emergent +0.007 | +0.009 | +0.008 | +0.008
        d_total ≈ d_w KHỚP tới <=0.01 ở CẢ BỐN zone => W* của cặp này đi qua
        ĐÚNG MỘT kênh: chính w_ij mà Phi đang mô hình hoá. KHÔNG thiếu cơ
        chế nào. (Kênh bảng-tra cũ ở nhánh else chỉ chạy khi
        use_sgtp_phi=False, nên không phải nó.)

        Vậy 4/4 lệch dấu đến từ đâu? Từ CHỖ LẤY MẪU:
          - Phi  = trung bình w_ij trên 12 state của MỘT bank seed cứng 777,
                   lấy dưới hành vi mặc định, KHÔNG BAO GIỜ đổi.
          - W*   = trung bình hiệu quả trên quỹ đạo state THỰC TẾ sau can
                   thiệp — phân bố hoàn toàn khác.
        Cùng công thức w_ij, hai phân bố state khác nhau. Bằng chứng trực
        tiếp: đo lại cặp này trên tập state khác cho dấu +0.26/-0.02/-0.18/
        +0.76, trong khi audit cho +0.91/+0.61/+0.72/+0.96 — ĐỔI CẢ DẤU chỉ
        vì đổi tập state.

        Chính docstring của measure_realized_phi_tiers (T6) đã cảnh báo đúng
        nguyên tắc này: "state bank KHÔNG thể là một tập cố định dùng lại".
        T6 áp dụng đúng, _measure_phi_from_sgtp thì quên.

        Sửa: bank xoay theo lần đo (bank_seed=None -> dẫn xuất từ self.seed
        và bộ đếm) + tăng n_states 12->48 để giảm phương sai Monte-Carlo.
        KHÔNG đụng công thức w_ij, chỉ đổi quy trình đo.
        """
        if bank_seed is None:
            self._phi_measure_count = getattr(self, "_phi_measure_count", 0) + 1
            bank_seed = (
                int(self.seed) * 7717 + self._phi_measure_count * 104729
            ) % (2 ** 31 - 1)
        self._in_phi_measurement = True
        snapshot = self.clone_state()
        try:
            bank = self.sample_state_bank(n_states, burn_in, bank_seed=bank_seed)
            acc, cnt = {}, 0
            for st in bank:
                self.restore_state(st)
                for (i, j), w in self._sgtp_influence_matrix().items():
                    acc[(i, j)] = acc.get((i, j), 0.0) + float(w)
                cnt += 1
        except Exception as e:
            print(f"[C2b][WARN] không đo được Φ từ công thức liên tục ({e}); "
                  f"giữ bảng tra cũ — corr(Φ,W*) sẽ KHÔNG đáng tin.")
            acc, cnt = {}, 0
        finally:
            # QUAN TRỌNG: restore_state() deep-copy LẠI gt_influence_by_ego từ
            # snapshot, nên phải ghi kết quả đo SAU khi restore — nếu ghi
            # trước thì nó bị xoá sạch và bảng tra cũ quay lại y nguyên (đã
            # dính đúng bẫy này một lần: Φ in ra vẫn đúng bằng PHI_* hằng số).
            self.restore_state(snapshot)
            self._in_phi_measurement = False

        if cnt:
            for (i, j), tot in acc.items():
                self.gt_influence_by_ego[j][i] = tot / cnt

    def _set_phi(self, src, dst, value):
        self.gt_influence_by_ego[dst][src] = float(value)

    # ============================================================
    # P1: delta_ij(s) — cổng kích hoạt phụ thuộc trạng thái, [0,1]
    # ============================================================

    def _compute_deltas(self):
        """
        Trả về dict {(i, j): delta_ij(s)} CHỈ cho các cặp có trong
        self.declared_pairs (những cặp khác coi như delta=0 luôn, vì phi=0).
        Được tính SAU khi vị trí đã cập nhật (state s' của bước này) để nhất
        quán với cách reward được tính trong step().

        P1 flag (enable_conditional_gates): khi OFF, delta_ij(s) collapse về
        1.0 cho MỌI cặp declared -- KHÔNG đánh giá bất kỳ điều kiện trạng
        thái nào (không phải "tính rồi bỏ qua"). Đây là nhánh riêng, không
        đi qua logic gate bên dưới.
        """
        if not self.enable_conditional_gates:
            return {(i, j): 1.0 for (i, j, _gate_name, _z) in self.declared_pairs}

        deltas = {}

        for (i, j, gate_name, z) in self.declared_pairs:
            ra = self.zone_role_agents[z]
            collector = ra[self.ROLE_COLLECTOR]
            gatekeeper = ra[self.ROLE_GATEKEEPER]
            blocker = ra[self.ROLE_BLOCKER]

            if self.enable_latency_ladder:
                d = self._gate_ladder(gate_name, z, ra, collector, gatekeeper, blocker)
            else:
                d = self._gate_flat(gate_name, z, ra, collector, gatekeeper, blocker)

            deltas[(i, j)] = float(d)

        return deltas

    def _gate_ladder(self, gate_name, z, ra, collector, gatekeeper, blocker):
        """
        P2 ON: cổng kích hoạt phân tầng theo 4 vai trò (blocker h=1,
        gatekeeper h=2-3, relay h=4-5, controller h=6+) -- xem Phần 2.2.
        """
        if gate_name == "gate_gk_collector":
            # QUAN TRỌNG: delta phải phụ thuộc vị trí THẬT của gatekeeper,
            # không chỉ trạng thái của collector -- nếu không, can thiệp
            # lên gatekeeper sẽ không đổi được kênh khai báo này chút nào
            # (đã phát hiện qua env_audit: corr(Phi,W*) sai dấu vì bug này).
            return 1.0 if (
                (not self.carrying[z])
                and self._dist(self.positions[collector], self.zone_gate[z]) <= 3
                and self._dist(self.positions[gatekeeper], self.zone_gate[z]) <= 1
            ) else 0.0

        if gate_name == "gate_relay_collector":
            lane_pos = self.zone_lane_a[z]
            relay_agent = ra[self.ROLE_RELAY]
            q = self._lane_queue(lane_pos, radius=2)
            relay_present = self._dist(self.positions[relay_agent], lane_pos) <= 1
            # [FIX-R2] q >= 1 CHƯA ĐỦ: relay dưới scripted policy gần như LUÔN
            # đứng ở lane_a, nên q >= 1 làm gate luôn bật => delta ≡ 1 =>
            # T3 CV(relay) VẪN = 0 (đo thật), chỉ đổi từ "chết vì luôn tắt"
            # sang "chết vì luôn bật", và W*(relay) bão hoà ~+2.0 kéo T5 SNR
            # lên 30.2 (ngoài band [3,20]).
            # Thêm điều kiện carrying[z]: relay chỉ thực sự quan trọng khi
            # collector ĐANG chở hàng qua lane. carrying là biến trạng thái đã
            # được chứng minh có biến thiên tốt (blocker CV = 0.845 sinh ra từ
            # chính nó), nên gate trở lại state-conditional mà vẫn quy được
            # nhân quả cho relay (vẫn đòi relay_present).
            #
            # Ghi chú lịch sử của ngưỡng q:
            # Sau khi _lane_queue lọc chỉ còn {collector, relay} (fix P-1 chống
            # rò ảnh hưởng qua thân gatekeeper), q >= 2 đòi CẢ collector LẪN
            # relay cùng đứng ở lane_a — gần như không bao giờ xảy ra vì
            # collector đi resource<->sink. Hệ quả đo được: kênh relay chết hẳn
            # (T3 CV relay 2.42 -> 0.0000, W*(relay) ~ -0.0075, T4 0.25 -> 0.0).
            # q >= 1 giữ gate phụ thuộc vị trí THẬT của relay (relay tự đếm)
            # cộng lane đang hoạt động -> vẫn state-conditional, vẫn quy được
            # nhân quả cho relay, mà không cần collector có mặt.
            return 1.0 if (
                self.active_lane[z] == "A"
                and q >= 1
                and relay_present
                and self.carrying[z]
            ) else 0.0

        if gate_name == "gate_blocker_collector":
            base_active = 1.0 if (
                self.carrying[z]
                and self._dist(self.positions[blocker], self.positions[collector]) <= 2
            ) else 0.0
            if self.active_lane[z] == "B":
                lane_pos = self.zone_lane_b[z]
                q = self._lane_queue(lane_pos, radius=2)
                blocker_present = self._dist(self.positions[blocker], lane_pos) <= 2
                lane_active = 1.0 if (q >= 2 and blocker_present) else 0.0
                # combine: khi có cả hai điều kiện lẫn lộn thì dùng max
                # (delta trong [0,1], không cộng dồn để tránh vượt 1.0)
                return max(base_active, lane_active)
            return base_active

        if gate_name == "gate_controller_collector":
            return 1.0 if self.low_priority_active[z] else 0.0

        if gate_name == "gate_collector_gatekeeper":
            return 1.0 if self._dist(self.positions[collector], self.zone_resource[z]) <= 1 else 0.0

        return 0.0

    def _gate_flat(self, gate_name, z, ra, collector, gatekeeper, blocker):
        """
        P2 OFF: một cửa sổ trễ mặc định, PHẲNG cho mọi vai trò (treat all
        pairs as if h=1) -- không đánh giá ngưỡng khoảng cách/hàng đợi phân
        tầng của _gate_ladder(). Chỉ còn "src ở sát đích liên quan hay
        không", không phân biệt độ trễ theo vai trò.
        """
        if gate_name == "gate_gk_collector":
            return 1.0 if (
                (not self.carrying[z])
                and self._dist(self.positions[gatekeeper], self.zone_gate[z]) <= 1
            ) else 0.0

        if gate_name == "gate_relay_collector":
            relay_agent = ra[self.ROLE_RELAY]
            lane_pos = self.zone_lane_a[z]
            return 1.0 if self._dist(self.positions[relay_agent], lane_pos) <= 1 else 0.0

        if gate_name == "gate_blocker_collector":
            return 1.0 if (
                self.carrying[z]
                and self._dist(self.positions[blocker], self.positions[collector]) <= 2
            ) else 0.0

        if gate_name == "gate_controller_collector":
            controller = ra[self.ROLE_CONTROLLER]
            return 1.0 if self._dist(self.positions[controller], self.zone_panel[z]) <= 1 else 0.0

        if gate_name == "gate_collector_gatekeeper":
            return 1.0 if self._dist(self.positions[collector], self.zone_resource[z]) <= 1 else 0.0

        return 0.0

    # ============================================================
    # P3 / RC-5: kênh [3] r_emergent — nổi lên, không quy được về cặp
    # ============================================================

    def _apply_emergent_congestion(self, r_emergent):
        """
        Ba nguồn tắc nghẽn, tất cả chỉ phụ thuộc MẬT ĐỘ VỊ TRÍ nên không cặp
        (i, j) khai báo nào "sở hữu" chúng — đúng định nghĩa kênh [3]:

          1. lane over-capacity : bottleneck vật lý tại lane A / lane B
          2. station queue      : tranh chấp resource và sink
          3. zone crowding      : coupling nền trơn 1/(1+d), phạm vi cùng zone

        (3) là bổ sung của RC-3(b) và nó BẮT BUỘC: (1) và (2) đều là ngưỡng
        rời rạc, hai agent không bao giờ đồng vị trí sẽ cho W* == 0 TUYỆT ĐỐI
        ⇒ std|W*|(control) == 0 ⇒ T5 SNR = ∞. Chỉ bơm biên độ (1)(2) không cứu
        được: cặp không chạm nhau vẫn bằng 0 dù biên độ to đến đâu. Cần một
        kênh liên tục theo khoảng cách để noise floor có giá trị hữu hạn.

        Độ phức tạp: O(n_zones * n_agents) cho (1)(2) + O(n_agents^2) cho (3).
        n_agents = 24 ⇒ ~600 phép tính/bước, không đáng kể so với chi phí
        rollout của oracle (H=8 x |A|=6 x n_states). Nếu scale lên hàng nghìn
        agent thì (3) phải đổi sang spatial hash theo zone.
        """
        for z in range(self.n_zones):
            for lane_pos in (self.zone_lane_a[z], self.zone_lane_b[z]):
                occupants = [
                    a for a in range(self.n_agents)
                    if self._dist(self.positions[a], lane_pos) <= self.LANE_CONGESTION_RADIUS
                ]
                if len(occupants) >= self.zone_lane_capacity[z]:   # [ZONE-ASYM]
                    for a in occupants:
                        r_emergent[a] -= self.LANE_CONGESTION_PENALTY

            # Bản cũ chỉ tính hàng đợi ở resource. Sink cũng là điểm nghẽn thật
            # (collector phải dừng đúng ô đó để giao hàng) nên phải tính cả hai.
            for station in (self.zone_resource[z], self.zone_sink[z]):
                waiting = [
                    a for a in range(self.n_agents)
                    if self._dist(self.positions[a], station) <= self.STATION_QUEUE_RADIUS
                ]
                if len(waiting) >= 2:
                    split = self.STATION_QUEUE_PENALTY / len(waiting)
                    for a in waiting:
                        r_emergent[a] -= split

        for a in range(self.n_agents):
            zone_a = self.agent_zone[a]
            crowding = 0.0
            for b in range(self.n_agents):
                if b == a or self.agent_zone[b] != zone_a:
                    continue
                crowding += 1.0 / (1.0 + self._dist(self.positions[a], self.positions[b]))
            r_emergent[a] -= self.ZONE_CROWDING_COEF * crowding

        # P3 / Phần 1.2: chặn biên độ kênh [3] SAU khi cộng đủ ba nguồn.
        for a in range(self.n_agents):
            r_emergent[a] = float(np.clip(
                r_emergent[a], -self.MAX_EMERGENT_MAGNITUDE, self.MAX_EMERGENT_MAGNITUDE
            ))

    def _aggregate_reward_by_role(self, rewards):
        """Mean reward theo vai trò — metric báo cáo thay cho mean toàn quần thể."""
        acc = {role: [] for role in self.ROLE_ORDER}
        for a in range(self.n_agents):
            acc[self.agent_role[a]].append(float(rewards[a]))
        return {role: (float(np.mean(v)) if v else 0.0) for role, v in acc.items()}

    def _lane_queue(self, lane_pos, radius=1):
        # [P-1 FINAL DEBUG] Chỉ đếm các vai trò THỰC SỰ xếp hàng ở lane
        # (collector/relay). Bản cũ đếm MỌI agent trong bán kính, kể cả thân
        # gatekeeper/controller đứng gần đó vì nhiệm vụ riêng => can thiệp lên
        # gatekeeper làm queue tụt dưới ngưỡng q>=2, bật/tắt gate của
        # blocker->collector (phi=-0.5) và relay->collector. Đo phân rã kênh
        # (base vs disengage, cặp gk->collector) cho thấy d_w(gk kênh khai báo)
        # = 0.000 ở MỌI state, toàn bộ W* âm -0.87 rò qua các gate khác =>
        # corr(Phi,W*) sập về 0.625. Lọc theo role cắt đứt đường rò này.
        queue_roles = (self.ROLE_COLLECTOR, self.ROLE_RELAY)
        count = 0
        for a in range(self.n_agents):
            if self.agent_role[a] not in queue_roles:
                continue
            if self._dist(self.positions[a], lane_pos) <= radius:
                count += 1
        return count

    # ============================================================
    # P4: structural shift — dời bottleneck, KHÔNG hoán đổi vai trò
    # ============================================================

    def _maybe_structural_shift(self):
        # P4 flag OFF: bottleneck relocation NEVER triggers -- only
        # behavioural_drift semantics remain reachable. Real branch, not a
        # "compute then discard": we return before even checking phase
        # boundaries, so tier_separation_ratio machinery stays at its
        # NaN-safe default (0/eps = 0, see tier_separation_ratio()).
        if not self.enable_structural_shift:
            self.delta_phi_frobenius_structural_last = 0.0
            self.delta_phi_frobenius_behavioural_last = 0.0
            return

        if self.mode != "structural_shift":
            return

        if self.episode_count > 0 and self.episode_count % self.phase_length == 0:
            self._do_structural_shift()
        else:
            self.delta_phi_frobenius_structural_last = 0.0
            self.delta_phi_frobenius_behavioural_last = 0.0

        """
        Thực thi dời bottleneck (P4). GUARD A (P2<->P4 trap): shift CHỈ
        được phép khi lệnh gọi này thật sự bắt nguồn từ bên trong reset()
        đang chạy -- kiểm tra bằng `self._in_reset`, KHÔNG PHẢI bằng
        `self.t`/`self.done`.

        LỊCH SỬ (xem docs/CIG-AMF_training_debug_master.md mục 5.6, và
        envs/test_p2_p4_guards.py): bản trước dùng
        `assert self.done or self.t == 0`. Điều kiện đó SAI TIỀN ĐỀ: nó
        coi self.t/self.done của EPISODE TRƯỚC (giá trị còn sót lại NGAY
        TRƯỚC khi reset() bắt đầu xử lý) là bằng chứng "đang ở ranh giới
        episode". Nhưng structure_value_tier0.run_tier0() CHỦ ĐỘNG gọi
        env.reset() sớm để tránh vượt max_steps giữa cửa sổ oracle
        (`if env.t + horizon + steps_between >= env.max_steps: env.reset()`)
        -- một reset() hoàn toàn hợp lệ dù self.t=48, self.done=False lúc
        đó -- nên assert cũ crash oan trên chính use case nó phải cho qua.
        Mặt khác test_guard_a_mid_episode_shift_asserts gọi thẳng
        `env._do_structural_shift()` sau `env.step()` (bỏ qua reset() hoàn
        toàn) để mô phỏng lệnh gọi SAI thật sự -- assert PHẢI nổ ở đây.

        Hai use case đó không thể phân biệt được bằng self.t/self.done
        (cùng là "t>0, done=False" tại thời điểm assert chạy). Tín hiệu
        đúng là "lệnh gọi có xuất phát từ bên trong reset() hay không" --
        đó là ý nghĩa thật của self._in_reset (đặt True ở đầu reset(),
        trước _maybe_structural_shift(), tắt lại ngay sau).

        P2<->P4 trap kiểu khác (shift lén xảy ra GIỮA một cửa sổ
        clone_state()/restore_state() ở nơi khác) vẫn được chặn độc lập:
        clone_state()/restore_state() chụp và khôi phục ĐẦY ĐỦ mọi field
        hàm này đụng tới (active_lane, current_phase, gt_influence_by_ego,
        delta_phi_frobenius_*, t, done, episode_count), nên nếu điều đó
        xảy ra thì restore_state() sau đó xoá sạch dấu vết của shift cùng
        phần state còn lại.
        """      
    def _do_structural_shift(self):
        # --- BẮT ĐẦU PATCH - GUARD A ---
        assert getattr(self, '_in_reset', False), \
            "GUARD A (P2<->P4 trap): Cannot apply structural shift mid-episode. Call this only via reset()!"
        # --- KẾT THÚC PATCH ---
        self.current_phase += 1
        prev_phi = copy.deepcopy(self.gt_influence_by_ego)

        for z in range(self.n_zones):
            self.active_lane[z] = "B" if self.active_lane[z] == "A" else "A"

        self._refresh_gt_graph()
        self.delta_phi_frobenius_structural_last = self._delta_phi_frobenius(prev_phi, self.gt_influence_by_ego)
        self.delta_phi_frobenius_behavioural_last = 0.0

        # RC-2: KHÔNG gán behavioural = 0.0 ở đây nữa. Hàm này không hề đo
        # behavioural drift, nên gán 0 là bịa số liệu — và đó chính là một
        # trong ba dòng đã biến T6 thành 1e12 theo cấu tạo.
        # Giá trị đúng chỉ đến từ measure_realized_phi_tiers().

    def _delta_phi_frobenius(self, phi_a, phi_b):
        total = 0.0
        for ego in range(self.n_agents):
            a = phi_a.get(ego, {})
            b = phi_b.get(ego, {})
            keys = set(a.keys()) | set(b.keys())
            for k in keys:
                total += (float(a.get(k, 0.0)) - float(b.get(k, 0.0))) ** 2
        return float(np.sqrt(total))

    def tier_separation_ratio(self):
        """
        RC-2: mẫu số PHẢI là ‖dΦ̃‖ đo được (measure_realized_phi_tiers), không
        phải hằng số 0.0 gán cứng. Bản cũ có ĐÚNG ba dòng gán
        `delta_phi_frobenius_behavioural_last = 0.0` và không dòng nào khác
        trong cả file gán giá trị khác ⇒ hàm này trả 1e12 THEO CẤU TẠO, và
        "T6 = 9.9e11" chưa bao giờ là kết quả đo.
        Khi chưa đo, trả inf để hỏng ồn ào thay vì trả 1e12 giả dạng số liệu.
        """
        num = self.delta_phi_frobenius_structural_last
        den = self.delta_phi_frobenius_behavioural_last
        if den <= 0.0:
            return float("inf")
        return float(num / den)

    # ============================================================
    # RC-2: Φ̃ = E_s[phi * delta] — ma trận ảnh hưởng ĐÃ HIỆN THỰC HOÁ
    # ============================================================

    def set_behaviour_override(self, mode_name):
        """Ép _behaviour_mode(). None = trả về lịch theo pha như bình thường."""
        self._behaviour_override = mode_name

    def realized_phi_matrix(self, state_bank):
        """
        Φ̃_ij = E_s[ phi_ij * delta_ij(s) ] trên state bank cho trước.

        TẠI SAO CẦN: bảng phi tĩnh mù tuyệt đối với hành vi — nó chỉ đổi tại
        ranh giới structural shift, nên ‖dΦ‖_behavioural = 0 là hệ quả của
        ĐỊNH NGHĨA, không phải phát hiện. Nhưng delta_ij(s) thì phụ thuộc vị
        trí THẬT của cả hai agent (xem _gate_ladder), nên behavioural drift
        CÓ làm đổi Φ̃ dù phi bất biến. Tín hiệu T6 cần đã nằm sẵn trong env
        từ đầu, chỉ là chưa ai đo — không cần thêm tầng "realization gate"
        nào cả, L2 chính là delta_ij(s).

        Trả về dict thưa {ego: {src: Φ̃}} — chỉ chứa các cặp declared, đúng
        miền mà _delta_phi_frobenius() cần.
        """
        if not state_bank:
            return {}

        # [C2c] NỐI T6 VÀO KÊNH ĐANG CHẠY.
        # Bằng chứng T6 đã tách khỏi env: sau khi thay TOÀN BỘ kênh ảnh hưởng
        # bằng công thức liên tục, T6 vẫn = 5.9004 và ‖dΦ̃‖ structural/
        # behavioural = 0.638402/0.108197 — GIỐNG TỪNG CHỮ SỐ so với trước.
        # Nguyên nhân: hàm này tính phi(bảng tra) × delta(gate nhị phân), tức
        # đo đúng thứ đã bị thay thế. Khi use_sgtp_phi bật, Φ̃ PHẢI là chính
        # w_ij(s) mà step() cộng vào reward.
        if getattr(self, "use_sgtp_phi", False):
            snapshot = self.clone_state()
            acc = {}
            try:
                for st in state_bank:
                    self.restore_state(st)
                    for (i, j), w in self._sgtp_influence_matrix().items():
                        row = acc.setdefault(j, {})
                        row[i] = row.get(i, 0.0) + float(w)
            finally:
                self.restore_state(snapshot)
            n = float(len(state_bank))
            return {j: {i: v / n for i, v in row.items()} for j, row in acc.items()}

        snapshot = self.clone_state()
        acc = {}
        try:
            for st in state_bank:
                self.restore_state(st)
                deltas = self._compute_deltas()
                for (i, j), d in deltas.items():
                    # phi lấy từ state đang restore: mỗi state mang theo
                    # active_lane + bảng phi của chính nó, nên so sánh
                    # cross-lane (structural) là hợp lệ.
                    phi = self.gt_influence_by_ego[j].get(i, 0.0)
                    row = acc.setdefault(j, {})
                    row[i] = row.get(i, 0.0) + phi * d
        finally:
            self.restore_state(snapshot)

        n = float(len(state_bank))
        return {j: {i: v / n for i, v in row.items()} for j, row in acc.items()}


    """
    Đo CẢ HAI ‖dΦ̃‖_F và ghi vào delta_phi_frobenius_*_last:

          structural  : đổi active_lane (phi đổi + delta đổi)  -> lớn
          behavioural : đổi _behaviour_mode() (phi BẤT BIẾN,
                        chỉ phân bố state đổi)                 -> nhỏ nhưng > 0

        KIỂM SOÁT NHIỄU: cả ba lần lấy mẫu dùng CHUNG bank_seed, chung
        n_states, chung burn_in. Nếu không, chênh lệch đo được chỉ là nhiễu
        lấy mẫu Monte-Carlo chứ không phải tín hiệu tầng.

        Lưu ý thiết kế: state bank KHÔNG thể là một tập cố định dùng lại cho
        cả hai chế độ hành vi — delta_ij(s) là hàm thuần của s, nên trên cùng
        một tập s thì Φ̃ giống hệt nhau và ta lại thu về 0. Thứ mà behavioural
        drift đổi là PHÂN BỐ state được ghé thăm. Vì vậy "cố định" ở đây có
        nghĩa: cố định seed và quy trình lấy mẫu, thả tự do phần state do
        chính hành vi sinh ra.

        Trả về (structural, behavioural).
    """

    def measure_realized_phi_tiers(
        self,
        n_states=32,
        burn_in=3,
        bank_seed=1234,
        behaviour_pair=("cooperative", "selfish"),
    ):
        mode_a, mode_b = behaviour_pair
        snapshot = self.clone_state()
        saved_mode = self.mode
        saved_override = getattr(self, "_behaviour_override", None)
        saved_lane = copy.deepcopy(self.active_lane)
        saved_role_agents = copy.deepcopy(self.zone_role_agents)
        saved_agent_zone = copy.deepcopy(self.agent_zone)

        try:
            self.set_behaviour_override(mode_a)
            phi_ref = self.realized_phi_matrix(
                self.sample_state_bank(n_states, burn_in, bank_seed=bank_seed))

            # --- STRUCTURAL SHIFT: relocate bottleneck only ---
            for z in range(self.n_zones):
                self.active_lane[z] = "B" if self.active_lane[z] == "A" else "A"

            self._refresh_gt_graph()
            
            phi_shifted = self.realized_phi_matrix(
                self.sample_state_bank(n_states, burn_in, bank_seed=bank_seed))
            structural = self._delta_phi_frobenius(phi_ref, phi_shifted)

            # Phục hồi nguyên trạng
            self.active_lane = copy.deepcopy(saved_lane)
            self.zone_role_agents = copy.deepcopy(saved_role_agents)
            self.agent_zone = copy.deepcopy(saved_agent_zone)
            self._refresh_gt_graph()

            # --- BEHAVIOURAL DRIFT ---
            # GIỮ NGUYÊN "selfish" như bạn yêu cầu!
            schedule = ["cooperative", "delayed", "zigzag", "lazy", "selfish"] 
            phis = {mode_a: phi_ref}
            for m in schedule:
                if m not in phis:
                    self.set_behaviour_override(m)
                    phis[m] = self.realized_phi_matrix(
                        self.sample_state_bank(n_states, burn_in, bank_seed=bank_seed))
            steps = [
                self._delta_phi_frobenius(phis[schedule[k]], phis[schedule[k + 1]])
                for k in range(len(schedule) - 1)
            ]
            if steps:
                behavioural = float(np.mean(steps))
            else:
                self.set_behaviour_override(mode_b)
                phi_drifted = self.realized_phi_matrix(
                    self.sample_state_bank(n_states, burn_in, bank_seed=bank_seed))
                behavioural = self._delta_phi_frobenius(phi_ref, phi_drifted)

        finally:
            self.set_behaviour_override(saved_override)
            self.mode = saved_mode
            self.active_lane = copy.deepcopy(saved_lane)
            self.zone_role_agents = copy.deepcopy(saved_role_agents)
            self.agent_zone = copy.deepcopy(saved_agent_zone)
            self._refresh_gt_graph()
            self.restore_state(snapshot)

        self.delta_phi_frobenius_structural_last = float(structural)
        self.delta_phi_frobenius_behavioural_last = float(behavioural)
        return float(structural), float(behavioural)


             # --- behavioural: phi y nguyên, chỉ đổi cách agent hành xử ---
            # [P-4 FINAL DEBUG] Đo bằng TRUNG BÌNH các bước drift KỀ NHAU trong
            # lịch thực tế của mode="behavioral_drift", thay vì một cặp cực đoan
            # (cooperative vs selfish) — cặp cực đoan đo "khoảng cách 4 pha dồn
            # một lần", thứ không agent nào trải nghiệm; drift mà learner thấy
            # là MỘT bước chuyển pha. behaviour_pair giữ lại cho tương thích
            # nhưng chỉ dùng làm fallback khi schedule < 2 mode.
    # ============================================================
    # API
    # ============================================================

    def get_supported_egos(self):
        return list(self.supported_egos)

    def get_gt_core_for_ego(self, ego_id):
        return set(self.gt_core_by_ego[ego_id])

    def get_gt_influence_for_ego(self, ego_id):
        """ Phi (tier A) — hằng số, KHÔNG gate theo state. """
        return dict(self.gt_influence_by_ego[ego_id])

    def get_conditional_influence_for_ego(self, ego_id):
        """
        w_ij(s) cho state HIỆN TẠI — dùng cho T3 (CV theo state).

        [C2] Khi use_sgtp_phi bật, phải trả về w LIÊN TỤC đang thực sự chi
        phối reward, không phải phi*delta cũ — nếu không T3 đo một kênh đã
        không còn tồn tại (đúng lỗi "đo lại giả định của chính mình").
        """
        if self.use_sgtp_phi:
            w_all = self._sgtp_influence_matrix()
            return {
                i: w for (i, j), w in w_all.items() if j == int(ego_id)
            }
        deltas = self._compute_deltas()
        out = {j: 0.0 for j in range(self.n_agents) if j != ego_id}
        for (i, j), d in deltas.items():
            if j == ego_id:
                out[i] = float(self.gt_influence_by_ego[ego_id].get(i, 0.0) * d)
        return out

    def get_obs_of_ego(self, obs_all, ego_id):
        return obs_all[ego_id]

    def get_reward_of_ego(self, rewards, ego_id):
        return float(rewards[ego_id])

    def get_obs_dim(self):
        return len(self._get_obs(0))

    def get_action_dim(self):
        return self.N_ACTIONS

    def clone_state(self):
        return {
            "positions": copy.deepcopy(self.positions),
            "gate_open": copy.deepcopy(self.gate_open),
            "resource_available": copy.deepcopy(self.resource_available),
            "carrying": copy.deepcopy(self.carrying),
            "low_priority_active": copy.deepcopy(self.low_priority_active),
            "active_lane": copy.deepcopy(self.active_lane),
            "last_actions": copy.deepcopy(self.last_actions),
            "t": self.t,
            "done": self.done,
            "episode_count": self.episode_count,
            "episode_deliveries": self.episode_deliveries,
            "current_phase": self.current_phase,
            "rng_state": self.rng.get_state(),
            "gt_core_by_ego": copy.deepcopy(self.gt_core_by_ego),
            "gt_influence_by_ego": copy.deepcopy(self.gt_influence_by_ego),
            "delta_phi_frobenius_structural_last": self.delta_phi_frobenius_structural_last,
            "delta_phi_frobenius_behavioural_last": self.delta_phi_frobenius_behavioural_last,
        }

    def restore_state(self, state):
        self.positions = copy.deepcopy(state["positions"])
        self.gate_open = copy.deepcopy(state["gate_open"])
        self.resource_available = copy.deepcopy(state["resource_available"])
        self.carrying = copy.deepcopy(state["carrying"])
        self.low_priority_active = copy.deepcopy(state["low_priority_active"])
        self.active_lane = copy.deepcopy(state["active_lane"])
        self.last_actions = copy.deepcopy(state["last_actions"])
        self.t = state["t"]
        self.done = state["done"]
        self.episode_count = state["episode_count"]
        # .get(): tương thích ngược với snapshot tạo trước RC-4.
        self.episode_deliveries = state.get("episode_deliveries", 0)
        self.current_phase = state["current_phase"]
        self.rng.set_state(state["rng_state"])
        self.gt_core_by_ego = copy.deepcopy(state["gt_core_by_ego"])
        self.gt_influence_by_ego = copy.deepcopy(state["gt_influence_by_ego"])
        self.delta_phi_frobenius_structural_last = state["delta_phi_frobenius_structural_last"]
        self.delta_phi_frobenius_behavioural_last = state["delta_phi_frobenius_behavioural_last"]

    def _clip(self, x):
        return max(0, min(self.grid_size - 1, x))

    def _clip_pos(self, pos):
        return [self._clip(pos[0]), self._clip(pos[1])]

    def _move(self, pos, action):
        r, c = pos
        if action == self.UP:
            r -= 1
        elif action == self.DOWN:
            r += 1
        elif action == self.LEFT:
            c -= 1
        elif action == self.RIGHT:
            c += 1
        return [self._clip(r), self._clip(c)]

    def _dist(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _frenet_sd(self, agent_id):
        """
        [C1 — CIG-AMF_SGTP_refactor_and_entropy_spec.md phần C1]
        Toạ độ Frenet (s, d) trên chuỗi gate -> resource -> sink của zone hiện
        tại. Hàm THUẦN ĐỌC STATE, không tác dụng phụ, không đổi reward — chỉ
        cấp toạ độ cho Φ (Phần C2, CHƯA làm trong bản này).

        s ∈ [0,1]: tiến độ dọc polyline (0=gate, 1=sink), qua chiếu điểm lên
            đoạn gần nhất trong 2 đoạn, chuẩn hoá arc-length tại điểm chiếu.
        d ∈ R: khoảng cách vuông góc CÓ DẤU. Dương = phía lane_b (cột lớn
            hơn), âm = phía lane_a — khớp quy ước zone_lane_a/zone_lane_b sẵn có.
        """
        z = self.agent_zone[agent_id]
        p = np.asarray(self.positions[agent_id], dtype=np.float64)
        waypoints = [
            np.asarray(self.zone_gate[z], dtype=np.float64),
            np.asarray(self.zone_resource[z], dtype=np.float64),
            np.asarray(self.zone_sink[z], dtype=np.float64),
        ]

        seg_lens = [float(np.linalg.norm(waypoints[k + 1] - waypoints[k]))
                    for k in range(len(waypoints) - 1)]
        total_len = max(sum(seg_lens), 1e-9)
        cum_len = [0.0]
        for L in seg_lens:
            cum_len.append(cum_len[-1] + L)

        best_dist, best_s, best_d = None, 0.0, 0.0
        for k in range(len(waypoints) - 1):
            a, b = waypoints[k], waypoints[k + 1]
            seg_vec = b - a
            seg_len = seg_lens[k]
            if seg_len < 1e-9:
                continue
            t = float(np.clip(np.dot(p - a, seg_vec) / (seg_len ** 2), 0.0, 1.0))
            proj = a + t * seg_vec
            perp = p - proj
            dist = float(np.linalg.norm(perp))
            cross = seg_vec[0] * perp[1] - seg_vec[1] * perp[0]
            signed_d = dist if cross >= 0 else -dist

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_s = (cum_len[k] + t * seg_len) / total_len
                # [UNIT-FIX] d PHẢI chuẩn hoá cùng mẫu số với s (total_len).
                # Bản cũ trả d thô theo Ô LƯỚI trong khi s đã ở [0,1] => mọi
                # ngưỡng lệch ngang (D_SAFE, D_LAT, D_LAT_OBS) và mọi biểu
                # thức 1/(1+dd) đang so hai đại lượng KHÁC ĐƠN VỊ.
                # Hệ quả đo được: quét W_OBSTRUCT qua {0,2,4,6} cho corr
                # GIỐNG HỆT tới 4 chữ số (-0.1627) => số hạng obstruction gần
                # như KHÔNG BAO GIỜ bắn, vì lane lệch ~3 ô mà ngưỡng là 0.25
                # "đơn vị s" (tức 1/4 chiều dài chuỗi). Cùng loại lỗi đơn vị
                # với k_sg/sigma_hi ở phần B.
                best_d = signed_d / total_len

        return float(best_s), float(best_d)

    # ================================================================
    # [C2 + C3] Φ LIÊN TỤC KIỂU SGTP — thay bảng tra role x role
    # ================================================================

    def _sgtp_pair_cost(self, s_i, d_i, s_j, d_j, zone_scale=1.0,
                        is_declared=False):
        """
        Chi phí tương tác của agent i LÊN agent j, viết theo đúng cấu trúc 4
        số hạng của SGTP `_compute_game_block_cost`, nhưng trên toạ độ (s, d)
        của chuỗi xử lý (xem _frenet_sd) thay vì Frenet của đường đua.

        Quy ước dấu: hàm trả về CHI PHÍ cho j. Nơi gọi cộng `-cost` vào
        reward[j], nên mọi số hạng dương ở đây là "i làm hại j".

            Δs = s_i - s_j   > 0  <=>  i đang ở TRƯỚC j trên chuỗi
            Δd = |d_i - d_j|      lệch ngang
            α  = 1/(1 + |Δs|/s_contest)     liên tục, KHÔNG BAO GIỜ = 0

          1. long    : + w_long · α · Δs
                       i ở trước và ở gần thì hại j; i ở sau (Δs<0) thì có
                       lợi cho j => số hạng này TỰ SINH ra cân bằng dấu (T2)
                       mà không cần khai báo cạnh âm/dương nào.
          2. contest : + w_contest · 1[|Δs| < s_contest]
          3. block   : + w_block/(1 + Δd)  NẾU  s_role < Δs < s_contest
                       ĐÂY LÀ ĐIỂM MẤU CHỐT: không có nhãn "blocker" nào cả.
                       Ai đang dẫn trước j trong dải [s_role, s_contest] thì
                       BƯỚC ĐÓ chính là blocker của j. Vai trò là hàm của
                       state, nên ‖dΦ̃‖_behavioural > 0 một cách tự nhiên và
                       T6 hết tautology.
          4. safety  : + w_safety · max(0, d_safe - dmin)^2   (hinge bậc 2)
                       Số hạng DUY NHẤT có ngưỡng cứng, và ngay cả nó cũng
                       được mượt hoá — đúng như SGTP.
        """
        ds = float(s_i) - float(s_j)
        dd = abs(float(d_i) - float(d_j))
        eps = 1e-6

        # ------------------------------------------------------------------
        # [ZS-SPLIT] zone_scale CHỈ nhân vào KÊNH DECLARED (block), KHÔNG nhân
        # vào NỀN (long/contest/safety) và KHÔNG nhân vào s_contest.
        #
        # Bản trước nhân zs vào cả nền lẫn bán kính tương tác => zone có zs
        # thấp bị dìm CẢ HAI phía: tín hiệu declared yếu đi mà nền cũng yếu
        # theo, nên SNR nội-zone của zone yếu không cải thiện, chỉ co lại.
        # Đo thật: zone0/zone3 gần như trơ (gatekeeper 0.51/0.57, blocker
        # −0.13/−0.18) trong khi zone1 mạnh (1.48/−1.23).
        # Giữ nền CHUNG THANG qua mọi zone thì zone_scale mới thực sự là
        # "độ mạnh cấu trúc của zone" chứ không phải "độ to của cả zone".
        # ------------------------------------------------------------------
        zs = float(zone_scale)
        alpha = 1.0 / (1.0 + abs(ds) / (self.S_CONTEST + eps))

        # --- NỀN: thang chung, không phụ thuộc zone ---
        cost = self.W_LONG * alpha * ds

        if abs(ds) < self.S_CONTEST:
            cost += self.W_CONTEST

        # ------------------------------------------------------------------
        # [OBSTRUCTION] TÁCH LÀM HAI SỐ HẠNG KHÁC BẢN CHẤT.
        #
        # Trong SGTP, blocking đòi ego DẪN TRƯỚC (Δs > s_role) vì xe chặn nằm
        # phía trước trên đường đua. Blocker của env này thì ĐUỔI và ĐỨNG ĐÈ
        # lên collector — hình học ngược hẳn. Đo thật lúc reset:
        #     z0..z3  relay ds=+0.000   blocker ds=+0.000   (mọi zone)
        # tức cả hai nằm NGOÀI mọi dải Δs. Không dải nào cứu được, và đó
        # KHÔNG phải lỗi chiếu polyline — Δs≈0 là ĐẶC TRƯNG VẬT LÝ ĐÚNG của
        # quan hệ blocker-collector. Vì vậy KHÔNG đưa lane vào polyline
        # (phương án (a)): nó chỉ đổi trục s, không sửa được điều này.
        #
        #   obstruction   : |Δs| < s_near  AND  Δd < d_lat   -> ĐỒNG VỊ TRÍ
        #   lane-blocking : s_role < Δs < s_contest          -> DẪN TRƯỚC
        #
        # Gộp hai thứ này vào một dải chính là chỗ sai. Relay cũng thuộc
        # nhóm đầu (đứng ở lane, lệch ngang, Δs ~ 0).
        # ------------------------------------------------------------------
        # [SCOPE-DECL] Ba số hạng VAI TRÒ (obstruct/block/support) chỉ áp cho
        # cặp DECLARED. Trước đây chúng không có ràng buộc role nào, nên sau
        # khi nới S_CONTEST 0.25->0.65 và D_LAT_OBS 0.25->0.45 chúng bắn cho
        # MỌI cặp đồng vị trí, kể cả control. Đo được: control mean|W*|
        # 0.16 -> 0.5279 (3.3x) và T1 Gini 0.74 -> 0.4977 — declared và
        # control về cùng bậc. Chính số hạng "sửa lỗi" lại tự tạo ra kênh
        # nhiễu THỨ BA, và đó là lý do corr không hồi khi hạ congestion.
        # Nền vẫn có coupling liên tục qua alpha_ij — đủ và đúng thiết kế.
        if is_declared:
            # ==============================================================
            # [PARTITION] PHÂN HOẠCH TRÊN (ds, dd) — MỘT ngưỡng ngang duy
            # nhất (D_LAT_OBS), không vùng chồng lấn, không vùng chết.
            #
            #   ds >  S_ROLE                      -> DẪN TRƯỚC   : block
            #   |ds| <= S_ROLE  hoặc  ds < -S_ROLE:
            #        dd <= D_LAT_OBS              -> BÁM ĐUÔI    : obstruct (hại)
            #        dd >  D_LAT_OBS              -> NHƯỜNG ĐƯỜNG: support  (lợi)
            #
            # TẠI SAO: điều kiện support cũ chỉ dùng MỘT trục (ds < -S_ROLE),
            # mà "ở phía sau" trong không gian 1-D không phân biệt được
            # NHƯỜNG ĐƯỜNG với BÁM ĐUÔI. Blocker của env này bám sát collector
            # từ phía sau — hành vi GÂY HẠI — nhưng lại ăn trọn kênh support.
            # Đo phân rã (z0, blocker->collector, 12 state):
            #     long -0.402 | contest +0.300 | obstruct +0.559
            #     block 0.000 | support -27.064 | safety 0.000
            # support át 50x => phi(blocker) = +0.399, tức "blocker giúp
            # collector". Mọi W_SUPPORT > 0 đều sẽ sai dấu ở zone mà blocker
            # bám sát nhiều — đây là LỖI MÔ HÌNH HOÁ, không phải lỗi thang.
            #
            # SGTP làm đúng chỗ này: c_block của nó đòi CẢ Δs trong dải LẪN
            # Δd nhỏ — hai trục cùng lúc. Bản cũ đưa hai-trục vào obstruct
            # nhưng để support một-trục; chính bất đối xứng đó là lỗ hổng.
            #
            # DÙNG CHUNG D_LAT_OBS cho cả hai nhánh (không phải hai hằng số
            # riêng): hai ngưỡng độc lập sẽ đẻ ra khe giữa mà một cặp nào đó
            # rơi vào, và lần sau lại mất một ngày đi tìm.
            # ==============================================================
            if ds > self.S_ROLE:
                if ds < self.S_CONTEST:
                    cost += self.W_BLOCK * zs / (1.0 + dd)
            else:
                if dd <= self.D_LAT_OBS:
                    cost += self.W_OBSTRUCT * zs / (1.0 + dd)
                elif ds > -self.S_CONTEST:
                    cost -= self.W_SUPPORT * zs * alpha / (1.0 + dd)

        dmin = float(np.sqrt(ds * ds + dd * dd))
        if dmin < self.D_SAFE:
            viol = self.D_SAFE - dmin
            cost += self.W_SAFETY * viol * viol

        return float(cost)

    def precondition_sites(self, z):
        """
        [ENABLE] DANH SÁCH ĐÓNG các TIỀN ĐIỀU KIỆN boolean mà một agent có
        thể bật/tắt trong zone z, trích TRỰC TIẾP từ step().

        Grep step() cho ra đúng hai chỗ agent ghi vào một cờ boolean:
            gate_open[z]           <- agent đứng tại zone_gate[z]  + OPEN
            low_priority_active[z] <- agent đứng tại zone_panel[z] + OPEN
        (resource_available do respawn/delivery, carrying là state của chính
        collector — cả hai KHÔNG phải thứ agent khác bật được.)

        Trả về list (site_pos, flag_name, has_consumer).

        has_consumer: cờ đó có THỰC SỰ chặn tiến trình task của ai không.
          - gate_open           : CÓ  (điều kiện pickup, dòng ~1805)
          - low_priority_active : KHÔNG. Sau C2 nó chỉ còn được đọc trong
            _gate_ladder (nhánh delta đã chết khi use_sgtp_phi bật), nên
            controller hiện KHÔNG có kênh nhân quả nào trong reward.
            Giữ nguyên sự thật này thay vì bịa kênh cho nó: nếu ta cộng
            enable cho controller trong khi cơ chế không tồn tại, đó chính
            là fit Phi vào W* — thứ kỷ luật chống-tautology cấm.
        """
        return [
            (self.zone_gate[z], "gate_open", True),
            (self.zone_panel[z], "low_priority_active", False),
        ]

    def _task_consumer_of_zone(self, z):
        """
        [ENABLE] Agent mà chuỗi task bị các tiền điều kiện trên CHẶN.

        Đọc từ CƠ CHẾ trong step(): khối pickup/delivery được viết cho
        ra[ROLE_COLLECTOR]. Đây là tra cứu cơ chế (ai thực hiện task), KHÔNG
        phải tra cứu bảng ảnh hưởng khai báo — gt_influence_by_ego vẫn không
        tham gia tính reward.
        """
        return self.zone_role_agents[z][self.ROLE_COLLECTOR]

    def _enable_cost(self, i, j, z, sd):
        """
        [ENABLE] Họ số hạng TIỀN ĐIỀU KIỆN — họ thứ hai của Phi.

        Phi được port từ SGTP, nơi MỌI ảnh hưởng đều là KHÔNG GIAN (xe đua
        chỉ tác động nhau qua vị trí). Resource-flow có HAI lớp cơ chế:

            không gian     : chiếm chỗ / cản đường / nhường đường
                             -> blocker, relay   (obstruct/block/support)
            tiền điều kiện : bật-tắt một cờ mà task của j cần
                             -> gatekeeper, controller   (SỐ HẠNG NÀY)

        Thiếu họ thứ hai là lỗi thiết kế từ C2. Bằng chứng: 8/20 cặp lệch
        dấu và CẢ TÁM đều là gatekeeper<->collector, nhất quán qua 4 zone và
        cả hai chiều — Phi nói "hại" (hình học: bám đuôi), W* nói "lợi"
        (mở cổng cho pickup). Đó là hệ thống, không phải dư lượng.

        Đây VẪN là kênh TRỰC TIẾP theo nghĩa A5: một bước, hai agent, không
        qua agent thứ ba. A5 giữ nguyên; chỉ cần ghi rằng Phi gồm hai họ.

        Viết theo CƠ CHẾ, không theo vai trò: bất kỳ agent nào đứng đủ gần
        một precondition site CÓ CONSUMER đều nhận số hạng này.
        """
        if j != self._task_consumer_of_zone(z):
            return 0.0

        cost = 0.0
        for site, _flag, has_consumer in self.precondition_sites(z):
            if not has_consumer:
                continue
            d_site = self._dist(self.positions[i], site)
            # mượt theo khoảng cách tới site, cùng dạng 1/(1+d) của các họ khác
            cost -= self.W_ENABLE / (1.0 + float(d_site))
        return cost

    def _sgtp_influence_matrix(self):
        """
        w_ij cho MỌI cặp cùng zone (i != j), theo _sgtp_pair_cost.

        Chỉ trong cùng zone: hai agent khác zone không chia sẻ chuỗi xử lý
        nào nên toạ độ s của chúng không so sánh được. Đây cũng đúng miền mà
        env_audit lấy mẫu control pair — và vì α liên tục, mọi cặp control
        giờ có coupling KHÁC 0, giảm dần theo |Δs|. Đó chính là "trường ảnh
        hưởng nền" mà RC-3 cần, và nó KHÔNG phụ thuộc P3 congestion nữa
        (gate S4 trong spec).

        Trả về dict {(i, j): w} với w là ảnh hưởng của i lên reward của j.
        """
        # [SCOPE-DECL] tập cặp declared, dựng một lần
        if not hasattr(self, "_declared_set") or self._declared_set is None:
            self._declared_set = {
                (int(i), int(j)) for (i, j, _g, _z) in self.declared_pairs
            }

        sd = {}
        for a in range(self.n_agents):
            sd[a] = self._frenet_sd(a)

        by_zone = {}
        for a in range(self.n_agents):
            by_zone.setdefault(int(self.agent_zone[a]), []).append(a)

        w = {}
        for _z, members in by_zone.items():
            zs = float(getattr(self, "zone_scale", {}).get(_z, 1.0))
            for j in members:
                s_j, d_j = sd[j]
                for i in members:
                    if i == j:
                        continue
                    s_i, d_i = sd[i]
                    is_dec = (i, j) in self._declared_set
                    cost = self._sgtp_pair_cost(
                        s_i, d_i, s_j, d_j, zone_scale=zs,
                        is_declared=is_dec,
                    )
                    # [ENABLE] hai họ CỘNG vào nhau, không loại trừ:
                    # gatekeeper có thể vừa mở cổng vừa chắn đường, cả hai
                    # đều thật.
                    if is_dec:
                        cost += self._enable_cost(i, j, _z, sd)
                    w[(i, j)] = -self.GAME_COST_WEIGHT * cost
        return w

    def close_coupling_pairs(self):
        """
        [C4] Số cặp đang ở trạng thái "coupling được kích hoạt": cùng zone,
        |Δs| < S_CONTEST và Δd < D_LAT. Dùng để tính CSD (Close-Coupling
        Duration) — metric đo coupling cấu trúc có bật hay không, ĐỘC LẬP
        với reward (mẫu CSD của SGTP, Bảng I).
        """
        sd = {a: self._frenet_sd(a) for a in range(self.n_agents)}
        by_zone = {}
        for a in range(self.n_agents):
            by_zone.setdefault(int(self.agent_zone[a]), []).append(a)

        n_active = 0
        for _z, members in by_zone.items():
            for k, j in enumerate(members):
                for i in members[k + 1:]:
                    ds = abs(sd[i][0] - sd[j][0])
                    dd = abs(sd[i][1] - sd[j][1])
                    if ds < self.S_CONTEST and dd < self.D_LAT:
                        n_active += 1
        return n_active

    def _frenet_s_with_task_offset(self, agent_id, s_geom, carry_bonus=0.15):
        """
        [C1 mở rộng — CHƯA XÁC MINH] self.carrying lưu THEO ZONE, không theo
        agent — hàm này giả định "agent carrying" = collector của đúng zone đó.
        Cần xác nhận lại với role-assignment thật trước khi dùng ở C2. Chưa
        gọi ở đâu trong code.
        """
        z = self.agent_zone[agent_id]
        if self.carrying.get(z, False):
            return float(np.clip(s_geom + carry_bonus, 0.0, 1.0))
        return float(s_geom)

    def _greedy_avoiding(self, agent_id, src, dst):
        """
        Greedy về phía dst, nhưng tránh đi vào ô đang bị agent KHÁC đứng yên
        chiếm giữ (nếu không, một agent "cắm trại" cố định -- vd relay đã
        tới lane_a rồi STAY mãi -- sẽ khoá vĩnh viễn đường đi của agent khác
        cần băng qua đúng ô đó). Nếu hướng ưu tiên bị chặn, thử hướng phụ
        (giảm khoảng cách theo trục còn lại); nếu cả hai bị chặn, đi ngang
        để tự gỡ kẹt thay vì đứng yên vô thời hạn.
        """
        sr, sc = src
        tr, tc = dst

        occupied = set()
        for a in range(self.n_agents):
            if a == agent_id:
                continue
            occupied.add(tuple(self.positions[a]))

        if (sr, sc) == (tr, tc):
            return self.STAY

        candidates = []
        if sr < tr:
            candidates.append(self.DOWN)
        elif sr > tr:
            candidates.append(self.UP)
        if sc < tc:
            candidates.append(self.RIGHT)
        elif sc > tc:
            candidates.append(self.LEFT)

        for act in candidates:
            nxt = tuple(self._move(src, act))
            if nxt not in occupied:
                return act

        # Cả hướng ưu tiên đều bị chặn -- thử đi NGANG (vuông góc với trục bị
        # chặn) trước, để đi vòng qua vật cản thay vì đi lùi (đi lùi dễ tạo
        # dao động vô hạn tiến-lùi khi hành lang một-ô-rộng bị chặn cố định).
        vertical_blocked = self.UP in candidates or self.DOWN in candidates
        horizontal_blocked = self.LEFT in candidates or self.RIGHT in candidates

        fallback_order = []
        if vertical_blocked:
            fallback_order += [self.LEFT, self.RIGHT]
        if horizontal_blocked:
            fallback_order += [self.UP, self.DOWN]
        fallback_order += [self.UP, self.DOWN, self.LEFT, self.RIGHT]

        tried = set(candidates)
        for act in fallback_order:
            if act in tried:
                continue
            tried.add(act)
            nxt = tuple(self._move(src, act))
            if nxt not in occupied:
                return act

        return self.STAY

    # ------------------------------------------------------------------
    # P0(d): common random numbers infra (dùng chung với tiny_oracle_dig)
    # ------------------------------------------------------------------

    def set_noise_buffer(self, buffer):
        self.noise_buffer = buffer
        self._noise_call_counter = {}

    def clear_noise_buffer(self):
        self.noise_buffer = None
        self._noise_call_counter = {}

    def _draw_noise(self, purpose):
        key = (self.t, purpose)
        if self.noise_buffer is not None and key in self.noise_buffer:
            val = self.noise_buffer[key]
            if isinstance(val, (list, tuple, np.ndarray)):
                idx = self._noise_call_counter.get(key, 0)
                self._noise_call_counter[key] = idx + 1
                idx = min(idx, len(val) - 1)
                return float(val[idx])
            return float(val)
        return float(self.rng.rand())

    def _noise_calls_per_step(self, purpose):
        if purpose == "resource_respawn":
            return self.n_zones
        return 1

    def _make_crn_buffer(self, horizon, seed_rng):
        buffer = {}
        for step in range(horizon):
            for purpose in self.NOISE_PURPOSES:
                n_calls = self._noise_calls_per_step(purpose)
                buffer[(step, purpose)] = [float(seed_rng.rand()) for _ in range(n_calls)]
        return buffer

    # ============================================================
    # Reset
    # ============================================================

    def reset(self):
        # self.t / self.done PHẢI được đặt lại TRƯỚC khi gọi
        # _maybe_structural_shift(): guard P2<->P4 (assert bẫy ranh giới
        # episode) coi self.t/self.done là nguồn sự thật cho "đang ở đầu
        # episode mới hay chưa" -- gọi shift trước khi reset hai biến này
        # khiến guard thấy trạng thái episode CŨ (t/done chưa reset) và
        # trip assert dù đây là lỗi thứ tự gọi, không phải lỗi của guard.
        self.t = 0
        self.done = False

        self._in_reset = True
        try:
            self._maybe_structural_shift()
        finally:
            self._in_reset = False

        self.positions = {}
        for z in range(self.n_zones):
            cr, cc = self.zone_centers[z]
            ra = self.zone_role_agents[z]
            self.positions[ra[self.ROLE_COLLECTOR]] = [self._clip(cr), self._clip(cc)]
            self.positions[ra[self.ROLE_GATEKEEPER]] = [self._clip(cr - 1), self._clip(cc + 1)]
            self.positions[ra[self.ROLE_RELAY]] = [self._clip(cr), self._clip(cc - 2)]
            self.positions[ra[self.ROLE_BLOCKER]] = [self._clip(cr), self._clip(cc + 1)]
            self.positions[ra[self.ROLE_CONTROLLER]] = [self._clip(cr + 3), self._clip(cc - 3)]

        for a in range(self.n_agents):
            if self.agent_role[a] == self.ROLE_DRIFTER:
                self.positions[a] = [
                    self.rng.randint(0, self.grid_size),
                    self.rng.randint(0, self.grid_size),
                ]

        self.gate_open = {z: False for z in range(self.n_zones)}
        self.resource_available = {z: True for z in range(self.n_zones)}
        self.carrying = {z: False for z in range(self.n_zones)}
        self.low_priority_active = {z: False for z in range(self.n_zones)}
        self.last_actions = {a: self.STAY for a in range(self.n_agents)}
        self.episode_deliveries = 0
        # self.t/self.done đã được đặt lại ở ĐẦU reset() (trước
        # _maybe_structural_shift()) -- không gán lại ở đây nữa.
        self.episode_count += 1
        return self._get_obs_all()

    # ============================================================
    # Behavioural drift (Phần 3.1) — Phi KHÔNG đổi, chỉ đổi cách thực thi
    # ============================================================

    def _behaviour_mode(self):
        # RC-2: override thắng tuyệt đối — phép đo Φ̃ cần cô lập trục hành vi
        # khỏi trục pha/episode_count.
        if self._behaviour_override is not None:
            return self._behaviour_override
        if self.mode == "behavioral_drift":
            phase = (self.episode_count // self.phase_length) % 5
            return ["cooperative", "delayed", "zigzag", "lazy", "selfish"][phase]
        return "cooperative"

    def scripted_policy(self, agent_id):
        role = self.agent_role[agent_id]
        z = self.agent_zone[agent_id]
        pos = self.positions[agent_id]

        gate = self.zone_gate[z]
        res = self.zone_resource[z]
        sink = self.zone_sink[z]
        lane_a = self.zone_lane_a[z]
        panel = self.zone_panel[z]
        checkpoint = self.zone_lane_b[z] if self.active_lane[z] == "B" else self.zone_checkpoint[z]

        def greedy(src, dst):
            return self._greedy_avoiding(agent_id, src, dst)

        mode = self._behaviour_mode()

        if role == self.ROLE_COLLECTOR:
            target = sink if self.carrying[z] else res
            return greedy(pos, target)

        if role == self.ROLE_GATEKEEPER:
            if tuple(pos) == gate:
                if mode == "cooperative":
                    return self.OPEN
                if mode == "delayed":
                    return self.OPEN if (self.t % 2 == 0) else self.STAY
                if mode == "zigzag":
                    return self.OPEN if (self.t % 3 != 1) else self.STAY
                if mode == "lazy":
                    return self.OPEN if (self.t % 3 != 1) else self.STAY
                # [P-4 FINAL DEBUG] selfish: duty suy giảm mạnh nhưng KHÔNG bỏ hẳn.
                # "Bỏ duty" = realized structure đổi (delta sập 0 hàng loạt) =>
                # behavioural dPhi~ (1.376) nuốt chửng structural (0.360), T6 đảo
                # ngược. Theo paper (Exp 2): drift = "dependencies fixed, only
                # policies move" — đổi CÁCH làm chứ không đổi VIỆC ai ảnh hưởng ai.
                return self.OPEN if (self.t % 2 == 0) else self.STAY  # selfish
            return greedy(pos, gate)

        if role == self.ROLE_RELAY:
            if mode == "zigzag" and self.t % 4 == 0:
                return self.STAY
            if mode == "selfish":
                # [P-4] vẫn hướng về lane nhưng lề mề (đi 1 nghỉ 1), không bỏ vị trí
                return self.STAY if (tuple(pos) == lane_a and self.t % 2 == 0) else greedy(pos, lane_a)
            return greedy(pos, lane_a)

        if role == self.ROLE_BLOCKER:
            if mode == "selfish":
                # [P-4] chậm nhịp quanh duty anchor, không bỏ anchor.
                return self.STAY if tuple(pos) == checkpoint else greedy(pos, checkpoint)
            # RC-4: scripted policy phải KHỚP với reward mới. Bản cũ
            # `greedy(pos, self.positions[collector])` là hành vi tối ưu cho
            # shaping đuổi bắt đã bị xoá; giữ nó lại thì baseline scripted vẫn
            # đuổi trong khi learner thì không, và hai đường số liệu không còn
            # so sánh được với nhau.
            if mode == "lazy":
                return self.STAY if tuple(pos) == checkpoint else greedy(pos, checkpoint)
            return greedy(pos, checkpoint)

        if role == self.ROLE_CONTROLLER:
            if mode == "lazy" or mode == "selfish":
                # [P-4] vẫn trực panel, chỉ kích hoạt thưa hơn — không bỏ duty
                if tuple(pos) == panel:
                    if mode == "lazy":
                        return self.OPEN if (self.t % 2 == 0) else self.STAY
                    return self.OPEN if (self.t % 3 == 0) else self.STAY
                return greedy(pos, panel)
            if tuple(pos) == panel:
                return self.OPEN
            return greedy(pos, panel)

        # drifter
        if agent_id % 3 == 0:
            return self.STAY
        if agent_id % 3 == 1:
            return self.rng.randint(0, self.N_ACTIONS - 1)
        return self.STAY if (self.t % 2 == 0) else self.rng.randint(0, self.N_ACTIONS - 1)

    # ============================================================
    # Step
    # ============================================================

    def step(self, actions, return_obs=True, return_info=True):
        if self.done:
            obs = self._get_obs_all() if return_obs else None
            return obs, [0.0] * self.n_agents, True, {}

        self._noise_call_counter = {}
        rewards = [-0.01 for _ in range(self.n_agents)]
        r_emergent = [0.0 for _ in range(self.n_agents)]
        deliveries_this_step = 0

        # ---- gate/panel activation (based on state BEFORE movement) ----
        for z in range(self.n_zones):
            ra = self.zone_role_agents[z]
            gk = ra[self.ROLE_GATEKEEPER]
            ctrl = ra[self.ROLE_CONTROLLER]

            self.gate_open[z] = (
                tuple(self.positions[gk]) == self.zone_gate[z] and actions[gk] == self.OPEN
            )
            if (
                tuple(self.positions[ctrl]) == self.zone_panel[z]
                and actions[ctrl] == self.OPEN
            ):
                self.low_priority_active[z] = True

        # ---- movement ----
        proposed = {}
        for a in range(self.n_agents):
            if actions[a] == self.OPEN:
                proposed[a] = list(self.positions[a])
            else:
                proposed[a] = self._move(self.positions[a], actions[a])

        occupancy = {}
        for a in range(self.n_agents):
            key = tuple(proposed[a])
            occupancy[key] = occupancy.get(key, 0) + 1

        # Movement resolution (basic grid physics -- two agents cannot end
        # up on the same tile) always applies regardless of enable_congestion.
        # P3 flag only gates the ASSOCIATED REWARD PENALTY (kênh [3]); it does
        # not gate whether collisions block movement, since that is env
        # physics, not the emergent reward channel the flag is about.
        for a in range(self.n_agents):
            if occupancy[tuple(proposed[a])] == 1:
                self.positions[a] = proposed[a]
            elif self.enable_congestion:
                # P3: va chạm -- KHÔNG quy được cho một cặp (i,j) cụ thể -> kênh [3]
                r_emergent[a] -= self.COLLISION_PENALTY

        self.last_actions = {a: int(actions[a]) for a in range(self.n_agents)}

        # ============================================================
        # RC-1 — ĐÃ XOÁ: hai khối reward hand-coded
        # ============================================================
        # (a) "TẦM NHÌN GATEKEEPER": `rewards[collector] -= 1.5` khi
        #     dist(collector, gatekeeper) <= gatekeeper_sight.
        #     KHÔNG phải dead code — chẩn đoán "0% nên vô hại" là SAI. Dưới
        #     scripted policy nó nổ 0% chỉ vì layout ép hai agent luôn cách
        #     nhau đúng 3 ô; dưới random policy nó nổ 7.5% collector-step, và
        #     76% số lần nổ TRÙNG bước mà kênh khai báo gate_gk_collector cũng
        #     nổ. Cạnh "gatekeeper GIÚP collector" (phi = +0.60) do đó có net
        #     reward = 0.60 − 1.5 = −0.90: DẤU NGƯỢC HẲN thiết kế.
        #     Đây là quả mìn tệ nhất trong cả file: learner càng giỏi thì
        #     collector càng tiến sát cổng, tần suất nổ càng tăng, tức ground
        #     truth nhân quả của env TỰ HỎNG DẦN theo tiến độ training.
        #     GIẢI QUYẾT DẤU (không xoá mù): cạnh gatekeeper->collector được
        #     chốt là DƯƠNG. Khối −1.5 mang ngữ nghĩa "giám sát/bắt quả tang",
        #     mâu thuẫn trực tiếp với ngữ nghĩa "mở cổng cho qua" của phi, nên
        #     nó bị loại; phi giữ nguyên +0.60 qua gate_gk_collector.
        #
        # (b) "ÁP LỰC BLOCKER": `rewards[collector] += -2.5` khi
        #     dist(collector, blocker) <= 2 (+ −0.7 khi <= 1).
        #     Đo thật: nổ 52.1% collector-step, tổng −397.2/episode = 4.2x
        #     TOÀN BỘ kênh khai báo cộng lại. Điều kiện của nó
        #     (`carrying and dist <= 2`) TRÙNG KHÍT gate_blocker_collector —
        #     nó là bản sao bẩn của logic đã tồn tại.
        #
        # (c) `rewards[collector] -= 0.05` khi đứng ô cổng lúc gate đóng:
        #     gate_open[z] do HÀNH ĐỘNG của gatekeeper quyết định ⇒ đây cũng
        #     là reward đa-agent trá hình r_solo. Giá trị của cổng đã nằm
        #     trong phi(gatekeeper->collector).
        #
        # BẤT BIẾN được khôi phục: mọi reward phụ thuộc từ 2 agent trở lên
        # PHẢI đi qua w_ij = phi_ij * delta_ij(s) và PHẢI có mặt trong
        # info["w_by_pair"] (trừ kênh [3] r_emergent, vốn không quy được về
        # một cặp và bị chặn biên độ bởi MAX_EMERGENT_MAGNITUDE).

        # ---- pickup / delivery (state after movement) ----
        for z in range(self.n_zones):
            ra = self.zone_role_agents[z]
            collector = ra[self.ROLE_COLLECTOR]
            cpos = tuple(self.positions[collector])

            if (
                (not self.carrying[z])
                and self.gate_open[z]
                and self.resource_available[z]
                and cpos == self.zone_resource[z]
            ):
                self.carrying[z] = True
                self.resource_available[z] = False
                rewards[collector] += 0.3

            if self.carrying[z] and cpos == self.zone_sink[z]:
                self.carrying[z] = False
                rewards[collector] += 0.7
                self.resource_available[z] = True
                # RC-4: delivery là metric nhiệm vụ DUY NHẤT không bị nhiễu bởi
                # phần zero-sum của reward. mean-reward toàn quần thể là metric
                # SAI cho env đối kháng — dùng cái này + reward-theo-vai-trò.
                deliveries_this_step += 1

        # P3 flag: khi OFF, kênh [3] không được ĐÁNH GIÁ chút nào (không phải
        # tính rồi zero-out) -- r_emergent giữ nguyên toàn 0.0 từ khởi tạo ở
        # đầu step(). Đây là nhánh riêng, đúng tinh thần "if/else sạch".
        if self.enable_congestion:
            self._apply_emergent_congestion(r_emergent)

        # ---- r_solo: chỉ phụ thuộc trạng thái/hành động CỦA CHÍNH ego ----
        for z in range(self.n_zones):
            ra = self.zone_role_agents[z]
            collector = ra[self.ROLE_COLLECTOR]
            gatekeeper = ra[self.ROLE_GATEKEEPER]
            relay = ra[self.ROLE_RELAY]
            blocker = ra[self.ROLE_BLOCKER]
            controller = ra[self.ROLE_CONTROLLER]

            target = self.zone_sink[z] if self.carrying[z] else self.zone_resource[z]
            d = self._dist(self.positions[collector], target)
            rewards[collector] += 0.02 / (d + 1)

            if tuple(self.positions[gatekeeper]) == self.zone_gate[z]:
                rewards[gatekeeper] += 0.05

            if tuple(self.positions[relay]) == self.zone_lane_a[z]:
                rewards[relay] += 0.05

            # RC-4: XOÁ `rewards[blocker] += 0.03 / (dist(blocker, collector) + 1)`.
            # Đó là gradient rõ ràng DUY NHẤT của blocker và nó trỏ thẳng vào
            # "đứng sát collector", trong khi collector mất −2.5 vì đúng việc
            # đó — bất đối xứng 83x. Ảnh hưởng cấu trúc trở thành MỤC TIÊU của
            # agent thay vì thuộc tính của môi trường, nên thứ audit đo được
            # là "cái blocker cố tình tạo ra", không phải cấu trúc env.
            # Thay bằng mục tiêu solo thuần vị trí, đối xứng với
            # relay/gatekeeper/controller: reward của blocker giờ KHÔNG còn
            # phụ thuộc vị trí collector chút nào.
            blocker_duty = (
                self.zone_lane_b[z]
                if self.active_lane[z] == "B"
                else self.zone_checkpoint[z]
            )
            if tuple(self.positions[blocker]) == blocker_duty:
                rewards[blocker] += 0.05

            if tuple(self.positions[controller]) == self.zone_panel[z]:
                rewards[controller] += 0.05

        # ---- kênh [2]: ảnh hưởng giữa các agent ----
        deltas = self._compute_deltas()

        if self.use_sgtp_phi:
            # [C2] Φ LIÊN TỤC. Không còn bảng tra role x role, không còn gate
            # nhị phân. w_ij là hàm mượt của (Δs, Δd) trên MỌI cặp cùng zone.
            #
            # gt_influence_by_ego / declared_pairs VẪN GIỮ nhưng từ đây chỉ
            # còn là NHÃN ĐỂ CHẤM ĐIỂM (ground truth cho Core F1 và
            # corr(Φ,W*)) — chúng KHÔNG còn tham gia tính reward. Đây chính
            # là điều làm T6 hết tautology: Φ̃ giờ phụ thuộc phân bố state mà
            # policy tạo ra, nên behavioural drift đổi nó một cách tự nhiên.
            w_by_pair = self._sgtp_influence_matrix()
            for (i, j), w in w_by_pair.items():
                rewards[j] += w
        else:
            # Đường cũ (bảng tra), giữ lại làm ABLATION "no-SGTP-phi".
            w_by_pair = {}
            for (i, j, gate_name, z) in self.declared_pairs:
                phi = self.gt_influence_by_ego[j].get(i, 0.0)
                d = deltas.get((i, j), 0.0)
                w = phi * d
                w_by_pair[(i, j)] = w
                rewards[j] += w

        # ---- resource respawn (stochastic, chỉ purpose có trong NOISE_PURPOSES) ----
        for z in range(self.n_zones):
            if (not self.resource_available[z]) and (not self.carrying[z]):
                # [ZONE-ASYM] chỉ đổi NGƯỠNG, số lần _draw_noise mỗi bước
                # giữ nguyên 1/zone => cấu trúc CRN không đổi.
                if self._draw_noise("resource_respawn") < (
                    0.03 / float(self.zone_respawn_delay[z])
                ):
                    self.resource_available[z] = True

        for a in range(self.n_agents):
            rewards[a] += r_emergent[a]

        self.t += 1
        self.done = (self.t >= self.max_steps)
        self.episode_deliveries += deliveries_this_step

        info = {}
        if return_info:
            info = {
                # RC-4: metric báo cáo ĐÚNG cho env đối kháng. mean(rewards) trộn
                # lẫn hai phía của một cạnh zero-sum nên nó bằng hằng số cộng nhiễu
                # — nhìn vào nó là nhìn vào chỗ không có tín hiệu.
                "reward_by_role": self._aggregate_reward_by_role(rewards),
                "deliveries_step": deliveries_this_step,
                "episode_deliveries": self.episode_deliveries,
                "gt_core_by_ego": copy.deepcopy(self.gt_core_by_ego),
                "gt_influence_by_ego": copy.deepcopy(self.gt_influence_by_ego),
                "delta_by_pair": {f"{i}->{j}": v for (i, j), v in deltas.items()},
                "w_by_pair": {f"{i}->{j}": v for (i, j), v in w_by_pair.items()},
                "r_emergent": list(r_emergent),
            # [C4] số cặp đang coupling — dùng dựng CSD ở tầng runner/audit.
            "n_close_coupling_pairs": self.close_coupling_pairs(),
                "active_lane": dict(self.active_lane),
                "current_phase": self.current_phase,
                "mode": self.mode,
                "tier_separation_ratio": self.tier_separation_ratio(),
                "delta_phi_frobenius_structural": self.delta_phi_frobenius_structural_last,
                "delta_phi_frobenius_behavioural": self.delta_phi_frobenius_behavioural_last,
                "evaluation_scope": "omni_arena_multi_ego",
            }

        obs = self._get_obs_all() if return_obs else None
        return obs, rewards, self.done, info

    # ============================================================
    # Observations
    # ============================================================

    def _get_obs(self, ego):
        own = self.positions[ego]
        z = self.agent_zone[ego]
        role = self.agent_role[ego]

        role_to_id = {r: i for i, r in enumerate(self.ROLE_ORDER)}

        obs = [
            own[0] / self.grid_size,
            own[1] / self.grid_size,
            z / max(1, self.n_zones - 1),
            role_to_id[role] / float(len(self.ROLE_ORDER) - 1),
        ]

        gate = self.zone_gate[z]
        res = self.zone_resource[z]
        sink = self.zone_sink[z]
        lane_a = self.zone_lane_a[z]
        panel = self.zone_panel[z]

        obs.extend([
            (gate[0] - own[0]) / self.grid_size,
            (gate[1] - own[1]) / self.grid_size,
            (res[0] - own[0]) / self.grid_size,
            (res[1] - own[1]) / self.grid_size,
            (sink[0] - own[0]) / self.grid_size,
            (sink[1] - own[1]) / self.grid_size,
            (lane_a[0] - own[0]) / self.grid_size,
            (lane_a[1] - own[1]) / self.grid_size,
            (panel[0] - own[0]) / self.grid_size,
            (panel[1] - own[1]) / self.grid_size,
            float(self.gate_open[z]),
            float(self.resource_available[z]),
            float(self.carrying[z]),
            float(self.low_priority_active[z]),
            1.0 if self.active_lane[z] == "A" else 0.0,
        ])

        neighbors = []
        for j in range(self.n_agents):
            if j == ego:
                continue
            if self._dist(self.positions[ego], self.positions[j]) <= self.obs_radius:
                pj = self.positions[j]
                neighbors.extend([
                    (pj[0] - own[0]) / self.grid_size,
                    (pj[1] - own[1]) / self.grid_size,
                    self.agent_zone[j] / max(1, self.n_zones - 1),
                    role_to_id[self.agent_role[j]] / float(len(self.ROLE_ORDER) - 1),
                    self.last_actions[j] / max(1, self.N_ACTIONS - 1),
                ])

        max_nb = min(8, self.n_agents - 1)
        target_len = max_nb * 5
        while len(neighbors) < target_len:
            neighbors.extend([-1.0, -1.0, -1.0, -1.0, -1.0])

        obs.extend(neighbors[:target_len])
        return np.array(obs, dtype=np.float32)

    def _get_obs_all(self):
        return [self._get_obs(i) for i in range(self.n_agents)]

    # ============================================================
    # Oracle rollout support (đồng bộ interface với tiny_oracle_dig.py)
    # ============================================================

    def rollout_from_current_state(self, forced=None, horizon=None):
        if horizon is None:
            horizon = self.causal_horizon

        # GUARD (P2<->P4 trap, mục 3b của user writeup): một forced-
        # intervention rollout không được phép BẮT ĐẦU ở vị trí mà cửa sổ
        # H-bước của nó có thể chạm/vượt ranh giới episode kế tiếp -- vì
        # ranh giới episode là nơi DUY NHẤT structural shift (P4) được phép
        # trigger (xem _do_structural_shift). Nếu oracle được gọi ở gần
        # cuối episode, người gọi có thể vô tình đo W* xuyên qua hai cấu
        # trúc Phi khác nhau một khi episode kế tiếp reset() và shift.
        # Runtime assert thật, không phải comment "best effort".
        if forced is not None:
            assert self.t + horizon <= self.max_steps, (
                f"oracle forced-intervention rollout window crosses episode "
                f"boundary: t={self.t} + horizon={horizon} > "
                f"max_steps={self.max_steps}. This risks the H-step window "
                f"spanning into a subsequent episode's structural-shift "
                f"boundary (P2<->P4 trap) and silently averaging W* over two "
                f"different Phi structures. Sample/restore a state earlier "
                f"in the episode (t <= max_steps - horizon) before forcing "
                f"an intervention rollout."
            )

        snapshot = self.clone_state()
        gamma = 0.95
        total = np.zeros(self.n_agents, dtype=np.float32)

        done = False
        local_t = 0
        while (not done) and local_t < horizon:
            acts = [self.scripted_policy(i) for i in range(self.n_agents)]
            if forced is not None:
                aid, forced_action, forced_step = forced
                if local_t == forced_step:
                    acts[aid] = forced_action

            _, rewards, done, _ = self.step(acts, return_obs=False, return_info=False)
            total += (gamma ** local_t) * np.array(rewards, dtype=np.float32)
            local_t += 1

        self.restore_state(snapshot)
        return total

    def sample_state_bank(self, n_states=24, burn_in=3, bank_seed=None):
        """
        bank_seed (RC-2): ép RNG về một hạt giống xác định trước khi lấy mẫu,
        rồi TRẢ LẠI trạng thái RNG cũ. Cần thiết để hai lần đo Φ̃ chỉ khác
        nhau đúng một biến (lane hoặc behaviour mode), phần ngẫu nhiên còn
        lại trùng khít — nếu không thì ‖dΦ̃‖ đo được chỉ là nhiễu lấy mẫu.
        """
        saved_rng_state = self.rng.get_state() if bank_seed is not None else None
        if bank_seed is not None:
            self.rng.seed(bank_seed)   # seed tại chỗ, giữ nguyên object

        try:
            bank = []
            self.reset()
            for _ in range(n_states):
                for _ in range(burn_in):
                    acts = [self.scripted_policy(i) for i in range(self.n_agents)]
                    _, _, done, _ = self.step(acts, return_obs=False, return_info=False)
                    if done:
                        self.reset()
                bank.append(self.clone_state())
            return bank
        finally:
            if saved_rng_state is not None:
                self.rng.set_state(saved_rng_state)

    def compute_oracle_influence_all_egos_from_current_state(
        self,
        agent_j,
        intervention_action,
        horizon=None,
        n_trials=1,
        forced_step=0,
        crn_seed=None,
    ):
        """
        [FIX-O1] Ảnh hưởng của MỘT can thiệp lên j, đo trên TẤT CẢ ego cùng lúc.

        rollout_from_current_state() vốn đã trả về VECTOR return theo agent
        (bản cũ chỉ lấy đúng một phần tử `[ego_id]` rồi vứt 23 phần tử còn
        lại, và lặp lại toàn bộ rollout cho từng ego). Giữ nguyên vector đó
        cho ta cả cột W*_{·,j} trong đúng một cặp rollout — KHÔNG xấp xỉ.

        Chi phí một lần refresh core: 1 rollout base + N rollout alt, thay vì
        N x (N-1) x (|A|+1) rollout của bản cũ.

        Trả về: np.ndarray [n_agents], phần tử ego = mean_trials(alt - base).
        Dấu được GIỮ (không abs) — nơi gọi tự quyết định dùng dấu hay độ lớn.
        """
        if horizon is None:
            horizon = self.causal_horizon

        snapshot = self.clone_state()
        acc = np.zeros((self.n_agents,), dtype=np.float64)

        try:
            for trial in range(int(n_trials)):
                crn_rng = np.random.RandomState(
                    ((crn_seed if crn_seed is not None else self.seed) * 9973 + trial)
                    % (2 ** 32)
                )
                buffer = self._make_crn_buffer(horizon, crn_rng)

                self.restore_state(snapshot)
                self.set_noise_buffer(buffer)
                base = np.asarray(
                    self.rollout_from_current_state(forced=None, horizon=horizon),
                    dtype=np.float64,
                )

                self.restore_state(snapshot)
                self.set_noise_buffer(buffer)
                alt = np.asarray(
                    self.rollout_from_current_state(
                        forced=(agent_j, intervention_action, forced_step),
                        horizon=horizon,
                    ),
                    dtype=np.float64,
                )

                acc += (alt - base)
        finally:
            self.clear_noise_buffer()
            self.restore_state(snapshot)

        return acc / float(max(1, int(n_trials)))

    def compute_oracle_influence_from_current_state(
        self,
        ego_id,
        agent_j,
        intervention_action,
        horizon=None,
        n_trials=1,
        forced_step=0,
        candidate_actions=None,
        crn_seed=None,
    ):
        """
        P0 oracle can thiệp — cùng thiết kế đã sửa (4.1a-d) như
        TinyOracleDIG.compute_oracle_influence_from_current_state(), áp dụng
        cho OmniArena (env P1-P4).
        """
        if horizon is None:
            horizon = self.causal_horizon

        if candidate_actions is None:
            candidate_actions = list(range(self.N_ACTIONS))
            if intervention_action not in candidate_actions:
                candidate_actions.append(intervention_action)

        snapshot = self.clone_state()

        per_action_trials = {a: [] for a in candidate_actions}
        base_trials = []

        for trial in range(n_trials):
            crn_rng = np.random.RandomState(
                ((crn_seed if crn_seed is not None else self.seed) * 9973 + trial) % (2**32)
            )
            buffer = self._make_crn_buffer(horizon, crn_rng)

            self.restore_state(snapshot)
            self.set_noise_buffer(buffer)
            base = self.rollout_from_current_state(forced=None, horizon=horizon)
            base_trials.append(float(base[ego_id]))

            for a in candidate_actions:
                self.restore_state(snapshot)
                self.set_noise_buffer(buffer)
                alt = self.rollout_from_current_state(
                    forced=(agent_j, a, forced_step),
                    horizon=horizon,
                )
                per_action_trials[a].append(float(alt[ego_id]))

        self.clear_noise_buffer()
        self.restore_state(snapshot)

        base_mean = float(np.mean(base_trials))
        per_action = {a: float(np.mean(vals) - base_mean) for a, vals in per_action_trials.items()}

        deltas = np.array(list(per_action.values()), dtype=np.float64)
        signed = float(np.mean(deltas))
        rng_ = float(np.max(deltas) - np.min(deltas)) if len(deltas) > 0 else 0.0
        best = float(np.max(deltas)) if len(deltas) > 0 else 0.0
        worst = float(np.min(deltas)) if len(deltas) > 0 else 0.0

        return OracleInfluenceProfile(
            signed=signed,
            range=rng_,
            best=best,
            worst=worst,
            per_action=per_action,
            base_return=base_mean,
        )

    def assert_behavioural_phi_invariance(self, n_phases=5):
        """
        Test tự động P4/3.1: chạy behavioural_drift qua đủ các pha, Phi tại
        episode 0 phải bằng Phi ở episode cuối, element-wise.
        """
        assert self.mode == "behavioral_drift", "chỉ hợp lệ trong behavioral_drift"
        phi0 = copy.deepcopy(self.gt_influence_by_ego)
        for _ in range(n_phases * self.phase_length):
            self.reset()
            done = False
            while not done:
                acts = [self.scripted_policy(i) for i in range(self.n_agents)]
                _, _, done, _ = self.step(acts, return_obs=False, return_info=False)
        phi1 = self.gt_influence_by_ego
        d = self._delta_phi_frobenius(phi0, phi1)
        assert d == 0.0, f"Phi bị đổi trong behavioural_drift! ||dPhi||_F = {d}"
        return True
