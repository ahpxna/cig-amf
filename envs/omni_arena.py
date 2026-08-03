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
    MAX_EMERGENT_MAGNITUDE = 0.18 * MIN_CORE_PHI

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
    LANE_CONGESTION_RADIUS = 2
    LANE_CAPACITY = 2                 # `>= 2` thay vì `> 2`
    LANE_CONGESTION_PENALTY = 0.010   # was 0.1 — sẽ clip ngay lập tức nếu giữ
    STATION_QUEUE_RADIUS = 2
    STATION_QUEUE_PENALTY = 0.020
    COLLISION_PENALTY = 0.015
    # Coupling nền TRƠN theo khoảng cách, phạm vi cùng-zone. Đây là thứ RC-3(b)
    # bắt buộc phải có: đo thật cho thấy các cặp control cùng zone KHÔNG chứa
    # blocker (relay->controller, relay->gatekeeper, ...) trả W* = 0 tuyệt đối
    # vì giữa chúng không tồn tại BẤT KỲ dòng code tương tác nào. Sampling
    # cùng zone là cần nhưng KHÔNG đủ — phải có kênh vật lý nối chúng.
    # [P-3 FINAL DEBUG] 0.012 -> 0.020: T5 SNR đo được 33.38 hơi vượt band
    # chấp nhận [3, 20] vì std|W*|(control) = 0.0197 hơi nhỏ. Tăng ~1.67x kênh
    # nền trơn kéo std control lên ~0.033 => SNR ~ 20. Nếu sau khi chạy lại
    # env_audit mà T1 Gini < 0.30 hoặc r_emergent clip thường xuyên, hạ về 0.016.
    ZONE_CROWDING_COEF = 0.020

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

        self._init_zone_layout()
        self._init_population_roles()
        self.reset()

    # ============================================================
    # Layout
    # ============================================================

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

                self.zone_gate[zone] = (self._clip(cr - 2), self._clip(cc_))
                self.zone_resource[zone] = (self._clip(cr + 1), self._clip(cc_))
                self.zone_sink[zone] = (self._clip(cr + 3), self._clip(cc_))
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
            return 1.0 if (self.active_lane[z] == "A" and q >= 2 and relay_present) else 0.0

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
                if len(occupants) >= self.LANE_CAPACITY:
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
        self.current_phase += 1
        prev_phi = copy.deepcopy(self.gt_influence_by_ego)

        for z in range(self.n_zones):
            self.active_lane[z] = "B" if self.active_lane[z] == "A" else "A"
            
            ra = self.zone_role_agents[z]
            ob, orl = ra[self.ROLE_BLOCKER], ra[self.ROLE_RELAY]
            ogk, oc = ra[self.ROLE_GATEKEEPER], ra[self.ROLE_CONTROLLER]
            
            # Đảo chéo 4 vai trò
            ra[self.ROLE_BLOCKER] = orl
            ra[self.ROLE_RELAY] = ob
            ra[self.ROLE_GATEKEEPER] = oc
            ra[self.ROLE_CONTROLLER] = ogk
            
            self.agent_role[ob] = self.ROLE_RELAY
            self.agent_role[orl] = self.ROLE_BLOCKER
            self.agent_role[ogk] = self.ROLE_CONTROLLER
            self.agent_role[oc] = self.ROLE_GATEKEEPER

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

            # --- STRUCTURAL SHIFT (Đảo 4 vai trò) ---
            for z in range(self.n_zones):
                self.active_lane[z] = "B" if self.active_lane[z] == "A" else "A"
                ra = self.zone_role_agents[z]
                ob, orl = ra[self.ROLE_BLOCKER], ra[self.ROLE_RELAY]
                ogk, oc = ra[self.ROLE_GATEKEEPER], ra[self.ROLE_CONTROLLER]
                
                ra[self.ROLE_BLOCKER] = orl
                ra[self.ROLE_RELAY] = ob
                ra[self.ROLE_GATEKEEPER] = oc
                ra[self.ROLE_CONTROLLER] = ogk
                
                self.agent_role[ob] = self.ROLE_RELAY
                self.agent_role[orl] = self.ROLE_BLOCKER
                self.agent_role[ogk] = self.ROLE_CONTROLLER
                self.agent_role[oc] = self.ROLE_GATEKEEPER
                
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
        w_ij(s) = phi_ij * delta_ij(s) cho state HIỆN TẠI — dùng cho T3.
        """
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
        checkpoint = self.zone_checkpoint[z]

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
                    return self.OPEN if (self.t % 3 != 1) else self.LEFT
                if mode == "lazy":
                    return self.OPEN if (self.t % 3 == 0) else self.STAY
                # [P-4 FINAL DEBUG] selfish: duty suy giảm mạnh nhưng KHÔNG bỏ hẳn.
                # "Bỏ duty" = realized structure đổi (delta sập 0 hàng loạt) =>
                # behavioural dPhi~ (1.376) nuốt chửng structural (0.360), T6 đảo
                # ngược. Theo paper (Exp 2): drift = "dependencies fixed, only
                # policies move" — đổi CÁCH làm chứ không đổi VIỆC ai ảnh hưởng ai.
                return self.OPEN if (self.t % 4 == 0) else self.STAY  # selfish
            return greedy(pos, gate)

        if role == self.ROLE_RELAY:
            if mode == "zigzag" and self.t % 4 == 0:
                return self.LEFT
            if mode == "selfish":
                # [P-4] vẫn hướng về lane nhưng lề mề (đi 1 nghỉ 1), không bỏ vị trí
                return self.STAY if (self.t % 2 == 0) else greedy(pos, lane_a)
            return greedy(pos, lane_a)

        if role == self.ROLE_BLOCKER:
            if mode == "selfish":
                # [P-4] nhiễu quanh checkpoint thay vì random toàn cục (bỏ vị trí)
                if self.rng.rand() < 0.5:
                    return self.rng.randint(0, self.N_ACTIONS - 1)
                return greedy(pos, checkpoint)
            # RC-4: scripted policy phải KHỚP với reward mới. Bản cũ
            # `greedy(pos, self.positions[collector])` là hành vi tối ưu cho
            # shaping đuổi bắt đã bị xoá; giữ nó lại thì baseline scripted vẫn
            # đuổi trong khi learner thì không, và hai đường số liệu không còn
            # so sánh được với nhau.
            if mode == "lazy":
                return self.STAY
            return greedy(pos, checkpoint)

        if role == self.ROLE_CONTROLLER:
            if mode == "lazy" or mode == "selfish":
                # [P-4] vẫn trực panel, chỉ kích hoạt thưa hơn — không bỏ duty
                if tuple(pos) == panel:
                    return self.OPEN if (self.t % 4 == 0) else self.STAY
                return self.STAY if (self.t % 2 == 0) else greedy(pos, panel)
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

    def step(self, actions):
        if self.done:
            return self._get_obs_all(), [0.0] * self.n_agents, True, {}

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
            if tuple(self.positions[blocker]) == self.zone_checkpoint[z]:
                rewards[blocker] += 0.05

            if tuple(self.positions[controller]) == self.zone_panel[z]:
                rewards[controller] += 0.05

        # ---- P1/P3: kênh [2] khai báo -- w_ij(s) = phi_ij * delta_ij(s) ----
        deltas = self._compute_deltas()
        w_by_pair = {}
        for (i, j, gate_name, z) in self.declared_pairs:
            phi = self.gt_influence_by_ego[j].get(i, 0.0)
            d = deltas.get((i, j), 0.0)
            w = phi * d  # psi(a_j) = 1 (xem báo cáo triển khai — không mã hoá
                         # thêm hàm hành động riêng, delta(s) đã bao hàm tính
                         # điều kiện trạng thái).
            w_by_pair[(i, j)] = w
            rewards[j] += w

        # ---- resource respawn (stochastic, chỉ purpose có trong NOISE_PURPOSES) ----
        for z in range(self.n_zones):
            if (not self.resource_available[z]) and (not self.carrying[z]):
                if self._draw_noise("resource_respawn") < 0.03:
                    self.resource_available[z] = True

        for a in range(self.n_agents):
            rewards[a] += r_emergent[a]

        self.t += 1
        self.done = (self.t >= self.max_steps)
        self.episode_deliveries += deliveries_this_step

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
            "active_lane": dict(self.active_lane),
            "current_phase": self.current_phase,
            "mode": self.mode,
            "tier_separation_ratio": self.tier_separation_ratio(),
            "delta_phi_frobenius_structural": self.delta_phi_frobenius_structural_last,
            "delta_phi_frobenius_behavioural": self.delta_phi_frobenius_behavioural_last,
            "evaluation_scope": "omni_arena_multi_ego",
        }

        return self._get_obs_all(), rewards, self.done, info

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

            _, rewards, done, _ = self.step(acts)
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
                    _, _, done, _ = self.step(acts)
                    if done:
                        self.reset()
                bank.append(self.clone_state())
            return bank
        finally:
            if saved_rng_state is not None:
                self.rng.set_state(saved_rng_state)

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
                _, _, done, _ = self.step(acts)
        phi1 = self.gt_influence_by_ego
        d = self._delta_phi_frobenius(phi0, phi1)
        assert d == 0.0, f"Phi bị đổi trong behavioural_drift! ||dPhi||_F = {d}"
        return True
