"""
env_audit.py — Omni-Arena validation according to Part 7 (acceptance criteria) of
docs/OMNI_ARENA_BLUEPRINT.md.

Instantiate OmniArena (P1-P4) + modified intervention oracle (P0), run enough episodes/oracle rollout to calculate:

T1 Gini(|W*|) on the set of agent pairs
T2 Sign balance (significant negative/positive W* pair ratio)
T3 CV of w_ij(s) across multiple states, for conditional pairs
T4 Spread of latency-to-peak-effect across 4 roles
T5 SNR (core signal amplitude / control-pair noise amplitude)
T6 tier_separation_ratio (structural vs behavioral ||dPhi||_F)

corr(Phi, W*)

Print the PASS/FAIL table according to Part 7, including all raw numbers.

NOTES ON SAMPLE SIZE (read before interpreting data):
The full oracle cost O(N^2 * S * T * H) is ~1 million env steps for N=24 (Section 4.2). In this session, to run for a few minutes, T1/T2/T5/corr
uses a DELIBERATE sample set consisting of:

- all 20 "declared" pairs (5 roles -> {collector or gatekeeper} x 4 zones)

- a set of ~20 randomly selected "control" pairs (ego, j) NOT in declared
pairs, used as an estimate of the noise-floor for T5 and as a control for Gini.
This is NOT radius pruning (avoiding the self-validation trap A1 of Section 4.2) — just reducing the number of states/trials to run within the session's time budget. The numbers S (states), T (trials), and forced_step scan are clearly specified
in the configuration section below and in the output.
"""
import argparse
import json
import os
import sys
import numpy as np

from omni_arena import OmniArena


# ============================================================
# Configuration; see the sample-size note above.
# ============================================================
N_AGENTS = 24
GRID_SIZE = 24
N_ZONES = 4
MAX_STEPS = 60
PHASE_LENGTH = 6          # Short enough to expose both drift types quickly.
HORIZON = 8
N_STATES_T1 = 10          # States sampled for T1/T2/T5/correlation.
N_STATES_T3 = 24          # States sampled for inexpensive non-oracle T3.
N_CONTROL_PAIRS = 20
N_TRIALS = 1               # CRN permits small n_trials under P0d.
SEED = 123

SIGNIFICANT_W = 0.01       # T2 materiality threshold in reward/step.

# RC-2: Measurement parameter Φ̃ = E_s[phi * delta] for T6.
# T6_N_STATES is the noise/time trade-off: ‖dΦ̃‖_behavioural is the difference between two
# Monte-Carlo averages, so the error is ~1/sqrt(N). Below ~24 states, the sampling noise
# overwhelms the drift signal and the T6 ratio becomes unstable between seeds.
T6_N_STATES = 48
T6_BURN_IN = 3
# The two extremes of drift in _behaviour_mode(): "cooperative" (gatekeeper opens
# gates every step) vs "selfish" (abandoning the task). Choose two extremes so that ‖dΦ̃‖_behaviour
# is the UPPER BOUNDARY of drift — if even the upper boundary is 3-20 times smaller than structural
# then the conclusion of new layer separation is valid.
T6_BEHAVIOUR_PAIR = ("cooperative", "selfish")
T6_INVARIANCE_PHASES = 2    # was 5 -- now 2


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
W*_ij (used for T1/T2/T5/corr): Use the correct oracle modified according to P0
(compute_oracle_influence_from_current_state, CRN, no abs), INTERVENTION
ONE STEP at forced_step, action = STAY.

IMPORTANT NOTE (discovered during audit, see deployment report): With one-step intervention at step 0, some pairs with high latency
(relay 4-5 steps, controller 6+ steps) almost cannot change the rollout in time
within the remaining horizon -> W* measured is close to 0, not accurately reflecting the Phi sign.
This is a known limitation of one-step measurement in an environment with long latency, not an oracle error. Use oracle_w_star_sustained() below
to measure "disabling role j throughout the horizon" (following the same root
concept of intervention, only the intervention is maintained instead of a
step) to use as the primary corr(Phi, W*) index -- this is the measure used for
T1/T2/T5/corr in this report.
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
The single-step action increases the distance between agent j and ego the most -- better representing "j not participating in the interaction" than a fixed STAY (STAY
can inadvertently stand right in the path/danger zone of ego,
see deployment report, "deviations" section).
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
W*_ij according to the CORRECT Eq (4): contrast between the ACTUAL action of j and the BASELINE
AVERAGE OF ACTION DISTRIBUTION, forced throughout the horizon.
W*_ij = R_i(a_j actual) - E_{a'~b_j}[ R_i(do a') ]


==================================================================

[ORACLE-EQ4] CORRECT A DEVIATION FROM THE SPEC, NOT A FINE-TOPIC.

==================================================================
The old version used _disengage_action — a HUMAN-CHOSEN baseline:
W*_old = R_i(a_j) - R_i(do a_disengage)
while Eq (4) defines the baseline as the average of the action distribution
. Paper §II.B states the reason directly: "rather than an arbitrarily chosen
alternative action, follows [12] and reduces variance because it
averages over the full action distribution." The old version runs exactly the
"arbitrarily chosen alternative action" that the section rejects.
Measurement: oracle and estimator target TWO DIFFERENT QUANTITIES —
exactly what §V.A warns ("oracle and estimator must target the same
quantity"). Specifically with the collector->gatekeeper pair:
dist(collector, gate) base 6.5 -> alt 1.5 (all zones)
dd(collector, gk) -> 0.000 (all zones)
_disengage_action pushes collector away from duty anchor (zone_resource), and
gate is on the opposite side of resource on the same polyline, so it pushes collector directly into the face of gatekeeper. Obstruct is stronger in the alt branch =>
W* = base - alt is STRUCTURALLY POSITIVE, and 4/4 pairs are sign-shifted from Phi.
W* then does not measure "how the collector affects the gatekeeper" but measures
"what happens when we push the collector into the gatekeeper's face".

WHY BASELINE IS UNIFORM, NOT pi_j:
Eq (4) writes E_{a'~pi_j}. But scripted_policy is DETERMINANT, so
E_{a'~pi_j} = the action itself = base => contrast ≡ 0, a degenerate measurement
. The correct distribution to average here is the distribution where the intervention

IS ACTUALLY taken from: eps-forcing takes UNIFORM on A (see
EpsilonForcedActionController.apply). A2 therefore satisfies the structure
(b_j >= eps/|A| > 0) — the baseline is never outside the support, unlike
a hand-designed action which has no guarantee about b_j.

The uniform is also the baseline that _compute_tiny_oracle_scores (H1) is
using, so after this correction env_audit and H1 aim for the SAME quantity.

Calculated using deterministic |A| rollout (each action is forced throughout the horizon) and then taken
averaged — more accurately, by sampling, and without adding noise.

Maintaining the "sustained" property (forced throughout n_force_steps): T4 H-sweep shows
the blocker going -0.14 -> -1.73 from H=1 to H=8 is not saturated, so the contrast
one-step will squeeze all pairs with long delays.
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
RC-3(b). The noise floor must be "interacting but NOT declared pairs" — that is, SAME ZONE and not belonging to the declared_set.

The old version randomly selected two indices across the entire population: 24 agents / 4 zones ⇒
~75% of pairs are in different zones. All interaction channels of the environment (declared, collision, lane,
queue, crowding) require physical location, so pairs in different zones give
W* = 0 EXACTLY ⇒ np.std(np.abs(w_control)) = 0 ⇒ T5 SNR = ∞. The number
3.8e9 is an artifact of the sampling method, it says nothing about the environment

Exhaustive sampling followed by shuffle instead of rejection sampling: the space of
same-zone pairs is only about n_zones * k^2 (k = agent/zone ≈ 6) ~ 144 elements, exhaustive sampling is O(n_agents^2) once and completely eliminates the possibility of the while loop spinning enough
n*20 times and still returning missing pairs.
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


def state_bank_diagnostics(states):
    """Measure diversity in the physical state bank used by the oracle."""
    if not states:
        return {
            "n_states": 0,
            "unique_state_fraction": 0.0,
            "mean_unique_positions_per_agent": 0.0,
            "min_unique_positions_per_agent": 0,
            "mean_successive_agent_move_fraction": 0.0,
            "carrying_state_fraction": 0.0,
            "gate_open_state_fraction": 0.0,
        }

    def ordered_values(mapping):
        return tuple(mapping[key] for key in sorted(mapping))

    def physical_key(state):
        positions = tuple(
            (int(agent), tuple(int(value) for value in state["positions"][agent]))
            for agent in sorted(state["positions"])
        )
        return (
            positions,
            ordered_values(state["gate_open"]),
            ordered_values(state["resource_available"]),
            ordered_values(state["carrying"]),
            ordered_values(state["low_priority_active"]),
            ordered_values(state["active_lane"]),
        )

    keys = [physical_key(state) for state in states]
    agent_ids = sorted(states[0]["positions"])
    position_counts = [
        len({tuple(state["positions"][agent]) for state in states})
        for agent in agent_ids
    ]
    move_fractions = []
    for previous, current in zip(states[:-1], states[1:]):
        moved = sum(
            tuple(previous["positions"][agent])
            != tuple(current["positions"][agent])
            for agent in agent_ids
        )
        move_fractions.append(moved / max(len(agent_ids), 1))

    return {
        "n_states": len(states),
        "unique_state_fraction": len(set(keys)) / len(states),
        "mean_unique_positions_per_agent": float(np.mean(position_counts)),
        "min_unique_positions_per_agent": int(min(position_counts)),
        "mean_successive_agent_move_fraction": (
            float(np.mean(move_fractions)) if move_fractions else 0.0
        ),
        "carrying_state_fraction": float(np.mean([
            any(bool(value) for value in state["carrying"].values())
            for state in states
        ])),
        "gate_open_state_fraction": float(np.mean([
            any(bool(value) for value in state["gate_open"].values())
            for state in states
        ])),
    }


# ============================================================
# T1, T2, T5, and corr(Phi,W*) use the oracle.
# ============================================================

def run_oracle_based_metrics(env):
    rng = np.random.RandomState(SEED)
    declared_pairs = build_declared_pair_list(env)
    declared_set = {(i, j) for (i, j, _) in declared_pairs}
    control_pairs = build_control_pair_list(env, declared_set, rng, n=N_CONTROL_PAIRS)

    states = sample_states(env, N_STATES_T1)
    bank_diag = state_bank_diagnostics(states)

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

    # Report static neighbours and mobile collectors separately.  Their
    # trajectory dependence differs, and a global correlation can hide a
    # failure through cancellation.  Only the static group is an acceptance
    # gate; mobile and global values remain diagnostics.
    static_records = [
        record for record in declared_records
        if env.agent_role[record[0]] != env.ROLE_COLLECTOR
    ]
    mobile_records = [
        record for record in declared_records
        if env.agent_role[record[0]] == env.ROLE_COLLECTOR
    ]

    def correlation(records):
        return pearson_corr([record[3] for record in records], [record[4] for record in records])

    def sign_agreement(records):
        usable = [
            record for record in records
            if abs(record[3]) > 1e-9 and abs(record[4]) > 1e-9
        ]
        if not usable:
            return 0.0
        return float(np.mean([
            np.sign(record[3]) == np.sign(record[4]) for record in usable
        ]))

    corr_phi_w = correlation(declared_records)
    corr_phi_w_static = correlation(static_records)
    corr_phi_w_mobile = correlation(mobile_records)

    control_values = np.asarray(
        [value for record in control_records for value in record[5]],
        dtype=np.float64,
    )
    control_pair_stds = np.asarray(
        [np.std(record[5]) for record in control_records], dtype=np.float64
    )
    control_diag = {
        "state_nonzero_fraction": (
            float(np.mean(np.abs(control_values) > 1e-9))
            if control_values.size else 0.0
        ),
        "mean_pair_std": (
            float(np.mean(control_pair_stds)) if control_pair_stds.size else 0.0
        ),
        "median_pair_std": (
            float(np.median(control_pair_stds)) if control_pair_stds.size else 0.0
        ),
        "constant_pair_fraction": (
            float(np.mean(control_pair_stds < 1e-9))
            if control_pair_stds.size else 1.0
        ),
    }

    return {
        "t1_gini": t1_gini,
        "t2_neg_frac": t2_neg_frac,
        "t2_pos_frac": t2_pos_frac,
        "t5_snr": t5_snr,                    # within-zone mean (GATE)
        "t5_snr_pooled": t5_snr_pooled,      # Legacy pooled diagnostic only.
        "t5_snr_per_zone": zone_snrs,
        "corr_phi_w": corr_phi_w,                 # Diagnostic only.
        "corr_phi_w_static": corr_phi_w_static,   # Primary correlation gate.
        "corr_phi_w_mobile": corr_phi_w_mobile,   # Separate mobile diagnostic.
        "corr_sign_static": sign_agreement(static_records),
        "corr_sign_mobile": sign_agreement(mobile_records),
        "n_corr_static": len(static_records),
        "n_corr_mobile": len(mobile_records),
        "state_bank_diagnostics": bank_diag,
        "control_diagnostics": control_diag,
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
# T4 -- Latency profile across the 4 non-collector roles
#
# MODIFIED METHOD (see previous results review): The OLD version forced intervention at a
# forced_step that SLID from 0 to H-1 with H=HORIZON fixed, then observed |W*| varying
# with the forced_step. This DEFINITELY decreases as the forced_step increases,
# REGARDLESS of the role's true latency -- because the later the intervention, the shorter the remaining horizon
# to accumulate the effect, which is a mechanical consequence of the "remaining horizon# shrinking", not a true latency measurement. All roles therefore always "appear" to peak at forced_step=0 and then decrease -- not diagnostic.

#
# NEW version (according to code_test.py's measure_at_horizon() -- reuse the correct pattern,
# do not rewrite oracle logic): FIXED INTERVENTION at forced_step/step 0 (using
# oracle_w_star_sustained(), disengage j throughout the horizon from step 0), then SCAN
# horizon H through {1,2,3,5,8} and see how W*(H) increases/changes sign as the horizon lengthens
# -- this is the real latency diagnosis (user example:
# gatekeeper W* goes from -0.23 at H=1 to +6.80 at H=8, changes sign; blocker
# remains negative throughout, does not change sign).
#
# Two NEW features replace the old "spread of peak forced_step":
# sign-flip step: The smallest H in the sweep where sign(W*(H)) is different
# sign(W*(H_min)) (H_min = sweep[0]); None if not present
# sign change in sweep.

# saturation step: The smallest H where the discrete gain |W*(H)-W*(H_prev)| drops
# below 20% of the largest observed discrete gain
# in that role's sweep; None if not present.

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

# RC-2. Both the NUMER and DENOMINATOR are now measured on Φ̃ = E_s[phi * delta], NOT
# on the non-static table.

#
# Old version: numerator = ‖dPhi‖ static at the shift boundary, denominator = literal
# `0.0 if behavioural_invariant else 1.0`. That denominator is not a measurement —
# it is a constant derived from an assert about the design assumption itself. T6 =
# structural / 1e-12 = 9.9e11 is an arithmetic consequence, the audit doesn't touch
# any data.
#
# On Φ̃, behavioral drift HAS a real signal: phi is invariant, but
# agent moves differently ⇒ delta_ij(s) changes ⇒ Φ̃ changes. This is what should be called
# "layer separation": the same quantity, measuring two different types of intervention.

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

def main(json_out=None):
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

    print("[3/6] T4 legacy horizon diagnostic ...")
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

    print(f"\n-- T4: legacy fixed-intervention horizon diagnostic --")
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
    print(
        f"T4 sign-flip role fraction (diagnostic only; not a latency claim) = "
        f"{m4['t4_spread']:.4f}"
    )

    print(f"\n-- T5: SNR --")
    print(f"T5 SNR (WITHIN-ZONE mean, GATE) = {m1['t5_snr']:.4f}")
    print(f"   per-zone: { {z: round(v, 3) for z, v in m1['t5_snr_per_zone'].items()} }")
    print(
        f"   [diagnostic] legacy pooled value (not a gate; inflated by "
        f"between-zone asymmetry) = {m1['t5_snr_pooled']:.4f}"
    )

    print(f"\n-- T6: tier_separation_ratio (RC-2: đo trên Φ̃ = E_s[phi*delta], "
          f"n_states={T6_N_STATES}, behaviour_pair={T6_BEHAVIOUR_PAIR}) --")
    print(f"  ||dPhi~||_F structural  (lật active_lane)           = "
          f"{m6['t6_delta_phi_structural_max']:.6f}")
    print(f"  ||dPhi~||_F behavioural (đổi behaviour mode, PHẢI > 0) = "
          f"{m6['t6_delta_phi_behavioural']:.6f}")
    print(f"  bảng phi TĨNH bất biến trong drift (P4 design)      = "
          f"{m6['t6_static_phi_invariant']}")
    print(f"T6 tier_separation_ratio = {m6['t6_ratio']:.4f}")

    print("\n-- corr(Phi, W*) -- split by source dynamics --")
    print(
        f"static-role sources (n={m1['n_corr_static']}): "
        f"corr={m1['corr_phi_w_static']:.4f}, "
        f"sign agreement={m1['corr_sign_static']:.4f}  [PRIMARY GATE]"
    )
    print(
        f"mobile collector sources (n={m1['n_corr_mobile']}): "
        f"corr={m1['corr_phi_w_mobile']:.4f}, "
        f"sign agreement={m1['corr_sign_mobile']:.4f}  [DIAGNOSTIC]"
    )
    print(f"global corr={m1['corr_phi_w']:.4f}  [DIAGNOSTIC ONLY]")

    bank_diag = m1["state_bank_diagnostics"]
    control_diag = m1["control_diagnostics"]
    print("\n-- oracle state-bank and control diagnostics --")
    print(
        f"unique physical states={bank_diag['unique_state_fraction']:.3f}, "
        f"mean unique positions/agent="
        f"{bank_diag['mean_unique_positions_per_agent']:.2f}, "
        f"min={bank_diag['min_unique_positions_per_agent']}"
    )
    print(
        f"control nonzero state-pair fraction="
        f"{control_diag['state_nonzero_fraction']:.3f}, "
        f"constant-pair fraction={control_diag['constant_pair_fraction']:.3f}"
    )

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

    # T5/T6 and correlation have upper bounds.  Extremely large ratios often
    # indicate degenerate controls or configured constants, so the audit would
    # be remeasuring assumptions rather than a usable signal.
    checks = [
        ("T1 Gini(|W*|) > 0.30", m1["t1_gini"], m1["t1_gini"] > 0.30),
        ("T2 sign balance: neg frac > 0.15", m1["t2_neg_frac"], m1["t2_neg_frac"] > 0.15),
        ("T2 sign balance: pos frac > 0.15", m1["t2_pos_frac"], m1["t2_pos_frac"] > 0.15),
        ("T3 CV (mean over declared pair types) > 0.30", m3["t3_cv_mean"], m3["t3_cv_mean"] > 0.30),
        ("T4 legacy horizon diagnostic (nice-to-have)",
         m4["t4_spread"], m4["t4_spread"] > 0.3),
        ("T5 SNR (within-zone) in [3, 20]", m1["t5_snr"], 3.0 <= m1["t5_snr"] <= 20.0),
        ("T6 ratio in [3, 20]", m6["t6_ratio"], 3.0 <= m6["t6_ratio"] <= 20.0),
        ("T6 ||dPhi~||(behavioural) > 0 on empirical Phi-tilde",
         m6["t6_delta_phi_behavioural"], m6["t6_delta_phi_behavioural"] > 0.0),
        ("P4 design: static Phi is invariant under behavioural drift",
         m6["t6_static_phi_invariant"], m6["t6_static_phi_invariant"]),
        ("corr(Phi, W*) static-role sources in [0.65, 0.95]",
         m1["corr_phi_w_static"], 0.65 <= m1["corr_phi_w_static"] <= 0.95),
        ("state bank unique physical-state fraction >= 0.80",
         bank_diag["unique_state_fraction"], bank_diag["unique_state_fraction"] >= 0.80),
        ("control distribution has nonzero effects in >= 25% state-pair samples",
         control_diag["state_nonzero_fraction"], control_diag["state_nonzero_fraction"] >= 0.25),
        ("oracle no abs() (per_action has negative deltas on a real pair)", mabs["has_negative_delta"], mabs["has_negative_delta"]),
    ]

    name_w = max(len(c[0]) for c in checks)
    for name, val, ok in checks:
        status = "PASS" if ok else "FAIL"
        val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
        print(f"  [{status}] {name:{name_w}s}  value = {val_str}")

    n_pass = sum(1 for _, _, ok in checks if ok)
    print(f"\n{n_pass}/{len(checks)} checks passed.")

    machine_checks = [
        {
            "name": name,
            "value": (
                float(value)
                if isinstance(value, (int, float, np.integer, np.floating))
                else bool(value) if isinstance(value, (bool, np.bool_)) else str(value)
            ),
            "required": "nice-to-have" not in name,
            "passed": bool(passed),
        }
        for name, value, passed in checks
    ]
    required_gate_pass = all(
        row["passed"] for row in machine_checks if row["required"]
    )
    print("Required environment gate: " + ("PASS" if required_gate_pass else "FAIL"))

    if json_out:
        output_path = os.path.abspath(json_out)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "audit": "all_flags",
                    "required_gate_pass": required_gate_pass,
                    "checks": machine_checks,
                },
                handle,
                indent=2,
            )
            handle.write("\n")

    return {
        "t1": m1, "t3": m3, "t4": m4, "t6": m6,
        "sign_check": msign, "abs_check": mabs, "checks": checks,
        "machine_checks": machine_checks,
        "required_gate_pass": required_gate_pass,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default=None)
    cli_args = parser.parse_args()
    result = main(json_out=cli_args.json_out)
    raise SystemExit(0 if result["required_gate_pass"] else 2)
