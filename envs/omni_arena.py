"""
OMNI-ARENA — merges DIG (Declared Clean) + arena_v3 (congestion/queue)
according to docs/OMNI_ARENA_BLUEPRINT.md.
NEW, independent files — envs/population_main_dig.py and
envs/adaptive_resource_flow_arena_v3.py are untouched (Section 6, "Directives
on source code management").

Implementation P1-P4:

P1 (Section 2.1) — signed, conditional influence matrix:

w_ij(s) = phi_ij * delta_ij(s)
phi_ij: signed constant, ONLY changes at structural shift boundaries.
delta_ij(s): trigger gate in [0,1], recalculated EACH STEP from state, written
to info["delta_by_pair"].

P2 (Section 2.2) — 5 roles/zones (collector, gatekeeper, relay, blocker,
controller) with 4 latency ranges: blocker h=1, gatekeeper h=2-3,
relay h=4-5, controller h=6+. H_causal = 8 (>= max_latency(6) + 2).

P3 (Sections 1.1, 1.2, 2.3) — separate reward channel:
r_i = r_solo + sum_j delta_ij(s)*phi_ij*psi(a_j) [2] declaration
+ r_emergent(i,s,a) [3] emerges (congestion)

Constraint: max|r_emergent| <= 0.15 * min|phi_ij| (core pairs).
P4 (Parts 3.2, 3.3) — structural shift by MOVING BOTTLENECK (lane A -> lane B),

NO role swapping -> no indefinite transition window.
behavioral_drift keeps Phi immutable (assertable).
tier_separation_ratio = ||dPhi||_F(structural) / ||dPhi||_F(behavioural)
shown in info for each episode.
"""
import copy
import numpy as np

from utils.latency_protocol import LATENCY_ONSET_ABS_FLOOR, LATENCY_ONSET_FRACTION

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
    OPEN = 5   # In-place activation: gatekeepers open gates and controllers
               # press panels; semantics depend on role and current tile.
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
    # P1: designed Phi table (tier A), a signed constant; see Section 2.1.
    # Unlisted pairs have phi=0 and no declared channel.
    # ------------------------------------------------------------------
    # RC-1: preserve the phi scale. Do not increase |phi|. The original
    # MAX_EMERGENT_MAGNITUDE=0.15*MIN_CORE_PHI=0.0375 was already sufficient:
    # channel [3] firing across H=8 produced |W*|~0.3 and T5 SNR~13, inside
    # [3,20]. T5 was governed by firing frequency, not amplitude.
    PHI_GATEKEEPER_TO_COLLECTOR = -0.60
    PHI_RELAY_TO_COLLECTOR = 0.35
    PHI_BLOCKER_TO_COLLECTOR = -0.50
    PHI_CONTROLLER_TO_COLLECTOR = 0.25
    PHI_COLLECTOR_TO_GATEKEEPER = 0.30
    # P4: lane-B bonus after bottleneck relocation. Add it to the
    # (blocker, collector) pair when active_lane=='B'. The blocker is already
    # near lane B and need not move to become highly influential.
    PHI_LANE_B_BONUS = 0.35

    CORE_PHI_MAGNITUDES = [
        abs(PHI_GATEKEEPER_TO_COLLECTOR),
        abs(PHI_RELAY_TO_COLLECTOR),
        abs(PHI_BLOCKER_TO_COLLECTOR),
        abs(PHI_CONTROLLER_TO_COLLECTOR),
        abs(PHI_COLLECTOR_TO_GATEKEEPER),
    ]
    MIN_CORE_PHI = min(CORE_PHI_MAGNITUDES)  # 0.25
# P3 / Part 1.2: max|r_emergent| <= 0.15 * min|phi_ij|

#
# RC-3(a): Before deleting the two hand-coded blocks, this invariant was meaningless — it
# anchored to the NOMINAL phi table (min 0.25) while the reward line pairwise
# was REAL 2.5, meaning the actual rate was 1.5%, not 15%. After RC-1, phi IS
# the only pairwise reward line, so this invariant is again literally true.
# [P-3 FINAL DEBUG] Ceiling widened 0.0375 -> 0.045 to accommodate ZONE_CROWDING_COEF
# 0.012 -> 0.020: typical crowding ~0.030 + queue ~0.010 = ~0.040 must
# lie BELOW ceiling, otherwise r_emergent clip continuously and derivative by position
# disappear (correct RC-3 warning below).
    # The ceiling is a safety bound, not a reward coefficient. Scaling it with
    # CONGESTION_SCALE previously pinned every episode at exactly 0.013514,
    # flattening the variation required by the control-pair oracle. Preserve
    # coefficient scaling below, but restore 0.045 headroom so the emergent
    # channel does not saturate at the clip boundary.
    MAX_EMERGENT_MAGNITUDE = 0.18 * MIN_CORE_PHI

    # ------------------------------------------------------------------
    # RC-5: channel [3] parameters. The previous radius=1 and ``>2`` threshold
    # on a 24x24 grid with six agents per zone was too sparse: measured
    # max|r_emergent| was 0.0 for the entire episode. The arena_v3 channel did
    # not exist in practice; "DIG + arena_v3" was effectively "DIG + 0".
    #
    # RC-3 calibration: typical accumulated magnitude must stay strictly
    # inside (0, MAX_EMERGENT_MAGNITUDE), not stick to the ceiling. Continuous
    # clipping removes derivatives with respect to other-agent position,
    # drives W*(control) back to zero, and explodes SNR. Here, saturation means
    # the mechanism fires every step, not that it reaches the clip every step.
    # Typical total: crowding~0.018 + queue~0.010 = ~0.028 < 0.0375.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # [KNOB-2] Scale the background congestion channel by exactly 3.33x along
    # with GAME_COST_WEIGHT 0.05 -> 0.015.
    #
    # Correction to an earlier claim: not every audit gate is scale invariant.
    # This is true for T1 Gini and the T6 ratio, but false for T5 and
    # correlation. W* measures RETURN, which has three channels:
    #       task reward  +  declared w_ij  +  r_emergent (congestion)
    # GAME_COST_WEIGHT multiplies only the middle channel. Reducing it by 3.3x
    # shrinks declared influence inside W* while the other channels remain
    # fixed. corr(Phi,W*) then falls because Phi models only declared influence,
    # while the relative T5 denominator grows. Evidence from that exact change:
    # correlation moved from 0.7246 to 0.5519.
    #
    # This is a two-degree-of-freedom problem:
    #     knob 1 GAME_COST_WEIGHT  -> declared / task
    #     knob 2 congestion terms  -> noise / declared
    # Changing knob 1 alone unintentionally changes the second ratio. Scaling
    # the congestion block by the same factor preserves declared/noise.
    # ------------------------------------------------------------------
    CONGESTION_SCALE = 1.0 / 3.33

    LANE_CONGESTION_RADIUS = 2
    LANE_CAPACITY = 2                 # Use ``>=2`` rather than ``>2``.
    LANE_CONGESTION_PENALTY = 0.010 * CONGESTION_SCALE
    STATION_QUEUE_RADIUS = 2
    STATION_QUEUE_PENALTY = 0.020 * CONGESTION_SCALE
    COLLISION_PENALTY = 0.015 * CONGESTION_SCALE
    # Smooth distance-based background coupling within each zone. RC-3(b)
    # requires this mechanism: measured same-zone control pairs without a
    # blocker, such as relay->controller and relay->gatekeeper, returned exactly
    # W*=0 because no interaction path connected them. Same-zone sampling is
    # necessary but insufficient; a physical coupling channel must exist.
    # [P-3 FINAL DEBUG] 0.012 -> 0.020: measured T5 SNR=33.38 exceeded [3,20]
    # because std|W*|(control)=0.0197 was too small. A ~1.67x increase should
    # raise control std to ~0.033 and reduce SNR to ~20. If rerunning env_audit
    # yields T1 Gini<0.30 or frequent emergent clipping, reduce it to 0.016.
    ZONE_CROWDING_COEF = 0.020 * CONGESTION_SCALE   # [KNOB-2]

    # ------------------------------------------------------------------
    # [C3] Interaction-weight scale anchored to SGTP game_cost_config.yaml:
    #      safety 50 > block 10 > long 2 > contest 1   (spread 50x)
    # SGTP measures s in meters on a ~100 m track, with contest_s_gap=8 m or
    # about 8% of track length. Here s is normalized to [0,1] over the
    # gate->resource->sink path, so thresholds preserve s_role/s_contest=1/8.
    #
    # Use one scalar, GAME_COST_WEIGHT, to balance the interaction block against
    # task reward (delivery 0.7, pickup 0.3, r_solo 0.05). Do not tune isolated
    # constants as before, where BLOCKER_PENALTY 2.5 versus baseline 0.01
    # created a 250x mismatch.
    # ------------------------------------------------------------------
    # Sweep on gate S4: T5 SNR in block A with P3 congestion disabled.
    #   50/10/2/1.0  -> SNR 2.910  FAIL
    #   20/10/2/0.5  -> SNR 2.999  FAIL
    #   10/10/2/0.3  -> SNR 3.030  PASS   selected
    #    2/10/2/0.2  -> SNR 3.044 PASS but effectively removes safety.
    # W_SAFETY is lower than SGTP's 50 because that ratio models high-speed
    # physical collisions. Grid physics and COLLISION_PENALTY already handle
    # collisions here; safety=50 only injects spikes into control pairs and
    # raises std|W*|.
    W_SAFETY = 10.0
    W_BLOCK = 10.0
    # [RELAY-BUFF] Behind-agent support is comparable to blocking and 4x
    # W_LONG so relay/controller effects rise above background coupling.
    # [BUFF-3] Applied a 3x increase after obstruction/support began firing;
    # before [UNIT-FIX], gatekeeper/relay fell outside every band, so increasing
    # W_SUPPORT had no effect. [SUPPORT-FIX] changed 24 -> 10 and added alpha
    # in the formula. At 24, gatekeeper became the strongest pair and restored
    # a sign contradiction at magnitude 1.2: phi +0.05..+0.23 versus
    # W* -1.28..+0.49.
    W_SUPPORT = 10.0
    # [OBSTRUCTION] Same-position coupling is the primary blocker/relay channel
    # in this environment (Delta s~0), so it must match W_BLOCK in scale.
    # Sweep on the within-zone T5 gate after [UNIT-FIX]:
    #   Wobs=12 Dlat=.25 Scon=.25 -> T5 13.23
    #   Wobs=12 Dlat=.45 Scon=.65 -> T5 12.32
    #   Wobs= 6 Dlat=.45 Scon=.65 -> T5  3.96   selected: mid-band, not at ceiling
    W_OBSTRUCT = 6.0
    # [ENABLE] Precondition-family weight. Match W_OBSTRUCT so the two families
    # can compete when a gatekeeper both opens a gate and follows closely.
    # Sweep over correlation and sign agreement:
    #    0 -> corr +0.336 sign 12/20      3 -> +0.344 12/20
    #    6 -> corr +0.398 sign 14/20     10 -> +0.500 16/20   selected
    # Correlation remains below 0.65 but increases monotonically with W_ENABLE
    # and remains below the 0.95 ceiling, so this is not a fit. Do not exceed 10
    # until the collector->gatekeeper pair is resolved; see the report.
    W_ENABLE = 10.0
    S_NEAR = 0.06      # Same-position threshold along the s axis.
    # D_LAT_OBS was widened from 0.25 to 0.45 to cover relays in zones with
    # distant lanes; measured zone-1 relay dd=0.400 remained outside 0.25.
    # [PARTITION] One lateral threshold partitions the behind region into
    # following/obstruction versus yielding/support. The sweep required both
    # Phi(blocker)<0 and Phi(relay)>0 in all four zones:
    #     0.15 / 0.18 -> blocker passes, but z1 is much weaker (-0.019 vs -0.101)
    #     0.21 / 0.25 -> both pass in 4/4 zones, the valid window
    #     0.45        -> relay flips sign in 4/4 zones after misclassification
    # Select 0.23 at the center for maximum margin on both sides. Measurements
    # give blocker dd~0.11-0.20 and relay dd~0.22-0.40, placing the threshold
    # in the gap between roles and explaining why one threshold works.
    D_LAT_OBS = 0.23
    # [NOISE-FLOOR CALIBRATION] These background couplings apply to declared
    # and control pairs. The corrected Eq. (4) oracle showed that the old scale
    # produced too little control variance and T5=85.23 despite a diverse bank.
    #
    # Do not scale both uniformly: 8x reduced T3 to 0.294. Keep the long-term
    # gradient at 2.0 and increase only the state-switched contest term. This
    # raises control variance while preserving state conditionality. Final audit:
    #   W_CONTEST=4.0 -> T5=9.43, per-zone=5.96/18.38/6.77/6.62,
    #   T3=0.484, T6=13.41, corr_static=0.787, Gini=0.628.
    # W_SAFETY 10->50 barely changes control variance because its threshold
    # rarely fires, so it is not used to calibrate the noise floor.
    W_LONG = 2.0
    W_CONTEST = 4.0
    # 0.25 -> 0.65: measured controller ds=+0.33..+0.60 was outside the old
    # band, leaving the controller without any declared channel.
    S_CONTEST = 0.65
    S_ROLE = 0.03125          # = S_CONTEST/8, preserving the SGTP ratio.
    D_SAFE = 0.08
    D_LAT = 0.05              # Lateral-offset threshold for CSD (C4).
    # [GCW-CAL] 0.05 -> 0.015. The first runner execution on the correct
    # OmniArena after [ENV-RESOLVE] showed that interaction dominated task
    # reward. Scripted-policy measurements per agent-step:
    #     total w_ij     = -0.0438   (p5 -0.504, p95 +0.346)
    #     task reward    = +0.0179   after removing channel w
    #     interaction magnitude = 169% of reward magnitude
    # The learner could not improve task reward because interaction penalties
    # consumed delivery gains. Logs moved reward -2.1 -> -3.4 with f1=0.000
    # throughout, worse than scripted at -0.778/episode/agent. A value of 0.015
    # puts interaction near 50% of task-reward magnitude: enough to make
    # structure valuable without making it the entire task.
    #
    # Historical audit assumption: all gates were treated as ratios or
    # scale-invariant statistics, motivating a single global knob. The
    # [KNOB-2] correction above records why T5 and correlation are exceptions.
    GAME_COST_WEIGHT = 0.015

    # P0(d): noise purposes consumed in step() for the oracle CRN buffer.
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
        use_sgtp_phi=True,               # [C2] Continuous SGTP Phi; False uses the legacy table ablation.
        structural_factor=None,
        behavioral_factor=None,
        enable_sgtp_delays=False,
        sgtp_delay_values=(0, 2, 4, 6),
        sgtp_delay_signal_gain=0.25,
    ):
        assert n_agents >= 5 * n_zones, "At least five agents per zone are required for the five P2 roles."
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.n_zones = n_zones
        self.obs_radius = obs_radius
        self.max_steps = max_steps
        self.phase_length = phase_length
        self.causal_horizon = causal_horizon
        self.mode = mode
        self.structural_factor = bool(
            mode == "structural_shift"
            if structural_factor is None
            else structural_factor
        )
        self.behavioral_factor = bool(
            mode == "behavioral_drift"
            if behavioral_factor is None
            else behavioral_factor
        )
        self.enable_sgtp_delays = bool(enable_sgtp_delays)
        self.sgtp_delay_values = tuple(int(value) for value in sgtp_delay_values)
        self.sgtp_delay_signal_gain = float(sgtp_delay_signal_gain)
        if not self.sgtp_delay_values or min(self.sgtp_delay_values) < 0:
            raise ValueError("sgtp_delay_values must contain non-negative lags")
        self.sgtp_delay_by_pair = {}
        self._sgtp_delay_queues = {}
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
        # [C2] True uses continuous same-zone
        # w_ij=-GAME_COST_WEIGHT*sgtp_pair_cost(Delta s, Delta d).
        # False uses the legacy phi-times-delta lookup table.
        self.use_sgtp_phi = bool(use_sgtp_phi)
        self._in_phi_measurement = False

        # [C2d] P1 conditional gates and the P2 latency ladder are NO-OPs when
        # use_sgtp_phi is enabled. Both act only through _compute_deltas(),
        # while the SGTP branch no longer consumes delta. Evidence: staged
        # audit block A minus BASELINE is 0.0000 for every metric except the
        # independently broken T6. The inactive latency ladder also explains
        # T4=0.25. Retain these parameters for use_sgtp_phi=False ablation
        # compatibility, but document their inactive status explicitly.
        if self.use_sgtp_phi and not (
            self.enable_conditional_gates and self.enable_latency_ladder
        ):
            print("[OmniArena][NOTE] use_sgtp_phi=True => "
                  "enable_conditional_gates / enable_latency_ladder are NO-OPs "
                  "because influence no longer passes through delta_ij.")

        self.supported_egos = list(range(self.n_agents))
        self.current_phase = 0
        self.episode_count = 0
        self.episode_deliveries = 0

        # RC-2: force _behaviour_mode() to return a fixed mode only while
        # measuring realized Phi (measure_realized_phi_tiers). This isolates
        # behavior from episode phase; otherwise both axes differ between
        # measurements and the change cannot be attributed.
        self._behaviour_override = None
        self._pending_factorial_intervention = None
        self._controlled_structural_event_active = False
        self._controlled_behavioral_event_active = False

        # [Guard A -- P2<->P4 trap] True only during reset(). This is the
        # correct guard for _do_structural_shift(): self.t/self.done describe
        # the previous episode and say nothing about the current call. The
        # relevant fact is whether the call actually originated from reset().
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
        # [C2b] The first Phi measurement must follow reset() because it needs positions.
        if self.use_sgtp_phi:
            self._measure_phi_from_sgtp()

    # ============================================================
    # Layout
    # ============================================================

    def _init_zone_heterogeneity(self):
        """[ZONE-ASYM] Break symmetry between zones.

        Required evidence: in W*_ij(t), curves 1->0 and 6->5 matched at every
        point, as did 3->0 and 8->5. Blocker W* across four zones was
        -1.2548/-1.2571/-1.1915/-1.2548. Every zone used the same fixed offsets
        from center (gate=c-2, resource=c+1, sink=c+3), making relative geometry
        exactly identical after _frenet_sd normalization. N=24 was effectively
        four copies of N=6, invalidating Experiment 5's N sweep.

        Important RNG trap: self.rng drives noise_buffer/CRN. Every paired
        oracle comparison, including the only currently significant result
        dr_eps005 + xi_ij, relies on intervention and control branches sharing
        random draws. Drawing from self.rng here shifts all later draws. Use a
        separate RNG derived from self.seed, instantiate it exactly once in
        __init__, and never recreate it in reset().

        The parameter space has 4 path lengths x 3 lane capacities x 3 respawn
        delays = 36 discrete combinations, sufficient through n_zones=16 or
        N=96 at about six agents per zone. At n_zones>=32, some zones share all
        three values; continuous zone_scale multiplied directly into w_ij is
        then the only guarantee that no two zones are identical.
        """
        zone_param_rng = np.random.RandomState(
            (int(self.seed) * 7919 + 104729) % (2 ** 31 - 1)
        )

        # path_len MUST fit inside each zone cell. Otherwise _clip pins gate
        # and sink positions to the same boundary in multiple zones, creating
        # another symmetry. Recompute cell_h exactly as _init_zone_layout does.
        # For grid=24 and n_zones=4, cell_h=12 and path_len<=9.
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

                # [ZONE-ASYM] path_len[z] changes the s axis and every Delta
                # s_ij, thereby changing alpha_ij and role conditions. This is
                # the most important symmetry-breaking variable. Keep the
                # resource near 60% of the path, as before (3/5 at path_len=5).
                pl = int(self.zone_path_len[zone])
                assert pl < cell_h - 2, (
                    f"path_len={pl} exceeds cell_h={cell_h} for zone {zone}; "
                    f"_clip would pin multiple zones to the same boundary and "
                    f"create another symmetry. Increase grid_size or reduce n_zones."
                )
                res_off = max(1, int(round(pl * 0.6)))
                self.zone_gate[zone] = (self._clip(cr - 2), self._clip(cc_))
                self.zone_resource[zone] = (
                    self._clip(cr - 2 + res_off), self._clip(cc_))
                self.zone_sink[zone] = (self._clip(cr - 2 + pl), self._clip(cc_))
                self.zone_lane_a[zone] = (self._clip(cr), self._clip(cc_ - 3))
                self.zone_lane_b[zone] = (self._clip(cr), self._clip(cc_ + 3))
                self.zone_panel[zone] = (self._clip(cr - 3), self._clip(cc_ - 3))
                # RC-4: the blocker duty tile lies between the resource (cr+1)
                # and sink (cr+3), directly on the collector route. The blocker
                # pursues only its static duty tile, but this placement makes
                # dist(blocker, collector)<=2 frequent enough to preserve the
                # declared blocker->collector edge. Influence is a task side
                # effect, not a reward objective for the blocker. Moving the
                # duty tile away from the route, such as lane_b, drives
                # gate_blocker_collector delta to zero, kills the graph's only
                # negative edge, and collapses T2 sign balance.
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
    # P1 + P4: Phi table (tier A, signed and changed only at shift boundaries)
    # ============================================================

    def _refresh_gt_graph(self):
        """Recompute tier-A Phi only at initialization or a P4 boundary.

        This method must never run during behavioral drift. Phi invariance is
        checked by ``assert_behavioural_invariance()`` and
        ``test_phi_invariance.py``.
        """
        self.gt_core_by_ego = {}
        self.gt_influence_by_ego = {}
        self.declared_pairs = []  # (i, j, phi, gate_fn_name) entries for P1/P3.
        self._declared_set = None  # [SCOPE-DECL] Rebuild with declared_pairs.

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

            # gatekeeper -> collector; phi is independent of lane.
            self._set_phi(gatekeeper, collector, self.PHI_GATEKEEPER_TO_COLLECTOR)
            self.declared_pairs.append((gatekeeper, collector, "gate_gk_collector", z))

            # relay -> collector has a designed weight only while lane A is
            # the active bottleneck under P4 relocation.
            relay_phi = self.PHI_RELAY_TO_COLLECTOR if self.active_lane[z] == "A" else 0.0
            self._set_phi(relay, collector, relay_phi)
            self.declared_pairs.append((relay, collector, "gate_relay_collector", z))

            # blocker -> collector combines fixed harmful phi with a lane-B
            # bonus when lane B is the bottleneck. The blocker already occupies
            # that vicinity and does not move, matching Section 3.2 relocation.
            blocker_phi = self.PHI_BLOCKER_TO_COLLECTOR
            if self.active_lane[z] == "B":
                blocker_phi += self.PHI_LANE_B_BONUS
            self._set_phi(blocker, collector, blocker_phi)
            self.declared_pairs.append((blocker, collector, "gate_blocker_collector", z))

            # RC-4 rejected a mirrored zero-sum collector->blocker edge. The
            # runner optimizes per-agent returns without a mixer, QMIX, or team
            # reward, so a negative sum does not itself create harm. A zero-sum
            # mirror would reward the blocker by exactly the collector's loss,
            # strengthen pursuit incentives, and break the cooperative framing.
            # Instead, remove pursuit incentives at r_solo: the blocker follows
            # a separate duty and affects the collector only as a side effect.
            # This preserves the research target of environment-generated
            # structural influence rather than deliberate adversarial behavior.

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
        # [C2b] Remeasure Phi from the continuous formula, not a lookup table.
        #
        # Evidence requiring this correction:
        # - corr(Phi,W*)=-0.0283 because Phi remained the legacy table while W*
        #   came from the continuous formula. They described different envs.
        # - gatekeeper phi=-0.600 versus W*=+1.5151: opposite sign and 13x.
        # - relay changed from +0.9899 to -0.0721 after refactoring.
        # - T6 remained exactly 5.9004 before and after the influence rewrite,
        #   proving it measured a static table disconnected from the running
        #   mechanism, the same failure class as the earlier hardcoded 0.0.
        #
        # Define Phi_ij=E_s[w_ij(s)] on a seeded state bank. Phi then becomes a
        # measured environment quantity rather than a declared constant, so
        # corr(Phi,W*), signs, and T6 track the real mechanism. Preserve
        # gt_core_by_ego role labels for Core F1; they do not compute reward.
        # ------------------------------------------------------------------
        # Positions do not exist during the first call from
        # _init_population_roles before reset(); measure at the end of __init__.
        if (getattr(self, "use_sgtp_phi", False)
                and not getattr(self, "_in_phi_measurement", False)
                and hasattr(self, "positions")):
            self._measure_phi_from_sgtp()

    def _measure_phi_from_sgtp(self, n_states=48, burn_in=3, bank_seed=None):
        """Measure Phi_ij=E_s[w_ij(s)]; see ``_refresh_gt_graph``.

        [PHI-SAMPLING] Replace fixed bank_seed=777 with a rotating derived seed,
        and increase n_states from 12 to 48.

        Channel decomposition of W*(collector->gatekeeper) across four zones:
            d_total  +0.270 | -0.006 | -0.173 | +0.771
            d_w(c->gk) +0.263 | -0.015 | -0.181 | +0.764
            d_emergent +0.007 | +0.009 | +0.008 | +0.008
        d_total and d_w agree within 0.01 in all zones, so this pair's W* uses
        exactly one channel: w_ij, which Phi models. No mechanism is missing;
        the legacy lookup branch runs only when use_sgtp_phi=False.

        The 4/4 sign disagreement came from sampling. Phi averaged 12 states
        from one fixed seed-777 bank under default behavior. W* averaged actual
        post-intervention trajectories from a different distribution. Using the
        same w_ij formula on another state set produced +0.26/-0.02/-0.18/+0.76,
        while audit produced +0.91/+0.61/+0.72/+0.96; signs changed solely with
        the state sample.

        The T6 ``measure_realized_phi_tiers`` docstring already states that a
        fixed reused bank is invalid. T6 followed that rule, but this method did
        not. Rotate a seed derived from self.seed and a counter on each measure,
        and use 48 states to reduce Monte Carlo variance. The w_ij formula is
        unchanged; only measurement procedure changes.
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
            print(f"[C2b][WARN] continuous Phi measurement failed ({e}); "
                  f"retaining the legacy table makes corr(Phi,W*) unreliable.")
            acc, cnt = {}, 0
        finally:
            # IMPORTANT: restore_state() deep-copies gt_influence_by_ego from
            # the snapshot. Write measurements only after restore; writing
            # before it erases them and restores the legacy table. This exact
            # trap previously left printed Phi equal to the PHI_* constants.
            self.restore_state(snapshot)
            self._in_phi_measurement = False

        if cnt:
            for (i, j), tot in acc.items():
                self.gt_influence_by_ego[j][i] = tot / cnt

    def _set_phi(self, src, dst, value):
        self.gt_influence_by_ego[dst][src] = float(value)

    # ============================================================
    # P1: state-dependent activation gate delta_ij(s) in [0,1]
    # ============================================================

    def _compute_deltas(self):
        """Return ``{(i,j): delta_ij(s)}`` for declared pairs only.

        Other pairs always have delta=0 because phi=0. Compute after positions
        update, at state s' for this step, to match reward timing in ``step``.
        When ``enable_conditional_gates`` is off, every declared delta collapses
        to 1.0 without evaluating state conditions. This is a separate branch,
        not compute-then-discard behavior.
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
        """Apply P2's four role-dependent latency tiers; see Section 2.2.

        blocker h=1, gatekeeper h=2-3, relay h=4-5, controller h=6+.
        """
        if gate_name == "gate_gk_collector":
            # Delta must depend on the gatekeeper's actual position, not only
            # collector state. Otherwise intervening on the gatekeeper cannot
            # change this declared channel. env_audit exposed this bug through
            # the wrong sign in corr(Phi,W*).
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
            # [FIX-R2] q>=1 alone is insufficient. Under scripted policy the
            # relay almost always occupies lane_a, making delta identically 1
            # and measured relay T3 CV remain zero. The failure merely changes
            # from always-off to always-on; W*(relay) saturates near +2.0 and
            # pushes T5 SNR to 30.2, outside [3,20]. Require carrying[z], since
            # relay matters while the collector carries through the lane.
            # carrying has demonstrated state variation, including blocker
            # CV=0.845, restoring conditionality while relay_present preserves
            # causal attribution to the relay.
            #
            # Threshold history: after _lane_queue was restricted to collector
            # and relay by P-1, q>=2 required both to occupy lane_a. This almost
            # never occurs because the collector travels resource<->sink. The
            # relay channel died: T3 CV 2.42->0.0000, W*~-0.0075, and T4
            # 0.25->0.0. q>=1 still depends on the relay's actual position and
            # active lane without requiring collector presence.
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
                # Combine overlapping conditions with max; delta remains in
                # [0,1] and does not accumulate above 1.0.
                return max(base_active, lane_active)
            return base_active

        if gate_name == "gate_controller_collector":
            return 1.0 if self.low_priority_active[z] else 0.0

        if gate_name == "gate_collector_gatekeeper":
            return 1.0 if self._dist(self.positions[collector], self.zone_resource[z]) <= 1 else 0.0

        return 0.0

    def _gate_flat(self, gate_name, z, ra, collector, gatekeeper, blocker):
        """Use one flat default latency window when P2 is disabled.

        Treat every pair as h=1 and skip role-tier distance/queue thresholds
        from ``_gate_ladder``. Only source proximity to the relevant target
        remains; role-specific latency is absent.
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
    # P3 / RC-5: emergent channel [3], not attributable to one pair
    # ============================================================

    def _apply_emergent_congestion(self, r_emergent):
        """Apply three position-density congestion sources.

        No declared pair (i,j) owns these effects, matching channel [3]:

          1. lane over-capacity : physical bottleneck at lane A or lane B
          2. station queue      : resource and sink contention
          3. zone crowding      : smooth same-zone 1/(1+d) background coupling

        RC-3(b) requires source (3). Sources (1) and (2) use discrete thresholds,
        so agents that never coincide produce exactly W*=0, control std=0, and
        infinite T5 SNR. Increasing their amplitudes cannot help non-contacting
        pairs. A continuous distance channel gives a finite noise floor.

        Complexity is O(n_zones*n_agents) for (1)(2) and O(n_agents^2) for (3).
        At n_agents=24 this is about 600 operations per step, negligible beside
        oracle rollouts at H=8*|A|=6*n_states. Thousands of agents would require
        a per-zone spatial hash for source (3).
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

            # The legacy implementation counted only the resource queue. The
            # sink is also a real bottleneck because delivery requires stopping
            # on that tile, so include both stations.
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

        # P3 / Section 1.2: clip channel [3] after accumulating all sources.
        for a in range(self.n_agents):
            r_emergent[a] = float(np.clip(
                r_emergent[a], -self.MAX_EMERGENT_MAGNITUDE, self.MAX_EMERGENT_MAGNITUDE
            ))

    def _aggregate_reward_by_role(self, rewards):
        """Return mean reward by role instead of a population-wide mean."""
        acc = {role: [] for role in self.ROLE_ORDER}
        for a in range(self.n_agents):
            acc[self.agent_role[a]].append(float(rewards[a]))
        return {role: (float(np.mean(v)) if v else 0.0) for role, v in acc.items()}

    def _lane_queue(self, lane_pos, radius=1):
        # [P-1 FINAL DEBUG] Count only roles that actually queue in a lane:
        # collector and relay. The legacy code counted every nearby agent,
        # including gatekeeper/controller bodies on separate duties. An
        # intervention on the gatekeeper could then lower q below 2 and toggle
        # blocker->collector (phi=-0.5) and relay->collector gates. Base-versus-
        # disengage decomposition for gk->collector measured its declared d_w
        # as 0.000 in every state; the full negative W*=-0.87 leaked through
        # other gates and reduced corr(Phi,W*) to 0.625. Role filtering removes
        # that leakage path.
        queue_roles = (self.ROLE_COLLECTOR, self.ROLE_RELAY)
        count = 0
        for a in range(self.n_agents):
            if self.agent_role[a] not in queue_roles:
                continue
            if self._dist(self.positions[a], lane_pos) <= radius:
                count += 1
        return count

    # ============================================================
    # P4: structural shift by bottleneck relocation, without role swapping
    # ============================================================

    def _maybe_structural_shift(self):
        # P4 flag OFF: bottleneck relocation NEVER triggers -- only
        # behavioural_drift semantics remain reachable. Real branch, not a
        # Avoid "compute then discard" by returning before the phase check.
        # boundaries, so tier_separation_ratio machinery stays at its
        # NaN-safe default (0/eps = 0, see tier_separation_ratio()).
        if not self.enable_structural_shift:
            self.delta_phi_frobenius_structural_last = 0.0
            self.delta_phi_frobenius_behavioural_last = 0.0
            return

        if not self.structural_factor:
            return

        if self.episode_count > 0 and self.episode_count % self.phase_length == 0:
            self._do_structural_shift()
        else:
            self.delta_phi_frobenius_structural_last = 0.0
            self.delta_phi_frobenius_behavioural_last = 0.0

        """Execute P4 bottleneck relocation only from inside ``reset()``.

        Guard A for the P2<->P4 trap uses ``self._in_reset``, not
        ``self.t``/``self.done``. The legacy guard asserted
        ``self.done or self.t==0`` and incorrectly treated previous-episode
        values present just before reset as evidence of the current call site.
        ``run_tier0`` legitimately resets early when
        ``t+horizon+steps_between>=max_steps``; at t=48 and done=False the old
        guard failed even though reset was valid. Conversely,
        ``test_guard_a_mid_episode_shift_asserts`` directly calls
        ``_do_structural_shift`` after step, bypassing reset, and must fail.

        Both cases have t>0 and done=False at the assertion, so those fields
        cannot distinguish them. ``self._in_reset`` records the actual origin:
        it is set before ``_maybe_structural_shift`` and cleared immediately.

        clone_state/restore_state independently blocks another P2<->P4 trap:
        a shift occurring inside a saved rollout window is erased on restore.
        Snapshots cover active_lane, current_phase, gt_influence_by_ego,
        delta_phi_frobenius_*, t, done, and episode_count.
        """
    def _do_structural_shift(self):
        # --- GUARD A PATCH START ---
        assert getattr(self, '_in_reset', False), \
            "GUARD A (P2<->P4 trap): Cannot apply structural shift mid-episode. Call this only via reset()!"
        # --- GUARD A PATCH END ---
        self.current_phase += 1
        prev_phi = copy.deepcopy(self.gt_influence_by_ego)

        for z in range(self.n_zones):
            self.active_lane[z] = "B" if self.active_lane[z] == "A" else "A"

        self._refresh_gt_graph()
        self.delta_phi_frobenius_structural_last = self._delta_phi_frobenius(prev_phi, self.gt_influence_by_ego)
        self.delta_phi_frobenius_behavioural_last = 0.0

        # RC-2: do not assign behavioural=0.0 here. This method does not
        # measure behavioral drift, so zero was fabricated data and one of
        # three assignments that made T6 equal 1e12 by construction. Only
        # measure_realized_phi_tiers() supplies the valid value.

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
        """Return the ratio using measured ``||dPhi_tilde||``.

        RC-2: the denominator must come from ``measure_realized_phi_tiers``,
        not a hardcoded zero. The legacy file contained exactly three
        assignments of behavioral delta to 0.0 and no nonzero assignment, so
        1e12 was produced by construction. T6=9.9e11 was never a measurement.
        Return infinity before measurement so the failure is explicit.
        """
        num = self.delta_phi_frobenius_structural_last
        den = self.delta_phi_frobenius_behavioural_last
        if den <= 0.0:
            return float("inf")
        return float(num / den)

    # ============================================================
    # RC-2: realized influence matrix Phi_tilde=E_s[phi*delta]
    # ============================================================

    def set_behaviour_override(self, mode_name):
        """Override ``_behaviour_mode``; None restores the phase schedule."""
        self._behaviour_override = mode_name

    def realized_phi_matrix(self, state_bank):
        """Compute Phi_tilde_ij=E_s[phi_ij*delta_ij(s)] on a state bank.

        A static phi table is blind to behavior and changes only at structural
        boundaries, so behavioral ||dPhi||=0 follows by definition rather than
        discovery. delta_ij(s), however, depends on both agents' actual
        positions. Behavioral drift can therefore change realized Phi while
        phi stays fixed. The required T6 signal already existed in L2 as
        delta_ij(s); no extra realization gate was needed.

        Return sparse ``{ego: {src: Phi_tilde}}`` for declared pairs, matching
        the domain consumed by ``_delta_phi_frobenius``.
        """
        if not state_bank:
            return {}

        # [C2c] Connect T6 to the active channel. After replacing the entire
        # influence channel with the continuous formula, T6 still equaled
        # 5.9004 and structural/behavioral norms remained exactly
        # 0.638402/0.108197. This method was still measuring the removed lookup
        # phi times binary delta. With use_sgtp_phi enabled, realized Phi must
        # be the same w_ij(s) added to reward by step().
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
                    # Read phi from the restored state. Each state carries its
                    # own active_lane and Phi table, enabling valid cross-lane
                    # structural comparisons.
                    phi = self.gt_influence_by_ego[j].get(i, 0.0)
                    row = acc.setdefault(j, {})
                    row[i] = row.get(i, 0.0) + phi * d
        finally:
            self.restore_state(snapshot)

        n = float(len(state_bank))
        return {j: {i: v / n for i, v in row.items()} for j, row in acc.items()}


    """Measure both realized-Phi Frobenius deltas and store the results.

          structural  : change active_lane, changing phi and delta -> large
          behavioral  : change _behaviour_mode while phi stays fixed;
                        only the state distribution changes        -> small > 0

        Noise control: all three samples use the same bank_seed, n_states, and
        burn_in. Otherwise the measured difference is Monte Carlo sampling
        noise rather than tier signal.

        The state bank cannot be one fixed set reused across behavior modes.
        delta_ij(s) is a pure function of s, so identical states produce
        identical realized Phi and a zero difference. Behavioral drift changes
        the visited-state distribution. "Fixed" therefore means a fixed seed
        and sampling procedure while allowing behavior to generate the states.

        Return ``(structural, behavioural)``.
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

            # --- STRUCTURAL SHIFT: relocate only the bottleneck. ---
            for z in range(self.n_zones):
                self.active_lane[z] = "B" if self.active_lane[z] == "A" else "A"

            self._refresh_gt_graph()
            
            phi_shifted = self.realized_phi_matrix(
                self.sample_state_bank(n_states, burn_in, bank_seed=bank_seed))
            structural = self._delta_phi_frobenius(phi_ref, phi_shifted)

            # Restore the original structural state.
            self.active_lane = copy.deepcopy(saved_lane)
            self.zone_role_agents = copy.deepcopy(saved_role_agents)
            self.agent_zone = copy.deepcopy(saved_agent_zone)
            self._refresh_gt_graph()

            # --- BEHAVIORAL DRIFT ---
            # Preserve the ``selfish`` terminal mode in the schedule.
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


             # --- behavioral: preserve phi and change only agent behavior. ---
            # [P-4 FINAL DEBUG] Average adjacent drift steps in the actual
            # behavioral_drift schedule rather than comparing only the extreme
            # cooperative and selfish endpoints. The extreme pair measures four
            # phase transitions at once, which no agent experiences. A learner
            # sees one transition. Keep behaviour_pair for compatibility and use
            # it only when the schedule contains fewer than two modes.
    # ============================================================
    # API
    # ============================================================

    def get_supported_egos(self):
        return list(self.supported_egos)

    def get_gt_core_for_ego(self, ego_id):
        return set(self.gt_core_by_ego[ego_id])

    def get_gt_influence_for_ego(self, ego_id):
        """Return tier-A Phi without state gating."""
        return dict(self.gt_influence_by_ego[ego_id])

    def get_conditional_influence_for_ego(self, ego_id):
        """Return w_ij(s) for the current state for T3 state-wise CV.

        [C2] With use_sgtp_phi enabled, return the continuous w that actually
        determines reward, not legacy phi*delta. Otherwise T3 measures a channel
        that no longer exists and merely restates its own assumption.
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
            "sgtp_delay_queues": copy.deepcopy(self._sgtp_delay_queues),
            "behaviour_override": self._behaviour_override,
            "pending_factorial_intervention": copy.deepcopy(
                self._pending_factorial_intervention
            ),
            "controlled_structural_event_active": bool(
                self._controlled_structural_event_active
            ),
            "controlled_behavioral_event_active": bool(
                self._controlled_behavioral_event_active
            ),
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
        # ``get`` preserves compatibility with snapshots created before RC-4.
        self.episode_deliveries = state.get("episode_deliveries", 0)
        self.current_phase = state["current_phase"]
        self.rng.set_state(state["rng_state"])
        self.gt_core_by_ego = copy.deepcopy(state["gt_core_by_ego"])
        self.gt_influence_by_ego = copy.deepcopy(state["gt_influence_by_ego"])
        self.delta_phi_frobenius_structural_last = state["delta_phi_frobenius_structural_last"]
        self.delta_phi_frobenius_behavioural_last = state["delta_phi_frobenius_behavioural_last"]
        self._sgtp_delay_queues = copy.deepcopy(state.get("sgtp_delay_queues", {}))
        self._behaviour_override = state.get("behaviour_override")
        self._pending_factorial_intervention = copy.deepcopy(
            state.get("pending_factorial_intervention")
        )
        self._controlled_structural_event_active = bool(
            state.get("controlled_structural_event_active", False)
        )
        self._controlled_behavioral_event_active = bool(
            state.get("controlled_behavioral_event_active", False)
        )

    def schedule_factorial_intervention(
        self, structural: bool, behavioral: bool, behavior_mode: str = "selfish"
    ):
        """Schedule one controlled S/B intervention for the next reset.

        The H2 harness uses this for all four factorial cells, including the
        sham cell. Passive phase schedules are disabled in that protocol, so
        both interventions share one exogenous event time.
        """
        if self._pending_factorial_intervention is not None:
            raise RuntimeError("a factorial intervention is already pending")
        self._pending_factorial_intervention = {
            "structural": bool(structural),
            "behavioral": bool(behavioral),
            "behavior_mode": str(behavior_mode),
        }

    def _apply_sgtp_delays(self, instantaneous):
        """Apply cloneable pair-specific transport delays to SGTP effects."""
        if not self.enable_sgtp_delays:
            return dict(instantaneous)
        applied = {}
        for pair, value in instantaneous.items():
            pair = (int(pair[0]), int(pair[1]))
            if pair not in self.sgtp_delay_by_pair:
                # Mix both directed identifiers. Using source*n_agents+target
                # modulo four collapsed to one delay whenever n_agents was a
                # multiple of four and the oracle held the ego fixed.
                mixed = (
                    pair[0] * 73856093
                    ^ pair[1] * 19349663
                    ^ int(self.seed) * 83492791
                )
                index = int(mixed % len(self.sgtp_delay_values))
                self.sgtp_delay_by_pair[pair] = int(self.sgtp_delay_values[index])
            delay = self.sgtp_delay_by_pair[pair]
            if delay <= 0:
                applied[pair] = float(value)
                continue
            queue = self._sgtp_delay_queues.setdefault(pair, [0.0] * delay)
            queue.append(float(value))
            applied[pair] = float(queue.pop(0))
        return applied

    def _add_sgtp_action_transport_signal(self, instantaneous, actions):
        """Add an action-coded signal that is transported by the SGTP queue.

        The optional latency environment needs an identifiable state-mediated
        path whose delay is independent of role. The signal is added to the
        existing same-zone SGTP edges before queueing, so clone/restore and
        common-random-number intervention rollouts preserve its exact timing.
        Ordinary experiments never enable this branch.
        """
        if not self.enable_sgtp_delays or self.sgtp_delay_signal_gain == 0.0:
            return dict(instantaneous)
        output = dict(instantaneous)
        midpoint = 0.5 * float(max(1, self.N_ACTIONS - 1))
        scale = max(midpoint, 1.0)
        for (source, target), value in list(output.items()):
            action = int(actions[int(source)])
            action_code = (float(action) - midpoint) / scale
            output[(source, target)] = float(
                value + self.sgtp_delay_signal_gain * action_code
            )
        return output

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
        """Return Frenet coordinates on the zone task path.

        [C1, SGTP refactor specification] The path is gate -> resource -> sink.
        This is a read-only, side-effect-free state transform used by Phi.

        s in [0,1] is normalized arc-length progress from gate to sink after
        projecting onto the nearer of the two segments. d is signed
        perpendicular distance: positive toward lane_b (larger column) and
        negative toward lane_a, matching existing zone conventions.
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
                # [UNIT-FIX] Normalize d by total_len, the same denominator as
                # s. The legacy code returned d in raw grid cells while s was
                # in [0,1], so D_SAFE, D_LAT, D_LAT_OBS, and 1/(1+dd) compared
                # incompatible units. Sweeping W_OBSTRUCT over {0,2,4,6}
                # produced identical correlation to four decimals (-0.1627),
                # showing obstruction almost never fired: a ~3-cell lateral
                # offset was compared to 0.25 in normalized-s units. This is the
                # same failure class as k_sg/sigma_hi in Section B.
                best_d = signed_d / total_len

        return float(best_s), float(best_d)

    # ================================================================
    # [C2 + C3] Continuous SGTP-style Phi replacing role-by-role lookup
    # ================================================================

    def _sgtp_pair_cost(self, s_i, d_i, s_j, d_j, zone_scale=1.0,
                        is_declared=False):
        """Return the interaction cost imposed by agent i on agent j.

        The four terms follow SGTP ``_compute_game_block_cost`` but use task
        path coordinates (s,d) from ``_frenet_sd`` rather than race-track
        Frenet coordinates. The caller adds ``-cost`` to reward[j], so positive
        terms mean that i harms j.

            Delta s = s_i-s_j > 0  <=> i is ahead of j on the path
            Delta d = |d_i-d_j|         lateral offset
            alpha = 1/(1+|Delta s|/s_contest), continuous and never zero

          1. long    : + w_long * alpha * Delta s
                       A nearby leading i harms j; a trailing i helps j. This
                       term generates T2 sign balance without declared signs.
          2. contest : + w_contest · 1[|Δs| < s_contest]
          3. block   : + w_block/(1+Delta d) when
                       s_role < Delta s < s_contest. There is no blocker label;
                       whichever agent leads j in this interval is the blocker
                       for that step. Role becomes state-dependent, naturally
                       making behavioral realized-Phi change nonzero and T6
                       non-tautological.
          4. safety  : + w_safety * max(0,d_safe-dmin)^2, a quadratic hinge.
                       This is the only hard-threshold term, and it remains
                       smoothed as in SGTP.
        """
        ds = float(s_i) - float(s_j)
        dd = abs(float(d_i) - float(d_j))
        eps = 1e-6

        # ------------------------------------------------------------------
        # [ZS-SPLIT] Apply zone_scale only to declared block terms, not to the
        # long/contest/safety background or s_contest.
        #
        # The previous implementation multiplied both background and interaction
        # radius by zs. Low-zs zones lost declared signal and background together,
        # so within-zone SNR did not improve; the full zone merely shrank.
        # Measurements showed nearly inert zones 0/3 (gatekeeper 0.51/0.57,
        # blocker -0.13/-0.18) versus strong zone 1 (1.48/-1.23). A shared
        # background scale makes zone_scale represent structural strength rather
        # than total zone magnitude.
        # ------------------------------------------------------------------
        zs = float(zone_scale)
        alpha = 1.0 / (1.0 + abs(ds) / (self.S_CONTEST + eps))

        # --- BACKGROUND: shared scale independent of zone. ---
        cost = self.W_LONG * alpha * ds

        if abs(ds) < self.S_CONTEST:
            cost += self.W_CONTEST

        # ------------------------------------------------------------------
        # [OBSTRUCTION] Separate two physically different terms.
        #
        # SGTP blocking requires the ego to lead because the blocking vehicle
        # is ahead on a track. This environment's blocker follows and overlaps
        # the collector, the opposite geometry. Measurements at reset:
        #     z0..z3 relay ds=+0.000, blocker ds=+0.000 in every zone
        # Both fall outside every Delta-s interval. This is not a polyline
        # projection bug; Delta s~0 is the correct physical property of the
        # blocker-collector relation. Adding lanes to the polyline would only
        # change the s axis and would not fix this.
        #
        #   obstruction   : |Delta s|<s_near and Delta d<d_lat -> colocated
        #   lane-blocking : s_role<Delta s<s_contest           -> leading
        #
        # Combining these relations into one interval was the modeling error.
        # Relays also belong to the first group: lane position, lateral offset,
        # and Delta s~0.
        # ------------------------------------------------------------------
        # [SCOPE-DECL] Apply obstruct/block/support role terms only to declared
        # pairs. They previously lacked role restrictions, so widening
        # S_CONTEST 0.25->0.65 and D_LAT_OBS 0.25->0.45 activated them for every
        # colocated pair, including controls. Measured control mean|W*| rose
        # 0.16->0.5279 (3.3x) and T1 Gini fell 0.74->0.4977, putting declared
        # and control effects on the same scale. The correction itself created
        # a third noise channel, explaining why correlation did not recover as
        # congestion fell. Continuous alpha_ij background coupling is sufficient.
        if is_declared:
            # ==============================================================
            # [PARTITION] Partition (ds,dd) with one lateral threshold,
            # D_LAT_OBS, with neither overlaps nor dead zones.
            #
            #   ds > S_ROLE                         -> leading: block
            #   |ds| <= S_ROLE or ds < -S_ROLE:
            #        dd <= D_LAT_OBS                 -> following: harmful obstruction
            #        dd > D_LAT_OBS                  -> yielding: beneficial support
            #
            # The legacy support condition used only ds<-S_ROLE. Being behind
            # in one dimension cannot distinguish yielding from following.
            # This environment's blocker follows closely from behind and harms
            # the collector, yet received the full support benefit.
            # Measured decomposition for zone 0 blocker->collector over 12 states:
            #     long -0.402 | contest +0.300 | obstruct +0.559
            #     block 0.000 | support -27.064 | safety 0.000
            # Support dominated by 50x and made phi(blocker)=+0.399, falsely
            # indicating that the blocker helped. Any W_SUPPORT>0 had the wrong
            # sign in zones with close following; this was a modeling error,
            # not a scale error.
            #
            # SGTP handles this correctly: c_block requires both Delta s in its
            # interval and small Delta d. The legacy model used two axes for
            # obstruction but one for support, creating the gap.
            #
            # Share D_LAT_OBS across both branches. Independent thresholds
            # create an unclassified gap that is difficult to diagnose.
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
        """List boolean preconditions that an agent can toggle in zone z.

        [ENABLE] This closed list is derived directly from ``step()``.

        Inspection of step() finds exactly two agent-controlled boolean writes:
            gate_open[z]           <- agent at zone_gate[z] plus OPEN
            low_priority_active[z] <- agent at zone_panel[z] plus OPEN
        resource_available is controlled by respawn/delivery, and carrying is
        collector state; neither is toggled by another agent.

        Return ``(site_pos, flag_name, has_consumer)`` entries. has_consumer
        records whether the flag actually blocks task progress. gate_open does
        because it gates pickup. low_priority_active does not: after C2 it is
        read only by _gate_ladder, whose delta branch is inactive under SGTP.
        The controller therefore has no causal reward channel. Preserve that
        fact rather than inventing one; adding controller enable cost without
        a mechanism would fit Phi to W*, violating anti-tautology discipline.
        """
        return [
            (self.zone_gate[z], "gate_open", True),
            (self.zone_panel[z], "low_priority_active", False),
        ]

    def _task_consumer_of_zone(self, z):
        """Return the agent whose task path is blocked by preconditions.

        [ENABLE] Derive this from the pickup/delivery mechanism in ``step()``,
        which targets ROLE_COLLECTOR. This is a mechanism lookup, not a declared
        influence-table lookup; gt_influence_by_ego does not compute reward.
        """
        return self.zone_role_agents[z][self.ROLE_COLLECTOR]

    def _enable_cost(self, i, j, z, sd):
        """Return the precondition term, the second Phi mechanism family.

        [ENABLE] SGTP contains only spatial influence between race vehicles.
        Resource flow has two mechanism classes:

            spatial        : occupancy / obstruction / yielding
                             -> blocker, relay   (obstruct/block/support)
            precondition   : toggle a flag required by j's task
                             -> gatekeeper, controller   (this term)

        Omitting the second class was the C2 design error. All eight of 20 sign
        mismatches were gatekeeper<->collector, consistent across four zones
        in both directions. Geometry-based Phi said harm from following, while
        W* said benefit from opening the pickup gate. This was systematic.

        This remains a direct A5 channel: one step, two agents, and no third
        agent. Phi simply contains two families. Define it by mechanism rather
        than role: any agent near a precondition site with a consumer receives
        the term.
        """
        if j != self._task_consumer_of_zone(z):
            return 0.0

        cost = 0.0
        for site, _flag, has_consumer in self.precondition_sites(z):
            if not has_consumer:
                continue
            d_site = self._dist(self.positions[i], site)
            # Smooth by site distance using the same 1/(1+d) form as other terms.
            cost -= self.W_ENABLE / (1.0 + float(d_site))
        return cost

    def _sgtp_influence_matrix(self):
        """Return w_ij for every same-zone pair with i!=j.

        Different zones do not share a task path, so their s coordinates are
        incomparable. This matches env_audit's control-pair domain. Continuous
        alpha gives every control pair nonzero coupling that decays with
        |Delta s|, providing the RC-3 background influence field independently
        of P3 congestion, as required by gate S4.

        Values map ``(i,j)`` to the influence of i on j's reward.
        """
        # [SCOPE-DECL] Build the declared-pair set once.
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
                    # [ENABLE] Add both mechanism families. A gatekeeper can
                    # open a gate while also obstructing a path; both are real.
                    if is_dec:
                        cost += self._enable_cost(i, j, _z, sd)
                    w[(i, j)] = -self.GAME_COST_WEIGHT * cost
        return w

    def close_coupling_pairs(self):
        """Count pairs with active close coupling for CSD.

        [C4] Active pairs share a zone and satisfy |Delta s|<S_CONTEST and
        Delta d<D_LAT. Close-Coupling Duration measures whether structural
        coupling is active independently of reward, following SGTP Table I.
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
        """Apply an unverified task offset to Frenet s.

        [C1 extension — UNVERIFIED] self.carrying is stored by zone, not agent.
        This method assumes the carrying agent is that zone's collector. Verify
        against actual role assignment before C2 use. The method is currently
        unused.
        """
        z = self.agent_zone[agent_id]
        if self.carrying.get(z, False):
            return float(np.clip(s_geom + carry_bonus, 0.0, 1.0))
        return float(s_geom)

    def _greedy_avoiding(self, agent_id, src, dst):
        """Move greedily toward dst while avoiding occupied stationary tiles.

        Without avoidance, a camping agent such as a relay staying on lane_a
        can permanently block another route. If the preferred direction is
        blocked, try the other distance-reducing axis. If both are blocked,
        move laterally to escape instead of waiting indefinitely.
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

        # If both preferred directions are blocked, move perpendicular to the
        # blocked axis before retreating. Retreating can create an infinite
        # forward/back oscillation in a one-tile corridor with a fixed blocker.
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
    # P0(d): common-random-number infrastructure shared with tiny_oracle_dig
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
        # Reset self.t/self.done before _maybe_structural_shift(). The earlier
        # P2<->P4 boundary guard treated those fields as evidence of a new
        # episode; calling shift first exposed stale previous-episode values and
        # tripped the assertion because of call order, not a real guard breach.
        self.t = 0
        self.done = False

        self.delta_phi_frobenius_structural_last = 0.0
        self.delta_phi_frobenius_behavioural_last = 0.0
        self._controlled_structural_event_active = False
        self._controlled_behavioral_event_active = False

        self._in_reset = True
        try:
            intervention = self._pending_factorial_intervention
            if intervention is None:
                self._maybe_structural_shift()
            else:
                if intervention["structural"]:
                    self._do_structural_shift()
                    self._controlled_structural_event_active = True
                if intervention["behavioral"]:
                    self._behaviour_override = intervention["behavior_mode"]
                    self._controlled_behavioral_event_active = True
                self._pending_factorial_intervention = None
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
        self._sgtp_delay_queues = {}
        self.episode_deliveries = 0
        # self.t/self.done were reset at the start, before the shift check.
        self.episode_count += 1
        return self._get_obs_all()

    # ============================================================
    # Behavioral drift (Section 3.1): preserve Phi and change execution only
    # ============================================================

    def _behaviour_mode(self):
        # RC-2: the override takes precedence so realized-Phi measurement can
        # isolate behavior from phase and episode_count.
        if self._behaviour_override is not None:
            return self._behaviour_override
        if self.behavioral_factor:
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
                # [P-4 FINAL DEBUG] Selfish behavior strongly reduces duty but
                # does not abandon it. Abandonment changes realized structure
                # by collapsing many deltas: behavioral dPhi~1.376 overwhelms
                # structural 0.360 and reverses T6. Paper Exp. 2 requires fixed
                # dependencies with moving policies: change how duties are
                # performed, not which agents influence each other.
                return self.OPEN if (self.t % 2 == 0) else self.STAY  # selfish
            return greedy(pos, gate)

        if role == self.ROLE_RELAY:
            if mode == "zigzag" and self.t % 4 == 0:
                return self.STAY
            if mode == "selfish":
                # [P-4] Continue toward the lane at half speed; do not abandon it.
                return self.STAY if (tuple(pos) == lane_a and self.t % 2 == 0) else greedy(pos, lane_a)
            return greedy(pos, lane_a)

        if role == self.ROLE_BLOCKER:
            if mode == "selfish":
                # [P-4] Slow down near the duty anchor without abandoning it.
                return self.STAY if tuple(pos) == checkpoint else greedy(pos, checkpoint)
            # RC-4: scripted policy must match the revised reward. The legacy
            # ``greedy(pos,self.positions[collector])`` optimized the removed
            # pursuit shaping. Retaining it makes the scripted baseline pursue
            # while the learner does not, invalidating comparison.
            if mode == "lazy":
                return self.STAY if tuple(pos) == checkpoint else greedy(pos, checkpoint)
            return greedy(pos, checkpoint)

        if role == self.ROLE_CONTROLLER:
            if mode == "lazy" or mode == "selfish":
                # [P-4] Remain at the panel but activate less often.
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

    def scripted_policy_distribution(self, agent_id):
        """Return the scripted action distribution without consuming RNG.

        This is used by the controlled H2 execution-policy adapter.  The
        deterministic roles preserve exactly the action selected by
        ``scripted_policy``.  Drifter branches that sample uniformly are
        represented as their analytic distribution, so observing the policy
        does not alter the random stream used by the environment.
        """
        agent_id = int(agent_id)
        role = self.agent_role[agent_id]
        if role != self.ROLE_DRIFTER:
            action = int(self.scripted_policy(agent_id))
            out = np.zeros(self.N_ACTIONS, dtype=np.float32)
            out[np.clip(action, 0, self.N_ACTIONS - 1)] = 1.0
            return out

        if agent_id % 3 == 0:
            action = self.STAY
            out = np.zeros(self.N_ACTIONS, dtype=np.float32)
            out[action] = 1.0
            return out
        if agent_id % 3 == 1:
            return np.full(self.N_ACTIONS, 1.0 / float(self.N_ACTIONS), dtype=np.float32)
        if self.t % 2 == 0:
            out = np.zeros(self.N_ACTIONS, dtype=np.float32)
            out[self.STAY] = 1.0
            return out
        return np.full(self.N_ACTIONS, 1.0 / float(self.N_ACTIONS), dtype=np.float32)

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
        # P3 flag only gates the ASSOCIATED REWARD PENALTY (channel [3]); it does
        # not gate whether collisions block movement, since that is env
        # physics, not the emergent reward channel the flag is about.
        for a in range(self.n_agents):
            if occupancy[tuple(proposed[a])] == 1:
                self.positions[a] = proposed[a]
            elif self.enable_congestion:
                # P3 collision is not attributable to one pair and belongs to channel [3].
                r_emergent[a] -= self.COLLISION_PENALTY

        self.last_actions = {a: int(actions[a]) for a in range(self.n_agents)}

        # ============================================================
        # RC-1 — REMOVED: two hand-coded reward blocks
        # ============================================================
        # (a) GATEKEEPER SIGHT subtracted 1.5 from collector reward when
        #     distance<=gatekeeper_sight. It was not dead code. A prior 0%
        #     diagnosis arose only because scripted layout kept the agents
        #     exactly three tiles apart. Under random policy it fired on 7.5%
        #     of collector steps, and 76% of those coincided with the declared
        #     gate_gk_collector channel. The +0.60 helping edge then had net
        #     reward 0.60-1.5=-0.90, opposite to design. As learning improved
        #     approach to the gate, firing increased and causal ground truth
        #     degraded during training. The sign was resolved explicitly:
        #     gatekeeper->collector is positive. The -1.5 surveillance/capture
        #     semantics contradicted gate-opening assistance and was removed;
        #     gate_gk_collector retains +0.60.
        #
        # (b) BLOCKER PRESSURE subtracted 2.5 at distance<=2 and another 0.7
        #     at distance<=1. It fired on 52.1% of collector steps and totaled
        #     -397.2 per episode, 4.2x all declared channels combined. Its
        #     carrying-and-distance condition exactly duplicated
        #     gate_blocker_collector.
        #
        # (c) Subtracting 0.05 when the collector occupied a closed gate tile
        #     was disguised multi-agent r_solo because gate_open depends on the
        #     gatekeeper action. Gate value already belongs to
        #     phi(gatekeeper->collector).
        #
        # Restored invariant: every reward depending on two or more agents must
        # pass through w_ij=phi_ij*delta_ij(s) and appear in info["w_by_pair"],
        # except pair-unattributable r_emergent channel [3], which is bounded by
        # MAX_EMERGENT_MAGNITUDE.

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
                # RC-4: delivery is the only task metric uncontaminated by
                # zero-sum reward components. Population mean reward is invalid
                # for this adversarial environment; use deliveries and
                # role-specific rewards.
                deliveries_this_step += 1

        # With P3 disabled, channel [3] is not evaluated; it is not computed and
        # then zeroed. r_emergent remains all-zero from initialization.
        if self.enable_congestion:
            self._apply_emergent_congestion(r_emergent)

        # ---- r_solo depends only on the ego's own state and action. ----
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

            # RC-4 removed the blocker reward 0.03/(distance-to-collector+1).
            # It was the blocker's only clear gradient and directly rewarded
            # proximity while the collector lost 2.5 for the same event, an
            # 83x asymmetry. Structural influence became an agent objective
            # rather than an environment property, so audit measured deliberate
            # blocker behavior. Replace it with a position-only solo duty like
            # relay/gatekeeper/controller; blocker reward no longer depends on
            # collector position.
            blocker_duty = (
                self.zone_lane_b[z]
                if self.active_lane[z] == "B"
                else self.zone_checkpoint[z]
            )
            if tuple(self.positions[blocker]) == blocker_duty:
                rewards[blocker] += 0.05

            if tuple(self.positions[controller]) == self.zone_panel[z]:
                rewards[controller] += 0.05

        # ---- Channel [2]: inter-agent influence. ----
        deltas = self._compute_deltas()

        if self.use_sgtp_phi:
            # [C2] Continuous Phi replaces role lookup and binary gates. w_ij
            # is a smooth function of (Delta s,Delta d) for every same-zone pair.
            #
            # Retain gt_influence_by_ego/declared_pairs only as scoring labels
            # for Core F1 and corr(Phi,W*); they do not compute reward. This
            # removes the T6 tautology because realized Phi now depends on the
            # policy-induced state distribution and changes under behavioral drift.
            instantaneous_w = self._sgtp_influence_matrix()
            instantaneous_w = self._add_sgtp_action_transport_signal(
                instantaneous_w, actions
            )
            w_by_pair = self._apply_sgtp_delays(instantaneous_w)
            for (i, j), w in w_by_pair.items():
                rewards[j] += w
        else:
            # Preserve the legacy lookup path for the no-SGTP-phi ablation.
            w_by_pair = {}
            for (i, j, gate_name, z) in self.declared_pairs:
                phi = self.gt_influence_by_ego[j].get(i, 0.0)
                d = deltas.get((i, j), 0.0)
                w = phi * d
                w_by_pair[(i, j)] = w
                rewards[j] += w

        # ---- Stochastic resource respawn; purpose is listed in NOISE_PURPOSES. ----
        for z in range(self.n_zones):
            if (not self.resource_available[z]) and (not self.carrying[z]):
                # [ZONE-ASYM] Change only the threshold. Keep one _draw_noise
                # call per zone-step so CRN structure is unchanged.
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
                # RC-4: valid metric for an adversarial environment. mean(rewards)
                # mixes both sides of zero-sum edges into a constant plus noise,
                # removing the signal of interest.
                "reward_by_role": self._aggregate_reward_by_role(rewards),
                "deliveries_step": deliveries_this_step,
                "episode_deliveries": self.episode_deliveries,
                "gt_core_by_ego": copy.deepcopy(self.gt_core_by_ego),
                "gt_influence_by_ego": copy.deepcopy(self.gt_influence_by_ego),
                "delta_by_pair": {f"{i}->{j}": v for (i, j), v in deltas.items()},
                "w_by_pair": {f"{i}->{j}": v for (i, j), v in w_by_pair.items()},
                "r_emergent": list(r_emergent),
            # [C4] Active coupling-pair count used for CSD in runner/audit.
            "n_close_coupling_pairs": self.close_coupling_pairs(),
                "active_lane": dict(self.active_lane),
                "current_phase": self.current_phase,
                "mode": self.mode,
                "tier_separation_ratio": self.tier_separation_ratio(),
                "delta_phi_frobenius_structural": self.delta_phi_frobenius_structural_last,
                "delta_phi_frobenius_behavioural": self.delta_phi_frobenius_behavioural_last,
                "controlled_structural_shift": int(
                    self._controlled_structural_event_active
                ),
                "controlled_behavioral_shift": int(
                    self._controlled_behavioral_event_active
                ),
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
    # Oracle rollout support aligned with the tiny_oracle_dig.py interface
    # ============================================================

    def _rollout_reward_sequence_from_current_state(
        self,
        forced=None,
        horizon=None,
        continuation_policy=None,
    ):
        """Roll out per-step rewards under a declared reference continuation.

        ``continuation_policy`` is the fixed reference policy :math:`rho` for
        a standardized response measurement.  It is held fixed across the
        intervention and reference branches; the policy may observe the state,
        but its definition does not depend on the intervened action.  The
        caller owns snapshot restoration and common-random-number setup.
        """
        if horizon is None:
            horizon = self.causal_horizon
        if continuation_policy is None:
            continuation_policy = self.scripted_policy

        rewards_by_lag = []
        done = False
        for local_t in range(int(horizon)):
            if done:
                break
            actions = [int(continuation_policy(agent)) for agent in range(self.n_agents)]
            if forced is not None:
                agent_id, intervention_action, forced_step = forced
                if local_t == int(forced_step):
                    actions[int(agent_id)] = int(intervention_action)
            _, rewards, done, _ = self.step(
                actions,
                return_obs=False,
                return_info=False,
            )
            rewards_by_lag.append(np.asarray(rewards, dtype=np.float64))

        out = np.zeros((int(horizon), self.n_agents), dtype=np.float64)
        if rewards_by_lag:
            out[:len(rewards_by_lag)] = np.stack(rewards_by_lag, axis=0)
        return out

    def compute_oracle_lag_response_from_current_state(
        self,
        ego_id,
        agent_j,
        intervention_action,
        horizon=None,
        n_trials=1,
        forced_step=0,
        continuation_policy=None,
        crn_seed=None,
        discount=0.95,
    ):
        """Measure the lag-specific causal response under fixed policy ``rho``.

        This is the latency oracle for the optional Paper-A contribution.  It
        compares an intervention only at ``forced_step`` with a reference
        rollout, then applies the same declared continuation policy to both
        branches.  Unlike the former horizon/sign-flip diagnostic, it returns
        the impulse-response vector directly.  The caller must establish an
        oracle gate before this quantity is introduced into a learned proxy.

        The returned ``onset_lag``, ``peak_lag``, and ``centre_of_mass_lag``
        are summaries of absolute response mass; ``None`` denotes a response
        with no measurable mass rather than an inferred zero-latency effect.
        """
        if horizon is None:
            horizon = self.causal_horizon
        horizon = int(horizon)
        ego_id = int(ego_id)
        agent_j = int(agent_j)
        forced_step = int(forced_step)
        if not 0 <= forced_step < horizon:
            raise ValueError("forced_step must lie in [0, horizon)")
        if self.t + horizon > self.max_steps:
            raise AssertionError(
                "lag-response oracle window crosses an episode boundary; "
                "sample a state with t + horizon <= max_steps"
            )

        snapshot = self.clone_state()
        accumulated = np.zeros(horizon, dtype=np.float64)
        try:
            for trial in range(max(1, int(n_trials))):
                crn_rng = np.random.RandomState(
                    ((crn_seed if crn_seed is not None else self.seed) * 65537 + trial)
                    % (2 ** 32)
                )
                buffer = self._make_crn_buffer(horizon, crn_rng)

                self.restore_state(snapshot)
                self.set_noise_buffer(buffer)
                base = self._rollout_reward_sequence_from_current_state(
                    horizon=horizon,
                    continuation_policy=continuation_policy,
                )[:, ego_id]

                self.restore_state(snapshot)
                self.set_noise_buffer(buffer)
                intervened = self._rollout_reward_sequence_from_current_state(
                    forced=(agent_j, intervention_action, forced_step),
                    horizon=horizon,
                    continuation_policy=continuation_policy,
                )[:, ego_id]
                accumulated += intervened - base
        finally:
            self.clear_noise_buffer()
            self.restore_state(snapshot)

        response = accumulated / float(max(1, int(n_trials)))
        mass = np.abs(response)
        total_mass = float(mass.sum())
        if total_mass <= 1e-12:
            onset_lag = None
            peak_lag = None
            centre_of_mass_lag = None
        else:
            # Use the same relative onset estimand as the learned proxy and
            # the standalone oracle validator. Keeping the constants shared
            # prevents a silent 5%-vs-10% paper/code drift.
            onset_threshold = max(
                LATENCY_ONSET_ABS_FLOOR,
                LATENCY_ONSET_FRACTION * float(mass.max()),
            )
            onset_lag = int(np.flatnonzero(mass >= onset_threshold)[0])
            peak_lag = int(np.argmax(mass))
            centre_of_mass_lag = float(
                np.dot(np.arange(horizon, dtype=np.float64), mass) / total_mass
            )

        discount_weights = float(discount) ** np.arange(
            horizon, dtype=np.float64
        )
        return {
            "per_lag_response": response.astype(np.float64),
            "discounted_response": float(np.dot(discount_weights, response)),
            "onset_lag": onset_lag,
            "peak_lag": peak_lag,
            "centre_of_mass_lag": centre_of_mass_lag,
            "response_mass": total_mass,
            "horizon": horizon,
            "forced_step": forced_step,
        }

    def rollout_from_current_state(self, forced=None, horizon=None):
        if horizon is None:
            horizon = self.causal_horizon

        # Guard for the P2<->P4 trap: a forced rollout cannot start where its
        # H-step window reaches the next episode boundary. P4 shifts can occur
        # only at that boundary. Starting near episode end could otherwise
        # measure W* across two Phi structures after reset and shift. Enforce
        # this with a runtime assertion rather than best-effort documentation.
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

    def sample_state_bank(
        self, n_states=24, burn_in=3, bank_seed=None, min_remaining_steps=0
    ):
        """Sample a state bank with optional common random numbers.

        ``min_remaining_steps`` is a hard oracle-safety contract: every saved
        state satisfies ``t + min_remaining_steps <= max_steps``.  This is
        required by H-step clone-state oracles and prevents state banks from
        containing late-episode states that cannot support the requested
        intervention horizon.

        RC-2: bank_seed temporarily fixes RNG and then restores its old state.
        Paired realized-Phi measurements then differ in exactly one variable,
        lane or behavior mode. Otherwise the norm measures sampling noise.
        """
        saved_rng_state = self.rng.get_state() if bank_seed is not None else None
        if bank_seed is not None:
            self.rng.seed(bank_seed)   # Seed in place; preserve the RNG object.

        try:
            n_states = int(n_states)
            burn_in = int(burn_in)
            min_remaining_steps = int(min_remaining_steps)
            if n_states <= 0 or burn_in < 0 or min_remaining_steps < 0:
                raise ValueError("invalid state-bank sampling arguments")
            if min_remaining_steps > int(self.max_steps):
                raise ValueError(
                    "min_remaining_steps cannot exceed the episode horizon"
                )
            bank = []
            self.reset()
            for _ in range(n_states):
                # Reset *before* burn-in if the requested oracle horizon would
                # cross the episode boundary.  Checking only after burn-in is
                # insufficient because a done/reset can otherwise leave the
                # state bank distribution dependent on accidental wraparound.
                if int(self.t) + burn_in + min_remaining_steps > int(self.max_steps):
                    self.reset()
                for _ in range(burn_in):
                    acts = [self.scripted_policy(i) for i in range(self.n_agents)]
                    _, _, done, _ = self.step(acts, return_obs=False, return_info=False)
                    if done:
                        self.reset()
                if int(self.t) + min_remaining_steps > int(self.max_steps):
                    self.reset()
                if int(self.t) + min_remaining_steps > int(self.max_steps):
                    raise RuntimeError("failed to sample an oracle-safe state")
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
        """Measure one intervention on j for all egos simultaneously.

        [FIX-O1] ``rollout_from_current_state`` already returns an agent-return
        vector. The legacy code retained only ``[ego_id]``, discarded 23 values,
        and repeated the rollout per ego. Keeping the vector gives the entire
        W*_{.,j} column from one rollout pair without approximation.

        One core refresh costs one base rollout plus N alternatives instead of
        N*(N-1)*(|A|+1). Return an n_agents vector of mean_trials(alt-base),
        preserving signs for the caller to interpret.
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
        """Run the corrected P0 intervention oracle for OmniArena P1-P4.

        The design follows corrected items 4.1a-d in
        ``TinyOracleDIG.compute_oracle_influence_from_current_state``.
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
        """Verify P4/3.1 Phi invariance across all behavioral-drift phases."""
        assert self.mode == "behavioral_drift", "valid only in behavioral_drift mode"
        phi0 = copy.deepcopy(self.gt_influence_by_ego)
        for _ in range(n_phases * self.phase_length):
            self.reset()
            done = False
            while not done:
                acts = [self.scripted_policy(i) for i in range(self.n_agents)]
                _, _, done, _ = self.step(acts, return_obs=False, return_info=False)
        phi1 = self.gt_influence_by_ego
        d = self._delta_phi_frobenius(phi0, phi1)
        assert d == 0.0, f"Phi changed during behavioral_drift: ||dPhi||_F={d}"
        return True
