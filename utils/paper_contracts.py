"""Frozen cross-script scientific contracts shared by confirmatory harnesses.

Keep values that define an estimand or experiment protocol in one importable
module so allocation, scaling, validators, and shell orchestration cannot drift
independently.
"""

# Paper B confirmatory allocation deliberately uses the one-step structural
# response surface.  Long-horizon/delayed-path claims are evaluated by the
# separate Paper-A latency panel with an explicit continuation regime.
PAPER_B_SELECTOR_ORACLE_HORIZON = 1

# Bump this whenever the candidate-recall oracle changes its state sampling,
# top-k rule, continuation regime, or action-mask semantics.  The manifest is
# a trust boundary: matching H=1 alone does not prove an old artifact used the
# same selector-oracle estimand.
PAPER_B_CANDIDATE_RECALL_PROTOCOL_VERSION = (
    "paper_b_candidate_recall_topk_crn_v1"
)

# Paper B H2b evaluates a promotion transient over one frozen decision window.
# This is intentionally independent of a run's end-to-end episode budget.
PAPER_B_PROMOTION_WINDOW_STEPS = 10

# Paper B's adaptive-budget result is an exact grid experiment: every fixed
# integer budget in [k_min, k_max] is observed, while one comparator is chosen
# solely from disjoint-pilot Adaptive-K representation cost.
PAPER_B_ADAPTIVE_BUDGET_PROTOCOL_VERSION = (
    "paper_b_adaptive_budget_v3_exact_grid"
)
PAPER_B_ADAPTIVE_MATCHING_RULE = (
    "nearest_integer_pilot_mean_core_cost_v1"
)

# External generalisation is evaluated only after matched training has ended.
# Training-time return is not a fair model comparator because CIG-AMF's
# causal-data acquisition deliberately executes epsilon-forced actions while
# PureMeanField does not.  G8 therefore consumes fresh paired frozen-policy
# evaluation episodes with learning/representation updates and forcing off.
EXTERNAL_GENERALIZATION_PROTOCOL_VERSION = (
    "external_matched_training_v4_frozen_policy_eval"
)
EXTERNAL_EVAL_SEED_OFFSET = 1_000_003
EXTERNAL_G8_MIN_EVAL_EPISODES = 20
EXTERNAL_G8_BOOTSTRAP_SEED = 4800
