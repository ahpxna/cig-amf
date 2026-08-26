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
