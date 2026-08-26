"""Frozen cross-script scientific contracts shared by confirmatory harnesses.

Keep values that define an estimand or experiment protocol in one importable
module so allocation, scaling, validators, and shell orchestration cannot drift
independently.
"""

# Paper B confirmatory allocation deliberately uses the one-step structural
# response surface.  Long-horizon/delayed-path claims are evaluated by the
# separate Paper-A latency panel with an explicit continuation regime.
PAPER_B_SELECTOR_ORACLE_HORIZON = 1
