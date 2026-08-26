from pathlib import Path

import numpy as np
import pytest

from models.structural_proxy import LocalCounterfactualProxyEnsemble
from scripts.validate_paper_b import (
    _validate_candidate_recall_protocol_horizon,
    _validate_candidate_recall_protocol_version,
)
from utils.paper_contracts import (
    PAPER_B_CANDIDATE_RECALL_PROTOCOL_VERSION,
    PAPER_B_SELECTOR_ORACLE_HORIZON,
)


def _strict_proxy():
    return LocalCounterfactualProxyEnsemble(
        obs_dim=2,
        action_dim=2,
        core_dim=1,
        periph_dim=1,
        belief_dim=1,
        n_ensemble=1,
        n_horizons=1,
        effect_mode="signed_policy_contrast",
        use_doubly_robust=False,
        use_vmap_ensemble=False,
        pair_feat_dim=1,
        context_item_dim=1,
        strict_typed_rows=True,
        device="cpu",
    )


def _base_row():
    return dict(
        ego_id=0,
        neighbor_id=1,
        obs_i=np.zeros(2, dtype=np.float32),
        action_i=0,
        observed_action_j=0,
        z_core_excl_j=np.zeros(1, dtype=np.float32),
        m_periph_excl_j=np.zeros(1, dtype=np.float32),
        belief_summary=np.zeros(1, dtype=np.float32),
        target_return_h=0.0,
        pair_feat=np.zeros(1, dtype=np.float32),
        context_items=np.zeros((1, 1), dtype=np.float32),
        context_mask=np.ones(1, dtype=np.float32),
    )


def test_paper_b_selector_oracle_horizon_is_frozen_to_one():
    assert PAPER_B_SELECTOR_ORACLE_HORIZON == 1


def test_paper_b_validator_rejects_noncanonical_candidate_horizon():
    assert _validate_candidate_recall_protocol_horizon({"horizon": "1"}) == 1
    for invalid in (
        {}, {"horizon": 0}, {"horizon": -1}, {"horizon": 1.5},
        {"horizon": True}, {"horizon": 8},
    ):
        with pytest.raises(ValueError, match="horizon"):
            _validate_candidate_recall_protocol_horizon(invalid)


def test_paper_b_validator_rejects_stale_candidate_oracle_protocol():
    assert _validate_candidate_recall_protocol_version({
        "protocol_version": PAPER_B_CANDIDATE_RECALL_PROTOCOL_VERSION,
    }) == PAPER_B_CANDIDATE_RECALL_PROTOCOL_VERSION
    assert _validate_candidate_recall_protocol_version({}) is None
    for invalid in ({"protocol_version": "old-candidate-oracle-v0"},):
        with pytest.raises(ValueError, match="protocol version"):
            _validate_candidate_recall_protocol_version(invalid)


def test_scaling_default_uses_shared_selector_oracle_horizon():
    source = Path("scripts/run_paper_b_scaling.py").read_text(encoding="utf-8")
    assert "default=int(PAPER_B_SELECTOR_ORACLE_HORIZON)" in source
    assert 'default=int(RE.default_cfg()["causal_horizon"])' not in source


def test_run_all_uses_pytest_as_canonical_regression_collector():
    source = Path("scripts/run_all.sh").read_text(encoding="utf-8")
    assert '"$CIG_PYTHON" -m pytest -q tests' in source
    assert "-m unittest discover" not in source


def test_run_all_defaults_candidate_recall_horizon_from_shared_contract():
    source = Path("scripts/run_all.sh").read_text(encoding="utf-8")
    assert "from utils.paper_contracts import PAPER_B_SELECTOR_ORACLE_HORIZON" in source
    assert 'CIG_PAPER_B_CANDIDATE_RECALL_HORIZON="${CIG_PAPER_B_CANDIDATE_RECALL_HORIZON:-$PAPER_B_SELECTOR_ORACLE_HORIZON}"' in source


def test_strict_proxy_rows_fail_closed_when_typed_contract_is_missing():
    proxy = _strict_proxy()
    with pytest.raises(ValueError, match="complete typed action-time treatment contract"):
        proxy.add_sample(**_base_row())


def test_strict_proxy_rows_accept_complete_typed_contract():
    proxy = _strict_proxy()
    row = _base_row()
    row.update(
        valid_action_mask=np.asarray([1, 1], dtype=bool),
        target_action_proposed=0,
        target_action_executed=0,
        target_pi=np.asarray([0.8, 0.2], dtype=np.float32),
        target_q=np.asarray([0.5, 0.5], dtype=np.float32),
        target_b=np.asarray([0.77, 0.23], dtype=np.float32),
        target_epsilon=0.1,
    )
    proxy.add_sample(**row)
    assert len(proxy.buffer) == 1
