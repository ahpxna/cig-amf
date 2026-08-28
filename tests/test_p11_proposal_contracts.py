import unittest
import json
import os
import tempfile
from types import SimpleNamespace
from unittest import mock

from models.disturbance_contracts import (
    DisturbanceRegime,
    PairedDisturbanceRecord,
    adjudicate_d0,
)
from models.query_contracts import QueryCard, QueryKind, QueryOracleRegistry
from models.belief_layer import BayesLightBeliefState
from scripts.run_paper_b_periphery import (
    _cfg,
    _common_pretrain_cfg,
    _seed_oracle_core,
)


class P11ProposalContractTests(unittest.TestCase):
    def test_scientific_precheck_main_forwards_oracle_trials_to_allocation(self):
        from scripts import run_scientific_prechecks as prechecks

        with tempfile.TemporaryDirectory() as root:
            experiment0 = os.path.join(root, "experiment0.json")
            out = os.path.join(root, "prechecks.json")
            with open(experiment0, "w", encoding="utf-8") as handle:
                json.dump({"required_gate_pass": True}, handle)
            allocation = mock.Mock(return_value={
                "oracle_C_minus_random": 1.0,
                "oracle_C_minus_mean_field": 1.0,
                "oracle_C_minus_absD": 1.0,
            })
            with mock.patch.object(prechecks, "_oracle_c_d_bank", return_value={
                "disagreement_fraction": 1.0,
                "disagreement_count": 1,
                "c_capture_advantage_on_disagreement_mean": 1.0,
            }), mock.patch.object(
                prechecks, "_oracle_allocation_value_seed", allocation
            ), mock.patch.object(
                prechecks, "_oracle_allocation_decision_fidelity_seed",
                return_value={
                    "oracle_C_minus_random_logit_fidelity_error": 1.0,
                    "oracle_C_minus_random_value_fidelity_error": 1.0,
                    "oracle_C_minus_random_action_agreement": 1.0,
                },
            ), mock.patch.object(prechecks, "_forced_only_h1", return_value={
                "q_spearman": 1.0,
                "q_normalized_rmse": 0.0,
                "q_nonconstant_surfaces": 1.0,
                "true_null_C_count": 1,
                "true_null_C_false_positives": 0,
                "capacity_prediction_threshold": 0.1,
                "forced_action_coverage_gate_pass": True,
                "forced_actions_seen": 2,
                "proxy_action_count": 2,
                "proxy_forced_only_training": True,
                "realised_forcing_rate": 0.5,
                "n_forced_proxy_samples": 2,
                "protocol_gate_pass": True,
            }), mock.patch.object(prechecks, "_g3_seed", return_value={
                "C_separation": 1.0,
                "beta_behavioral_absD": 1.0,
            }):
                payload = prechecks.main([
                    "--experiment0", experiment0,
                    "--seeds", "17",
                    "--protocol-mode", "quick",
                    "--out", out,
                    "--n-agents", "4",
                    "--core-k", "1",
                    "--oracle-trials", "7",
                    "--allocation-episodes", "1",
                    "--allocation-pretrain-episodes", "1",
                    "--allocation-state-count", "1",
                    "--allocation-max-steps", "1",
                    "--allocation-core-refresh-every", "1",
                    "--allocation-final-window", "1",
                ])
        self.assertEqual(payload["overall_status"], "SMOKE_ONLY")
        self.assertEqual(allocation.call_args.kwargs["trials"], 7)

    def test_periphery_panel_freezes_oracle_fixed_allocation(self):
        cfg = _cfg(seed=0, core_budget=2)
        self.assertEqual(cfg["core_selection_mode"], "oracle_capacity")
        self.assertTrue(cfg["semantic_router_frozen"])
        self.assertFalse(cfg["freeze_graph_updates"])
        self.assertTrue(cfg["force_graph_update_every_episode"])

    def test_common_pretrain_does_not_require_future_oracle_table(self):
        cfg = _common_pretrain_cfg(seed=0, core_budget=2)
        self.assertEqual(cfg["core_selection_mode"], "structural_capacity")
        self.assertEqual(cfg["min_core_size"], 2)
        self.assertEqual(cfg["max_core_size"], 2)
        self.assertTrue(cfg["force_graph_update_every_episode"])

    def test_full_explicit_seed_overrides_restored_k_core_limits(self):
        belief = BayesLightBeliefState(
            ego_id=0,
            neighbor_ids=[1, 2, 3],
            min_core_size=1,
            max_core_size=1,
            adaptive_k=False,
            adaptive_k_min=1,
        )
        belief.set_fixed_core([1])
        runner = SimpleNamespace(
            belief_modules={0: belief},
            pair_rel_module=SimpleNamespace(reconcile_core_sets=mock.Mock()),
        )
        _seed_oracle_core(
            runner,
            core_budget=1,
            table={0: {1: 3.0, 2: 2.0, 3: 1.0}},
            full_explicit=True,
        )
        self.assertEqual(set(belief.get_core_set()), {1, 2, 3})
        self.assertEqual(belief.min_core_size, 3)
        self.assertEqual(belief.max_core_size, 3)
        belief.reconcile_neighbors([1, 2, 3])
        self.assertEqual(set(belief.get_core_set()), {1, 2, 3})
        self.assertEqual(belief.min_core_size, 3)
        self.assertEqual(belief.max_core_size, 3)

    def test_query_card_rejects_fractional_and_boolean_budgets(self):
        kwargs = dict(
            query_id="response",
            kind=QueryKind.RESPONSE_RANGE,
            estimand_key="e",
            support_key="s",
            candidate_ids=("r0", "r1"),
            generator_sha256="1" * 64,
            oracle_source_sha256="2" * 64,
        )
        with self.assertRaises(TypeError):
            QueryCard(budget=1.5, **kwargs)
        with self.assertRaises(TypeError):
            QueryCard(budget=True, **kwargs)

    def _card(self, query_id, kind, *, support="support-a"):
        return QueryCard(
            query_id=query_id,
            kind=kind,
            estimand_key="history/report/h1/rho-fixed",
            support_key=support,
            candidate_ids=("r0", "r1"),
            budget=1,
            generator_sha256="1" * 64,
            oracle_source_sha256="2" * 64,
        )

    def test_query_registry_keeps_response_and_information_distinct(self):
        registry = QueryOracleRegistry()
        registry.register(
            self._card("response", QueryKind.RESPONSE_RANGE),
            lambda: {"r0": 1.0, "r1": 1.0},
        )
        registry.register(
            self._card("information", QueryKind.INFORMATION),
            lambda: {"r0": 0.0, "r1": 3.0},
        )
        registry.require_kinds(QueryKind.RESPONSE_RANGE, QueryKind.INFORMATION)
        # The response tie is deterministic, while the information query has
        # a different winner.  A single untyped score cannot impersonate both.
        self.assertEqual(registry.evaluate("response").selected, ("r0",))
        self.assertEqual(registry.evaluate("information").selected, ("r1",))

    def test_query_registry_fails_closed_on_candidate_mismatch(self):
        registry = QueryOracleRegistry()
        registry.register(
            self._card("response", QueryKind.RESPONSE_RANGE),
            lambda: {"r0": 1.0},
        )
        with self.assertRaises(ValueError):
            registry.evaluate("response")

    def test_query_fingerprint_changes_with_support(self):
        first = self._card("response", QueryKind.RESPONSE_RANGE, support="a")
        second = self._card("response", QueryKind.RESPONSE_RANGE, support="b")
        self.assertNotEqual(first.fingerprint(), second.fingerprint())

    def _disturbance_rows(self, live_b=0.7):
        rows = []
        for regime, values in {
            DisturbanceRegime.RESET: (0.0, 0.0),
            DisturbanceRegime.FROZEN_POLICY: (0.1, 0.4),
            DisturbanceRegime.LIVE_LEARNING: (0.2, live_b),
        }.items():
            for replicate in (0, 1):
                for arm, value in zip(("a", "b"), values):
                    rows.append(PairedDisturbanceRecord(
                        regime=regime,
                        arm=arm,
                        replicate=replicate,
                        target_key="fixed-baseline-v1",
                        immediate_cost=1.0,
                        future_state_distance=value,
                        future_response_shift=0.0,
                        future_policy_distance=0.0,
                    ))
        return rows

    def _d0_kwargs(self):
        return dict(
            metric_thresholds={
                "future_state_distance": 0.2,
                "future_response_shift": 1.0,
                "future_policy_distance": 1.0,
            },
            minimum_arm_spreads={
                "future_state_distance": 0.2,
                "future_response_shift": 1.0,
                "future_policy_distance": 1.0,
            },
            immediate_cost_tolerance=0.0,
        )

    def test_d0_gate_requires_nontrivial_arm_heterogeneity(self):
        result = adjudicate_d0(
            self._disturbance_rows(),
            **self._d0_kwargs(),
        )
        self.assertEqual(result.status, "CONFIRMED")

    def test_d0_gate_fails_closed_without_live_regime(self):
        rows = [
            row for row in self._disturbance_rows()
            if row.regime != DisturbanceRegime.LIVE_LEARNING
        ]
        result = adjudicate_d0(
            rows,
            **self._d0_kwargs(),
        )
        self.assertEqual(result.status, "FAILED")
        self.assertTrue(any("missing regimes" in reason for reason in result.reasons))

    def test_d0_gate_fails_closed_on_incomplete_regime_arm_cells(self):
        rows = [
            row for row in self._disturbance_rows()
            if not (row.regime == DisturbanceRegime.FROZEN_POLICY and row.arm == "b")
        ]
        result = adjudicate_d0(rows, **self._d0_kwargs())
        self.assertEqual(result.status, "FAILED")
        self.assertTrue(any("incomplete regime×arm" in reason for reason in result.reasons))

    def test_d0_gate_rejects_invalid_thresholds(self):
        kwargs = self._d0_kwargs()
        kwargs["metric_thresholds"]["future_state_distance"] = float("nan")
        with self.assertRaises(ValueError):
            adjudicate_d0(self._disturbance_rows(), **kwargs)


if __name__ == "__main__":
    unittest.main()
