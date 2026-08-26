"""Focused regression tests for the H3 protocol and slot diagnostics."""

import unittest

import numpy as np
import torch

import run_experiment as RE
from models.influence_signature import (
    CausalPairSignal,
    ROLE_UNCERTAIN,
    ROLE_BENEFICIAL,
    ROLE_HARMFUL,
    ROLE_NEUTRAL,
)
from models.peripheral_memory import (
    FULL_ITEM_DIM,
    ITEM_LATENCY_NORM,
    ITEM_LATENCY_VALID,
    PeripheralMultiMemory,
)
from models.single_mean_memory import SingleMeanPeripheral
from runners.final_runner import FinalCIGAMFRunner
from runners.h3_ablation_runner import H3NoMultiMemoryRunner
from scripts import run_h3_slots as H3


def role_items():
    """Two examples for each semantic role in the full 13D layout."""
    items = np.zeros((8, FULL_ITEM_DIM), dtype=np.float32)
    items[:, 0] = np.arange(8) % 4
    items[:, 1:6] = np.asarray(
        [
            [0.50, 0.52, 0.05, 0.02, 0.01],
            [0.35, 0.38, 0.08, 0.03, 0.02],
            [0.50, -0.52, 0.05, 0.02, 0.01],
            [0.35, -0.38, 0.08, 0.03, 0.02],
            [0.00, 0.01, 0.05, 0.01, 0.01],
            [0.01, 0.02, 0.08, 0.02, 0.01],
            [0.40, 0.50, 0.05, 0.60, 0.50],
            [0.40, -0.50, 0.05, 0.55, 0.45],
        ],
        dtype=np.float32,
    )
    items[:, 6] = 0.25
    items[:, 7] = np.linspace(0.0, 1.0, 8)
    items[:, 8] = 1.0
    items[:, 9:] = np.linspace(-0.5, 0.5, 8)[:, None]
    return items


class FakeEnv:
    n_agents = 3
    grid_size = 10
    n_zones = 2
    positions = [[0, 0], [2, 3], [4, 5]]
    agent_zone = [0, 0, 1]
    last_actions = [0, 1, 2]


class TestSignatureInputProtocol(unittest.TestCase):
    def test_full_signature_mapping_builds_13d_rows(self):
        module = PeripheralMultiMemory(
            action_dim=4,
            num_slots=6,
            require_full_signature=True,
            allow_legacy_items=False,
        )
        belief = {
            1: {"mu_bar": 99.0, "sigma_bar": 99.0, "p_core": 0.2},
            2: {"mu_bar": 99.0, "sigma_bar": 99.0, "p_core": 0.3},
        }
        signatures = {
            1: [0.2, 0.3, 0.1, 0.04, 0.05],
            2: [-0.4, 0.5, 0.2, 0.06, 0.07],
        }
        rows = module.build_inputs(
            ego_id=0,
            peripheral_ids=[1, 2],
            env=FakeEnv(),
            belief_state=belief,
            influence_signatures=signatures,
        )
        self.assertEqual(rows.shape, (2, FULL_ITEM_DIM))
        np.testing.assert_allclose(rows[0, 1:6], signatures[1])
        self.assertEqual(module.last_signature_source, "full_profile")
        self.assertEqual(module.signature_full_items_seen, 2)
        self.assertEqual(module.signature_legacy_items_seen, 0)

    def test_typed_latency_is_retained_with_a_distinct_validity_mask(self):
        module = PeripheralMultiMemory(
            action_dim=4, num_slots=6, require_full_signature=True,
            allow_legacy_items=False,
        )
        belief = {1: {"mu_bar": 0.0, "sigma_bar": 0.0, "p_core": 0.2}}

        def signal(*, latency_cm, latency_valid, peak, onset):
            return CausalPairSignal(
                ego_id=0, target_id=1, timestamp=4, structure_regime_id=0,
                capacity=0.2, direction=-0.1, sigma_capacity=0.03,
                sigma_direction=0.04, context_variation=0.05,
                context_valid=True, latency_onset=onset, latency_peak=peak,
                latency_cm=latency_cm, latency_valid=latency_valid,
                latency_onset_valid=latency_valid, support_valid=True,
                valid_action_count=3, latency_horizon=5,
            )

        zero_lag = module.build_inputs(
            ego_id=0, peripheral_ids=[1], env=FakeEnv(), belief_state=belief,
            causal_pair_signals={1: signal(
                latency_cm=0.0, latency_valid=True, peak=0, onset=0,
            )},
        )
        unsupported = module.build_inputs(
            ego_id=0, peripheral_ids=[1], env=FakeEnv(), belief_state=belief,
            causal_pair_signals={1: signal(
                latency_cm=0.0, latency_valid=False, peak=-1, onset=-1,
            )},
        )
        self.assertEqual(float(zero_lag[0, ITEM_LATENCY_NORM]), 0.0)
        self.assertEqual(float(zero_lag[0, ITEM_LATENCY_VALID]), 1.0)
        self.assertEqual(float(unsupported[0, ITEM_LATENCY_NORM]), 0.0)
        self.assertEqual(float(unsupported[0, ITEM_LATENCY_VALID]), 0.0)

    def test_strict_mode_rejects_missing_tracker_signature(self):
        module = PeripheralMultiMemory(
            action_dim=4,
            num_slots=4,
            require_full_signature=True,
            allow_legacy_items=False,
        )
        belief = {
            1: {"mu_bar": 0.2, "sigma_bar": 0.1, "p_core": 0.2},
        }
        with self.assertRaisesRegex(ValueError, "required but missing"):
            module.build_inputs(
                ego_id=0,
                peripheral_ids=[1],
                env=FakeEnv(),
                belief_state=belief,
            )

    def test_legacy_array_is_explicitly_derived_or_rejected(self):
        legacy = np.zeros((2, 9), dtype=np.float32)
        legacy[:, 1] = [0.2, -0.3]
        legacy[:, 2] = 0.1

        compatible = PeripheralMultiMemory(
            action_dim=4,
            num_slots=4,
            require_full_signature=False,
            allow_legacy_items=True,
        )
        compatible.forward_full(legacy)
        diag = compatible.get_slot_diagnostics()
        self.assertEqual(diag["signature_source"], "legacy_derived")
        self.assertEqual(diag["signature_full_fraction"], 0.0)

        strict = PeripheralMultiMemory(
            action_dim=4,
            num_slots=4,
            require_full_signature=True,
            allow_legacy_items=False,
        )
        with self.assertRaisesRegex(ValueError, "legacy 9D"):
            strict.forward_full(legacy)


class TestRoutingAndCollapseDiagnostics(unittest.TestCase):
    def test_orthogonality_excludes_empty_slots(self):
        module = PeripheralMultiMemory(action_dim=4, num_slots=4)
        memories = torch.randn(4, module.memory_dim)
        loss = module._orthogonality_loss(
            memories,
            slot_support=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )
        self.assertEqual(float(loss), 0.0)

    def test_diagnostics_can_be_recomputed_on_a_heldout_pass(self):
        module = PeripheralMultiMemory(action_dim=4, num_slots=4)
        module.forward_full(role_items())
        self.assertGreater(module.get_slot_diagnostics()["diagnostic_updates"], 0)
        module.reset_slot_diagnostics()
        reset = module.get_slot_diagnostics()
        self.assertEqual(reset["diagnostic_updates"], 0)
        self.assertEqual(reset["signature_source"], "none")

    def test_semantic_roles_and_scalar_ablation_are_distinct(self):
        items = role_items()
        full = PeripheralMultiMemory(
            action_dim=4,
            num_slots=4,
            tau_role=0.05,
            sigma_hi=0.5,
            signature_mode="full",
        )
        full_roles = full.forward_full(items)["semantic_probs"].argmax(1).tolist()
        self.assertEqual(
            full_roles,
            [
                ROLE_BENEFICIAL,
                ROLE_BENEFICIAL,
                ROLE_HARMFUL,
                ROLE_HARMFUL,
                ROLE_NEUTRAL,
                ROLE_NEUTRAL,
                ROLE_UNCERTAIN,
                ROLE_UNCERTAIN,
            ],
        )

        scalar = PeripheralMultiMemory(
            action_dim=4,
            num_slots=4,
            tau_role=0.05,
            sigma_hi=0.5,
            signature_mode="scalar",
        )
        scalar_probs = scalar.forward_full(items)["semantic_probs"]
        self.assertTrue(torch.allclose(scalar_probs[:, ROLE_UNCERTAIN], torch.zeros(8)))
        self.assertNotEqual(
            scalar_probs.argmax(1).tolist(),
            full_roles,
            "Scalar-Only must not retain sigma through the anomalous gate",
        )

    def test_diffuse_uniform_assignment_is_detected(self):
        module = PeripheralMultiMemory(
            action_dim=4,
            num_slots=4,
            routing_mode="unconstrained",
        )
        with torch.no_grad():
            for parameter in module.unconstrained_router.parameters():
                parameter.zero_()
        result = module.forward_full(role_items())
        expected = torch.full_like(result["slot_probs"], 0.25)
        self.assertTrue(torch.allclose(result["slot_probs"], expected))
        diag = module.get_slot_diagnostics()
        self.assertLess(diag["assignment_mutual_info_ratio"], 1e-6)
        self.assertTrue(diag["diffuse_assignment_collapse"])
        self.assertTrue(diag["collapse_detected"])

    def test_monopoly_assignment_is_detected(self):
        module = PeripheralMultiMemory(
            action_dim=4,
            num_slots=4,
            routing_mode="unconstrained",
        )
        with torch.no_grad():
            for parameter in module.unconstrained_router.parameters():
                parameter.zero_()
            module.unconstrained_router[-1].bias.copy_(
                torch.tensor([20.0, -20.0, -20.0, -20.0])
            )
        module.forward_full(role_items())
        diag = module.get_slot_diagnostics()
        self.assertLess(diag["hard_usage_entropy_ratio"], 1e-6)
        self.assertTrue(diag["monopoly_collapse"])
        self.assertTrue(diag["collapse_detected"])

    def test_identical_slot_content_is_detected_even_with_role_usage(self):
        module = PeripheralMultiMemory(action_dim=4, num_slots=4)
        with torch.no_grad():
            for parameter in module.item_encoder.parameters():
                parameter.zero_()
            module.item_encoder[-1].bias.fill_(1.0)
        module.forward_full(role_items())
        diag = module.get_slot_diagnostics()
        self.assertGreater(diag["usage_entropy_ratio"], 0.5)
        self.assertGreater(diag["mean_offdiag_cosine"], 0.99)
        self.assertTrue(diag["uniform_content_collapse"])

    def test_load_balance_has_a_gradient_only_for_trainable_slots(self):
        semantic_only = PeripheralMultiMemory(action_dim=4, num_slots=4)
        fixed = semantic_only.forward_full(role_items())
        self.assertIsNone(fixed["balance_probs"])
        self.assertEqual(float(fixed["lb_loss"]), 0.0)

        hybrid = PeripheralMultiMemory(action_dim=4, num_slots=6)
        routed = hybrid.forward_full(role_items())
        self.assertEqual(tuple(routed["balance_probs"].shape), (8, 2))
        hybrid.zero_grad()
        routed["lb_loss"].backward()
        grad_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in hybrid.slot_router.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)


class TestH3Configuration(unittest.TestCase):
    def test_project_defaults_disable_uniform_copy_and_keep_free_slots(self):
        cfg = RE.default_cfg()
        self.assertFalse(cfg["periph_use_uniform_mix"])
        self.assertEqual(cfg["periph_uniform_mix"], 0.0)
        self.assertEqual(cfg["num_memory_slots"], 6)

    def test_ablations_change_exactly_the_claimed_switches(self):
        by_name = {spec["name"]: spec for spec in H3.VARIANTS}
        self.assertEqual(
            by_name["Unconstrained-NoSemantic"]["overrides"]["periph_routing_mode"],
            "unconstrained",
        )
        self.assertEqual(
            by_name["Scalar-Only"]["overrides"]["periph_signature_mode"],
            "scalar",
        )
        fixed = by_name["Fixed-Cardinality"]["overrides"]
        self.assertFalse(fixed["belief_adaptive_k"])
        self.assertEqual(fixed["min_core_size"], fixed["max_core_size"])
        self.assertEqual(
            by_name["NoMultiMemory-SingleMean"]["runner_model"],
            "H3FinalSingleMean",
        )
        self.assertTrue(issubclass(H3NoMultiMemoryRunner, FinalCIGAMFRunner))


class TestSingleMeanAblation(unittest.TestCase):
    def test_single_mean_uses_full_items_and_vectorized_leave_one_out(self):
        module = SingleMeanPeripheral(
            action_dim=4,
            memory_dim=8,
            out_dim=6,
            require_full_signature=True,
            allow_legacy_items=False,
        )
        items = role_items()[:3]
        output = module.forward_full(items)
        self.assertEqual(tuple(output["memory"].shape), (6,))
        self.assertEqual(float(output["aux_loss"]), 0.0)
        excluded = module.forward_excluding_all(items, [10, 11, 12])
        for index, item_id in enumerate([10, 11, 12]):
            keep = np.delete(items, index, axis=0)
            expected = module(keep)
            self.assertTrue(torch.allclose(excluded[item_id], expected, atol=1e-6))

    def test_single_mean_gradient_reaches_active_encoder(self):
        module = SingleMeanPeripheral(action_dim=4, memory_dim=8, out_dim=6)
        loss = module.forward_full(role_items())["memories"].square().sum()
        loss.backward()
        gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient, 0.0)


def supported_gate_rows():
    rows = []
    for seed in range(5):
        for spec in H3.VARIANTS:
            name = spec["name"]
            full = name == "Full-CIGAMF"
            fixed = name == "Fixed-Cardinality"
            collapsed_control = name in {
                "Unconstrained-NoSemantic",
                "No-AuxLoss",
            }
            rows.append(
                {
                    "variant": name,
                    "seed": seed,
                    "run_complete": True,
                    "signature_protocol_valid": True,
                    "signature_empirical_support_valid": True,
                    "collapse_detected": collapsed_control,
                    "hard_usage_entropy_ratio": 0.8 if full else 0.3,
                    "assignment_mutual_info_ratio": 0.3 if full else 0.05,
                    "slot_cos_offdiag": 0.4 if full else 0.98,
                    "frac_k_at_kmax": 0.4 if full else (1.0 if fixed else 0.7),
                    "core_size_std": 0.3 if full else (0.0 if fixed else 0.2),
                    "mean_f1": 0.70 if full else 0.50,
                    "mean_reward": 1.0 if full else 0.8,
                    "heldout_agent_states_per_second": (
                        120.0 if full else (200.0 if name == "NoMultiMemory-SingleMean" else 100.0)
                    ),
                }
            )
    return rows


class TestH3FalsifiableGates(unittest.TestCase):
    def test_each_declared_ablation_has_a_separate_gate(self):
        gate = H3._build_hypothesis_gate(supported_gate_rows())
        expected = {
            "full_vs_scalar_signature",
            "full_vs_unconstrained_routing",
            "full_vs_no_auxiliary_losses",
            "adaptive_vs_fixed_cardinality",
            "full_vs_single_mean_decision_cost",
        }
        self.assertEqual(set(gate["gates"]), expected)
        self.assertTrue(all(item["supported"] for item in gate["gates"].values()))
        self.assertTrue(gate["h3_claim_supported"])

    def test_one_failed_ablation_does_not_overwrite_other_gate_results(self):
        rows = supported_gate_rows()
        for row in rows:
            if row["variant"] == "Scalar-Only":
                row["mean_f1"] = 0.80
        gate = H3._build_hypothesis_gate(rows)
        self.assertFalse(gate["gates"]["full_vs_scalar_signature"]["supported"])
        self.assertTrue(gate["gates"]["full_vs_unconstrained_routing"]["supported"])
        self.assertTrue(
            gate["gates"]["full_vs_single_mean_decision_cost"]["supported"]
        )


if __name__ == "__main__":
    unittest.main()
