import json
import os
import tempfile
import unittest

from scripts.validate_p11_scope import validate_scope


class P11ScopeReadinessTests(unittest.TestCase):
    def _write(self, root, name, payload):
        path = os.path.join(root, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def test_current_style_artifacts_allow_smokes_but_block_scientific_grids(self):
        with tempfile.TemporaryDirectory() as root:
            query = self._write(root, "query.json", {
                "protocol_version": "p11_typed_query_exact_gates_v1",
                "overall_status": "PASS",
                "gate_pass": {f"Q{i}": True for i in range(5)},
            })
            prechecks = self._write(root, "prechecks.json", {
                "protocol_version": "scientific_prechecks_g0_g4_v4_allocation_fidelity",
                "protocol_mode": "quick",
                "overall_status": "SMOKE_ONLY",
                "core_prechecks_pass": False,
                "gates": {f"G{i}": {} for i in range(5)},
            })
            h1 = self._write(root, "h1.json", {"h1_claim_gate_pass": False})
            periphery = self._write(root, "periphery.json", {
                "experiment": "paper_b_fixed_core_periphery",
                "complete": True,
                "summary_row_count": 4,
            })
            result = validate_scope(
                query=query, prechecks=prechecks, h1=h1, periphery=periphery
            )
        self.assertEqual(result["engineering_alignment"]["passed"], 3)
        self.assertTrue(result["allowed_now"]["paper01_03_quick_allocation_diagnostic"])
        self.assertTrue(result["allowed_now"]["paper08_plumbing_smoke"])
        self.assertFalse(result["can_start_unbounded_or_confirmatory_grid"])
        self.assertEqual(result["paper_status"]["paper06"], "BLOCKED_H1")
        self.assertEqual(
            result["paper_status"]["paper08"],
            "PLUMBING_ONLY_SCIENCE_BLOCKED",
        )

    def test_d0_confirmed_does_not_authorize_algorithm_without_theory(self):
        with tempfile.TemporaryDirectory() as root:
            query = self._write(root, "query.json", {
                "protocol_version": "p11_typed_query_exact_gates_v1",
                "overall_status": "PASS",
                "gate_pass": {f"Q{i}": True for i in range(5)},
            })
            prechecks = self._write(root, "prechecks.json", {
                "protocol_version": "scientific_prechecks_g0_g4_v4_allocation_fidelity",
                "protocol_mode": "confirmatory",
                "overall_status": "PASS",
                "core_prechecks_pass": True,
                "gates": {f"G{i}": {} for i in range(5)},
            })
            h1 = self._write(root, "h1.json", {"h1_claim_gate_pass": True})
            periphery = self._write(root, "periphery.json", {
                "experiment": "paper_b_fixed_core_periphery",
                "complete": True,
                "summary_row_count": 4,
                "information_channel_gate_pass": True,
                "same_q_opposite_information_gate_pass": True,
                "information_oracle_sha256": "a" * 64,
            })
            d0 = self._write(root, "d0.json", {
                "d0_status": "CONFIRMED", "collection_complete": True,
            })
            result = validate_scope(
                query=query, prechecks=prechecks, h1=h1,
                periphery=periphery, d0=d0,
            )
        self.assertEqual(result["engineering_alignment"]["fraction"], 1.0)
        self.assertFalse(result["scientific_support_contracts"][
            "paper07_09_t0_t1_t2_formalized"
        ])
        self.assertTrue(result["blocked_now"]["paper07_09_adaptive_algorithm"])
        self.assertFalse(result["can_start_unbounded_or_confirmatory_grid"])


if __name__ == "__main__":
    unittest.main()
