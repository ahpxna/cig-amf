"""Regression tests for confirmatory claim gates."""

import unittest

from scripts import validate_claims as claims


def _h2_rows():
    rows = []
    for seed in range(5):
        for model, sr, latency in (
            ("Final-CIGAMF", 4.0, 0.5),
            ("CorrelationMeanField", 1.0, 1.0),
            ("NoTwoTimescale", 2.0, 1.5),
        ):
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "claim_evaluable": True,
                    "episodes": 400,
                    "policy_learning_frozen": True,
                    "representation_learning_frozen": True,
                    "frozen_representation_unchanged": True,
                    "SR_C": sr,
                    "capacity_beta_structural": 0.40 if model != "CorrelationMeanField" else 0.12,
                    "capacity_beta_behavioral": 0.05 if model != "CorrelationMeanField" else 0.08,
                    "direction_beta_structural": 0.02,
                    "direction_beta_behavioral": 0.30,
                    "estimand_capacity_beta_structural": 0.50,
                    "estimand_capacity_beta_behavioral": 0.02,
                    "estimand_direction_beta_behavioral": 0.40,
                    "behavioral_false_trigger_rate": 0.01,
                    "recovery_latency": latency,
                    "n_shift_events": 2,
                    "n_recovered_shifts": (
                        2 if model == "Final-CIGAMF" else 1
                    ),
                    "n_shift_with_trigger": 2,
                    "n_complete_structural_windows": 2,
                    "n_complete_behavioral_windows": 2,
                    "direction_manipulation_pass": model != "CorrelationMeanField",
                    "direction_behav": 0.2 if model != "CorrelationMeanField" else 0.0,
                }
            )
    return rows


class H2ClaimGateTests(unittest.TestCase):
    def test_requires_factor_selectivity_and_association_comparator(self):
        report = claims._h2_status(_h2_rows(), h1_supported=True)
        self.assertTrue(report["supported"])
        self.assertTrue(
            report["conditions"][
                "causal_capacity_is_more_structurally_selective_than_correlation"
            ]
        )

    def test_missing_capacity_selectivity_prevents_support(self):
        rows = _h2_rows()
        for row in rows:
            if row["model"] == "Final-CIGAMF":
                row["capacity_beta_behavioral"] = 0.60
        report = claims._h2_status(rows, h1_supported=True)
        self.assertFalse(report["supported"])
        self.assertFalse(
            report["conditions"][
                "capacity_prefers_structural_over_behavioral_change"
            ]
        )


if __name__ == "__main__":
    unittest.main()
