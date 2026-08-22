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
                    "SR_cross_run": sr,
                    "recovery_latency": latency,
                    "n_shift_events": 2,
                    "n_recovered_shifts": 2,
                    "n_shift_with_trigger": 2,
                    "n_complete_structural_windows": 2,
                    "n_complete_behavioral_windows": 2,
                }
            )
    return rows


class H2ClaimGateTests(unittest.TestCase):
    def test_requires_complete_recovery_and_scheduler_effect(self):
        report = claims._h2_status(_h2_rows(), h1_supported=True)
        self.assertTrue(report["supported"])
        self.assertTrue(
            report["conditions"][
                "two_timescale_scheduler_beats_every_episode_control"
            ]
        )

    def test_one_unrecovered_shift_prevents_support(self):
        rows = _h2_rows()
        for row in rows:
            if row["model"] == "Final-CIGAMF" and row["seed"] == 0:
                row["n_recovered_shifts"] = 1
        report = claims._h2_status(rows, h1_supported=True)
        self.assertFalse(report["supported"])
        self.assertFalse(
            report["conditions"][
                "final_recovers_and_triggers_for_every_structural_shift"
            ]
        )


if __name__ == "__main__":
    unittest.main()
