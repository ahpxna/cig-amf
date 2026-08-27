"""Integration contracts for the Paper-B population/budget scaling grid."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_paper_b_scaling import VARIANTS
from scripts.validate_paper_b import _load_scaling


class PaperBScalingMatrixContractTests(unittest.TestCase):
    def _rows(self, seeds, agent_counts, budgets):
        return [
            {
                "seed": str(seed),
                "n_agents": str(n_agents),
                "requested_core_budget": str(budget),
                "core_budget": str(min(budget, n_agents - 1)),
                "variant": variant,
            }
            for seed in seeds
            for n_agents in agent_counts
            for budget in budgets
            for variant in VARIANTS
        ]

    def _manifest(self, seeds, agent_counts, budgets):
        return {
            "complete": True,
            "seeds": list(seeds),
            "agent_counts": list(agent_counts),
            "core_budgets": list(budgets),
            "variants": list(VARIANTS),
            "candidate_max_degree": 8,
            "candidate_policy": (
                "dynamic adapter-owned policy-independent pre-measurement "
                "candidate set"
            ),
            "candidate_update_protocol": "refresh_before_measurement_each_step",
            "candidate_max_cell_occupancy": 16,
            "candidate_oracle_recall": {
                "minimum": 0.8,
                "states": 1,
                "horizon": 1,
                "trials": 1,
                "independent_replicates": 1,
                "target_k_rule": (
                    "max tested explicit core budget clipped to d_max and N-1"
                ),
                "stability_min": 0.8,
                "stable_fraction_min": 0.8,
            },
            "candidate_degradation_gate": {
                "reference_variant": "Semantic-Free-Unrestricted",
                "max_relative_reward_drop": 0.1,
                "max_relative_logit_error_increase": 0.25,
                "max_relative_value_error_increase": 0.25,
                "max_action_agreement_drop": 0.05,
                "reward_denominator_floor": 1.0,
                "error_denominator_floor": 0.1,
            },
        }

    def _load(self, rows, manifest, expected_seeds):
        with tempfile.TemporaryDirectory() as root:
            panel_root = Path(root) / "paper_b_scaling"
            panel_root.mkdir()
            with (panel_root / "manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            with (
                panel_root / "summary_paper_b_scaling.csv"
            ).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            return _load_scaling(root, expected_seeds, protocol_mode="quick")

    def test_scaling_loader_accepts_multiple_budgets(self):
        seeds, populations, budgets = [7], [12], [2, 3]
        self._load(
            self._rows(seeds, populations, budgets),
            self._manifest(seeds, populations, budgets),
            seeds,
        )

    def test_scaling_loader_accepts_multiple_populations(self):
        seeds, populations, budgets = [7], [12, 24], [2]
        self._load(
            self._rows(seeds, populations, budgets),
            self._manifest(seeds, populations, budgets),
            seeds,
        )

    def test_scaling_loader_accepts_full_cartesian_sweep(self):
        seeds, populations, budgets = [7, 8], [12, 24], [2, 3]
        self._load(
            self._rows(seeds, populations, budgets),
            self._manifest(seeds, populations, budgets),
            seeds,
        )

    def test_scaling_loader_rejects_duplicate_exact_cell(self):
        seeds, populations, budgets = [7], [12], [2]
        rows = self._rows(seeds, populations, budgets)
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate logical cells"):
            self._load(rows, self._manifest(seeds, populations, budgets), seeds)

    def test_scaling_loader_rejects_missing_manifest_population(self):
        seeds, populations, budgets = [7], [12, 24], [2]
        rows = self._rows(seeds, [12], budgets)
        with self.assertRaisesRegex(ValueError, "matrix mismatch"):
            self._load(rows, self._manifest(seeds, populations, budgets), seeds)

    def test_scaling_loader_rejects_clipping_collision(self):
        seeds, populations, budgets = [7], [5], [4, 5]
        rows = self._rows(seeds, populations, budgets)
        with self.assertRaisesRegex(ValueError, "collapses after clipping"):
            self._load(rows, self._manifest(seeds, populations, budgets), seeds)


if __name__ == "__main__":
    unittest.main()
