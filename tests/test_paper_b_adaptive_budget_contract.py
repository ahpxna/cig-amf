"""Fail-closed exact-grid contracts for Paper-B adaptive-budget artifacts."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_paper_b import _load_adaptive_budget
from utils.paper_contracts import (
    PAPER_B_ADAPTIVE_BUDGET_PROTOCOL_VERSION,
    PAPER_B_ADAPTIVE_MATCHING_RULE,
)


class PaperBAdaptiveBudgetContractTests(unittest.TestCase):
    def _variants(self, k_min=2, k_max=3):
        return [
            "Adaptive-K",
            *[f"Fixed-K-{k}" for k in range(k_min, k_max + 1)],
            "Full-Explicit",
        ]

    def _rows(self, seeds, *, k_min=2, k_max=3, matched="Fixed-K-2"):
        rows = []
        for seed in seeds:
            for variant in self._variants(k_min, k_max):
                rows.append({
                    "variant": variant,
                    "seed": str(seed),
                    "matched_to_adaptive": "1" if variant == matched else "0",
                    "mean_core_cost_per_ego": "2.0" if variant == "Adaptive-K" else "2.0",
                    "mean_core_cost_gap_to_adaptive": "0.0" if variant.startswith("Fixed-K-") else "nan",
                    "mean_K_t": "24.0",
                    "boundary_saturation_rate": "0.2" if variant == "Adaptive-K" else "0.0",
                })
        return rows

    def _manifest(self, seeds, *, k_min=2, k_max=3, matched="Fixed-K-2"):
        return {
            "complete": True,
            "protocol_version": PAPER_B_ADAPTIVE_BUDGET_PROTOCOL_VERSION,
            "seeds": list(seeds),
            "k_min": k_min,
            "k_max": k_max,
            "variants": self._variants(k_min, k_max),
            "matching_rule_id": PAPER_B_ADAPTIVE_MATCHING_RULE,
            "matching_seeds": [101, 102],
            "matching_seed_disjoint": True,
            "pilot_adaptive_core_cost_per_ego": 2.0,
            "matched_fixed_variant": matched,
            "max_boundary_saturation_fraction": 0.8,
        }

    def _load(self, rows, manifest, expected_seeds, *, mode="quick"):
        with tempfile.TemporaryDirectory() as root:
            panel_root = Path(root) / "paper_b_adaptive_budget"
            panel_root.mkdir()
            summary_path = panel_root / "summary_paper_b_adaptive_budget.csv"
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            manifest = dict(manifest)
            manifest["summary_row_count"] = len(rows)
            manifest["summary_sha256"] = hashlib.sha256(
                summary_path.read_bytes()
            ).hexdigest()
            with (panel_root / "manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            return _load_adaptive_budget(root, expected_seeds, protocol_mode=mode)

    def test_adaptive_accepts_full_fixed_k_grid(self):
        seeds = [7, 8]
        self._load(self._rows(seeds), self._manifest(seeds), seeds)

    def test_adaptive_rejects_missing_nonmatched_fixed_k(self):
        seeds = [7]
        rows = [row for row in self._rows(seeds) if row["variant"] != "Fixed-K-3"]
        with self.assertRaisesRegex(ValueError, "matrix mismatch"):
            self._load(rows, self._manifest(seeds), seeds)

    def test_adaptive_rejects_extra_fixed_k(self):
        seeds = [7]
        rows = self._rows(seeds)
        extra = dict(rows[0])
        extra["variant"] = "Fixed-K-4"
        rows.append(extra)
        with self.assertRaisesRegex(ValueError, "matrix mismatch"):
            self._load(rows, self._manifest(seeds), seeds)

    def test_adaptive_rejects_duplicate_exact_cell(self):
        seeds = [7]
        rows = self._rows(seeds)
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate logical cells"):
            self._load(rows, self._manifest(seeds), seeds)

    def test_adaptive_rejects_manifest_variant_grid_mismatch(self):
        seeds = [7]
        manifest = self._manifest(seeds)
        manifest["variants"] = ["Adaptive-K", "Fixed-K-2", "Full-Explicit"]
        with self.assertRaisesRegex(ValueError, "variant grid mismatch"):
            self._load(self._rows(seeds), manifest, seeds)

    def test_adaptive_rejects_pilot_cost_matched_variant_mismatch(self):
        seeds = [7]
        manifest = self._manifest(seeds, matched="Fixed-K-3")
        with self.assertRaisesRegex(ValueError, "pilot-cost rule"):
            self._load(self._rows(seeds, matched="Fixed-K-3"), manifest, seeds)

    def test_adaptive_rejects_wrong_protocol_version_in_confirmatory_mode(self):
        seeds = [7]
        manifest = self._manifest(seeds)
        manifest["protocol_version"] = "legacy"
        with self.assertRaisesRegex(ValueError, "protocol version mismatch"):
            self._load(self._rows(seeds), manifest, seeds, mode="confirmatory")

    def test_adaptive_rejects_overlapping_matching_seeds(self):
        seeds = [7]
        manifest = self._manifest(seeds)
        manifest["matching_seeds"] = [7]
        with self.assertRaisesRegex(ValueError, "must be non-empty, unique, and disjoint"):
            self._load(self._rows(seeds), manifest, seeds)


if __name__ == "__main__":
    unittest.main()
