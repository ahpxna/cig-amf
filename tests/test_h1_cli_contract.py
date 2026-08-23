"""Regression contract for the executable one-step H1 configuration."""

import json
import os
import tempfile
import unittest
from unittest import mock

from scripts import run_h1_calibration as H1


class H1CliContractTests(unittest.TestCase):
    def test_one_step_runner_horizons_and_fingerprint_inputs_match(self):
        thresholds = {
            "capacity_active_threshold": 0.01,
            "capacity_prediction_threshold": 0.01,
            "direction_active_threshold": 0.005,
            "direction_prediction_threshold": 0.005,
        }
        cfg = H1.build_h1_config({"eps": 0.05}, seed=101,
                                  threshold_calibration=thresholds)
        self.assertEqual(cfg["causal_horizon"], 1)
        self.assertEqual(cfg["proxy_n_horizons"], 1)
        self.assertEqual(cfg["h1_capacity_active_threshold"], 0.01)
        self.assertEqual(cfg["h1_direction_active_threshold"], 0.005)

    def test_cli_attempt_passes_a_matched_one_step_contract_to_runner(self):
        observed = {}

        def fake_tiny_task(args, cfg, device, out_dir, run_label):
            del args, device, out_dir, run_label
            observed.update(cfg)
            return {"seed": 101, "protocol_gate_pass": True}

        with tempfile.TemporaryDirectory() as directory:
            calibration = os.path.join(directory, "thresholds.json")
            with open(calibration, "w", encoding="utf-8") as handle:
                json.dump({
                    "oracle_only": True,
                    "development_seeds": [1],
                    "capacity_active_threshold": 0.01,
                    "capacity_prediction_threshold": 0.01,
                    "direction_active_threshold": 0.005,
                    "direction_prediction_threshold": 0.005,
                }, handle)
            with mock.patch.object(H1.RE, "run_tiny_task", fake_tiny_task), \
                 mock.patch.object(
                     H1, "_claim_gate", return_value={"h1_claim_gate_pass": False}
                 ):
                H1.main([
                    "--seeds", "101", "--variants", "plugin_eps005",
                    "--tiny-proxy-train-episodes", "1", "--tiny-states", "1",
                    "--out-root", directory, "--run-id", "contract",
                    "--threshold-calibration", calibration,
                ])
        self.assertEqual(observed["causal_horizon"], 1)
        self.assertEqual(observed["proxy_n_horizons"], 1)


if __name__ == "__main__":
    unittest.main()
