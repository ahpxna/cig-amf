import os
import tempfile
import unittest

from scripts.run_p11_query_gates import PROTOCOL_VERSION, run_exact_query_gates
from scripts.scientific_gate_common import atomic_json, load_json


class P11QueryGateTests(unittest.TestCase):
    def test_exact_query_gate_suite_passes_all_required_cells(self):
        result = run_exact_query_gates()
        self.assertEqual(result["protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(result["evidence_class"], "DETERMINISTIC_SYNTHETIC_GATE_ONLY")
        self.assertEqual(result["overall_status"], "PASS")
        self.assertEqual(set(result["gate_pass"]), {"Q0", "Q1", "Q2", "Q3", "Q4"})
        self.assertTrue(all(result["gate_pass"].values()))

    def test_transfer_matrix_has_matched_zero_and_mismatched_positive_regret(self):
        result = run_exact_query_gates()
        rows = result["gates"]["Q2"]["metrics"]["transfer_rows"]
        matched = [row for row in rows if row["selector"] == row["target_query"]]
        mismatched = [row for row in rows if row["selector"] != row["target_query"]]
        self.assertTrue(all(row["regret"] == 0.0 for row in matched))
        self.assertTrue(any(row["regret"] > 0.0 for row in mismatched))

    def test_gate_payload_roundtrips_as_strict_json(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "query_gates.json")
            atomic_json(path, run_exact_query_gates())
            loaded = load_json(path, "query gate")
        self.assertEqual(loaded["overall_status"], "PASS")
        self.assertEqual(len(loaded["generator_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
