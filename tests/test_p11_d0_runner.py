import unittest

from scripts.run_p11_d0 import run


class P11D0RunnerTests(unittest.TestCase):
    def test_quick_is_complete_but_never_claim_confirmed(self):
        result = run(mode="quick", replicates=2)
        self.assertTrue(result["collection_complete"])
        self.assertEqual(result["d0_status"], "SMOKE_ONLY")
        self.assertEqual(len(result["records"]), 12)
        self.assertTrue(all(row["action_applied"] for row in result["records"]))
        self.assertTrue(all(row["paired_baseline_verified"] for row in result["records"]))

    def test_common_target_and_exact_cost_match(self):
        result = run(mode="quick", replicates=2)
        self.assertEqual({row["target_key"] for row in result["records"]},
                         {"fixed_two_state_target_v1"})
        self.assertEqual({row["immediate_cost"] for row in result["records"]}, {1.0})

    def test_confirmatory_rejects_too_few_replicates(self):
        with self.assertRaisesRegex(ValueError, "at least 30"):
            run(mode="confirmatory", replicates=2)


if __name__ == "__main__":
    unittest.main()
