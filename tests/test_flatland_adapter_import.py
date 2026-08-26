"""Import contract for the optional Flatland integration layer."""

import copy
import importlib
import unittest

import numpy as np


class _RailEnvCloneFromDouble:
    """Mimic pinned Flatland's destination.clone_from(source) contract."""

    number_of_agents = 2

    def __init__(self):
        self.agents = [object(), object()]
        self.payload = {"step": 7, "nested": [1, 2, 3]}
        self.clone_from_calls = 0

    def clone_from(self, env):
        self.clone_from_calls += 1
        self.payload = copy.deepcopy(env.payload)
        # Pinned Flatland mutates self and returns None.
        return None


class FlatlandAdapterImportTests(unittest.TestCase):
    def test_adapter_module_imports_without_optional_flatland_install(self):
        module = importlib.import_module("envs.flatland_adapter")
        self.assertTrue(hasattr(module, "FlatlandCIGEnvironment"))

    def test_clone_state_honours_destination_clone_from_contract(self):
        module = importlib.import_module("envs.flatland_adapter")
        raw = _RailEnvCloneFromDouble()
        env = module.FlatlandCIGEnvironment(raw, observation_width=4)
        env._obs = [np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32) for _ in range(2)]

        state = env.clone_state()
        cloned_raw = state[0]

        self.assertIsNot(cloned_raw, raw)
        self.assertEqual(cloned_raw.clone_from_calls, 1)
        self.assertEqual(cloned_raw.payload, raw.payload)
        raw.payload["nested"].append(99)
        self.assertNotEqual(cloned_raw.payload, raw.payload)


if __name__ == "__main__":
    unittest.main()
