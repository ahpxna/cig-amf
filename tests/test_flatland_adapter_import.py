"""Import contract for the optional Flatland integration layer."""

import importlib
import unittest


class FlatlandAdapterImportTests(unittest.TestCase):
    def test_adapter_module_imports_without_optional_flatland_install(self):
        module = importlib.import_module("envs.flatland_adapter")
        self.assertTrue(hasattr(module, "FlatlandCIGEnvironment"))


if __name__ == "__main__":
    unittest.main()
