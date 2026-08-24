"""Fail-closed external-runtime bootstrap and install-status contracts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from envs.external.runtime import maybe_reexec_in_external_runtime
from envs.external import registry


class ExternalRuntimeTests(unittest.TestCase):
    def test_required_runtime_does_not_fall_back_to_main_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {"CIG_EXTERNAL_ENVS_DIR": directory},
                clear=False,
            ):
                os.environ.pop("CIG_EXTERNAL_RUNTIME_ACTIVE", None)
                with self.assertRaises(SystemExit) as caught:
                    maybe_reexec_in_external_runtime(require_ready=True)
        message = str(caught.exception)
        self.assertIn("external benchmark runtime is not ready", message)
        self.assertIn("setup_external_envs.sh --install", message)

    def test_runtime_import_failure_is_reported_before_adapter_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            (runtime / "bin").mkdir(parents=True)
            (runtime / "bin" / "python").write_text("", encoding="utf-8")
            (runtime / ".cig-external-runtime-ready").write_text("", encoding="utf-8")
            (runtime / "runtime.json").write_text(
                json.dumps({
                    "imports": {
                        "robotic-warehouse": {
                            "ok": False,
                            "module": "rware",
                            "error": "ModuleNotFoundError: No module named 'gymnasium'",
                        }
                    }
                }),
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {
                    "CIG_EXTERNAL_ENVS_DIR": str(root),
                    "CIG_EXTERNAL_RUNTIME_ACTIVE": "1",
                },
                clear=False,
            ):
                with self.assertRaises(RuntimeError) as caught:
                    registry._require_runtime_import("rware")
        message = str(caught.exception)
        self.assertIn("not operational for rware", message)
        self.assertIn("gymnasium", message)

    def test_runtime_import_success_allows_registry_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            (runtime / "bin").mkdir(parents=True)
            (runtime / "bin" / "python").write_text("", encoding="utf-8")
            (runtime / ".cig-external-runtime-ready").write_text("", encoding="utf-8")
            (runtime / "runtime.json").write_text(
                json.dumps({"imports": {"robotic-warehouse": {"ok": True, "module": "rware"}}}),
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {
                    "CIG_EXTERNAL_ENVS_DIR": str(root),
                    "CIG_EXTERNAL_RUNTIME_ACTIVE": "1",
                },
                clear=False,
            ):
                registry._require_runtime_import("rware")

    def test_registry_never_falls_back_to_legacy_root_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "robotic-warehouse"
            legacy.mkdir(parents=True)
            with mock.patch.dict(os.environ, {"CIG_EXTERNAL_ENVS_DIR": str(root)}, clear=False):
                self.assertEqual(registry.repo_path("rware"), root / "repos" / "robotic-warehouse")
                self.assertTrue(registry.legacy_repo_path("rware").exists())
                with self.assertRaises(FileNotFoundError):
                    registry.ensure_repo_on_path("rware")

    def test_runtime_active_without_import_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            (runtime / "bin").mkdir(parents=True)
            (runtime / "bin" / "python").write_text("", encoding="utf-8")
            (runtime / ".cig-external-runtime-ready").write_text("", encoding="utf-8")
            with mock.patch.dict(
                os.environ,
                {"CIG_EXTERNAL_ENVS_DIR": str(root), "CIG_EXTERNAL_RUNTIME_ACTIVE": "1"},
                clear=False,
            ):
                with self.assertRaises(RuntimeError) as caught:
                    registry._require_runtime_import("flatland")
        self.assertIn("no import verification", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
