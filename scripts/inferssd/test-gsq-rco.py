#!/usr/bin/env python3
"""Failure-state contracts for the bounded allocation monitor; no provider calls."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("gsq_check", Path(__file__).with_name("check-gsq-rco.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MonitorTests(unittest.TestCase):
    def test_cached_allocation_and_invalid_coverage(self):
        raw = MODULE.BASELINE.read_bytes()
        self.assertEqual(len(MODULE.allocation(raw)), 1043)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.allocation(raw + b"output.weight: Q5_K\n")
        with self.assertRaisesRegex(ValueError, "coverage"):
            MODULE.allocation(raw.replace(b"output.weight: Q5_K\n", b""))
        # Preserve the total type counts while violating an expert up/gate pair.
        changed = raw.replace(b"blk.0.ffn_up_exps.weight: IQ1_M", b"blk.0.ffn_up_exps.weight: IQ2_XS")
        changed = changed.replace(b"blk.17.ffn_up_exps.weight: IQ2_XS", b"blk.17.ffn_up_exps.weight: IQ1_M")
        with self.assertRaisesRegex(ValueError, "up/gate"):
            MODULE.allocation(changed)

    def test_unavailable_source_is_persisted_and_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch("sys.argv", ["check", "--refresh", "--output", temporary]), mock.patch.object(
                MODULE.urllib.request, "urlopen", side_effect=OSError("source unavailable")
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(MODULE.main(), 1)
            result = json.loads(next(Path(temporary).glob("*/result.json")).read_text())
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["source_availability"], "unavailable")
            self.assertNotIn("tensor_count", result)

    def test_available_but_invalid_source_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            response = io.BytesIO(b"<html>login required</html>")
            response.headers = {}
            with mock.patch("sys.argv", ["check", "--refresh", "--output", temporary]), mock.patch.object(
                MODULE.urllib.request, "urlopen", return_value=response
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(MODULE.main(), 1)
            result = json.loads(next(Path(temporary).glob("*/result.json")).read_text())
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["source_availability"], "available")
            self.assertEqual(next(Path(temporary).glob("*/allocation.txt")).read_bytes(), b"<html>login required</html>")

    def test_missing_cached_evidence_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch("sys.argv", ["check", "--output", temporary]), mock.patch.object(
                MODULE, "BASELINE", Path(temporary) / "missing"
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(MODULE.main(), 1)
            result = json.loads(next(Path(temporary).glob("*/result.json")).read_text())
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["source_availability"], "not_requested")


if __name__ == "__main__":
    unittest.main()
