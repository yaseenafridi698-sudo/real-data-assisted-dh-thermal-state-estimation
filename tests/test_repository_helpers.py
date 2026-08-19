from __future__ import annotations

import csv
import unittest
from pathlib import Path

import numpy as np

from src.dataset import contiguous_window_starts
from src.proxy_causality_audit import FULL_STATE_CAUSALITY_VERSION, verify_preserved_audit


ROOT = Path(__file__).resolve().parents[1]


class DatasetWindowTests(unittest.TestCase):
    def test_windows_do_not_cross_trajectory_break(self) -> None:
        starts = np.array([True, False, False, True, False, False, False], dtype=bool)
        self.assertEqual(contiguous_window_starts(starts, 3), [0, 3, 4])

    def test_window_may_begin_at_segment_start(self) -> None:
        starts = np.array([True, False, False], dtype=bool)
        self.assertEqual(contiguous_window_starts(starts, 3), [0])

    def test_invalid_window_length_rejected(self) -> None:
        with self.assertRaises(ValueError):
            contiguous_window_starts(np.array([True, False]), 0)


class CausalityAuditTests(unittest.TestCase):
    def test_preserved_audit_passes(self) -> None:
        verify_preserved_audit(ROOT / "results" / "proxy_causality_audit.csv")

    def test_recorded_version_matches_protocol(self) -> None:
        with (ROOT / "results" / "proxy_causality_audit.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        version = next(r["value"] for r in rows if r["audit"] == "full_state_causality_version_recorded")
        self.assertEqual(version, FULL_STATE_CAUSALITY_VERSION)


if __name__ == "__main__":
    unittest.main()
