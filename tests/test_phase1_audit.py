from __future__ import annotations

import unittest

import pandas as pd

from veri_madenciligi.services.phase1_audit import (
    determine_go_decision,
    normalize_binary_like_values,
)


class Phase1AuditHelpersTestCase(unittest.TestCase):
    def test_normalize_binary_like_values_handles_string_and_float_variants(self) -> None:
        series = pd.Series(["0", 1.0, "1.0", 0, None])

        self.assertEqual(normalize_binary_like_values(series), [0, 1])

    def test_determine_go_decision_returns_conditional_go_for_duplicates(self) -> None:
        decision = determine_go_decision(
            schema_issues=False,
            target_valid=True,
            duplicate_rate=0.1,
            has_high_risk_feature=False,
        )

        self.assertEqual(decision, "Conditional Go")

    def test_determine_go_decision_returns_no_go_for_invalid_target(self) -> None:
        decision = determine_go_decision(
            schema_issues=False,
            target_valid=False,
            duplicate_rate=0.0,
            has_high_risk_feature=False,
        )

        self.assertEqual(decision, "No Go")


if __name__ == "__main__":
    unittest.main()