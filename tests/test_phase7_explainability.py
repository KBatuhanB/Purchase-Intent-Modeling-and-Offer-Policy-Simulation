import unittest

import numpy as np
import pandas as pd

from veri_madenciligi.services.phase4_baseline import BaselineModelResult
from veri_madenciligi.services.phase7_explainability import (
    build_fairness_alerts,
    compute_group_fairness_metrics,
    extract_top_local_contributions,
    select_reference_scenarios,
    summarize_fairness_by_group,
)


def make_result(predictions: list[int], probabilities: list[float]) -> BaselineModelResult:
    return BaselineModelResult(
        name="test_model",
        family="test",
        metrics={
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.0,
            "pr_auc": 0.0,
            "balanced_accuracy": 0.0,
            "brier_score": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
        },
        calibration={"bin_true_rate": [], "bin_predicted_probability": [], "mean_absolute_gap": None},
        note="",
        probabilities=probabilities,
        predictions=predictions,
    )


class Phase7ExplainabilityHelperTests(unittest.TestCase):
    def test_compute_group_fairness_metrics_includes_rate_fields(self) -> None:
        metrics = compute_group_fairness_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0, 1, 1, 1]),
            np.asarray([0.1, 0.6, 0.8, 0.9]),
        )

        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(metrics["positive_prediction_rate"], 0.75)
        self.assertAlmostEqual(metrics["false_positive_rate"], 0.5)

    def test_build_fairness_alerts_flags_large_gaps(self) -> None:
        summary = {
            "groups": {
                "Gender": {
                    "gender_0": {
                        "gaps_vs_overall": {
                            "recall": 0.13,
                            "precision": 0.05,
                            "positive_prediction_rate": 0.02,
                            "false_positive_rate": 0.01,
                            "false_negative_rate": 0.0,
                        }
                    }
                }
            }
        }

        alerts = build_fairness_alerts(summary, gap_threshold=0.12)

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["metric_name"], "recall")

    def test_extract_top_local_contributions_orders_by_absolute_value(self) -> None:
        row = pd.Series({"Age": 30, "AnnualIncome": 75000, "LoyaltyProgram": 1})
        contributions = np.asarray([0.1, -0.45, 0.25])

        top_contributions = extract_top_local_contributions(row, contributions, top_k=2)

        self.assertEqual(top_contributions[0]["feature_name"], "AnnualIncome")
        self.assertEqual(top_contributions[1]["feature_name"], "LoyaltyProgram")

    def test_select_reference_scenarios_returns_unique_rows(self) -> None:
        evaluation_frame = pd.DataFrame(
            {
                "Age": [22, 24, 26, 32, 35, 41, 45, 52, 55, 60],
                "Gender": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                "AnnualIncome": [30000, 35000, 40000, 60000, 62000, 70000, 82000, 95000, 110000, 120000],
                "NumberOfPurchases": [0, 1, 0, 2, 3, 5, 1, 7, 2, 8],
                "ProductCategory": [0, 1, 2, 1, 0, 3, 2, 4, 1, 0],
                "TimeSpentOnWebsite": [5, 8, 15, 18, 22, 30, 44, 50, 55, 58],
                "LoyaltyProgram": [0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
                "DiscountsAvailed": [0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
                "PurchaseStatus": [0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
            }
        )
        probabilities = np.asarray([0.05, 0.12, 0.82, 0.48, 0.76, 0.61, 0.72, 0.91, 0.33, 0.44])
        predictions = (probabilities >= 0.5).astype(int)

        scenarios = select_reference_scenarios(evaluation_frame, probabilities, predictions, max_scenarios=8)

        self.assertEqual(len({scenario["row_position"] for scenario in scenarios}), len(scenarios))
        self.assertTrue(any(scenario["scenario_id"] == "boundary_case" for scenario in scenarios))

    def test_summarize_fairness_by_group_collects_group_metrics(self) -> None:
        evaluation_frame = pd.DataFrame(
            {
                "Age": [22, 23, 24, 40, 41, 42],
                "Gender": [0, 0, 0, 1, 1, 1],
                "AnnualIncome": [30000, 32000, 34000, 80000, 82000, 84000],
                "NumberOfPurchases": [0, 1, 0, 4, 5, 6],
                "ProductCategory": [0, 0, 1, 2, 2, 3],
                "TimeSpentOnWebsite": [10, 12, 11, 35, 36, 38],
                "LoyaltyProgram": [0, 0, 0, 1, 1, 1],
                "DiscountsAvailed": [0, 0, 0, 1, 1, 1],
                "PurchaseStatus": [0, 0, 1, 1, 1, 1],
            }
        )
        result = make_result(predictions=[0, 1, 1, 1, 1, 1], probabilities=[0.1, 0.7, 0.8, 0.9, 0.95, 0.96])

        summary = summarize_fairness_by_group(evaluation_frame, result, min_group_size=2)

        self.assertIn("overall", summary)
        self.assertIn("Gender", summary["groups"])
        self.assertIn("gender_0", summary["groups"]["Gender"])


if __name__ == "__main__":
    unittest.main()