import unittest

import numpy as np
import pandas as pd

from veri_madenciligi.services.phase4_baseline import BaselineModelResult
from veri_madenciligi.services.phase6_advanced_modeling import (
    choose_champion_and_challenger,
    choose_model_variant,
    compute_slice_metrics,
    summarize_cross_validation,
    summarize_segment_performance,
)


def make_result(
    name: str,
    pr_auc: float,
    balanced_accuracy: float,
    roc_auc: float,
    brier_score: float,
    calibration_gap: float,
) -> BaselineModelResult:
    return BaselineModelResult(
        name=name,
        family="test",
        metrics={
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "balanced_accuracy": balanced_accuracy,
            "brier_score": brier_score,
            "confusion_matrix": [[0, 0], [0, 0]],
        },
        calibration={
            "bin_true_rate": [],
            "bin_predicted_probability": [],
            "mean_absolute_gap": calibration_gap,
        },
        note="",
        probabilities=[0.1, 0.9],
        predictions=[0, 1],
    )


class Phase6AdvancedModelingHelperTests(unittest.TestCase):
    def test_choose_model_variant_prefers_calibrated_when_brier_gain_is_clear(self) -> None:
        base_result = make_result("random_forest_base", 0.80, 0.74, 0.82, 0.19, 0.09)
        calibrated_result = make_result("random_forest_sigmoid_calibrated", 0.798, 0.739, 0.821, 0.17, 0.05)

        decision = choose_model_variant(base_result, calibrated_result)

        self.assertEqual(decision["selected_variant"], "sigmoid_calibrated")

    def test_choose_model_variant_keeps_base_when_pr_auc_drops_too_much(self) -> None:
        base_result = make_result("hist_gradient_boosting_base", 0.84, 0.78, 0.85, 0.16, 0.07)
        calibrated_result = make_result("hist_gradient_boosting_sigmoid_calibrated", 0.82, 0.78, 0.84, 0.15, 0.05)

        decision = choose_model_variant(base_result, calibrated_result)

        self.assertEqual(decision["selected_variant"], "base")

    def test_choose_champion_and_challenger_prefers_simpler_family_when_gap_is_small(self) -> None:
        results = {
            "hist_gradient_boosting": make_result("hist_gradient_boosting_base", 0.84, 0.78, 0.85, 0.16, 0.05),
            "random_forest": make_result("random_forest_base", 0.835, 0.776, 0.84, 0.161, 0.05),
            "gradient_boosting": make_result("gradient_boosting_base", 0.80, 0.74, 0.82, 0.18, 0.07),
        }

        decision = choose_champion_and_challenger(results)

        self.assertEqual(decision["champion_key"], "random_forest")
        self.assertEqual(decision["challenger_key"], "hist_gradient_boosting")

    def test_summarize_cross_validation_returns_mean_and_std(self) -> None:
        summary = summarize_cross_validation(
            {
                "test_pr_auc": np.asarray([0.81, 0.82, 0.80]),
                "test_balanced_accuracy": np.asarray([0.75, 0.74, 0.76]),
                "test_roc_auc": np.asarray([0.84, 0.85, 0.83]),
                "fit_time": np.asarray([0.1, 0.2, 0.3]),
                "score_time": np.asarray([0.01, 0.01, 0.02]),
            }
        )

        self.assertAlmostEqual(summary["metrics"]["pr_auc"]["mean"], 0.81)
        self.assertAlmostEqual(summary["metrics"]["balanced_accuracy"]["std"], 0.008165, places=6)

    def test_compute_slice_metrics_returns_confusion_matrix(self) -> None:
        metrics = compute_slice_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0, 1, 1, 1]),
            np.asarray([0.2, 0.7, 0.8, 0.9]),
        )

        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(metrics["precision"], 0.666667)

    def test_summarize_segment_performance_filters_small_slices(self) -> None:
        evaluation_frame = pd.DataFrame(
            {
                "Age": [25, 26, 40, 41, 52, 53],
                "Gender": [0, 0, 1, 1, 0, 1],
                "AnnualIncome": [40000, 42000, 65000, 68000, 110000, 112000],
                "NumberOfPurchases": [1, 2, 3, 4, 5, 6],
                "ProductCategory": [0, 0, 1, 1, 2, 2],
                "TimeSpentOnWebsite": [10, 12, 20, 22, 35, 40],
                "LoyaltyProgram": [0, 0, 1, 1, 1, 1],
                "DiscountsAvailed": [0, 0, 1, 1, 0, 1],
                "PurchaseStatus": [0, 0, 1, 1, 1, 1],
            }
        )
        result = BaselineModelResult(
            name="rf",
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
            probabilities=[0.1, 0.2, 0.7, 0.8, 0.9, 0.95],
            predictions=[0, 0, 1, 1, 1, 1],
        )

        summary = summarize_segment_performance(evaluation_frame, result, min_slice_size=2)

        self.assertIn("Gender", summary)
        self.assertNotIn("unknown", summary.get("AgeBand", {}))


if __name__ == "__main__":
    unittest.main()