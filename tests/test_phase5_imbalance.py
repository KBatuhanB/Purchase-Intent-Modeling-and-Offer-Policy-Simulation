import unittest

import numpy as np
import pandas as pd

from veri_madenciligi.services.phase4_baseline import BaselineModelResult
from veri_madenciligi.services.phase5_imbalance import (
    build_decision_bands,
    choose_optimal_threshold,
    choose_recommended_imbalance_strategy,
    compute_threshold_metrics,
    profile_imbalance,
)


def make_result(name: str, pr_auc: float, recall: float, balanced_accuracy: float, precision: float) -> BaselineModelResult:
    return BaselineModelResult(
        name=name,
        family="test",
        metrics={
            "accuracy": 0.0,
            "precision": precision,
            "recall": recall,
            "f1": 0.0,
            "roc_auc": 0.0,
            "pr_auc": pr_auc,
            "balanced_accuracy": balanced_accuracy,
            "brier_score": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
        },
        calibration={"bin_true_rate": [], "bin_predicted_probability": [], "mean_absolute_gap": None},
        note="",
        probabilities=[0.1, 0.9],
        predictions=[0, 1],
    )


class Phase5ImbalanceHelperTests(unittest.TestCase):
    def test_profile_imbalance_reports_mild_ratio(self) -> None:
        profile = profile_imbalance(pd.Series([0] * 80 + [1] * 60))
        self.assertEqual(profile["severity"], "mild")
        self.assertAlmostEqual(profile["minority_to_majority_ratio"], 0.75)

    def test_compute_threshold_metrics_returns_expected_confusion_matrix(self) -> None:
        metrics = compute_threshold_metrics(
            pd.Series([0, 0, 1, 1]),
            np.asarray([0.1, 0.7, 0.6, 0.9]),
            0.65,
        )
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)

    def test_choose_optimal_threshold_returns_ordered_bands(self) -> None:
        selection = choose_optimal_threshold(
            pd.Series([0, 0, 1, 1, 1, 0]),
            np.asarray([0.12, 0.28, 0.52, 0.63, 0.81, 0.47]),
        )
        bands = selection["decision_bands"]
        self.assertLess(bands["low_action_threshold"], bands["binary_decision_threshold"])
        self.assertLess(bands["binary_decision_threshold"], bands["high_confidence_no_discount_threshold"])

    def test_choose_recommended_strategy_prefers_reference_when_gain_is_small(self) -> None:
        results = {
            "logistic_reference": make_result("logistic_reference", 0.79, 0.70, 0.72, 0.74),
            "logistic_class_weight": make_result("logistic_class_weight", 0.80, 0.72, 0.73, 0.71),
            "smote_logistic": make_result("smote_logistic", 0.805, 0.73, 0.735, 0.70),
        }
        recommendation = choose_recommended_imbalance_strategy(results)
        self.assertEqual(recommendation["recommended_strategy"], "logistic_reference")

    def test_build_decision_bands_stays_ordered_near_edges(self) -> None:
        bands = build_decision_bands(0.22)
        self.assertLess(bands["low_action_threshold"], bands["binary_decision_threshold"])
        self.assertLess(bands["binary_decision_threshold"], bands["high_confidence_no_discount_threshold"])


if __name__ == "__main__":
    unittest.main()