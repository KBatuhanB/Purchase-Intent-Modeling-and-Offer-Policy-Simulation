from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from veri_madenciligi.services.phase4_baseline import (
    BaselineModelResult,
    RuleBasedBaselineClassifier,
    build_calibration_summary,
    choose_recommended_baseline,
    evaluate_binary_classifier,
)


class Phase4BaselineHelpersTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = pd.DataFrame(
            {
                "NumberOfPurchases": [0, 1, 8, 12],
                "TimeSpentOnWebsite": [10.0, 20.0, 35.0, 40.0],
                "LoyaltyProgram": [0, 0, 1, 1],
            }
        )

    def test_rule_based_baseline_classifier_outputs_probabilities_in_unit_interval(self) -> None:
        classifier = RuleBasedBaselineClassifier().fit(self.dataset)

        probabilities = classifier.predict_proba(self.dataset)

        self.assertEqual(probabilities.shape, (4, 2))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))

    def test_evaluate_binary_classifier_returns_expected_metric_keys(self) -> None:
        result = evaluate_binary_classifier(
            name="dummy",
            family="test",
            y_true=pd.Series([0, 0, 1, 1]),
            predictions=np.array([0, 1, 1, 1]),
            probabilities=np.array([0.1, 0.6, 0.8, 0.9]),
            note="test",
        )

        self.assertIn("roc_auc", result.metrics)
        self.assertIn("pr_auc", result.metrics)
        self.assertIn("confusion_matrix", result.metrics)

    def test_choose_recommended_baseline_prefers_logistic_when_gap_is_small(self) -> None:
        logistic = BaselineModelResult(
            name="logistic_regression",
            family="linear_model",
            metrics={
                "accuracy": 0.74,
                "precision": 0.71,
                "recall": 0.70,
                "f1": 0.705,
                "roc_auc": 0.82,
                "pr_auc": 0.79,
                "balanced_accuracy": 0.74,
                "brier_score": 0.17,
                "confusion_matrix": [[70, 10], [12, 50]],
            },
            calibration=build_calibration_summary(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.7, 0.9])),
            note="logistic",
            probabilities=[0.1, 0.4, 0.7, 0.9],
            predictions=[0, 0, 1, 1],
        )
        tree = BaselineModelResult(
            name="decision_tree",
            family="tree_model",
            metrics={
                "accuracy": 0.75,
                "precision": 0.72,
                "recall": 0.71,
                "f1": 0.715,
                "roc_auc": 0.83,
                "pr_auc": 0.80,
                "balanced_accuracy": 0.751,
                "brier_score": 0.185,
                "confusion_matrix": [[71, 9], [11, 51]],
            },
            calibration=build_calibration_summary(np.array([0, 0, 1, 1]), np.array([0.05, 0.3, 0.8, 0.95])),
            note="tree",
            probabilities=[0.05, 0.3, 0.8, 0.95],
            predictions=[0, 0, 1, 1],
        )

        recommendation = choose_recommended_baseline(
            {
                "logistic_regression": logistic,
                "decision_tree": tree,
            }
        )

        self.assertEqual(recommendation["recommended_model"], "logistic_regression")


if __name__ == "__main__":
    unittest.main()