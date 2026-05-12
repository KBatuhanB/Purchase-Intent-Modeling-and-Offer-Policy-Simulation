import unittest

import pandas as pd

from veri_madenciligi.services.phase8_policy import (
    assign_score_band,
    build_guardrails,
    compute_proxy_business_value,
    evaluate_policy_scenarios,
    recommend_policy_action,
    summarize_scenario_policy_checks,
)


DECISION_BANDS = {
    "low_action_threshold": 0.2,
    "binary_decision_threshold": 0.3,
    "high_confidence_no_discount_threshold": 0.45,
}


class Phase8PolicyHelperTests(unittest.TestCase):
    def test_assign_score_band_uses_phase5_thresholds(self) -> None:
        self.assertEqual(assign_score_band(0.1, DECISION_BANDS), "low_intent_holdout")
        self.assertEqual(assign_score_band(0.25, DECISION_BANDS), "lower_targeting_band")
        self.assertEqual(assign_score_band(0.35, DECISION_BANDS), "upper_targeting_band")
        self.assertEqual(assign_score_band(0.7, DECISION_BANDS), "high_confidence_no_discount")

    def test_recommend_policy_action_downgrades_discount_for_heavy_history(self) -> None:
        recommendation = recommend_policy_action(
            probability=0.24,
            discounts_availed=3,
            decision_bands=DECISION_BANDS,
        )

        self.assertEqual(recommendation["recommended_action"], "targeted_light_discount")
        self.assertIn("discount_intensity_downgraded", recommendation["guardrail_flags"])

    def test_recommend_policy_action_requires_manual_review_when_discount_history_is_saturated(self) -> None:
        recommendation = recommend_policy_action(
            probability=0.29,
            discounts_availed=4,
            decision_bands=DECISION_BANDS,
        )

        self.assertEqual(recommendation["recommended_action"], "manual_review_discount_cap")
        self.assertTrue(recommendation["requires_manual_review"])

    def test_compute_proxy_business_value_returns_positive_delta_for_target_band(self) -> None:
        value_summary = compute_proxy_business_value(
            probability=0.24,
            discount_rate=0.1,
            expected_uplift=0.15,
            contact_cost=0.5,
        )

        self.assertGreater(value_summary["proxy_incremental_value"], 0.0)

    def test_build_guardrails_includes_loyalty_monitor_alert(self) -> None:
        phase7_summary = {
            "phase7_closeout": {"champion_key": "random_forest"},
            "model_explanations": {
                "random_forest": {
                    "fairness_summary": {
                        "alerts": [
                            {
                                "message": "LoyaltyProgram:loyalty_0 icin recall farki yuksek bulundu.",
                            }
                        ]
                    }
                }
            },
        }
        policy_frame = pd.DataFrame(
            {
                "LoyaltyProgram": [0, 0, 1, 1],
                "discount_offered": [True, True, False, False],
                "requires_manual_review": [False, False, True, False],
            }
        )

        guardrails = build_guardrails(phase7_summary=phase7_summary, policy_frame=policy_frame)
        loyalty_guardrail = next(item for item in guardrails if item["guardrail_id"] == "loyalty_fairness_monitor")

        self.assertEqual(loyalty_guardrail["status"], "alert")
        self.assertIn("discount_action_share_by_loyalty", loyalty_guardrail["monitor"])

    def test_evaluate_policy_scenarios_marks_boundary_case_for_review(self) -> None:
        scenario_catalog = [
            {
                "scenario_id": "boundary_case",
                "title": "Esik Yakini Belirsiz Vaka",
                "predicted_probability": 0.500736,
                "true_label": 1,
                "feature_snapshot": {"DiscountsAvailed": 4},
            },
            {
                "scenario_id": "false_positive",
                "title": "Yanlis Pozitif",
                "predicted_probability": 0.913726,
                "true_label": 0,
                "feature_snapshot": {"DiscountsAvailed": 2},
            },
        ]
        champion_local_explanations = [
            {
                "scenario_id": "boundary_case",
                "top_contributions": [
                    {"feature_name": "NumberOfPurchases"},
                    {"feature_name": "LoyaltyProgram"},
                ],
            },
            {
                "scenario_id": "false_positive",
                "top_contributions": [
                    {"feature_name": "Age"},
                    {"feature_name": "TimeSpentOnWebsite"},
                ],
            },
        ]

        evaluations = evaluate_policy_scenarios(
            scenario_catalog=scenario_catalog,
            champion_local_explanations=champion_local_explanations,
            decision_bands=DECISION_BANDS,
        )
        checks = summarize_scenario_policy_checks(evaluations)

        boundary_case = next(item for item in evaluations if item["scenario_id"] == "boundary_case")
        false_positive = next(item for item in evaluations if item["scenario_id"] == "false_positive")

        self.assertTrue(boundary_case["requires_manual_review"])
        self.assertEqual(false_positive["recommended_discount_rate"], 0.0)
        self.assertTrue(checks["false_positive_margin_protection"])


if __name__ == "__main__":
    unittest.main()