import unittest

import numpy as np
import pandas as pd

from veri_madenciligi.core.exceptions import DataAnalysisError
from veri_madenciligi.services.phase3_preprocessing import build_feature_role_matrix
from veri_madenciligi.services.phase9_simulation import (
    build_demo_runbook,
    build_demo_scenario_catalog,
    build_input_validation_rules,
    build_simulation_log_frame,
    sanitize_simulation_input,
    simulate_customer_profile,
)


DECISION_BANDS = {
    "low_action_threshold": 0.2,
    "binary_decision_threshold": 0.3,
    "high_confidence_no_discount_threshold": 0.45,
}


class StubChampionModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = []
        for row in frame.itertuples(index=False):
            if row.LoyaltyProgram == 1 and row.TimeSpentOnWebsite >= 30:
                probability = 0.72
            elif row.TimeSpentOnWebsite >= 40:
                probability = 0.26
            else:
                probability = 0.12
            probabilities.append([1.0 - probability, probability])
        return np.asarray(probabilities, dtype=float)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(frame)[:, 1] >= 0.5).astype(int)


def _build_reference_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": [18, 35, 70],
            "Gender": [0, 1, 0],
            "AnnualIncome": [20000.0, 80000.0, 150000.0],
            "NumberOfPurchases": [0, 8, 20],
            "ProductCategory": [0, 2, 4],
            "TimeSpentOnWebsite": [1.0, 35.0, 60.0],
            "LoyaltyProgram": [0, 1, 1],
            "DiscountsAvailed": [0, 2, 5],
            "PurchaseStatus": [0, 1, 1],
        }
    )


class Phase9SimulationHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = _build_reference_dataset()
        self.validation_rules = build_input_validation_rules(self.dataset)
        self.role_matrix = build_feature_role_matrix()
        self.stub_model = StubChampionModel()

    def test_build_input_validation_rules_exposes_allowed_values_and_ranges(self) -> None:
        self.assertEqual(self.validation_rules["Age"]["min_value"], 18)
        self.assertEqual(self.validation_rules["Age"]["max_value"], 70)
        self.assertEqual(self.validation_rules["ProductCategory"]["allowed_values"], [0, 2, 4])

    def test_sanitize_simulation_input_rejects_out_of_range_values(self) -> None:
        with self.assertRaises(DataAnalysisError):
            sanitize_simulation_input(
                {
                    "Age": 99,
                    "Gender": 0,
                    "AnnualIncome": 80000.0,
                    "NumberOfPurchases": 2,
                    "ProductCategory": 0,
                    "TimeSpentOnWebsite": 10.0,
                    "LoyaltyProgram": 0,
                    "DiscountsAvailed": 1,
                },
                self.validation_rules,
            )

    def test_build_demo_scenario_catalog_adds_synthetic_variants(self) -> None:
        reference_scenarios = [
            {
                "scenario_id": "reference_a",
                "title": "Referans A",
                "selection_reason": "Tipik pozitif senaryo.",
                "predicted_probability": 0.7,
                "true_label": 1,
                "feature_snapshot": self.dataset.iloc[1].to_dict(),
            },
            {
                "scenario_id": "reference_b",
                "title": "Referans B",
                "selection_reason": "Tipik negatif senaryo.",
                "predicted_probability": 0.1,
                "true_label": 0,
                "feature_snapshot": self.dataset.iloc[0].to_dict(),
            },
        ]

        catalog = build_demo_scenario_catalog(reference_scenarios, self.validation_rules)

        self.assertEqual(len(catalog), 6)
        self.assertEqual(len({item["scenario_id"] for item in catalog}), 6)

    def test_simulate_customer_profile_returns_standardized_output(self) -> None:
        scenario = {
            "scenario_id": "target_offer",
            "title": "Hedefli Teklif Vaka",
            "scenario_origin": "phase9_synthetic",
            "selection_reason": "Orta band senaryosu.",
            "input_payload": {
                "Age": 35,
                "Gender": 0,
                "AnnualIncome": 80000.0,
                "NumberOfPurchases": 4,
                "ProductCategory": 2,
                "TimeSpentOnWebsite": 45.0,
                "LoyaltyProgram": 0,
                "DiscountsAvailed": 1,
            },
        }

        result = simulate_customer_profile(
            scenario=scenario,
            champion_model=self.stub_model,
            role_matrix=self.role_matrix,
            validation_rules=self.validation_rules,
            decision_bands=DECISION_BANDS,
            global_driver_hints=["TimeSpentOnWebsite", "LoyaltyProgram"],
        )

        self.assertEqual(result["recommended_action"], "targeted_standard_discount")
        self.assertIn("purchase_score", result["standard_output"])
        self.assertEqual(result["standard_output"]["risk_band"], "lower_targeting_band")

    def test_build_demo_runbook_selects_core_demo_steps(self) -> None:
        simulation_results = [
            {
                "scenario_id": "margin_case",
                "title": "Marj Koruma",
                "recommended_action": "protect_margin_no_discount",
                "recommended_discount_rate": 0.0,
                "requires_manual_review": False,
                "guardrail_flags": [],
                "standard_output": {"recommended_action": "protect_margin_no_discount"},
            },
            {
                "scenario_id": "offer_case",
                "title": "Teklif",
                "recommended_action": "targeted_standard_discount",
                "recommended_discount_rate": 0.1,
                "requires_manual_review": False,
                "guardrail_flags": [],
                "standard_output": {"recommended_action": "targeted_standard_discount"},
            },
            {
                "scenario_id": "review_case",
                "title": "Manual Review",
                "recommended_action": "manual_review_discount_cap",
                "recommended_discount_rate": 0.05,
                "requires_manual_review": True,
                "guardrail_flags": ["historical_discount_saturation"],
                "standard_output": {"recommended_action": "manual_review_discount_cap"},
            },
        ]

        runbook = build_demo_runbook(simulation_results)

        self.assertEqual(len(runbook), 3)
        self.assertEqual(runbook[0]["scenario_id"], "margin_case")
        self.assertEqual(runbook[1]["scenario_id"], "offer_case")
        self.assertEqual(runbook[2]["scenario_id"], "review_case")

    def test_build_simulation_log_frame_includes_alignment_columns(self) -> None:
        log_frame = build_simulation_log_frame(
            [
                {
                    "scenario_id": "scenario_a",
                    "title": "Senaryo A",
                    "scenario_origin": "phase7_reference",
                    "predicted_probability": 0.72,
                    "predicted_label": 1,
                    "score_band": "high_confidence_no_discount",
                    "recommended_action": "protect_margin_no_discount",
                    "recommended_discount_rate": 0.0,
                    "requires_manual_review": False,
                    "guardrail_flags": [],
                    "top_driver_hints": ["LoyaltyProgram"],
                    "phase8_reference_alignment": {
                        "action_match": True,
                        "band_match": True,
                        "probability_delta": 0.0,
                    },
                }
            ]
        )

        self.assertIn("phase8_action_match", log_frame.columns)
        self.assertIn("phase8_probability_delta", log_frame.columns)


if __name__ == "__main__":
    unittest.main()