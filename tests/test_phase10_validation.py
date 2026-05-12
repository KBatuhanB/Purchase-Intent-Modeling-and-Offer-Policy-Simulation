import unittest

import numpy as np
import pandas as pd

from veri_madenciligi.services.phase10_validation import (
    build_edge_case_catalog,
    build_limitations_catalog,
    build_readiness_checklist,
    build_sensitivity_variants,
    benchmark_runtime_surface,
    evaluate_edge_case_matrix,
    evaluate_reproducibility_surface,
    evaluate_sensitivity_suite,
)
from veri_madenciligi.services.phase3_preprocessing import build_feature_role_matrix
from veri_madenciligi.services.phase9_simulation import build_input_validation_rules


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


def _build_reference_scenarios() -> list[dict[str, object]]:
    return [
        {
            "scenario_id": "high_confidence_true_positive",
            "title": "Yuksek Guvenli Dogru Pozitif",
            "scenario_origin": "phase7_reference",
            "selection_reason": "Referans pozitif vaka.",
            "input_payload": {
                "Age": 35,
                "Gender": 1,
                "AnnualIncome": 80000.0,
                "NumberOfPurchases": 8,
                "ProductCategory": 2,
                "TimeSpentOnWebsite": 35.0,
                "LoyaltyProgram": 1,
                "DiscountsAvailed": 2,
            },
        },
        {
            "scenario_id": "boundary_case",
            "title": "Sinir Vaka",
            "scenario_origin": "phase7_reference",
            "selection_reason": "Sinir vakasi.",
            "input_payload": {
                "Age": 18,
                "Gender": 0,
                "AnnualIncome": 90000.0,
                "NumberOfPurchases": 0,
                "ProductCategory": 0,
                "TimeSpentOnWebsite": 45.0,
                "LoyaltyProgram": 0,
                "DiscountsAvailed": 4,
            },
        },
        {
            "scenario_id": "non_loyal_high_time",
            "title": "Sadakatsiz Ama Yuksek Ilgi",
            "scenario_origin": "phase9_synthetic",
            "selection_reason": "Hedefli teklif yuzeyi.",
            "input_payload": {
                "Age": 35,
                "Gender": 0,
                "AnnualIncome": 120000.0,
                "NumberOfPurchases": 3,
                "ProductCategory": 2,
                "TimeSpentOnWebsite": 48.0,
                "LoyaltyProgram": 0,
                "DiscountsAvailed": 1,
            },
        },
        {
            "scenario_id": "synthetic_loyal_discount_fatigue",
            "title": "Sadik Ama Yorgun",
            "scenario_origin": "phase9_synthetic",
            "selection_reason": "Guardrail testi.",
            "input_payload": {
                "Age": 70,
                "Gender": 0,
                "AnnualIncome": 120000.0,
                "NumberOfPurchases": 20,
                "ProductCategory": 4,
                "TimeSpentOnWebsite": 60.0,
                "LoyaltyProgram": 1,
                "DiscountsAvailed": 5,
            },
        },
    ]


class Phase10ValidationHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = _build_reference_dataset()
        self.validation_rules = build_input_validation_rules(self.dataset)
        self.role_matrix = build_feature_role_matrix()
        self.stub_model = StubChampionModel()
        self.baseline_scenarios = _build_reference_scenarios()
        self.baseline_results_lookup = {}
        from veri_madenciligi.services.phase9_simulation import simulate_customer_profile

        for scenario in self.baseline_scenarios:
            result = simulate_customer_profile(
                scenario=scenario,
                champion_model=self.stub_model,
                role_matrix=self.role_matrix,
                validation_rules=self.validation_rules,
                decision_bands=DECISION_BANDS,
            )
            self.baseline_results_lookup[scenario["scenario_id"]] = result

    def test_build_edge_case_catalog_contains_valid_and_invalid_cases(self) -> None:
        catalog = build_edge_case_catalog(self.validation_rules)

        self.assertGreaterEqual(len(catalog), 8)
        self.assertTrue(any(item["expected_valid"] for item in catalog))
        self.assertTrue(any(not item["expected_valid"] for item in catalog))

    def test_evaluate_edge_case_matrix_marks_invalid_inputs_as_controlled_rejections(self) -> None:
        catalog = build_edge_case_catalog(self.validation_rules)
        results = evaluate_edge_case_matrix(
            edge_cases=catalog,
            champion_model=self.stub_model,
            role_matrix=self.role_matrix,
            validation_rules=self.validation_rules,
            decision_bands=DECISION_BANDS,
        )

        invalid_result = next(item for item in results if item["case_id"] == "invalid_age_below_min")
        valid_result = next(item for item in results if item["case_id"] == "valid_min_boundary_profile")

        self.assertEqual(invalid_result["observed_status"], "passed")
        self.assertEqual(valid_result["observed_status"], "passed")

    def test_build_sensitivity_variants_returns_unique_bounded_variants(self) -> None:
        base_input = self.baseline_scenarios[0]["input_payload"]
        variants = build_sensitivity_variants(base_input, self.validation_rules)

        self.assertGreaterEqual(len(variants), 4)
        unique_payloads = {tuple(sorted(item["input_payload"].items())) for item in variants}
        self.assertEqual(len(unique_payloads), len(variants))

    def test_evaluate_sensitivity_suite_produces_variant_summaries(self) -> None:
        results = evaluate_sensitivity_suite(
            baseline_scenarios=self.baseline_scenarios,
            baseline_results_lookup=self.baseline_results_lookup,
            champion_model=self.stub_model,
            role_matrix=self.role_matrix,
            validation_rules=self.validation_rules,
            decision_bands=DECISION_BANDS,
        )

        self.assertTrue(results)
        self.assertIn("variants", results[0])
        self.assertIn(results[0]["stability_assessment"], {"stable", "monitor"})

    def test_evaluate_reproducibility_surface_reports_deterministic_behavior(self) -> None:
        summary = evaluate_reproducibility_surface(
            baseline_scenarios=self.baseline_scenarios,
            baseline_results_lookup=self.baseline_results_lookup,
            champion_model=self.stub_model,
            role_matrix=self.role_matrix,
            validation_rules=self.validation_rules,
            decision_bands=DECISION_BANDS,
            reruns=2,
        )

        self.assertTrue(summary["deterministic"])
        self.assertEqual(summary["action_mismatch_count"], 0)

    def test_benchmark_and_readiness_helpers_return_expected_structure(self) -> None:
        performance_summary = benchmark_runtime_surface(
            baseline_scenarios=self.baseline_scenarios,
            champion_model=self.stub_model,
            role_matrix=self.role_matrix,
            validation_rules=self.validation_rules,
            decision_bands=DECISION_BANDS,
        )
        limitations = build_limitations_catalog(
            raw_row_count=1500,
            deduplicated_row_count=1388,
            phase7_summary={"phase7_closeout": {}},
            phase8_summary={"phase7_context": {"fairness_alerts": []}},
        )
        readiness_checklist = build_readiness_checklist(
            edge_case_results=[{"observed_status": "passed"}],
            sensitivity_results=[{"scenario_id": "a"}],
            performance_summary=performance_summary,
            reproducibility_summary={
                "deterministic": True,
                "max_probability_delta": 0.0,
            },
            limitations=limitations,
            phase8_summary={"phase7_context": {"fairness_alerts": []}},
            phase9_summary={"simulation_results": [{"scenario_id": "a"}]},
        )

        self.assertIn("single_scenario_avg_ms", performance_summary)
        self.assertTrue(limitations)
        self.assertEqual(readiness_checklist[0]["status"], "hazir")


if __name__ == "__main__":
    unittest.main()