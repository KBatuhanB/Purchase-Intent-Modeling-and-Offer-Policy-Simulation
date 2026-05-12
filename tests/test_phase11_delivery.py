import tempfile
import unittest
from pathlib import Path

from veri_madenciligi.core.exceptions import DataAnalysisError
from veri_madenciligi.services.phase11_delivery import (
    build_delivery_manifest,
    build_demo_report_mapping,
    build_future_work_items,
    build_presentation_layers,
    build_visual_asset_library,
    validate_phase11_prerequisites,
)


class Phase11DeliveryHelperTests(unittest.TestCase):
    def test_validate_phase11_prerequisites_rejects_champion_mismatch(self) -> None:
        with self.assertRaises(DataAnalysisError):
            validate_phase11_prerequisites(
                dataset_row_count=10,
                deduplicated_row_count=8,
                phase1_summary={
                    "row_count": 10,
                    "duplicate_summary": {"duplicate_row_count": 2},
                },
                phase6_summary={"deduplicated_row_count": 8},
                phase7_summary={
                    "deduplicated_row_count": 8,
                    "phase6_context": {"champion_selection": {"champion_model": "model_a"}},
                },
                phase8_summary={
                    "deduplicated_row_count": 8,
                    "phase6_context": {"champion_selection": {"champion_model": "model_b"}},
                },
                phase9_summary={"deduplicated_row_count": 8, "champion_model_name": "model_a"},
                phase10_summary={
                    "deduplicated_row_count": 8,
                    "champion_model_name": "model_a",
                    "phase10_closeout": {"overall_status": "hazir"},
                },
            )

    def test_build_presentation_layers_returns_all_expected_levels(self) -> None:
        layers = build_presentation_layers(
            phase1_summary={"duplicate_summary": {"duplicate_rate": 0.074667}, "final_decision": "Conditional Go"},
            phase6_summary={},
            phase8_summary={
                "phase6_context": {
                    "validated_champion_metrics": {
                        "accuracy": 0.73,
                        "pr_auc": 0.83,
                        "roc_auc": 0.85,
                        "precision": 0.71,
                        "recall": 0.72,
                    }
                }
            },
            phase9_summary={"scenario_catalog_overview": {"total_scenarios": 12}},
            phase10_summary={"phase10_closeout": {"overall_status": "hazir"}},
            champion_model_name="rf_calibrated",
        )

        self.assertIn("thirty_second_pitch", layers)
        self.assertEqual(len(layers["three_minute_pitch"]), 3)
        self.assertGreaterEqual(len(layers["ten_minute_defense_outline"]), 4)

    def test_build_demo_report_mapping_preserves_runbook_order(self) -> None:
        mapping = build_demo_report_mapping(
            phase9_summary={
                "demo_runbook": [
                    {
                        "step_number": 2,
                        "step_title": "Demo",
                        "scenario_id": "case_2",
                        "scenario_title": "Ikinci",
                        "takeaway": "orta risk",
                        "expected_output": "indirim",
                    },
                    {
                        "step_number": 1,
                        "step_title": "Demo",
                        "scenario_id": "case_1",
                        "scenario_title": "Birinci",
                        "takeaway": "yuksek guven",
                        "expected_output": "marj koru",
                    },
                ],
                "simulation_results": [
                    {
                        "scenario_id": "case_1",
                        "scenario_origin": "phase7_reference",
                        "predicted_probability": 0.91,
                        "recommended_action": "protect_margin_no_discount",
                        "requires_manual_review": False,
                    },
                    {
                        "scenario_id": "case_2",
                        "scenario_origin": "phase9_synthetic",
                        "predicted_probability": 0.28,
                        "recommended_action": "targeted_standard_discount",
                        "requires_manual_review": True,
                    },
                ],
            }
        )

        self.assertEqual(mapping[0]["scenario_id"], "case_2")
        self.assertIn("Demo ve simülasyon", mapping[0]["report_sections"])

    def test_build_future_work_items_translates_limitations(self) -> None:
        items = build_future_work_items(
            limitations=[
                {"limitation_id": "discount_history_not_causal_signal", "severity": "high"},
                {"limitation_id": "processed_dataset_generalization_risk", "severity": "medium"},
            ],
            readiness_status="hazir",
        )

        self.assertEqual(items[0]["priority"], "high")
        self.assertIn("Uplift", items[0]["workstream"])

    def test_build_visual_asset_library_collects_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            plot_dir = root_dir / "artifacts" / "phase_2" / "plots"
            scenario_dir = root_dir / "artifacts" / "phase_9" / "scenarios"
            plot_dir.mkdir(parents=True)
            scenario_dir.mkdir(parents=True)
            (plot_dir / "eda_plot.png").write_text("png-placeholder", encoding="utf-8")
            (scenario_dir / "01_demo.md").write_text("demo-placeholder", encoding="utf-8")

            asset_library = build_visual_asset_library(
                root_dir=root_dir,
                asset_groups=[
                    ("phase_2", "plots", plot_dir),
                    ("phase_9", "scenario_cards", scenario_dir),
                ],
            )

            self.assertEqual(asset_library["total_asset_count"], 2)
            self.assertEqual(asset_library["phase_counts"]["phase_2"], 1)

    def test_build_delivery_manifest_references_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            generated_output = root_dir / "artifacts" / "phase_11" / "final_report.md"
            source_dependency = root_dir / "artifacts" / "phase_10" / "validation_report.md"
            generated_output.parent.mkdir(parents=True)
            source_dependency.parent.mkdir(parents=True)
            generated_output.write_text("report", encoding="utf-8")
            source_dependency.write_text("validation", encoding="utf-8")

            manifest = build_delivery_manifest(
                root_dir=root_dir,
                dataset_path=root_dir / "customer_purchase_data.csv",
                champion_model_name="rf_calibrated",
                readiness_status="hazir",
                generated_outputs=[generated_output],
                source_dependencies=[source_dependency],
                key_metrics={"accuracy": 0.73},
                visual_asset_count=4,
                demo_mapping_count=8,
            )

            self.assertTrue(manifest["ready_for_submission"])
            self.assertEqual(manifest["generated_outputs"][0], "artifacts/phase_11/final_report.md")


if __name__ == "__main__":
    unittest.main()