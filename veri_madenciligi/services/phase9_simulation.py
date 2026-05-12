from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from ..config import CATEGORICAL_COLUMNS, EXPECTED_COLUMNS, INTEGER_LIKE_COLUMNS, TARGET_COLUMN, ProjectPaths
from ..core.exceptions import ArtifactWriteError, DataAnalysisError
from ..core.serialization import to_json_safe
from .csv_loader import CsvDatasetLoader
from .phase2_eda import deduplicate_dataset
from .phase3_preprocessing import build_feature_role_matrix, validate_required_columns
from .phase8_policy import recommend_policy_action, validate_decision_bands

SIMULATION_INPUT_COLUMNS = tuple(column for column in EXPECTED_COLUMNS if column != TARGET_COLUMN)

COLUMN_PURPOSES = {
    "Age": "Demografik yas bilgisi; segment ve risk baglamini tasir.",
    "Gender": "Kodlanmis kategorik alan; fairness ve segment izlemesi icin korunur.",
    "AnnualIncome": "Gelir seviyesi; teklif maliyeti ve satin alma niyeti icin vekil sinyaldir.",
    "NumberOfPurchases": "Gecmis satin alma yogunlugu; baglilik ve aliskanlik sinyali verir.",
    "ProductCategory": "Kategori ilgisi; modelin kategori bazli davranisini ayristirmaya yardim eder.",
    "TimeSpentOnWebsite": "Ilgi seviyesi ve karar asamasina yakinlik icin davranissal sinyaldir.",
    "LoyaltyProgram": "Sadakat programi uyeligi; policy fairness guardrail'leri icin de izlenir.",
    "DiscountsAvailed": "Ana model girdisi degil; Faz 9'da yalnizca guardrail ve teklif yogunlugu kontrolu icin kullanilir.",
}


@dataclass(frozen=True)
class Phase9Artifacts:
    markdown_report_path: Path
    json_summary_path: Path
    input_schema_path: Path
    simulation_log_path: Path
    demo_runbook_path: Path
    scenarios_dir: Path


def build_input_validation_rules(dataset: pd.DataFrame) -> dict[str, dict[str, Any]]:
    validate_required_columns(dataset, EXPECTED_COLUMNS)
    rules: dict[str, dict[str, Any]] = {}

    for column in SIMULATION_INPUT_COLUMNS:
        numeric_series = pd.to_numeric(dataset[column], errors="coerce")
        if numeric_series.isna().all():
            raise DataAnalysisError(f"Faz 9 icin {column} sutununda gecerli sayisal deger bulunamadi.")

        base_rule = {
            "required": True,
            "purpose": COLUMN_PURPOSES[column],
        }
        if column in CATEGORICAL_COLUMNS:
            allowed_values = sorted({int(value) for value in numeric_series.dropna().unique()})
            rules[column] = {
                **base_rule,
                "kind": "categorical_integer",
                "allowed_values": allowed_values,
                "example_value": allowed_values[0],
            }
        elif column in INTEGER_LIKE_COLUMNS:
            min_value = int(numeric_series.min())
            max_value = int(numeric_series.max())
            rules[column] = {
                **base_rule,
                "kind": "integer_range",
                "min_value": min_value,
                "max_value": max_value,
                "example_value": int(round((min_value + max_value) / 2)),
            }
        else:
            min_value = float(numeric_series.min())
            max_value = float(numeric_series.max())
            rules[column] = {
                **base_rule,
                "kind": "float_range",
                "min_value": round(min_value, 6),
                "max_value": round(max_value, 6),
                "example_value": round((min_value + max_value) / 2.0, 6),
            }

    return rules


def sanitize_simulation_input(
    raw_input: Mapping[str, Any],
    validation_rules: dict[str, dict[str, Any]],
) -> dict[str, int | float]:
    if not isinstance(raw_input, Mapping):
        raise DataAnalysisError("Faz 9 simülasyon girdisi key/value yapisinda olmalidir.")

    unexpected_keys = sorted(set(raw_input) - set(validation_rules))
    if unexpected_keys:
        raise DataAnalysisError(f"Faz 9 icin beklenmeyen girdi alanlari alindi: {unexpected_keys}")

    missing_columns = [column for column, rule in validation_rules.items() if rule["required"] and column not in raw_input]
    if missing_columns:
        raise DataAnalysisError(f"Faz 9 simülasyon girdisinde zorunlu alanlar eksik: {missing_columns}")

    sanitized_payload: dict[str, int | float] = {}
    for column, rule in validation_rules.items():
        numeric_value = _coerce_numeric_scalar(column, raw_input[column])

        if rule["kind"] == "categorical_integer":
            if not float(numeric_value).is_integer():
                raise DataAnalysisError(f"{column} yalnizca tam sayi kategorik deger kabul eder.")
            candidate_value = int(numeric_value)
            if candidate_value not in rule["allowed_values"]:
                raise DataAnalysisError(
                    f"{column} icin izin verilen degerler {rule['allowed_values']} disinda bir deger alindi: {candidate_value}"
                )
            sanitized_payload[column] = candidate_value
            continue

        if rule["kind"] == "integer_range":
            if not float(numeric_value).is_integer():
                raise DataAnalysisError(f"{column} icin tam sayi bir deger bekleniyor.")
            candidate_value = int(numeric_value)
            if candidate_value < rule["min_value"] or candidate_value > rule["max_value"]:
                raise DataAnalysisError(
                    f"{column} icin izin verilen aralik {rule['min_value']} ile {rule['max_value']} arasindadir."
                )
            sanitized_payload[column] = candidate_value
            continue

        candidate_value = float(numeric_value)
        if candidate_value < float(rule["min_value"]) or candidate_value > float(rule["max_value"]):
            raise DataAnalysisError(
                f"{column} icin izin verilen aralik {rule['min_value']} ile {rule['max_value']} arasindadir."
            )
        sanitized_payload[column] = round(candidate_value, 6)

    return sanitized_payload


def build_demo_scenario_catalog(
    reference_scenarios: list[dict[str, Any]],
    validation_rules: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    scenario_catalog: list[dict[str, Any]] = []

    for scenario in reference_scenarios:
        feature_snapshot = scenario.get("feature_snapshot", {})
        input_payload = {
            column: feature_snapshot[column]
            for column in SIMULATION_INPUT_COLUMNS
            if column in feature_snapshot
        }
        scenario_catalog.append(
            {
                "scenario_id": str(scenario["scenario_id"]),
                "title": str(scenario["title"]),
                "selection_reason": str(scenario.get("selection_reason", "Referans senaryo olarak korundu.")),
                "scenario_origin": "phase7_reference",
                "input_payload": sanitize_simulation_input(input_payload, validation_rules),
                "reference_probability": scenario.get("predicted_probability"),
                "reference_label": scenario.get("true_label"),
            }
        )

    # Bu sentetik vakalar, gecerli alan sinirlarini bozmadan demoda gosterilecek koseleri genisletmek icin uretilir.
    synthetic_scenarios = [
        {
            "scenario_id": "synthetic_min_signal_new_visitor",
            "title": "Yeni ve Dusuk Sinyalli Ziyaretci",
            "selection_reason": "Minimum ilgi sinyaline yakin kullanici icin sistemin agressif teklif vermedigini gostermek icin eklendi.",
            "scenario_origin": "phase9_synthetic",
            "input_payload": {
                "Age": _interpolate_rule_value(validation_rules["Age"], 0.0),
                "Gender": _pick_allowed_value(validation_rules["Gender"], 0),
                "AnnualIncome": _interpolate_rule_value(validation_rules["AnnualIncome"], 0.25),
                "NumberOfPurchases": _interpolate_rule_value(validation_rules["NumberOfPurchases"], 0.0),
                "ProductCategory": _pick_allowed_value(validation_rules["ProductCategory"], 0),
                "TimeSpentOnWebsite": _interpolate_rule_value(validation_rules["TimeSpentOnWebsite"], 0.02),
                "LoyaltyProgram": _pick_allowed_value(validation_rules["LoyaltyProgram"], 0),
                "DiscountsAvailed": _interpolate_rule_value(validation_rules["DiscountsAvailed"], 0.0),
            },
        },
        {
            "scenario_id": "synthetic_premium_window_shopper",
            "title": "Premium Gelirli Ama Kararsiz Gezgin",
            "selection_reason": "Geliri yuksek ancak baglilik sinyali zayif musteri profilinde hedefli teklif ihtiyacini gostermek icin eklendi.",
            "scenario_origin": "phase9_synthetic",
            "input_payload": {
                "Age": _interpolate_rule_value(validation_rules["Age"], 0.55),
                "Gender": _pick_allowed_value(validation_rules["Gender"], 1),
                "AnnualIncome": _interpolate_rule_value(validation_rules["AnnualIncome"], 1.0),
                "NumberOfPurchases": _interpolate_rule_value(validation_rules["NumberOfPurchases"], 0.35),
                "ProductCategory": _pick_allowed_value(validation_rules["ProductCategory"], -1),
                "TimeSpentOnWebsite": _interpolate_rule_value(validation_rules["TimeSpentOnWebsite"], 0.95),
                "LoyaltyProgram": _pick_allowed_value(validation_rules["LoyaltyProgram"], 0),
                "DiscountsAvailed": _interpolate_rule_value(validation_rules["DiscountsAvailed"], 0.2),
            },
        },
        {
            "scenario_id": "synthetic_loyal_discount_fatigue",
            "title": "Sadik Ama Indirim Yorgunu Musteri",
            "selection_reason": "Cok fazla gecmis indirim kullanimi olan kullanicida guardrail'in nasil devreye girdigini gostermek icin eklendi.",
            "scenario_origin": "phase9_synthetic",
            "input_payload": {
                "Age": _interpolate_rule_value(validation_rules["Age"], 0.7),
                "Gender": _pick_allowed_value(validation_rules["Gender"], 0),
                "AnnualIncome": _interpolate_rule_value(validation_rules["AnnualIncome"], 0.6),
                "NumberOfPurchases": _interpolate_rule_value(validation_rules["NumberOfPurchases"], 0.85),
                "ProductCategory": _pick_allowed_value(validation_rules["ProductCategory"], 1),
                "TimeSpentOnWebsite": _interpolate_rule_value(validation_rules["TimeSpentOnWebsite"], 0.7),
                "LoyaltyProgram": _pick_allowed_value(validation_rules["LoyaltyProgram"], -1),
                "DiscountsAvailed": _interpolate_rule_value(validation_rules["DiscountsAvailed"], 1.0),
            },
        },
        {
            "scenario_id": "synthetic_young_high_attention_boundary",
            "title": "Genc ve Yuksek Dikkatli Sinir Vaka",
            "selection_reason": "Yuksek ilgi ama kisa gecmis satinalma sinyalinde manuel review ihtiyacini tartismak icin eklendi.",
            "scenario_origin": "phase9_synthetic",
            "input_payload": {
                "Age": _interpolate_rule_value(validation_rules["Age"], 0.08),
                "Gender": _pick_allowed_value(validation_rules["Gender"], 1),
                "AnnualIncome": _interpolate_rule_value(validation_rules["AnnualIncome"], 0.75),
                "NumberOfPurchases": _interpolate_rule_value(validation_rules["NumberOfPurchases"], 0.1),
                "ProductCategory": _pick_allowed_value(validation_rules["ProductCategory"], 2),
                "TimeSpentOnWebsite": _interpolate_rule_value(validation_rules["TimeSpentOnWebsite"], 0.98),
                "LoyaltyProgram": _pick_allowed_value(validation_rules["LoyaltyProgram"], -1),
                "DiscountsAvailed": _interpolate_rule_value(validation_rules["DiscountsAvailed"], 0.8),
            },
        },
    ]
    for scenario in synthetic_scenarios:
        scenario["input_payload"] = sanitize_simulation_input(scenario["input_payload"], validation_rules)
        scenario_catalog.append(scenario)

    scenario_ids = [scenario["scenario_id"] for scenario in scenario_catalog]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise DataAnalysisError("Faz 9 demo senaryo katalogunda tekrar eden scenario_id bulundu.")
    return scenario_catalog


def simulate_customer_profile(
    *,
    scenario: dict[str, Any],
    champion_model: Any,
    role_matrix: Any,
    validation_rules: dict[str, dict[str, Any]],
    decision_bands: dict[str, float],
    reference_policy_lookup: dict[str, dict[str, Any]] | None = None,
    global_driver_hints: list[str] | None = None,
) -> dict[str, Any]:
    sanitized_input = sanitize_simulation_input(scenario["input_payload"], validation_rules)
    model_frame = pd.DataFrame([{column: sanitized_input[column] for column in role_matrix.base_input_features}])

    probability = float(np.clip(np.asarray(champion_model.predict_proba(model_frame), dtype=float)[0, 1], 0.0, 1.0))
    predicted_label = int(np.asarray(champion_model.predict(model_frame), dtype=int)[0])
    policy_decision = recommend_policy_action(
        probability=probability,
        discounts_availed=sanitized_input["DiscountsAvailed"],
        decision_bands=decision_bands,
    )

    reference_policy_lookup = reference_policy_lookup or {}
    reference_policy = reference_policy_lookup.get(scenario["scenario_id"], {})
    top_driver_hints = list(reference_policy.get("top_drivers") or (global_driver_hints or [])[:2])

    commentary_parts = [policy_decision["rationale"]]
    if top_driver_hints:
        commentary_parts.append("Aciklama ipuclari: " + ", ".join(top_driver_hints) + ".")
    if scenario.get("selection_reason"):
        commentary_parts.append(f"Senaryo amaci: {scenario['selection_reason']}")

    alignment_summary = None
    if reference_policy:
        reference_probability = reference_policy.get("predicted_probability")
        alignment_summary = {
            "action_match": policy_decision["recommended_action"] == reference_policy.get("recommended_action"),
            "band_match": policy_decision["score_band"] == reference_policy.get("score_band"),
            "manual_review_match": policy_decision["requires_manual_review"] == reference_policy.get("requires_manual_review"),
            "probability_delta": round(abs(probability - float(reference_probability)), 6)
            if reference_probability is not None
            else None,
        }

    standard_output = {
        "purchase_score": round(probability, 6),
        "predicted_label": predicted_label,
        "risk_band": policy_decision["score_band"],
        "recommended_action": policy_decision["recommended_action"],
        "recommended_discount_rate": policy_decision["recommended_discount_rate"],
        "requires_manual_review": policy_decision["requires_manual_review"],
        "guardrail_flags": policy_decision["guardrail_flags"],
        "short_rationale": " ".join(commentary_parts),
    }

    return {
        "scenario_id": scenario["scenario_id"],
        "title": scenario["title"],
        "scenario_origin": scenario["scenario_origin"],
        "selection_reason": scenario.get("selection_reason"),
        "simulation_input": sanitized_input,
        "predicted_probability": round(probability, 6),
        "predicted_label": predicted_label,
        "score_band": policy_decision["score_band"],
        "recommended_action": policy_decision["recommended_action"],
        "recommended_discount_rate": policy_decision["recommended_discount_rate"],
        "requires_manual_review": policy_decision["requires_manual_review"],
        "guardrail_flags": policy_decision["guardrail_flags"],
        "top_driver_hints": top_driver_hints,
        "commentary": " ".join(commentary_parts),
        "phase8_reference_alignment": alignment_summary,
        "standard_output": standard_output,
    }


def build_simulation_log_frame(simulation_results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in simulation_results:
        alignment = result.get("phase8_reference_alignment") or {}
        rows.append(
            {
                "scenario_id": result["scenario_id"],
                "title": result["title"],
                "scenario_origin": result["scenario_origin"],
                "predicted_probability": result["predicted_probability"],
                "predicted_label": result["predicted_label"],
                "score_band": result["score_band"],
                "recommended_action": result["recommended_action"],
                "recommended_discount_rate": result["recommended_discount_rate"],
                "requires_manual_review": result["requires_manual_review"],
                "guardrail_flags": "|".join(result["guardrail_flags"]),
                "top_driver_hints": "|".join(result["top_driver_hints"]),
                "phase8_action_match": alignment.get("action_match"),
                "phase8_band_match": alignment.get("band_match"),
                "phase8_probability_delta": alignment.get("probability_delta"),
            }
        )
    return pd.DataFrame(rows)


def build_demo_runbook(simulation_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_steps: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    def add_step(step_title: str, takeaway: str, predicate: Any) -> None:
        for result in simulation_results:
            if result["scenario_id"] in used_ids:
                continue
            if not predicate(result):
                continue
            selected_steps.append(
                {
                    "step_number": len(selected_steps) + 1,
                    "step_title": step_title,
                    "scenario_id": result["scenario_id"],
                    "scenario_title": result["title"],
                    "takeaway": takeaway,
                    "expected_output": result["standard_output"],
                }
            )
            used_ids.add(result["scenario_id"])
            return

    add_step(
        "Marj Koruma Gosterimi",
        "Yuksek guvenli vakalarda sistemin gereksiz indirim vermedigini gosterir.",
        lambda item: item["recommended_action"] == "protect_margin_no_discount" and not item["requires_manual_review"],
    )
    add_step(
        "Hedefli Teklif Gosterimi",
        "Orta risk bandinda kontrollu ve veri destekli teklif onerildigini gosterir.",
        lambda item: item["recommended_discount_rate"] > 0.0 and not item["requires_manual_review"],
    )
    add_step(
        "Manual Review Gosterimi",
        "Esik yakini veya riskli senaryolarda otomasyonun frenlendiginin gosterimidir.",
        lambda item: item["requires_manual_review"],
    )
    add_step(
        "Guardrail Doygunluk Gosterimi",
        "Gecmis indirim doygunlugunda teklif siddetinin sinirlandirildigini anlatir.",
        lambda item: "historical_discount_saturation" in item["guardrail_flags"] or item["recommended_action"] == "manual_review_discount_cap",
    )
    return selected_steps


class Phase9SimulationService:
    def __init__(
        self,
        loader: CsvDatasetLoader,
        logger: logging.Logger,
    ) -> None:
        self._loader = loader
        self._logger = logger

    def run(self, project_paths: ProjectPaths) -> Phase9Artifacts:
        try:
            self._logger.info("Faz 9 prototip ve simülasyon katmani baslatildi: %s", project_paths.dataset_path)
            dataset = self._loader.load(project_paths.dataset_path)
            deduplicated_dataset = deduplicate_dataset(dataset)
            validate_required_columns(deduplicated_dataset, EXPECTED_COLUMNS)

            phase7_summary = self._load_json_context(project_paths.phase_7_dir / "explainability_summary.json")
            phase8_summary = self._load_json_context(project_paths.phase_8_dir / "policy_summary.json")
            if phase7_summary is None or phase8_summary is None:
                raise DataAnalysisError("Faz 9 icin Faz 7 ve Faz 8 artefaktlari eksiksiz bulunmalidir.")

            reference_scenarios = phase7_summary.get("scenario_catalog", [])
            if len(reference_scenarios) < 8:
                raise DataAnalysisError("Faz 9 icin Faz 7'den gelen referans senaryo katalogu beklenenden kucuk.")

            role_matrix = build_feature_role_matrix()
            validation_rules = build_input_validation_rules(deduplicated_dataset)
            decision_bands = validate_decision_bands(phase8_summary["phase5_context"]["decision_bands"])
            champion_model_name = phase8_summary["phase6_context"]["champion_selection"]["champion_model"]
            champion_model = self._load_champion_model(project_paths.phase_6_dir / "models" / f"{champion_model_name}.joblib")
            scenario_catalog = build_demo_scenario_catalog(reference_scenarios, validation_rules)

            champion_key = phase7_summary.get("phase7_closeout", {}).get("champion_key")
            global_driver_hints = [
                item["feature_name"]
                for item in phase7_summary.get("model_explanations", {}).get(champion_key, {}).get("shap_global_summary", [])[:3]
            ]
            reference_policy_lookup = {
                item["scenario_id"]: item
                for item in phase8_summary.get("scenario_policy_validation", [])
            }

            # Faz 9 yeni karar mantigi uretmez; ayni model ve ayni bantlarla simülasyon katmaninin tekrar uretilebilirligini kanitlar.
            simulation_results = [
                simulate_customer_profile(
                    scenario=scenario,
                    champion_model=champion_model,
                    role_matrix=role_matrix,
                    validation_rules=validation_rules,
                    decision_bands=decision_bands,
                    reference_policy_lookup=reference_policy_lookup,
                    global_driver_hints=global_driver_hints,
                )
                for scenario in scenario_catalog
            ]
            self._validate_phase8_reference_alignment(simulation_results)

            simulation_log_frame = build_simulation_log_frame(simulation_results)
            demo_runbook = build_demo_runbook(simulation_results)
            artifacts = self._ensure_phase9_directories(project_paths.phase_9_dir)
            summary = self._build_summary(
                dataset_path=project_paths.dataset_path,
                deduplicated_row_count=len(deduplicated_dataset),
                champion_model_name=champion_model_name,
                validation_rules=validation_rules,
                decision_bands=decision_bands,
                simulation_results=simulation_results,
                simulation_log_frame=simulation_log_frame,
                demo_runbook=demo_runbook,
            )

            self._write_json_file(artifacts.json_summary_path, summary)
            self._write_json_file(artifacts.input_schema_path, validation_rules)
            self._write_simulation_log(artifacts.simulation_log_path, simulation_log_frame)
            self._write_demo_runbook(artifacts.demo_runbook_path, demo_runbook)
            self._write_scenario_cards(artifacts.scenarios_dir, simulation_results)
            self._write_markdown_report(artifacts.markdown_report_path, summary)
            self._logger.info("Faz 9 prototip ve simülasyon katmani tamamlandi.")
            return artifacts
        except DataAnalysisError:
            self._logger.exception("Faz 9 prototip ve simülasyon katmani veri dogrulamasi nedeniyle durdu.")
            raise
        except (KeyError, OSError, TypeError, ValueError) as error:
            self._logger.exception("Faz 9 prototip ve simülasyon katmani beklenmeyen bir hata ile durdu.")
            raise DataAnalysisError("Faz 9 prototip ve simülasyon katmani tamamlanamadi.") from error

    def _ensure_phase9_directories(self, phase_9_dir: Path) -> Phase9Artifacts:
        scenarios_dir = phase_9_dir / "scenarios"
        for directory in (phase_9_dir, scenarios_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return Phase9Artifacts(
            markdown_report_path=phase_9_dir / "simulation_report.md",
            json_summary_path=phase_9_dir / "simulation_summary.json",
            input_schema_path=phase_9_dir / "input_schema.json",
            simulation_log_path=phase_9_dir / "simulation_log.csv",
            demo_runbook_path=phase_9_dir / "demo_runbook.md",
            scenarios_dir=scenarios_dir,
        )

    def _load_json_context(self, file_path: Path) -> dict[str, Any] | None:
        if not file_path.exists():
            return None
        try:
            with file_path.open("r", encoding="utf-8") as input_file:
                return json.load(input_file)
        except (OSError, json.JSONDecodeError) as error:
            self._logger.warning("Faz 9 baglam dosyasi okunamadi: %s | %s", file_path, error)
            return None

    def _load_champion_model(self, model_path: Path) -> Any:
        if not model_path.exists():
            raise DataAnalysisError(f"Faz 9 icin champion model artefakti bulunamadi: {model_path}")
        try:
            return joblib.load(model_path)
        except (OSError, ValueError) as error:
            raise DataAnalysisError(f"Faz 9 icin champion model artefakti yuklenemedi: {model_path}") from error

    def _validate_phase8_reference_alignment(self, simulation_results: list[dict[str, Any]]) -> None:
        for result in simulation_results:
            if result["scenario_origin"] != "phase7_reference":
                continue
            alignment = result.get("phase8_reference_alignment")
            if alignment is None:
                raise DataAnalysisError(f"Faz 9 referans senaryosu Faz 8 karsilastirmasi eksik: {result['scenario_id']}")
            if not alignment["action_match"] or not alignment["band_match"] or not alignment["manual_review_match"]:
                raise DataAnalysisError(f"Faz 9 ile Faz 8 politika yeniden oynatimi uyusmadi: {result['scenario_id']}")
            probability_delta = alignment.get("probability_delta")
            if probability_delta is not None and probability_delta > 1e-6:
                raise DataAnalysisError(f"Faz 9 ile Faz 8 skor tekrarimi uyusmadi: {result['scenario_id']}")

    def _build_summary(
        self,
        *,
        dataset_path: Path,
        deduplicated_row_count: int,
        champion_model_name: str,
        validation_rules: dict[str, dict[str, Any]],
        decision_bands: dict[str, float],
        simulation_results: list[dict[str, Any]],
        simulation_log_frame: pd.DataFrame,
        demo_runbook: list[dict[str, Any]],
    ) -> dict[str, Any]:
        origin_counts = (
            pd.Series([result["scenario_origin"] for result in simulation_results])
            .value_counts(dropna=False)
            .to_dict()
        )
        action_counts = (
            pd.Series([result["recommended_action"] for result in simulation_results])
            .value_counts(dropna=False)
            .to_dict()
        )
        manual_review_share = round(
            float(np.mean([bool(result["requires_manual_review"]) for result in simulation_results])),
            6,
        )

        return {
            "dataset_path": str(dataset_path),
            "deduplicated_row_count": deduplicated_row_count,
            "champion_model_name": champion_model_name,
            "input_validation_rules": validation_rules,
            "decision_bands": decision_bands,
            "scenario_catalog_overview": {
                "total_scenarios": len(simulation_results),
                "origin_counts": {str(key): int(value) for key, value in origin_counts.items()},
                "manual_review_share": manual_review_share,
            },
            "action_distribution": {str(key): int(value) for key, value in action_counts.items()},
            "simulation_output_contract": {
                "input_keys": list(SIMULATION_INPUT_COLUMNS),
                "output_keys": [
                    "purchase_score",
                    "predicted_label",
                    "risk_band",
                    "recommended_action",
                    "recommended_discount_rate",
                    "requires_manual_review",
                    "guardrail_flags",
                    "short_rationale",
                ],
            },
            "simulation_log_schema": list(simulation_log_frame.columns),
            "simulation_results": simulation_results,
            "demo_runbook": demo_runbook,
            "phase9_closeout": {
                "summary": "Faz 9'da ayni champion model ve Faz 8 politika katmani kullanilarak validasyonlu demo girdileri ve standart cikti sozlesmesi olusturuldu.",
                "next_step": "Faz 10'da bu prototip yuzeyi edge-case, sensitivity ve tekrar uretilebilirlik testleriyle stres altina alinmalidir.",
            },
        }

    def _write_json_file(self, output_path: Path, payload: dict[str, Any]) -> None:
        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                json.dump(to_json_safe(payload), output_file, ensure_ascii=False, indent=2)
        except OSError as error:
            raise ArtifactWriteError(f"Faz 9 JSON artefakti yazilamadi: {output_path}") from error

    def _write_simulation_log(self, output_path: Path, simulation_log_frame: pd.DataFrame) -> None:
        try:
            simulation_log_frame.to_csv(output_path, index=False, encoding="utf-8")
        except OSError as error:
            raise ArtifactWriteError(f"Faz 9 simülasyon logu yazilamadi: {output_path}") from error

    def _write_demo_runbook(self, output_path: Path, demo_runbook: list[dict[str, Any]]) -> None:
        lines = [
            "# Faz 9 Demo Runbook",
            "",
            "- Bu belge, prototip gösteriminde hangi senaryolarin hangi sirayla anlatilacagini sabitler.",
            "",
        ]
        for step in demo_runbook:
            lines.extend(
                [
                    f"## Adim {step['step_number']} - {step['step_title']}",
                    "",
                    f"- Senaryo: {step['scenario_title']} ({step['scenario_id']})",
                    f"- Mesaj: {step['takeaway']}",
                    f"- Beklenen cikti: {step['expected_output']}",
                    "",
                ]
            )
        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                output_file.write("\n".join(lines))
        except OSError as error:
            raise ArtifactWriteError(f"Faz 9 demo runbook yazilamadi: {output_path}") from error

    def _write_scenario_cards(self, scenarios_dir: Path, simulation_results: list[dict[str, Any]]) -> None:
        for index, result in enumerate(simulation_results, start=1):
            lines = [
                f"# Faz 9 Senaryo {index:02d} - {result['title']}",
                "",
                f"- Senaryo kimligi: {result['scenario_id']}",
                f"- Kaynak: {result['scenario_origin']}",
                f"- Olasilik skoru: {result['predicted_probability']}",
                f"- Risk bandi: {result['score_band']}",
                f"- Onerilen aksiyon: {result['recommended_action']}",
                f"- Onerilen indirim orani: {result['recommended_discount_rate']}",
                f"- Manual review: {result['requires_manual_review']}",
                f"- Guardrail bayraklari: {result['guardrail_flags']}",
                f"- Yorum: {result['commentary']}",
                "",
                "## Girdi Ozeti",
                "",
            ]
            for feature_name, feature_value in result["simulation_input"].items():
                lines.append(f"- {feature_name}: {feature_value}")
            lines.extend([
                "",
                "## Standart Cikti",
                "",
                f"- {result['standard_output']}",
            ])
            output_path = scenarios_dir / f"{index:02d}_{result['scenario_id']}.md"
            try:
                with output_path.open("w", encoding="utf-8") as output_file:
                    output_file.write("\n".join(lines))
            except OSError as error:
                raise ArtifactWriteError(f"Faz 9 senaryo karti yazilamadi: {output_path}") from error

    def _write_markdown_report(self, output_path: Path, summary: dict[str, Any]) -> None:
        lines = [
            "# Faz 9 Prototip ve Simülasyon Raporu",
            "",
            f"- Veri seti: `{summary['dataset_path']}`",
            f"- Tekillestirilmis satir sayisi: {summary['deduplicated_row_count']}",
            f"- Champion model: {summary['champion_model_name']}",
            "",
            "## Girdi Sozlesmesi",
            "",
            f"- Girdi anahtarlari: {summary['simulation_output_contract']['input_keys']}",
            f"- Girdi kurallari: {summary['input_validation_rules']}",
            "",
            "## Cikti Sozlesmesi",
            "",
            f"- Cikti alanlari: {summary['simulation_output_contract']['output_keys']}",
            f"- Karar bantlari: {summary['decision_bands']}",
            "",
            "## Senaryo Ozeti",
            "",
            f"- Senaryo genel gorunumu: {summary['scenario_catalog_overview']}",
            f"- Aksiyon dagilimi: {summary['action_distribution']}",
            f"- Simülasyon log semasi: {summary['simulation_log_schema']}",
            "",
            "## Demo Runbook",
            "",
        ]
        for step in summary["demo_runbook"]:
            lines.append(
                f"- Adim {step['step_number']}: {step['scenario_title']} ({step['scenario_id']}) | mesaj={step['takeaway']} | beklenen={step['expected_output']}"
            )

        lines.extend([
            "",
            "## Senaryo Sonuclari",
            "",
        ])
        for result in summary["simulation_results"]:
            lines.append(
                f"- {result['scenario_id']}: skor={result['predicted_probability']}, band={result['score_band']}, aksiyon={result['recommended_action']}, review={result['requires_manual_review']}"
            )

        lines.extend([
            "",
            "## Faz 9 Kapanis",
            "",
            f"- Ozet: {summary['phase9_closeout']['summary']}",
            f"- Sonraki adim: {summary['phase9_closeout']['next_step']}",
        ])
        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                output_file.write("\n".join(lines))
        except OSError as error:
            raise ArtifactWriteError(f"Faz 9 markdown raporu yazilamadi: {output_path}") from error


def _coerce_numeric_scalar(column: str, raw_value: Any) -> float:
    if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
        raise DataAnalysisError(f"{column} bos bir deger alamaz.")
    if isinstance(raw_value, (dict, list, set, tuple)):
        raise DataAnalysisError(f"{column} yalnizca tekil skaler deger kabul eder.")

    candidate_value = raw_value.strip() if isinstance(raw_value, str) else raw_value
    numeric_value = pd.to_numeric(pd.Series([candidate_value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value) or not np.isfinite(float(numeric_value)):
        raise DataAnalysisError(f"{column} icin gecerli bir sayisal deger saglanmadi.")
    return float(numeric_value)


def _interpolate_rule_value(rule: dict[str, Any], ratio: float) -> int | float:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    if rule["kind"] == "categorical_integer":
        return _pick_allowed_value(rule, int(round(ratio * (len(rule["allowed_values"]) - 1))))

    min_value = float(rule["min_value"])
    max_value = float(rule["max_value"])
    value = min_value + ((max_value - min_value) * ratio)
    if rule["kind"] == "integer_range":
        return int(round(value))
    return round(float(value), 6)


def _pick_allowed_value(rule: dict[str, Any], position: int) -> int:
    allowed_values = list(rule["allowed_values"])
    if not allowed_values:
        raise DataAnalysisError("Kategorik alan icin izin verilen deger listesi bos olamaz.")
    normalized_position = position % len(allowed_values)
    return int(allowed_values[normalized_position])