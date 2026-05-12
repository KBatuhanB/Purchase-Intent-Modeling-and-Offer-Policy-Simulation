from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import TARGET_COLUMN, ProjectPaths
from ..core.exceptions import ArtifactWriteError, DataAnalysisError
from ..core.serialization import to_json_safe
from .csv_loader import CsvDatasetLoader
from .phase2_eda import deduplicate_dataset
from .phase3_preprocessing import build_feature_role_matrix, split_dataset_stratified, validate_required_columns
from .phase4_baseline import BaselineModelResult, evaluate_binary_classifier

REFERENCE_ORDER_VALUE = 100.0
REFERENCE_MARGIN_RATE = 0.35
LOW_THRESHOLD_REVIEW_MARGIN = 0.03
BINARY_THRESHOLD_REVIEW_MARGIN = 0.03
HIGH_THRESHOLD_REVIEW_MARGIN = 0.06
LOYALTY_ACTION_GAP_THRESHOLD = 0.12

ACTION_POLICY_SPECS: dict[str, dict[str, Any]] = {
    "holdout_low_cost_nurture": {
        "title": "Dusuk Maliyetli Nurture",
        "description": "Dusuk niyet grubunda indirim harcamasini tutar, sadece dusuk maliyetli hatirlatma uygular.",
        "default_discount_rate": 0.0,
        "expected_uplift": 0.01,
        "contact_cost": 0.1,
        "primary_channel": "email_or_push",
        "target_bands": ["low_intent_holdout"],
    },
    "targeted_standard_discount": {
        "title": "Standart Hedefli Indirim",
        "description": "Alt hedefleme bandinda kontrollu ancak anlamli bir teklif ile donusumu kaldirmayi hedefler.",
        "default_discount_rate": 0.1,
        "expected_uplift": 0.15,
        "contact_cost": 0.5,
        "primary_channel": "email_plus_web_banner",
        "target_bands": ["lower_targeting_band"],
    },
    "targeted_light_discount": {
        "title": "Hafif Indirim veya Bundle",
        "description": "Ust hedefleme bandinda satin alma ihtimali zaten yuksek oldugu icin marji koruyan hafif teklif onerir.",
        "default_discount_rate": 0.05,
        "expected_uplift": 0.08,
        "contact_cost": 0.4,
        "primary_channel": "onsite_or_email",
        "target_bands": ["upper_targeting_band"],
    },
    "protect_margin_no_discount": {
        "title": "Marj Koru, Indirim Verme",
        "description": "Yuksek guvenli bandda musteri zaten satin almaya yakin oldugu icin gereksiz marj kaybini engeller.",
        "default_discount_rate": 0.0,
        "expected_uplift": 0.0,
        "contact_cost": 0.0,
        "primary_channel": "none",
        "target_bands": ["high_confidence_no_discount"],
    },
    "manual_review_discount_cap": {
        "title": "Manual Review ve Indirim Cap",
        "description": "Yuksek indirim gecmisi veya esik yakini belirsizlik nedeniyle otomatik karari durdurur ve en fazla hafif teklif siniri koyar.",
        "default_discount_rate": 0.05,
        "expected_uplift": 0.05,
        "contact_cost": 0.6,
        "primary_channel": "manual_queue",
        "target_bands": ["lower_targeting_band", "upper_targeting_band", "high_confidence_no_discount"],
    },
}

BASE_ACTION_BY_BAND = {
    "low_intent_holdout": "holdout_low_cost_nurture",
    "lower_targeting_band": "targeted_standard_discount",
    "upper_targeting_band": "targeted_light_discount",
    "high_confidence_no_discount": "protect_margin_no_discount",
}


@dataclass(frozen=True)
class Phase8Artifacts:
    markdown_report_path: Path
    json_summary_path: Path
    plots_dir: Path
    scenarios_dir: Path


def validate_decision_bands(decision_bands: dict[str, float]) -> dict[str, float]:
    required_keys = {
        "low_action_threshold",
        "binary_decision_threshold",
        "high_confidence_no_discount_threshold",
    }
    if not required_keys.issubset(decision_bands):
        missing_keys = sorted(required_keys - set(decision_bands))
        raise DataAnalysisError(f"Faz 8 icin karar bantlari eksik: {missing_keys}")

    low_threshold = float(decision_bands["low_action_threshold"])
    binary_threshold = float(decision_bands["binary_decision_threshold"])
    high_threshold = float(decision_bands["high_confidence_no_discount_threshold"])
    if not (0.0 < low_threshold < binary_threshold < high_threshold < 1.0):
        raise DataAnalysisError("Faz 8 karar bantlari 0 ile 1 arasinda ve artan sirada olmalidir.")

    return {
        "low_action_threshold": low_threshold,
        "binary_decision_threshold": binary_threshold,
        "high_confidence_no_discount_threshold": high_threshold,
    }


def assign_score_band(probability: float, decision_bands: dict[str, float]) -> str:
    decision_bands = validate_decision_bands(decision_bands)
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise DataAnalysisError("Faz 8 icin olasilik degeri 0 ile 1 arasinda olmalidir.")

    if probability < decision_bands["low_action_threshold"]:
        return "low_intent_holdout"
    if probability < decision_bands["binary_decision_threshold"]:
        return "lower_targeting_band"
    if probability < decision_bands["high_confidence_no_discount_threshold"]:
        return "upper_targeting_band"
    return "high_confidence_no_discount"


def should_require_threshold_review(probability: float, decision_bands: dict[str, float]) -> bool:
    decision_bands = validate_decision_bands(decision_bands)
    probability = float(probability)
    threshold_windows = (
        (decision_bands["low_action_threshold"], LOW_THRESHOLD_REVIEW_MARGIN),
        (decision_bands["binary_decision_threshold"], BINARY_THRESHOLD_REVIEW_MARGIN),
        (decision_bands["high_confidence_no_discount_threshold"], HIGH_THRESHOLD_REVIEW_MARGIN),
    )
    return any(abs(probability - threshold) <= margin for threshold, margin in threshold_windows)


def compute_proxy_business_value(
    *,
    probability: float,
    discount_rate: float,
    expected_uplift: float,
    contact_cost: float,
    reference_order_value: float = REFERENCE_ORDER_VALUE,
    margin_rate: float = REFERENCE_MARGIN_RATE,
) -> dict[str, float]:
    probability = float(probability)
    discount_rate = float(discount_rate)
    expected_uplift = float(expected_uplift)
    contact_cost = float(contact_cost)
    reference_order_value = float(reference_order_value)
    margin_rate = float(margin_rate)

    if not 0.0 <= probability <= 1.0:
        raise DataAnalysisError("Proxy fayda hesabinda olasilik 0 ile 1 arasinda olmalidir.")
    if not 0.0 <= discount_rate <= 0.5:
        raise DataAnalysisError("Proxy fayda hesabinda indirim orani 0 ile 0.5 arasinda olmalidir.")
    if not 0.0 <= expected_uplift <= 0.5:
        raise DataAnalysisError("Proxy fayda hesabinda uplift 0 ile 0.5 arasinda olmalidir.")
    if reference_order_value <= 0.0 or margin_rate <= 0.0:
        raise DataAnalysisError("Proxy fayda hesabinda referans siparis degeri ve marj pozitif olmalidir.")

    margin_without_discount = reference_order_value * margin_rate
    margin_with_discount = max(0.0, margin_without_discount - (reference_order_value * discount_rate))
    expected_probability_after_action = min(0.99, probability + expected_uplift)

    baseline_expected_margin = probability * margin_without_discount
    action_expected_margin = (expected_probability_after_action * margin_with_discount) - contact_cost
    proxy_incremental_value = action_expected_margin - baseline_expected_margin

    return {
        "baseline_expected_margin": round(baseline_expected_margin, 6),
        "action_expected_margin": round(action_expected_margin, 6),
        "expected_probability_after_action": round(expected_probability_after_action, 6),
        "proxy_incremental_value": round(proxy_incremental_value, 6),
    }


def recommend_policy_action(
    *,
    probability: float,
    discounts_availed: float | int | None,
    decision_bands: dict[str, float],
) -> dict[str, Any]:
    decision_bands = validate_decision_bands(decision_bands)
    probability = float(probability)
    if discounts_availed is None or pd.isna(discounts_availed):
        discount_history = 0.0
    else:
        discount_history = float(discounts_availed)
    if discount_history < 0.0:
        raise DataAnalysisError("DiscountsAvailed negatif olamaz.")

    score_band = assign_score_band(probability, decision_bands)
    base_action = BASE_ACTION_BY_BAND[score_band]
    recommended_action = base_action
    guardrail_flags: list[str] = []
    requires_manual_review = False

    # Bu kural, model skorunu kampanya gevekligine cevirmemek icin indirim gecmisini ikinci bir emniyet kemeri olarak kullanir.
    if base_action in {"targeted_standard_discount", "targeted_light_discount"} and discount_history >= 4.0:
        recommended_action = "manual_review_discount_cap"
        requires_manual_review = True
        guardrail_flags.append("historical_discount_saturation")
    elif base_action == "targeted_standard_discount" and discount_history >= 3.0:
        recommended_action = "targeted_light_discount"
        guardrail_flags.append("discount_intensity_downgraded")

    if should_require_threshold_review(probability, decision_bands):
        requires_manual_review = True
        guardrail_flags.append("threshold_boundary_review")

    action_spec = ACTION_POLICY_SPECS[recommended_action]
    proxy_value = compute_proxy_business_value(
        probability=probability,
        discount_rate=action_spec["default_discount_rate"],
        expected_uplift=action_spec["expected_uplift"],
        contact_cost=action_spec["contact_cost"],
    )

    rationale_parts = [
        f"Skor {score_band} bandina dustugu icin temel aksiyon {BASE_ACTION_BY_BAND[score_band]} olarak secildi.",
    ]
    if recommended_action != base_action:
        rationale_parts.append(
            f"DiscountsAvailed={round(discount_history, 2)} nedeniyle aksiyon {recommended_action} seviyesine cekildi."
        )
    if "threshold_boundary_review" in guardrail_flags:
        rationale_parts.append("Skor karar esiklerinden birine yakin oldugu icin otomasyon ustune manual review eklendi.")

    return {
        "score_band": score_band,
        "base_action": base_action,
        "recommended_action": recommended_action,
        "recommended_discount_rate": round(float(action_spec["default_discount_rate"]), 6),
        "requires_manual_review": requires_manual_review,
        "guardrail_flags": guardrail_flags,
        "action_title": action_spec["title"],
        "action_description": action_spec["description"],
        "primary_channel": action_spec["primary_channel"],
        "rationale": " ".join(rationale_parts),
        **proxy_value,
    }


def build_guardrails(
    *,
    phase7_summary: dict[str, Any],
    policy_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    phase7_closeout = phase7_summary.get("phase7_closeout", {})
    champion_key = phase7_closeout.get("champion_key")
    champion_summary = phase7_summary.get("model_explanations", {}).get(champion_key, {})
    fairness_alerts = champion_summary.get("fairness_summary", {}).get("alerts", [])

    loyalty_gap = None
    loyalty_monitor: dict[str, Any] = {}
    if not policy_frame.empty and "LoyaltyProgram" in policy_frame.columns:
        loyalty_frame = policy_frame.copy(deep=True)
        loyalty_frame["LoyaltyProgram"] = pd.to_numeric(loyalty_frame["LoyaltyProgram"], errors="coerce")
        grouped = loyalty_frame.groupby("LoyaltyProgram", dropna=True)["discount_offered"].mean().to_dict()
        loyalty_monitor = {str(int(key)): round(float(value), 6) for key, value in grouped.items()}
        if 0.0 in grouped and 1.0 in grouped:
            loyalty_gap = round(abs(float(grouped[1.0]) - float(grouped[0.0])), 6)

    guardrails = [
        {
            "guardrail_id": "discount_history_cap",
            "severity": "high",
            "rule": "DiscountsAvailed >= 4 olan kayitlarda otomatik indirim verilmez; karar manual review kuyruğuna gider.",
            "reason": "Gecmis kampanya yogunlugunu tekrar odullendirmek marj asindirmasi ve politika bagimliligi riski yaratir.",
            "status": "active",
        },
        {
            "guardrail_id": "threshold_boundary_review",
            "severity": "medium",
            "rule": "Skor karar esiklerine yakin ise aksiyon otomatik uygulanmaz, manual review veya dusuk riskli deney tercih edilir.",
            "reason": "Esik yakinindaki vakalar policy churn ve hata maliyeti uretmeye daha yatkindir.",
            "status": "active",
        },
        {
            "guardrail_id": "margin_protection_high_confidence",
            "severity": "medium",
            "rule": "Yuksek guvenli bandda varsayilan aksiyon indirim vermemektir.",
            "reason": "Bu bandda gereksiz indirim vermek marj kacagi yaratir ve causal uplift kaniti olmadan savunulamaz.",
            "status": "active",
        },
        {
            "guardrail_id": "loyalty_fairness_monitor",
            "severity": "high" if fairness_alerts or (loyalty_gap is not None and loyalty_gap > LOYALTY_ACTION_GAP_THRESHOLD) else "medium",
            "rule": "LoyaltyProgram sinyali indirim derinligini tek basina artiramaz; aksiyon dagilimi her kosuda izlenir.",
            "reason": (
                fairness_alerts[0]["message"]
                if fairness_alerts
                else "Faz 7 fairness-lite ciktilari LoyaltyProgram segmentlerinde ek izleme gerektirdigini gosteriyor."
            ),
            "status": "alert" if fairness_alerts or (loyalty_gap is not None and loyalty_gap > LOYALTY_ACTION_GAP_THRESHOLD) else "watch",
            "monitor": {
                "discount_action_share_by_loyalty": loyalty_monitor,
                "observed_gap": loyalty_gap,
                "alert_threshold": LOYALTY_ACTION_GAP_THRESHOLD,
            },
        },
    ]
    return guardrails


def evaluate_policy_scenarios(
    *,
    scenario_catalog: list[dict[str, Any]],
    champion_local_explanations: list[dict[str, Any]],
    decision_bands: dict[str, float],
) -> list[dict[str, Any]]:
    explanation_lookup = {item["scenario_id"]: item for item in champion_local_explanations}
    evaluations: list[dict[str, Any]] = []

    for scenario in scenario_catalog:
        snapshot = scenario.get("feature_snapshot", {})
        evaluation = recommend_policy_action(
            probability=float(scenario["predicted_probability"]),
            discounts_availed=snapshot.get("DiscountsAvailed"),
            decision_bands=decision_bands,
        )
        local_explanation = explanation_lookup.get(scenario["scenario_id"], {})
        top_drivers = [
            contribution["feature_name"]
            for contribution in local_explanation.get("top_contributions", [])[:2]
        ]

        commentary = evaluation["rationale"]
        if top_drivers:
            commentary += " En baskin lokal suruculer: " + ", ".join(top_drivers) + "."

        evaluations.append(
            {
                "scenario_id": scenario["scenario_id"],
                "title": scenario["title"],
                "predicted_probability": round(float(scenario["predicted_probability"]), 6),
                "true_label": int(scenario["true_label"]),
                "recommended_action": evaluation["recommended_action"],
                "recommended_discount_rate": evaluation["recommended_discount_rate"],
                "requires_manual_review": evaluation["requires_manual_review"],
                "score_band": evaluation["score_band"],
                "guardrail_flags": evaluation["guardrail_flags"],
                "top_drivers": top_drivers,
                "commentary": commentary,
                "feature_snapshot": snapshot,
            }
        )
    return evaluations


def summarize_scenario_policy_checks(scenario_evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_lookup = {item["scenario_id"]: item for item in scenario_evaluations}

    def scenario_check(scenario_id: str, predicate: Any) -> bool | None:
        scenario = scenario_lookup.get(scenario_id)
        if scenario is None:
            return None
        return bool(predicate(scenario))

    return {
        "false_positive_margin_protection": scenario_check(
            "false_positive",
            lambda item: item["recommended_discount_rate"] == 0.0,
        ),
        "boundary_case_manual_review": scenario_check(
            "boundary_case",
            lambda item: item["requires_manual_review"],
        ),
        "non_loyal_high_time_not_ignored": scenario_check(
            "non_loyal_high_time",
            lambda item: item["recommended_discount_rate"] > 0.0 or item["requires_manual_review"],
        ),
    }


class Phase8PolicyService:
    def __init__(
        self,
        loader: CsvDatasetLoader,
        logger: logging.Logger,
    ) -> None:
        self._loader = loader
        self._logger = logger

    def run(self, project_paths: ProjectPaths) -> Phase8Artifacts:
        dataset = self._loader.load(project_paths.dataset_path)
        deduplicated_dataset = deduplicate_dataset(dataset)
        role_matrix = build_feature_role_matrix()
        validate_required_columns(
            deduplicated_dataset,
            role_matrix.base_input_features + (role_matrix.target_column,),
        )

        phase5_summary = self._load_json_context(project_paths.phase_5_dir / "imbalance_summary.json")
        phase6_summary = self._load_json_context(project_paths.phase_6_dir / "advanced_modeling_summary.json")
        phase7_summary = self._load_json_context(project_paths.phase_7_dir / "explainability_summary.json")
        if phase5_summary is None or phase6_summary is None or phase7_summary is None:
            raise DataAnalysisError("Faz 8 icin Faz 5, Faz 6 ve Faz 7 artefaktlari eksiksiz bulunmalidir.")

        decision_bands = self._extract_decision_bands(phase5_summary)
        split_config = self._extract_phase6_split_config(phase6_summary)
        artifacts = self._ensure_phase8_directories(project_paths.phase_8_dir)

        self._logger.info("Faz 8 politika ve aksiyon katmani baslatildi: %s", project_paths.dataset_path)
        # Faz 8 modeli yeniden egitmez; ayni Faz 6 champion skorlarini tekrar kullanir ki karar mantigi birebir ayni yuzeyi temsil etsin.
        _outer_train, outer_test = split_dataset_stratified(
            deduplicated_dataset,
            role_matrix.target_column,
            test_size=split_config["test_size"],
            random_state=split_config["random_state"],
        )
        champion_key, champion_model_name, champion_model = self._load_champion_model(phase6_summary, project_paths)

        X_eval = outer_test.loc[:, role_matrix.base_input_features]
        probabilities = champion_model.predict_proba(X_eval)[:, 1]
        predictions = champion_model.predict(X_eval)
        holdout_result = evaluate_binary_classifier(
            name=champion_model_name,
            family=champion_key,
            y_true=outer_test[TARGET_COLUMN],
            predictions=predictions,
            probabilities=probabilities,
            note="Faz 8 politika katmani Faz 6 champion model skorlarini tekrar kullanir.",
        )
        self._validate_phase6_alignment(phase6_summary, holdout_result, champion_key)

        policy_frame = self._build_policy_frame(
            evaluation_frame=outer_test,
            probabilities=np.asarray(probabilities, dtype=float),
            predictions=np.asarray(predictions, dtype=int),
            decision_bands=decision_bands,
        )
        action_catalog = self._build_action_catalog()
        policy_band_summary = self._summarize_policy_bands(policy_frame)
        action_mix_summary = self._summarize_action_mix(policy_frame)
        loyalty_monitor = self._summarize_loyalty_monitor(policy_frame)
        proxy_value_summary = self._summarize_proxy_value(policy_frame)
        guardrails = build_guardrails(phase7_summary=phase7_summary, policy_frame=policy_frame)

        champion_local_explanations = (
            phase7_summary
            .get("model_explanations", {})
            .get(phase7_summary.get("phase7_closeout", {}).get("champion_key"), {})
            .get("local_explanations", [])
        )
        scenario_evaluations = evaluate_policy_scenarios(
            scenario_catalog=phase7_summary.get("scenario_catalog", []),
            champion_local_explanations=champion_local_explanations,
            decision_bands=decision_bands,
        )
        scenario_checks = summarize_scenario_policy_checks(scenario_evaluations)
        policy_insights = self._build_policy_insights(policy_band_summary, proxy_value_summary, guardrails, scenario_checks)

        summary = {
            "dataset_path": str(project_paths.dataset_path),
            "deduplicated_row_count": len(deduplicated_dataset),
            "phase5_context": {
                "recommended_strategy": phase5_summary.get("recommended_strategy"),
                "decision_bands": decision_bands,
                "policy_band_summary": phase5_summary.get("threshold_strategy", {}).get("policy_band_summary"),
            },
            "phase6_context": {
                "champion_selection": phase6_summary.get("champion_selection"),
                "split_summary": phase6_summary.get("split_summary"),
                "validated_champion_metrics": holdout_result.metrics,
            },
            "phase7_context": {
                "business_insights": phase7_summary.get("business_insights", []),
                "fairness_alerts": (
                    phase7_summary
                    .get("model_explanations", {})
                    .get(phase7_summary.get("phase7_closeout", {}).get("champion_key"), {})
                    .get("fairness_summary", {})
                    .get("alerts", [])
                ),
            },
            "action_catalog": action_catalog,
            "policy_band_summary": policy_band_summary,
            "action_mix_summary": action_mix_summary,
            "loyalty_monitor": loyalty_monitor,
            "proxy_value_summary": proxy_value_summary,
            "guardrails": guardrails,
            "scenario_policy_validation": scenario_evaluations,
            "scenario_policy_checks": scenario_checks,
            "policy_insights": policy_insights,
            "phase8_closeout": {
                "summary": "Faz 8'de champion model skoru, Faz 5 karar bantlari ve Faz 7 fairness/aciklama notlari birlestirilerek aksiyon politikasina cevrildi.",
                "next_step": "Gercek is etkisi icin Faz 9'da uplift veya deney tasarimi ile bu proxy fayda varsayimlari canli veride sinanmalidir.",
            },
        }

        self._write_plots(artifacts.plots_dir, policy_band_summary, action_mix_summary, loyalty_monitor)
        self._write_scenario_cards(artifacts.scenarios_dir, scenario_evaluations)
        self._write_json_summary(artifacts.json_summary_path, summary)
        self._write_markdown_report(artifacts.markdown_report_path, summary)
        self._logger.info("Faz 8 politika ve aksiyon katmani tamamlandi.")
        return artifacts

    def _ensure_phase8_directories(self, phase_8_dir: Path) -> Phase8Artifacts:
        plots_dir = phase_8_dir / "plots"
        scenarios_dir = phase_8_dir / "scenarios"
        for directory in (phase_8_dir, plots_dir, scenarios_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return Phase8Artifacts(
            markdown_report_path=phase_8_dir / "policy_report.md",
            json_summary_path=phase_8_dir / "policy_summary.json",
            plots_dir=plots_dir,
            scenarios_dir=scenarios_dir,
        )

    def _extract_decision_bands(self, phase5_summary: dict[str, Any]) -> dict[str, float]:
        threshold_strategy = phase5_summary.get("threshold_strategy", {})
        validation_selection = threshold_strategy.get("validation_selection", {})
        decision_bands = validation_selection.get("decision_bands")
        if not decision_bands:
            raise DataAnalysisError("Faz 8 icin Faz 5 karar bantlari bulunamadi.")
        return validate_decision_bands(decision_bands)

    def _extract_phase6_split_config(self, phase6_summary: dict[str, Any]) -> dict[str, Any]:
        split_summary = phase6_summary.get("split_summary")
        if not split_summary:
            raise DataAnalysisError("Faz 8 icin Faz 6 split bilgisi bulunamadi.")
        return {
            "test_size": float(split_summary["test_size"]),
            "random_state": int(split_summary["random_state"]),
        }

    def _load_champion_model(
        self,
        phase6_summary: dict[str, Any],
        project_paths: ProjectPaths,
    ) -> tuple[str, str, Any]:
        champion_selection = phase6_summary.get("champion_selection", {})
        champion_key = champion_selection.get("champion_key")
        champion_model_name = champion_selection.get("champion_model")
        if not champion_key or not champion_model_name:
            raise DataAnalysisError("Faz 8 icin Faz 6 champion bilgisi eksik.")

        model_path = project_paths.phase_6_dir / "models" / f"{champion_model_name}.joblib"
        if not model_path.exists():
            raise DataAnalysisError(f"Faz 8 champion model artefakti bulunamadi: {model_path}")

        try:
            model_object = joblib.load(model_path)
        except (OSError, ValueError) as error:
            raise DataAnalysisError(f"Faz 8 champion model artefakti yuklenemedi: {model_path}") from error
        return str(champion_key), str(champion_model_name), model_object

    def _validate_phase6_alignment(
        self,
        phase6_summary: dict[str, Any],
        recalculated_result: BaselineModelResult,
        champion_key: str,
    ) -> None:
        stored_result = phase6_summary.get("candidate_families", {}).get(champion_key, {}).get("holdout_result")
        if not stored_result:
            raise DataAnalysisError("Faz 8 champion holdout referansi Faz 6 ozetinde bulunamadi.")

        for metric_name in ("pr_auc", "balanced_accuracy", "brier_score"):
            stored_value = stored_result["metrics"].get(metric_name)
            recalculated_value = recalculated_result.metrics.get(metric_name)
            if stored_value is None or recalculated_value is None:
                continue
            if abs(float(stored_value) - float(recalculated_value)) > 1e-6:
                raise DataAnalysisError(
                    f"Faz 8 yeniden hesaplanan champion sonucu Faz 6 ile uyusmuyor: {metric_name}"
                )

    def _build_policy_frame(
        self,
        *,
        evaluation_frame: pd.DataFrame,
        probabilities: np.ndarray,
        predictions: np.ndarray,
        decision_bands: dict[str, float],
    ) -> pd.DataFrame:
        frame = evaluation_frame.reset_index(drop=True).copy(deep=True)
        frame["predicted_probability"] = np.asarray(probabilities, dtype=float)
        frame["predicted_label"] = np.asarray(predictions, dtype=int)

        recommendation_rows: list[dict[str, Any]] = []
        for row in frame.itertuples(index=False):
            recommendation_rows.append(
                recommend_policy_action(
                    probability=float(row.predicted_probability),
                    discounts_availed=getattr(row, "DiscountsAvailed", None),
                    decision_bands=decision_bands,
                )
            )

        recommendation_frame = pd.DataFrame(recommendation_rows)
        merged_frame = pd.concat([frame, recommendation_frame], axis=1)
        merged_frame["discount_offered"] = merged_frame["recommended_discount_rate"] > 0.0
        return merged_frame

    def _build_action_catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "action_key": action_key,
                **to_json_safe(action_spec),
            }
            for action_key, action_spec in ACTION_POLICY_SPECS.items()
        ]

    def _summarize_policy_bands(self, policy_frame: pd.DataFrame) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        grouped = policy_frame.groupby("score_band", sort=False)
        for score_band, group in grouped:
            summary[str(score_band)] = {
                "count": int(group.shape[0]),
                "share": round(float(group.shape[0] / len(policy_frame)), 6),
                "observed_purchase_rate": round(float(group[TARGET_COLUMN].mean()), 6),
                "average_predicted_probability": round(float(group["predicted_probability"].mean()), 6),
                "average_discount_rate": round(float(group["recommended_discount_rate"].mean()), 6),
                "manual_review_share": round(float(group["requires_manual_review"].mean()), 6),
                "average_proxy_incremental_value": round(float(group["proxy_incremental_value"].mean()), 6),
            }
        return summary

    def _summarize_action_mix(self, policy_frame: pd.DataFrame) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        grouped = policy_frame.groupby("recommended_action", sort=False)
        for action_key, group in grouped:
            summary[str(action_key)] = {
                "count": int(group.shape[0]),
                "share": round(float(group.shape[0] / len(policy_frame)), 6),
                "average_predicted_probability": round(float(group["predicted_probability"].mean()), 6),
                "average_discount_rate": round(float(group["recommended_discount_rate"].mean()), 6),
                "average_proxy_incremental_value": round(float(group["proxy_incremental_value"].mean()), 6),
            }
        return summary

    def _summarize_loyalty_monitor(self, policy_frame: pd.DataFrame) -> dict[str, Any]:
        if policy_frame.empty:
            return {}

        frame = policy_frame.copy(deep=True)
        frame["LoyaltyProgram"] = pd.to_numeric(frame["LoyaltyProgram"], errors="coerce")
        grouped = frame.groupby("LoyaltyProgram", dropna=True)

        summary = {
            str(int(group_name)): {
                "count": int(group.shape[0]),
                "discount_action_share": round(float(group["discount_offered"].mean()), 6),
                "manual_review_share": round(float(group["requires_manual_review"].mean()), 6),
                "average_proxy_incremental_value": round(float(group["proxy_incremental_value"].mean()), 6),
            }
            for group_name, group in grouped
        }
        if "0" in summary and "1" in summary:
            summary["observed_discount_action_gap"] = round(
                abs(float(summary["1"]["discount_action_share"]) - float(summary["0"]["discount_action_share"])),
                6,
            )
        return summary

    def _summarize_proxy_value(self, policy_frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "portfolio_average_proxy_incremental_value": round(float(policy_frame["proxy_incremental_value"].mean()), 6),
            "portfolio_total_proxy_incremental_value": round(float(policy_frame["proxy_incremental_value"].sum()), 6),
            "positive_proxy_share": round(float((policy_frame["proxy_incremental_value"] > 0.0).mean()), 6),
        }

    def _build_policy_insights(
        self,
        policy_band_summary: dict[str, Any],
        proxy_value_summary: dict[str, Any],
        guardrails: list[dict[str, Any]],
        scenario_checks: dict[str, Any],
    ) -> list[str]:
        insights: list[str] = []

        high_confidence_band = policy_band_summary.get("high_confidence_no_discount")
        if high_confidence_band:
            insights.append(
                "Yuksek guvenli bandda gozlenen satin alma orani "
                f"{high_confidence_band['observed_purchase_rate']} oldugu icin marj koruma kuralinin veri destegi guclu gorunuyor."
            )

        lower_target_band = policy_band_summary.get("lower_targeting_band")
        if lower_target_band:
            insights.append(
                "Alt hedefleme bandinin ortalama proxy faydasi "
                f"{lower_target_band['average_proxy_incremental_value']} olarak hesaplandi; bu band kontrollu teklif icin ana adaydir."
            )

        insights.append(
            "Portfoy ortalama proxy incremental value "
            f"{proxy_value_summary['portfolio_average_proxy_incremental_value']} oldu; bu deger causal uplift degil, yalnizca politika kiyaslama sinyalidir."
        )

        active_guardrail = next((item for item in guardrails if item["guardrail_id"] == "loyalty_fairness_monitor"), None)
        if active_guardrail is not None:
            insights.append(active_guardrail["reason"])

        if scenario_checks.get("false_positive_margin_protection") is True:
            insights.append("False positive referans senaryosunda politika indirim bloklayarak gereksiz maliyet riskini dusurdu.")
        if scenario_checks.get("boundary_case_manual_review") is True:
            insights.append("Boundary case referans senaryosu manual review kuyruğuna alinarak sert otomasyon kirildi.")

        return insights

    def _write_plots(
        self,
        plots_dir: Path,
        policy_band_summary: dict[str, Any],
        action_mix_summary: dict[str, Any],
        loyalty_monitor: dict[str, Any],
    ) -> None:
        self._plot_band_distribution(plots_dir / "policy_band_distribution.png", policy_band_summary)
        self._plot_action_mix(plots_dir / "action_mix_proxy_value.png", action_mix_summary)
        self._plot_loyalty_monitor(plots_dir / "loyalty_action_monitor.png", loyalty_monitor)

    def _plot_band_distribution(self, output_path: Path, policy_band_summary: dict[str, Any]) -> None:
        labels = list(policy_band_summary)
        counts = [policy_band_summary[label]["count"] for label in labels]
        proxy_values = [policy_band_summary[label]["average_proxy_incremental_value"] for label in labels]

        figure, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(labels, counts)
        axes[0].set_title("Faz 8 Politika Bant Sayilari")
        axes[0].tick_params(axis="x", rotation=20)

        axes[1].bar(labels, proxy_values)
        axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
        axes[1].set_title("Faz 8 Bant Bazli Proxy Fayda")
        axes[1].tick_params(axis="x", rotation=20)
        figure.tight_layout()
        self._save_figure(figure, output_path)

    def _plot_action_mix(self, output_path: Path, action_mix_summary: dict[str, Any]) -> None:
        labels = list(action_mix_summary)
        shares = [action_mix_summary[label]["share"] for label in labels]
        proxy_values = [action_mix_summary[label]["average_proxy_incremental_value"] for label in labels]

        figure, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(labels, shares)
        axes[0].set_title("Faz 8 Aksiyon Paylari")
        axes[0].tick_params(axis="x", rotation=20)

        axes[1].bar(labels, proxy_values)
        axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
        axes[1].set_title("Faz 8 Aksiyon Bazli Proxy Fayda")
        axes[1].tick_params(axis="x", rotation=20)
        figure.tight_layout()
        self._save_figure(figure, output_path)

    def _plot_loyalty_monitor(self, output_path: Path, loyalty_monitor: dict[str, Any]) -> None:
        group_keys = [key for key in loyalty_monitor if key in {"0", "1"}]
        figure, axis = plt.subplots(figsize=(8, 5))

        if not group_keys:
            axis.text(0.5, 0.5, "Loyalty monitor verisi bulunamadi.", ha="center", va="center")
            axis.set_axis_off()
        else:
            discount_shares = [loyalty_monitor[group_key]["discount_action_share"] for group_key in group_keys]
            manual_review_shares = [loyalty_monitor[group_key]["manual_review_share"] for group_key in group_keys]
            x_positions = np.arange(len(group_keys))
            width = 0.36
            axis.bar(x_positions - (width / 2), discount_shares, width=width, label="discount_action_share")
            axis.bar(x_positions + (width / 2), manual_review_shares, width=width, label="manual_review_share")
            axis.set_xticks(x_positions, [f"loyalty_{group_key}" for group_key in group_keys])
            axis.set_title("Loyalty Segment Aksiyon Izleme")
            axis.legend()

        figure.tight_layout()
        self._save_figure(figure, output_path)

    def _write_scenario_cards(self, scenarios_dir: Path, scenario_evaluations: list[dict[str, Any]]) -> None:
        for index, scenario in enumerate(scenario_evaluations, start=1):
            lines = [
                f"# Faz 8 Senaryo {index:02d} - {scenario['title']}",
                "",
                f"- Senaryo kimligi: {scenario['scenario_id']}",
                f"- Referans olasilik: {scenario['predicted_probability']}",
                f"- Onerilen aksiyon: {scenario['recommended_action']}",
                f"- Onerilen indirim orani: {scenario['recommended_discount_rate']}",
                f"- Manual review: {scenario['requires_manual_review']}",
                f"- Guardrail bayraklari: {scenario['guardrail_flags']}",
                f"- Politik aciklama: {scenario['commentary']}",
                "",
                "## Ozellik Ozeti",
                "",
            ]
            for feature_name, feature_value in scenario["feature_snapshot"].items():
                lines.append(f"- {feature_name}: {feature_value}")
            output_path = scenarios_dir / f"{index:02d}_{scenario['scenario_id']}.md"
            try:
                with output_path.open("w", encoding="utf-8") as output_file:
                    output_file.write("\n".join(lines))
            except OSError as error:
                raise ArtifactWriteError(f"Faz 8 senaryo karti yazilamadi: {output_path}") from error

    def _save_figure(self, figure: plt.Figure, output_path: Path) -> None:
        try:
            figure.savefig(output_path, dpi=200, bbox_inches="tight")
        except OSError as error:
            raise ArtifactWriteError(f"Faz 8 grafigi kaydedilemedi: {output_path}") from error
        finally:
            plt.close(figure)

    def _load_json_context(self, file_path: Path) -> dict[str, Any] | None:
        if not file_path.exists():
            return None
        try:
            with file_path.open("r", encoding="utf-8") as input_file:
                return json.load(input_file)
        except (OSError, json.JSONDecodeError) as error:
            self._logger.warning("Baglam dosyasi okunamadi: %s | %s", file_path, error)
            return None

    def _write_json_summary(self, output_path: Path, summary: dict[str, Any]) -> None:
        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                json.dump(to_json_safe(summary), output_file, ensure_ascii=False, indent=2)
        except OSError as error:
            raise ArtifactWriteError(f"Faz 8 JSON ozeti yazilamadi: {output_path}") from error

    def _write_markdown_report(self, output_path: Path, summary: dict[str, Any]) -> None:
        lines = [
            "# Faz 8 Politika ve Aksiyon Raporu",
            "",
            f"- Veri seti: `{summary['dataset_path']}`",
            f"- Tekillestirilmis satir sayisi: {summary['deduplicated_row_count']}",
            f"- Faz 6 champion: {summary['phase6_context']['champion_selection']['champion_model']}",
            "",
            "## Karar Bantlari",
            "",
            f"- Faz 5 karar bantlari: {summary['phase5_context']['decision_bands']}",
            f"- Faz 5 holdout policy bant ozeti: {summary['phase5_context']['policy_band_summary']}",
            "",
            "## Aksiyon Katalogu",
            "",
        ]
        for action in summary["action_catalog"]:
            lines.append(
                f"- {action['action_key']}: indirim={action['default_discount_rate']}, uplift={action['expected_uplift']}, kanal={action['primary_channel']}, aciklama={action['description']}"
            )

        lines.extend([
            "",
            "## Politika Ozetleri",
            "",
            f"- Bant ozeti: {summary['policy_band_summary']}",
            f"- Aksiyon miksi: {summary['action_mix_summary']}",
            f"- Loyalty monitor: {summary['loyalty_monitor']}",
            f"- Proxy deger ozeti: {summary['proxy_value_summary']}",
            "",
            "## Guardrail Listesi",
            "",
        ])
        for guardrail in summary["guardrails"]:
            lines.append(
                f"- {guardrail['guardrail_id']} | seviye={guardrail['severity']} | durum={guardrail['status']} | kural={guardrail['rule']} | neden={guardrail['reason']}"
            )

        lines.extend([
            "",
            "## Senaryo Bazli Politika Dogrulamasi",
            "",
            f"- Senaryo kontrolleri: {summary['scenario_policy_checks']}",
        ])
        for scenario in summary["scenario_policy_validation"]:
            lines.append(
                f"- {scenario['scenario_id']}: aksiyon={scenario['recommended_action']}, indirim={scenario['recommended_discount_rate']}, review={scenario['requires_manual_review']}, yorum={scenario['commentary']}"
            )

        lines.extend([
            "",
            "## Politika Icgoruleri",
            "",
        ])
        for insight in summary["policy_insights"]:
            lines.append(f"- {insight}")

        lines.extend([
            "",
            "## Faz 8 Kapanis",
            "",
            f"- Ozet: {summary['phase8_closeout']['summary']}",
            f"- Sonraki adim: {summary['phase8_closeout']['next_step']}",
        ])

        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                output_file.write("\n".join(lines))
        except OSError as error:
            raise ArtifactWriteError(f"Faz 8 markdown raporu yazilamadi: {output_path}") from error