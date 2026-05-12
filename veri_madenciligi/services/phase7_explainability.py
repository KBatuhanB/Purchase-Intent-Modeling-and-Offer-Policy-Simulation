from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from ..config import TARGET_COLUMN, ProjectPaths
from ..core.exceptions import ArtifactWriteError, DataAnalysisError
from ..core.serialization import to_json_safe
from .csv_loader import CsvDatasetLoader
from .phase2_eda import deduplicate_dataset
from .phase3_preprocessing import build_feature_role_matrix, split_dataset_stratified, validate_required_columns
from .phase4_baseline import BaselineModelResult, evaluate_binary_classifier
from .phase6_advanced_modeling import compute_slice_metrics

FAIRNESS_GAP_THRESHOLD = 0.12


@dataclass(frozen=True)
class Phase7Artifacts:
    markdown_report_path: Path
    json_summary_path: Path
    plots_dir: Path
    scenarios_dir: Path


@dataclass
class Phase7ModelBundle:
    family_key: str
    model_name: str
    model_path: Path
    model_object: Any = field(repr=False)
    holdout_result: BaselineModelResult | None = None
    stored_holdout_result: dict[str, Any] | None = None
    permutation_importance: list[dict[str, Any]] = field(default_factory=list)
    shap_global_summary: list[dict[str, Any]] = field(default_factory=list)
    local_explanations: list[dict[str, Any]] = field(default_factory=list)
    fairness_summary: dict[str, Any] = field(default_factory=dict)
    global_feature_frame: pd.DataFrame | None = field(default=None, repr=False)
    global_shap_values: np.ndarray | None = field(default=None, repr=False)


def sample_frame(frame: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if max_rows <= 0:
        raise DataAnalysisError("Ornekleme boyutu sifirdan buyuk olmalidir.")
    if len(frame) <= max_rows:
        return frame.copy(deep=True)
    return frame.sample(n=max_rows, random_state=random_state).sort_index().copy(deep=True)


def compute_group_fairness_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    metrics = compute_slice_metrics(y_true, predictions, probabilities)
    confusion = np.asarray(metrics["confusion_matrix"], dtype=int)
    true_negative, false_positive, false_negative, true_positive = confusion.ravel()

    positive_prediction_rate = float(np.asarray(predictions, dtype=int).mean()) if len(predictions) else 0.0
    false_positive_rate = (
        false_positive / (false_positive + true_negative)
        if (false_positive + true_negative)
        else None
    )
    false_negative_rate = (
        false_negative / (false_negative + true_positive)
        if (false_negative + true_positive)
        else None
    )

    metrics.update(
        {
            "positive_prediction_rate": round(positive_prediction_rate, 6),
            "false_positive_rate": round(float(false_positive_rate), 6) if false_positive_rate is not None else None,
            "false_negative_rate": round(float(false_negative_rate), 6) if false_negative_rate is not None else None,
        }
    )
    return metrics


def summarize_fairness_by_group(
    evaluation_frame: pd.DataFrame,
    result: BaselineModelResult,
    *,
    min_group_size: int = 20,
) -> dict[str, Any]:
    frame = evaluation_frame.reset_index(drop=True).copy(deep=True)
    predictions = np.asarray(result.predictions, dtype=int)
    probabilities = np.asarray(result.probabilities, dtype=float)
    y_true = frame[TARGET_COLUMN].to_numpy(dtype=int)

    if len(frame) != len(predictions):
        raise DataAnalysisError("Fairness ozeti icin veri ve tahmin boyutlari esit olmalidir.")

    overall = compute_group_fairness_metrics(y_true, predictions, probabilities)
    age_band = pd.cut(
        pd.to_numeric(frame["Age"], errors="coerce"),
        bins=[17, 29, 44, 70],
        labels=["18_29", "30_44", "45_plus"],
        include_lowest=True,
    )

    group_registry = {
        "Gender": frame["Gender"].astype(str).map(lambda value: f"gender_{value}"),
        "LoyaltyProgram": frame["LoyaltyProgram"].astype(str).map(lambda value: f"loyalty_{value}"),
        "ProductCategory": frame["ProductCategory"].astype(str).map(lambda value: f"category_{value}"),
        "AgeBand": age_band.astype("object").where(age_band.notna(), "unknown").astype(str),
    }

    grouped_summary: dict[str, Any] = {}
    for group_name, group_series in group_registry.items():
        group_details: dict[str, Any] = {}
        for group_value in sorted(group_series.dropna().unique()):
            mask = group_series == group_value
            row_count = int(mask.sum())
            if row_count < min_group_size:
                continue

            metrics = compute_group_fairness_metrics(
                y_true=y_true[mask.to_numpy()],
                predictions=predictions[mask.to_numpy()],
                probabilities=probabilities[mask.to_numpy()],
            )
            metrics["gaps_vs_overall"] = {
                metric_name: round(abs(float(metrics[metric_name]) - float(overall[metric_name])), 6)
                if metrics.get(metric_name) is not None and overall.get(metric_name) is not None
                else None
                for metric_name in (
                    "recall",
                    "precision",
                    "positive_prediction_rate",
                    "false_positive_rate",
                    "false_negative_rate",
                )
            }
            group_details[str(group_value)] = metrics

        if group_details:
            grouped_summary[group_name] = group_details

    summary = {
        "overall": overall,
        "groups": grouped_summary,
    }
    summary["alerts"] = build_fairness_alerts(summary, gap_threshold=FAIRNESS_GAP_THRESHOLD)
    return summary


def build_fairness_alerts(summary: dict[str, Any], *, gap_threshold: float) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for group_name, group_details in summary.get("groups", {}).items():
        for group_value, metrics in group_details.items():
            for metric_name, gap_value in metrics.get("gaps_vs_overall", {}).items():
                if gap_value is None or gap_value < gap_threshold:
                    continue
                alerts.append(
                    {
                        "group_name": group_name,
                        "group_value": group_value,
                        "metric_name": metric_name,
                        "gap_value": gap_value,
                        "message": (
                            f"{group_name}:{group_value} icin {metric_name} farki {gap_value} oldu; "
                            "bu segment final sunumunda dikkatle yorumlanmalidir."
                        ),
                    }
                )
    return alerts


def extract_top_local_contributions(
    feature_row: pd.Series,
    shap_values: np.ndarray,
    *,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    feature_names = list(feature_row.index)
    contributions = np.asarray(shap_values, dtype=float)
    if contributions.shape[0] != len(feature_names):
        raise DataAnalysisError("Yerel SHAP katkilari ile ozellik sayisi uyusmuyor.")

    ranked_indices = np.argsort(np.abs(contributions))[::-1][:top_k]
    top_contributions: list[dict[str, Any]] = []
    for feature_index in ranked_indices:
        shap_value = float(contributions[feature_index])
        top_contributions.append(
            {
                "feature_name": feature_names[feature_index],
                "feature_value": to_json_safe(feature_row.iloc[feature_index]),
                "shap_value": round(shap_value, 6),
                "direction": "satinalma_olasiligini_artiriyor" if shap_value >= 0.0 else "satinalma_olasiligini_azaltiyor",
            }
        )
    return top_contributions


def select_reference_scenarios(
    evaluation_frame: pd.DataFrame,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    *,
    max_scenarios: int = 8,
) -> list[dict[str, Any]]:
    frame = evaluation_frame.reset_index(drop=True).copy(deep=True)
    frame["row_position"] = frame.index.astype(int)
    frame["predicted_probability"] = np.asarray(probabilities, dtype=float)
    frame["predicted_label"] = np.asarray(predictions, dtype=int)
    frame["distance_to_boundary"] = np.abs(frame["predicted_probability"] - 0.5)

    high_time_threshold = float(pd.to_numeric(frame["TimeSpentOnWebsite"], errors="coerce").quantile(0.75))
    low_history_threshold = float(pd.to_numeric(frame["NumberOfPurchases"], errors="coerce").quantile(0.25))
    used_positions: set[int] = set()
    scenarios: list[dict[str, Any]] = []

    def add_scenario(
        scenario_id: str,
        title: str,
        selection_reason: str,
        mask: pd.Series,
        *,
        sort_by: list[str],
        ascending: list[bool],
    ) -> None:
        candidate_frame = frame.loc[mask & ~frame["row_position"].isin(used_positions)].sort_values(
            by=sort_by,
            ascending=ascending,
        )
        if candidate_frame.empty:
            return

        selected_row = candidate_frame.iloc[0]
        row_position = int(selected_row["row_position"])
        used_positions.add(row_position)
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "title": title,
                "selection_reason": selection_reason,
                "row_position": row_position,
                "true_label": int(selected_row[TARGET_COLUMN]),
                "predicted_label": int(selected_row["predicted_label"]),
                "predicted_probability": round(float(selected_row["predicted_probability"]), 6),
                "feature_snapshot": {
                    column_name: to_json_safe(selected_row[column_name])
                    for column_name in evaluation_frame.columns
                },
            }
        )

    # Bu senaryolar, final demoda hem tipik hem de hata yapan karar davranisini savunabilmek icin sabit kategorilere ayrilir.
    add_scenario(
        "high_confidence_true_positive",
        "Yuksek Guvenli Dogru Pozitif",
        "Modelin cok guvendigi ve gercekte satin alma yapan musteriyi gostermek icin secildi.",
        (frame[TARGET_COLUMN] == 1) & (frame["predicted_label"] == 1) & (frame["predicted_probability"] >= 0.8),
        sort_by=["predicted_probability"],
        ascending=[False],
    )
    add_scenario(
        "high_confidence_true_negative",
        "Yuksek Guvenli Dogru Negatif",
        "Modelin satin alma niyeti dusuk musterileri nasil ayristirdigini gostermek icin secildi.",
        (frame[TARGET_COLUMN] == 0) & (frame["predicted_label"] == 0) & (frame["predicted_probability"] <= 0.2),
        sort_by=["predicted_probability"],
        ascending=[True],
    )
    add_scenario(
        "boundary_case",
        "Esik Yakini Belirsiz Vaka",
        "Karar bandina en yakin vaka, politika katmaninin neden kritik oldugunu gostermek icin secildi.",
        pd.Series(True, index=frame.index),
        sort_by=["distance_to_boundary"],
        ascending=[True],
    )
    add_scenario(
        "false_positive",
        "Yanlis Pozitif",
        "Modelin gereksiz mudahale onerebilecegi pahali bir hata tipini gostermek icin secildi.",
        (frame[TARGET_COLUMN] == 0) & (frame["predicted_label"] == 1),
        sort_by=["predicted_probability"],
        ascending=[False],
    )
    add_scenario(
        "false_negative",
        "Yanlis Negatif",
        "Modelin kacirdigi ama gercekte satin alma yapan musteri profilini gostermek icin secildi.",
        (frame[TARGET_COLUMN] == 1) & (frame["predicted_label"] == 0),
        sort_by=["predicted_probability"],
        ascending=[False],
    )
    add_scenario(
        "loyalty_positive_intent",
        "Sadakat Programli Yuksek Niyet",
        "Sadakat programi bulunan kullanicilarda modelin hangi sinyalleri one cikardigini gormek icin secildi.",
        (frame["LoyaltyProgram"] == 1) & (frame["predicted_probability"] >= 0.65),
        sort_by=["predicted_probability"],
        ascending=[False],
    )
    add_scenario(
        "non_loyal_high_time",
        "Sadakatsiz Ama Yuksek Ilgi",
        "Uzun sure sitede kalan fakat sadakat programinda olmayan kararsiz kullaniciyi aciklamak icin secildi.",
        (frame["LoyaltyProgram"] == 0) & (frame["TimeSpentOnWebsite"] >= high_time_threshold),
        sort_by=["TimeSpentOnWebsite", "predicted_probability"],
        ascending=[False, False],
    )
    add_scenario(
        "low_history_surprise_purchase",
        "Dusuk Gecmis Satin Alma Ama Pozitif Vaka",
        "Gecmis satin alma sinyali zayifken modelin baska sinyallerle nasil pozitif karar verdigini gostermek icin secildi.",
        (frame["NumberOfPurchases"] <= low_history_threshold) & (frame[TARGET_COLUMN] == 1),
        sort_by=["predicted_probability"],
        ascending=[False],
    )

    if len(scenarios) < max_scenarios:
        fallback_candidates = frame.loc[~frame["row_position"].isin(used_positions)].sort_values(
            by=["distance_to_boundary"],
            ascending=[True],
        )
        for _, fallback_row in fallback_candidates.iterrows():
            if len(scenarios) >= max_scenarios:
                break
            scenarios.append(
                {
                    "scenario_id": f"fallback_{len(scenarios) + 1}",
                    "title": "Yedek Belirsiz Vaka",
                    "selection_reason": "Birincil kategoriler dolmadi; bu yuzden sinir bolgesine yakin bir vaka eklendi.",
                    "row_position": int(fallback_row["row_position"]),
                    "true_label": int(fallback_row[TARGET_COLUMN]),
                    "predicted_label": int(fallback_row["predicted_label"]),
                    "predicted_probability": round(float(fallback_row["predicted_probability"]), 6),
                    "feature_snapshot": {
                        column_name: to_json_safe(fallback_row[column_name])
                        for column_name in evaluation_frame.columns
                    },
                }
            )
            used_positions.add(int(fallback_row["row_position"]))

    return scenarios[:max_scenarios]


def build_business_insights(
    champion_bundle: Phase7ModelBundle,
    challenger_bundle: Phase7ModelBundle,
) -> list[str]:
    insights: list[str] = []

    if champion_bundle.shap_global_summary:
        top_feature = champion_bundle.shap_global_summary[0]
        insights.append(
            f"Champion modelde en baskin sinyal {top_feature['feature_name']} oldu; ortalama mutlak SHAP etkisi {top_feature['mean_abs_shap']} seviyesinde." 
        )

    champion_top_features = {item["feature_name"] for item in champion_bundle.shap_global_summary[:3]}
    challenger_top_features = {item["feature_name"] for item in challenger_bundle.shap_global_summary[:3]}
    common_features = sorted(champion_top_features & challenger_top_features)
    if common_features:
        insights.append(
            "Champion ve challenger modellerin ilk uc aciklayici sinyalinde ortusen ozellikler goruldu: "
            + ", ".join(common_features)
            + "."
        )

    champion_alerts = champion_bundle.fairness_summary.get("alerts", [])
    if champion_alerts:
        first_alert = champion_alerts[0]
        insights.append(first_alert["message"])
    else:
        insights.append(
            f"Champion model icin {FAIRNESS_GAP_THRESHOLD} esigini asan belirgin fairness-lite alarmi tespit edilmedi."
        )

    boundary_case = next(
        (item for item in champion_bundle.local_explanations if item["scenario_id"] == "boundary_case"),
        None,
    )
    if boundary_case and boundary_case["top_contributions"]:
        dominant_drivers = ", ".join(
            contribution["feature_name"] for contribution in boundary_case["top_contributions"][:2]
        )
        insights.append(
            f"Esik yakinindaki kararsiz vakalarda karar agirlikli olarak {dominant_drivers} sinyalleriyle sekillendi."
        )

    return insights


class Phase7ExplainabilityService:
    def __init__(
        self,
        loader: CsvDatasetLoader,
        logger: logging.Logger,
        *,
        background_sample_size: int = 20,
        shap_sample_size: int = 60,
        min_fairness_group_size: int = 20,
    ) -> None:
        self._loader = loader
        self._logger = logger
        self._background_sample_size = background_sample_size
        self._shap_sample_size = shap_sample_size
        self._min_fairness_group_size = min_fairness_group_size

    def run(self, project_paths: ProjectPaths) -> Phase7Artifacts:
        dataset = self._loader.load(project_paths.dataset_path)
        deduplicated_dataset = deduplicate_dataset(dataset)
        role_matrix = build_feature_role_matrix()
        validate_required_columns(
            deduplicated_dataset,
            role_matrix.base_input_features + (role_matrix.target_column,),
        )

        phase6_summary = self._load_json_context(project_paths.phase_6_dir / "advanced_modeling_summary.json")
        if phase6_summary is None:
            raise DataAnalysisError("Faz 7 icin Faz 6 ozeti bulunamadi; once phase6 calistirilmalidir.")

        artifacts = self._ensure_phase7_directories(project_paths.phase_7_dir)
        split_config = self._extract_phase6_split_config(phase6_summary)
        self._logger.info("Faz 7 aciklanabilirlik ve guvenilirlik analizi baslatildi: %s", project_paths.dataset_path)

        # Faz 7, Faz 6 champion/challenger modellerini ayni split uzerinde tekrar kullanir; boylece aciklamalar birebir ayni karar yuzeyini temsil eder.
        outer_train, outer_test = split_dataset_stratified(
            deduplicated_dataset,
            role_matrix.target_column,
            test_size=split_config["test_size"],
            random_state=split_config["random_state"],
        )
        model_bundles = self._load_phase6_model_bundles(
            phase6_summary=phase6_summary,
            project_paths=project_paths,
            outer_test=outer_test,
            role_matrix=role_matrix,
        )

        champion_key = phase6_summary["champion_selection"]["champion_key"]
        challenger_key = phase6_summary["champion_selection"]["challenger_key"]
        champion_bundle = model_bundles[champion_key]
        challenger_bundle = model_bundles[challenger_key]

        scenario_catalog = select_reference_scenarios(
            evaluation_frame=outer_test,
            probabilities=np.asarray(champion_bundle.holdout_result.probabilities, dtype=float),
            predictions=np.asarray(champion_bundle.holdout_result.predictions, dtype=int),
            max_scenarios=8,
        )
        background_frame = sample_frame(
            outer_train.loc[:, role_matrix.base_input_features],
            max_rows=self._background_sample_size,
            random_state=split_config["random_state"],
        )
        global_sample_positions = list(
            sample_frame(outer_test, max_rows=self._shap_sample_size, random_state=split_config["random_state"]).index
        )
        explanation_positions = list(
            dict.fromkeys(global_sample_positions + [scenario["row_position"] for scenario in scenario_catalog])
        )
        explanation_frame = outer_test.loc[explanation_positions, role_matrix.base_input_features].copy(deep=True)

        for bundle in model_bundles.values():
            bundle.permutation_importance = self._compute_permutation_importance(
                model=bundle.model_object,
                evaluation_frame=outer_test.loc[:, role_matrix.base_input_features],
                y_true=outer_test[TARGET_COLUMN],
                random_state=split_config["random_state"],
            )
            shap_payload = self._compute_shap_payload(
                model=bundle.model_object,
                background_frame=background_frame,
                explanation_frame=explanation_frame,
                global_sample_positions=global_sample_positions,
                scenario_catalog=scenario_catalog,
            )
            bundle.shap_global_summary = shap_payload["global_summary"]
            bundle.global_feature_frame = shap_payload["global_feature_frame"]
            bundle.global_shap_values = shap_payload["global_shap_values"]
            bundle.local_explanations = shap_payload["local_explanations"]
            bundle.fairness_summary = summarize_fairness_by_group(
                evaluation_frame=outer_test,
                result=bundle.holdout_result,
                min_group_size=self._min_fairness_group_size,
            )

        business_insights = build_business_insights(champion_bundle, challenger_bundle)
        summary = self._build_summary(
            project_paths=project_paths,
            phase6_summary=phase6_summary,
            split_config=split_config,
            deduplicated_row_count=len(deduplicated_dataset),
            model_bundles=model_bundles,
            scenario_catalog=scenario_catalog,
            business_insights=business_insights,
        )

        self._write_plots(artifacts.plots_dir, champion_bundle, challenger_bundle)
        self._write_scenario_cards(artifacts.scenarios_dir, scenario_catalog, champion_bundle, challenger_bundle)
        self._write_json_summary(artifacts.json_summary_path, summary)
        self._write_markdown_report(artifacts.markdown_report_path, summary)
        self._logger.info("Faz 7 aciklanabilirlik ve guvenilirlik analizi tamamlandi.")
        return artifacts

    def _ensure_phase7_directories(self, phase_7_dir: Path) -> Phase7Artifacts:
        plots_dir = phase_7_dir / "plots"
        scenarios_dir = phase_7_dir / "scenarios"
        for directory in (phase_7_dir, plots_dir, scenarios_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return Phase7Artifacts(
            markdown_report_path=phase_7_dir / "explainability_report.md",
            json_summary_path=phase_7_dir / "explainability_summary.json",
            plots_dir=plots_dir,
            scenarios_dir=scenarios_dir,
        )

    def _extract_phase6_split_config(self, phase6_summary: dict[str, Any]) -> dict[str, Any]:
        split_summary = phase6_summary.get("split_summary")
        if not split_summary:
            raise DataAnalysisError("Faz 6 ozeti split bilgisi icermiyor.")
        return {
            "test_size": float(split_summary["test_size"]),
            "validation_size": float(split_summary["validation_size"]),
            "random_state": int(split_summary["random_state"]),
        }

    def _load_phase6_model_bundles(
        self,
        *,
        phase6_summary: dict[str, Any],
        project_paths: ProjectPaths,
        outer_test: pd.DataFrame,
        role_matrix: Any,
    ) -> dict[str, Phase7ModelBundle]:
        candidate_families = phase6_summary.get("candidate_families", {})
        champion_key = phase6_summary.get("champion_selection", {}).get("champion_key")
        challenger_key = phase6_summary.get("champion_selection", {}).get("challenger_key")
        required_keys = [key for key in (champion_key, challenger_key) if key]
        if len(required_keys) != 2:
            raise DataAnalysisError("Faz 7 icin Faz 6 champion/challenger bilgisi eksik.")

        bundles: dict[str, Phase7ModelBundle] = {}
        for family_key in required_keys:
            family_summary = candidate_families.get(family_key)
            if family_summary is None:
                raise DataAnalysisError(f"Faz 6 ozetinde beklenen aile bulunamadi: {family_key}")

            model_name = str(family_summary["holdout_result"]["name"])
            model_path = project_paths.phase_6_dir / "models" / f"{model_name}.joblib"
            if not model_path.exists():
                raise DataAnalysisError(f"Faz 6 model artefakti bulunamadi: {model_path}")

            try:
                model_object = joblib.load(model_path)
            except (OSError, ValueError) as error:
                raise DataAnalysisError(f"Faz 6 model artefakti yuklenemedi: {model_path}") from error

            X_eval = outer_test.loc[:, role_matrix.base_input_features]
            y_eval = outer_test[TARGET_COLUMN]
            probabilities = model_object.predict_proba(X_eval)[:, 1]
            predictions = model_object.predict(X_eval)
            holdout_result = evaluate_binary_classifier(
                name=model_name,
                family=family_key,
                y_true=y_eval,
                predictions=predictions,
                probabilities=probabilities,
                note=str(family_summary["holdout_result"]["note"]),
            )

            self._validate_phase6_alignment(family_summary["holdout_result"], holdout_result, family_key)
            bundles[family_key] = Phase7ModelBundle(
                family_key=family_key,
                model_name=model_name,
                model_path=model_path,
                model_object=model_object,
                holdout_result=holdout_result,
                stored_holdout_result=family_summary["holdout_result"],
            )
        return bundles

    def _validate_phase6_alignment(
        self,
        stored_result: dict[str, Any],
        recalculated_result: BaselineModelResult,
        family_key: str,
    ) -> None:
        for metric_name in ("pr_auc", "balanced_accuracy", "brier_score"):
            stored_value = stored_result["metrics"].get(metric_name)
            recalculated_value = recalculated_result.metrics.get(metric_name)
            if stored_value is None or recalculated_value is None:
                continue
            if abs(float(stored_value) - float(recalculated_value)) > 1e-6:
                raise DataAnalysisError(
                    f"Faz 7 yeniden hesaplanan {family_key} sonucu Faz 6 ile uyusmuyor: {metric_name}"
                )

    def _compute_permutation_importance(
        self,
        *,
        model: Any,
        evaluation_frame: pd.DataFrame,
        y_true: pd.Series,
        random_state: int,
    ) -> list[dict[str, Any]]:
        try:
            importance = permutation_importance(
                model,
                evaluation_frame,
                y_true,
                scoring="average_precision",
                n_repeats=8,
                random_state=random_state,
                n_jobs=1,
            )
        except ValueError as error:
            raise DataAnalysisError(f"Permutation importance hesaplanamadi: {error}") from error

        records = []
        for feature_name, mean_value, std_value in zip(
            evaluation_frame.columns,
            importance.importances_mean,
            importance.importances_std,
            strict=True,
        ):
            records.append(
                {
                    "feature_name": str(feature_name),
                    "importance_mean": round(float(mean_value), 6),
                    "importance_std": round(float(std_value), 6),
                }
            )
        return sorted(records, key=lambda item: item["importance_mean"], reverse=True)

    def _compute_shap_payload(
        self,
        *,
        model: Any,
        background_frame: pd.DataFrame,
        explanation_frame: pd.DataFrame,
        global_sample_positions: list[int],
        scenario_catalog: list[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            # Ham ozellik uzayinda model-agnostic SHAP kullanimi, pipeline ve kalibrasyon sarmalini bozmadan iki modeli de ayni dilde aciklamamizi saglar.
            explainer = shap.Explainer(model.predict_proba, background_frame)
            explanation = explainer(explanation_frame, silent=True)
        except Exception as error:  # noqa: BLE001 - SHAP farkli backend hatalari uretebilir.
            raise DataAnalysisError(f"SHAP aciklamasi uretilemedi: {error}") from error

        positive_class_values = np.asarray(explanation.values[:, :, 1], dtype=float)
        position_to_index = {int(position): index for index, position in enumerate(explanation_frame.index)}
        global_indices = [position_to_index[position] for position in global_sample_positions if position in position_to_index]

        global_feature_frame = explanation_frame.loc[global_sample_positions].copy(deep=True)
        global_shap_values = positive_class_values[global_indices, :]
        mean_abs_values = np.abs(global_shap_values).mean(axis=0)
        mean_signed_values = global_shap_values.mean(axis=0)

        global_summary = sorted(
            [
                {
                    "feature_name": str(feature_name),
                    "mean_abs_shap": round(float(mean_abs_value), 6),
                    "mean_signed_shap": round(float(mean_signed_value), 6),
                }
                for feature_name, mean_abs_value, mean_signed_value in zip(
                    global_feature_frame.columns,
                    mean_abs_values,
                    mean_signed_values,
                    strict=True,
                )
            ],
            key=lambda item: item["mean_abs_shap"],
            reverse=True,
        )

        local_explanations: list[dict[str, Any]] = []
        for scenario in scenario_catalog:
            row_position = int(scenario["row_position"])
            explanation_index = position_to_index[row_position]
            feature_row = explanation_frame.loc[row_position]
            top_contributions = extract_top_local_contributions(
                feature_row=feature_row,
                shap_values=positive_class_values[explanation_index, :],
            )
            local_explanations.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "title": scenario["title"],
                    "selection_reason": scenario["selection_reason"],
                    "row_position": row_position,
                    "predicted_probability": round(float(model.predict_proba(feature_row.to_frame().T)[:, 1][0]), 6),
                    "predicted_label": int(model.predict(feature_row.to_frame().T)[0]),
                    "base_value": round(float(explanation.base_values[explanation_index, 1]), 6),
                    "top_contributions": top_contributions,
                }
            )

        return {
            "global_summary": global_summary,
            "global_feature_frame": global_feature_frame,
            "global_shap_values": global_shap_values,
            "local_explanations": local_explanations,
        }

    def _build_summary(
        self,
        *,
        project_paths: ProjectPaths,
        phase6_summary: dict[str, Any],
        split_config: dict[str, Any],
        deduplicated_row_count: int,
        model_bundles: dict[str, Phase7ModelBundle],
        scenario_catalog: list[dict[str, Any]],
        business_insights: list[str],
    ) -> dict[str, Any]:
        champion_key = phase6_summary["champion_selection"]["champion_key"]
        challenger_key = phase6_summary["champion_selection"]["challenger_key"]
        return {
            "dataset_path": str(project_paths.dataset_path),
            "deduplicated_row_count": deduplicated_row_count,
            "phase6_context": {
                "champion_selection": phase6_summary.get("champion_selection"),
                "library_strategy": phase6_summary.get("library_strategy"),
                "split_summary": phase6_summary.get("split_summary"),
            },
            "explanation_config": {
                "background_sample_size": self._background_sample_size,
                "global_shap_sample_size": self._shap_sample_size,
                "min_fairness_group_size": self._min_fairness_group_size,
                "shap_version": shap.__version__,
                "random_state": split_config["random_state"],
            },
            "model_explanations": {
                family_key: {
                    "model_name": bundle.model_name,
                    "holdout_result": self._result_summary(bundle.holdout_result),
                    "permutation_importance": bundle.permutation_importance,
                    "shap_global_summary": bundle.shap_global_summary,
                    "fairness_summary": bundle.fairness_summary,
                    "local_explanations": bundle.local_explanations,
                }
                for family_key, bundle in model_bundles.items()
            },
            "scenario_catalog": scenario_catalog,
            "business_insights": business_insights,
            "phase7_closeout": {
                "champion_key": champion_key,
                "challenger_key": challenger_key,
                "summary": "Faz 7'de champion ve challenger ayni holdout verisi uzerinde global importance, SHAP, yerel aciklama ve fairness-lite lensleriyle okundu.",
                "next_step": "Faz 8'de bu aciklamalar skor bandi ve aksiyon kurallarina baglanacaktir.",
            },
        }

    def _result_summary(self, result: BaselineModelResult | None) -> dict[str, Any] | None:
        if result is None:
            return None
        return {
            "name": result.name,
            "family": result.family,
            "metrics": result.metrics,
            "calibration": result.calibration,
            "note": result.note,
        }

    def _write_plots(
        self,
        plots_dir: Path,
        champion_bundle: Phase7ModelBundle,
        challenger_bundle: Phase7ModelBundle,
    ) -> None:
        self._plot_permutation_importance(
            plots_dir / "champion_permutation_importance.png",
            champion_bundle.model_name,
            champion_bundle.permutation_importance,
        )
        self._plot_permutation_importance(
            plots_dir / "challenger_permutation_importance.png",
            challenger_bundle.model_name,
            challenger_bundle.permutation_importance,
        )
        self._plot_shap_global_comparison(
            plots_dir / "shap_global_comparison.png",
            champion_bundle,
            challenger_bundle,
        )
        self._plot_dependence(
            plots_dir / "champion_top_feature_dependence.png",
            champion_bundle,
        )
        self._plot_dependence(
            plots_dir / "challenger_top_feature_dependence.png",
            challenger_bundle,
        )
        self._plot_fairness_overview(
            plots_dir / "champion_fairness_overview.png",
            champion_bundle,
        )

    def _plot_permutation_importance(
        self,
        output_path: Path,
        model_name: str,
        importance_records: list[dict[str, Any]],
    ) -> None:
        top_records = importance_records[:7]
        labels = [record["feature_name"] for record in reversed(top_records)]
        values = [record["importance_mean"] for record in reversed(top_records)]
        stds = [record["importance_std"] for record in reversed(top_records)]

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.barh(labels, values, xerr=stds, capsize=4)
        axis.set_title(f"Permutation Importance | {model_name}")
        axis.set_xlabel("Ortalama AP Dusu")
        figure.tight_layout()
        self._save_figure(figure, output_path)

    def _plot_shap_global_comparison(
        self,
        output_path: Path,
        champion_bundle: Phase7ModelBundle,
        challenger_bundle: Phase7ModelBundle,
    ) -> None:
        all_features = [item["feature_name"] for item in champion_bundle.shap_global_summary]
        champion_lookup = {item["feature_name"]: item["mean_abs_shap"] for item in champion_bundle.shap_global_summary}
        challenger_lookup = {item["feature_name"]: item["mean_abs_shap"] for item in challenger_bundle.shap_global_summary}
        x_positions = np.arange(len(all_features))
        width = 0.36

        figure, axis = plt.subplots(figsize=(14, 6))
        axis.bar(
            x_positions - (width / 2),
            [champion_lookup.get(feature_name, 0.0) for feature_name in all_features],
            width=width,
            label=champion_bundle.model_name,
        )
        axis.bar(
            x_positions + (width / 2),
            [challenger_lookup.get(feature_name, 0.0) for feature_name in all_features],
            width=width,
            label=challenger_bundle.model_name,
        )
        axis.set_title("Faz 7 Ortalama Mutlak SHAP Karsilastirmasi")
        axis.set_ylabel("Mean |SHAP|")
        axis.set_xticks(x_positions, all_features, rotation=20, ha="right")
        axis.legend()
        figure.tight_layout()
        self._save_figure(figure, output_path)

    def _plot_dependence(self, output_path: Path, bundle: Phase7ModelBundle) -> None:
        if not bundle.shap_global_summary or bundle.global_feature_frame is None or bundle.global_shap_values is None:
            return
        top_feature = bundle.shap_global_summary[0]["feature_name"]
        feature_index = list(bundle.global_feature_frame.columns).index(top_feature)
        feature_values = pd.to_numeric(bundle.global_feature_frame[top_feature], errors="coerce")
        shap_values = bundle.global_shap_values[:, feature_index]

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.scatter(feature_values, shap_values, alpha=0.75)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"SHAP Dependence | {bundle.model_name} | {top_feature}")
        axis.set_xlabel(top_feature)
        axis.set_ylabel("SHAP etkisi")
        figure.tight_layout()
        self._save_figure(figure, output_path)

    def _plot_fairness_overview(self, output_path: Path, bundle: Phase7ModelBundle) -> None:
        records: list[dict[str, Any]] = []
        for group_name, group_details in bundle.fairness_summary.get("groups", {}).items():
            for group_value, metrics in group_details.items():
                records.append(
                    {
                        "segment": f"{group_name}:{group_value}",
                        "recall_gap": metrics["gaps_vs_overall"].get("recall") or 0.0,
                        "false_positive_rate_gap": metrics["gaps_vs_overall"].get("false_positive_rate") or 0.0,
                    }
                )

        records = sorted(records, key=lambda item: max(item["recall_gap"], item["false_positive_rate_gap"]), reverse=True)[:8]
        figure, axis = plt.subplots(figsize=(12, 6))

        if not records:
            axis.text(0.5, 0.5, "Gosterilebilir fairness farki bulunamadi.", ha="center", va="center")
            axis.set_axis_off()
        else:
            x_positions = np.arange(len(records))
            width = 0.36
            axis.bar(
                x_positions - (width / 2),
                [record["recall_gap"] for record in records],
                width=width,
                label="recall_gap",
            )
            axis.bar(
                x_positions + (width / 2),
                [record["false_positive_rate_gap"] for record in records],
                width=width,
                label="false_positive_rate_gap",
            )
            axis.axhline(FAIRNESS_GAP_THRESHOLD, color="black", linestyle="--", linewidth=1, label="alert_threshold")
            axis.set_xticks(x_positions, [record["segment"] for record in records], rotation=25, ha="right")
            axis.set_ylabel("Mutlak fark")
            axis.set_title(f"Fairness-lite gap gorunumu | {bundle.model_name}")
            axis.legend()

        figure.tight_layout()
        self._save_figure(figure, output_path)

    def _write_scenario_cards(
        self,
        scenarios_dir: Path,
        scenario_catalog: list[dict[str, Any]],
        champion_bundle: Phase7ModelBundle,
        challenger_bundle: Phase7ModelBundle,
    ) -> None:
        champion_lookup = {item["scenario_id"]: item for item in champion_bundle.local_explanations}
        challenger_lookup = {item["scenario_id"]: item for item in challenger_bundle.local_explanations}

        for index, scenario in enumerate(scenario_catalog, start=1):
            lines = [
                f"# Senaryo {index:02d} - {scenario['title']}",
                "",
                f"- Senaryo kimligi: {scenario['scenario_id']}",
                f"- Secim nedeni: {scenario['selection_reason']}",
                f"- Gercek etiket: {scenario['true_label']}",
                f"- Faz 6 champion referans olasiligi: {scenario['predicted_probability']}",
                "",
                "## Ozellik Ozeti",
                "",
            ]
            for feature_name, feature_value in scenario["feature_snapshot"].items():
                lines.append(f"- {feature_name}: {feature_value}")

            lines.extend(["", f"## {champion_bundle.model_name}", ""])
            for contribution in champion_lookup[scenario["scenario_id"]]["top_contributions"]:
                lines.append(
                    f"- {contribution['feature_name']}: {contribution['feature_value']} | {contribution['direction']} | SHAP={contribution['shap_value']}"
                )

            lines.extend(["", f"## {challenger_bundle.model_name}", ""])
            for contribution in challenger_lookup[scenario["scenario_id"]]["top_contributions"]:
                lines.append(
                    f"- {contribution['feature_name']}: {contribution['feature_value']} | {contribution['direction']} | SHAP={contribution['shap_value']}"
                )

            output_path = scenarios_dir / f"{index:02d}_{scenario['scenario_id']}.md"
            try:
                with output_path.open("w", encoding="utf-8") as output_file:
                    output_file.write("\n".join(lines))
            except OSError as error:
                raise ArtifactWriteError(f"Faz 7 senaryo karti yazilamadi: {output_path}") from error

    def _save_figure(self, figure: plt.Figure, output_path: Path) -> None:
        try:
            figure.savefig(output_path, dpi=200, bbox_inches="tight")
        except OSError as error:
            raise ArtifactWriteError(f"Faz 7 grafigi kaydedilemedi: {output_path}") from error
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
            raise ArtifactWriteError(f"Faz 7 JSON ozeti yazilamadi: {output_path}") from error

    def _write_markdown_report(self, output_path: Path, summary: dict[str, Any]) -> None:
        champion_key = summary["phase7_closeout"]["champion_key"]
        challenger_key = summary["phase7_closeout"]["challenger_key"]
        champion_summary = summary["model_explanations"][champion_key]
        challenger_summary = summary["model_explanations"][challenger_key]

        lines = [
            "# Faz 7 Aciklanabilirlik ve Guvenilirlik Raporu",
            "",
            f"- Veri seti: `{summary['dataset_path']}`",
            f"- Tekillestirilmis satir sayisi: {summary['deduplicated_row_count']}",
            f"- SHAP surumu: {summary['explanation_config']['shap_version']}",
            "",
            "## Faz 6 Baglami",
            "",
            f"- Champion: {summary['phase6_context']['champion_selection']['champion_model']}",
            f"- Challenger: {summary['phase6_context']['champion_selection']['challenger_model']}",
            f"- Kutuphane stratejisi: {summary['phase6_context']['library_strategy']['rationale']}",
            "",
            "## Champion Global Icgoruler",
            "",
            f"- Holdout metrikleri: {champion_summary['holdout_result']['metrics']}",
            f"- En guclu SHAP sinyalleri: {champion_summary['shap_global_summary'][:5]}",
            f"- Fairness-lite alarmlari: {champion_summary['fairness_summary']['alerts']}",
            "",
            "## Challenger Global Icgoruler",
            "",
            f"- Holdout metrikleri: {challenger_summary['holdout_result']['metrics']}",
            f"- En guclu SHAP sinyalleri: {challenger_summary['shap_global_summary'][:5]}",
            f"- Fairness-lite alarmlari: {challenger_summary['fairness_summary']['alerts']}",
            "",
            "## Yerel Aciklama Senaryolari",
            "",
        ]

        for scenario in summary["scenario_catalog"]:
            champion_local = next(
                item for item in champion_summary["local_explanations"] if item["scenario_id"] == scenario["scenario_id"]
            )
            lines.append(f"### {scenario['title']}")
            lines.append(f"- Secim nedeni: {scenario['selection_reason']}")
            lines.append(f"- Champion olasiligi: {champion_local['predicted_probability']}")
            lines.append(f"- Champion ana suruculer: {champion_local['top_contributions']}")
            lines.append("")

        lines.extend(["## Is Icgoruleri", ""])
        for insight in summary["business_insights"]:
            lines.append(f"- {insight}")

        lines.extend(
            [
                "",
                "## Faz 7 Kapanis",
                "",
                f"- Ozet: {summary['phase7_closeout']['summary']}",
                f"- Sonraki adim: {summary['phase7_closeout']['next_step']}",
            ]
        )

        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                output_file.write("\n".join(lines))
        except OSError as error:
            raise ArtifactWriteError(f"Faz 7 markdown raporu yazilamadi: {output_path}") from error