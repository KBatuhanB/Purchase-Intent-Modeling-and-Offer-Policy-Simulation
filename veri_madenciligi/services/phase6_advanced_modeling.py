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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from ..config import TARGET_COLUMN, ProjectPaths
from ..core.exceptions import ArtifactWriteError, DataAnalysisError
from ..core.serialization import to_json_safe
from .csv_loader import CsvDatasetLoader
from .phase2_eda import deduplicate_dataset
from .phase3_preprocessing import (
    FeatureRoleMatrix,
    build_feature_role_matrix,
    build_linear_preprocessing_pipeline,
    build_tree_preprocessing_pipeline,
    split_dataset_stratified,
    validate_required_columns,
)
from .phase4_baseline import BaselineModelResult, evaluate_binary_classifier

MODEL_FAMILY_LABELS = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "hist_gradient_boosting": "HistGradientBoosting",
}

MODEL_FAMILY_COMPLEXITY = {
    "random_forest": 0,
    "hist_gradient_boosting": 1,
    "gradient_boosting": 2,
}


@dataclass(frozen=True)
class Phase6Artifacts:
    markdown_report_path: Path
    json_summary_path: Path
    plots_dir: Path
    models_dir: Path


@dataclass
class ModelFamilyOutcome:
    family_key: str
    family_label: str
    best_params: dict[str, Any]
    tuning_history: list[dict[str, Any]]
    selected_variant: str
    variant_rationale: str
    validation_variants: dict[str, BaselineModelResult]
    validation_result: BaselineModelResult
    holdout_result: BaselineModelResult | None = None
    holdout_model: Any = field(default=None, repr=False)


def _metric_or_default(value: float | None, default: float = -1.0) -> float:
    return float(value) if value is not None else default


def rank_model_result(result: BaselineModelResult) -> tuple[float, float, float, float, float]:
    calibration_gap = result.calibration.get("mean_absolute_gap")
    calibration_score = -float(calibration_gap) if calibration_gap is not None else -1.0
    return (
        _metric_or_default(result.metrics.get("pr_auc")),
        _metric_or_default(result.metrics.get("balanced_accuracy")),
        _metric_or_default(result.metrics.get("roc_auc")),
        -float(result.metrics.get("brier_score", 1.0)),
        calibration_score,
    )


def choose_model_variant(base_result: BaselineModelResult, calibrated_result: BaselineModelResult) -> dict[str, str]:
    base_pr_auc = _metric_or_default(base_result.metrics.get("pr_auc"), default=0.0)
    calibrated_pr_auc = _metric_or_default(calibrated_result.metrics.get("pr_auc"), default=0.0)
    base_brier = float(base_result.metrics.get("brier_score", 1.0))
    calibrated_brier = float(calibrated_result.metrics.get("brier_score", 1.0))
    base_gap = float(base_result.calibration.get("mean_absolute_gap") or 1.0)
    calibrated_gap = float(calibrated_result.calibration.get("mean_absolute_gap") or 1.0)

    if calibrated_pr_auc >= base_pr_auc + 0.005 and calibrated_brier <= base_brier + 0.01:
        return {
            "selected_variant": "sigmoid_calibrated",
            "rationale": "Kalibre varyant PR-AUC kazanci sagladigi ve olasilik kalitesini bozmadigi icin tercih edildi.",
        }

    if calibrated_pr_auc + 0.01 >= base_pr_auc and calibrated_brier + 0.005 < base_brier and calibrated_gap <= base_gap:
        return {
            "selected_variant": "sigmoid_calibrated",
            "rationale": "Kalibre varyant PR-AUC'u korurken Brier score ve kalibrasyon boslugunu iyilestirdigi icin tercih edildi.",
        }

    return {
        "selected_variant": "base",
        "rationale": "Baz varyant, kalibrasyon sonrasi olusan ek karmasikliga ragmen daha iyi ayrisma profili korudugu icin secildi.",
    }


def choose_champion_and_challenger(results: dict[str, BaselineModelResult]) -> dict[str, Any]:
    if len(results) < 2:
        raise DataAnalysisError("Champion/challenger karari icin en az iki aday sonuc gerekir.")

    sorted_candidates = sorted(results.items(), key=lambda item: rank_model_result(item[1]), reverse=True)
    champion_key, champion_result = sorted_candidates[0]
    challenger_key, challenger_result = sorted_candidates[1]

    pr_auc_gap = _metric_or_default(champion_result.metrics.get("pr_auc"), 0.0) - _metric_or_default(
        challenger_result.metrics.get("pr_auc"),
        0.0,
    )
    balanced_accuracy_gap = float(champion_result.metrics.get("balanced_accuracy", 0.0)) - float(
        challenger_result.metrics.get("balanced_accuracy", 0.0)
    )

    if (
        pr_auc_gap <= 0.01
        and balanced_accuracy_gap <= 0.01
        and MODEL_FAMILY_COMPLEXITY.get(challenger_key, 99) < MODEL_FAMILY_COMPLEXITY.get(champion_key, 99)
    ):
        champion_key, champion_result, challenger_key, challenger_result = (
            challenger_key,
            challenger_result,
            champion_key,
            champion_result,
        )
        rationale = "Validation farki marjinal oldugu icin daha sade ve daha savunulabilir aile champion olarak one cekildi."
    else:
        rationale = "Champion secimi validation PR-AUC, balanced accuracy ve kalibrasyon kalitesi birlikte okunarak yapildi."

    return {
        "champion_key": champion_key,
        "challenger_key": challenger_key,
        "champion_model": champion_result.name,
        "challenger_model": challenger_result.name,
        "rationale": rationale,
        "ordered_candidates": [
            {
                "family_key": family_key,
                "model_name": result.name,
                "pr_auc": result.metrics.get("pr_auc"),
                "balanced_accuracy": result.metrics.get("balanced_accuracy"),
                "brier_score": result.metrics.get("brier_score"),
            }
            for family_key, result in sorted_candidates
        ],
    }


def summarize_cross_validation(cv_results: dict[str, Any]) -> dict[str, Any]:
    metric_summary: dict[str, Any] = {}
    for metric_name in ("pr_auc", "balanced_accuracy", "roc_auc"):
        values = np.asarray(cv_results.get(f"test_{metric_name}", []), dtype=float)
        if values.size == 0:
            continue
        metric_summary[metric_name] = {
            "mean": round(float(values.mean()), 6),
            "std": round(float(values.std(ddof=0)), 6),
            "fold_scores": [round(float(value), 6) for value in values],
        }

    fit_times = np.asarray(cv_results.get("fit_time", []), dtype=float)
    score_times = np.asarray(cv_results.get("score_time", []), dtype=float)
    return {
        "metrics": metric_summary,
        "fit_time_mean": round(float(fit_times.mean()), 6) if fit_times.size else None,
        "score_time_mean": round(float(score_times.mean()), 6) if score_times.size else None,
    }


def compute_slice_metrics(y_true: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    y_true_array = np.asarray(y_true, dtype=int)
    prediction_array = np.asarray(predictions, dtype=int)
    probability_array = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)

    if len(y_true_array) != len(prediction_array) or len(y_true_array) != len(probability_array):
        raise DataAnalysisError("Dilime gore performans hesabi icin uzunluklar uyusmalidir.")

    confusion = confusion_matrix(y_true_array, prediction_array, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = confusion.ravel()

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    specificity_denominator = true_negative + false_positive

    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    specificity = true_negative / specificity_denominator if specificity_denominator else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    balanced_accuracy = ((recall + specificity) / 2.0) if specificity is not None else None

    pr_auc_value: float | None
    roc_auc_value: float | None
    if np.unique(y_true_array).shape[0] < 2:
        pr_auc_value = None
        roc_auc_value = None
    else:
        pr_auc_value = round(float(average_precision_score(y_true_array, probability_array)), 6)
        roc_auc_value = round(float(roc_auc_score(y_true_array, probability_array)), 6)

    return {
        "row_count": int(len(y_true_array)),
        "positive_rate": round(float(y_true_array.mean()), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "balanced_accuracy": round(float(balanced_accuracy), 6) if balanced_accuracy is not None else None,
        "pr_auc": pr_auc_value,
        "roc_auc": roc_auc_value,
        "confusion_matrix": confusion.astype(int).tolist(),
    }


def summarize_segment_performance(
    evaluation_frame: pd.DataFrame,
    result: BaselineModelResult,
    *,
    min_slice_size: int = 20,
) -> dict[str, Any]:
    frame = evaluation_frame.reset_index(drop=True)
    predictions = np.asarray(result.predictions, dtype=int)
    probabilities = np.asarray(result.probabilities, dtype=float)
    y_true = frame[TARGET_COLUMN].to_numpy(dtype=int)

    if len(frame) != len(predictions):
        raise DataAnalysisError("Segment performansi icin veri ve tahmin sayisi esit olmalidir.")

    age_band = pd.cut(
        pd.to_numeric(frame["Age"], errors="coerce"),
        bins=[17, 29, 44, 70],
        labels=["18_29", "30_44", "45_plus"],
        include_lowest=True,
    )
    income_band = pd.cut(
        pd.to_numeric(frame["AnnualIncome"], errors="coerce"),
        bins=[0, 50000, 90000, float("inf")],
        labels=["income_low", "income_mid", "income_high"],
        include_lowest=True,
    )

    segment_registry = {
        "Gender": frame["Gender"].astype(str).map(lambda value: f"gender_{value}"),
        "LoyaltyProgram": frame["LoyaltyProgram"].astype(str).map(lambda value: f"loyalty_{value}"),
        "ProductCategory": frame["ProductCategory"].astype(str).map(lambda value: f"category_{value}"),
        "AgeBand": age_band.astype("object").where(age_band.notna(), "unknown").astype(str),
        "IncomeBand": income_band.astype("object").where(income_band.notna(), "unknown").astype(str),
    }

    summary: dict[str, Any] = {}
    for segment_name, segment_series in segment_registry.items():
        segment_details: dict[str, Any] = {}
        for segment_value in sorted(segment_series.dropna().unique()):
            mask = segment_series == segment_value
            if int(mask.sum()) < min_slice_size:
                continue
            segment_details[str(segment_value)] = compute_slice_metrics(
                y_true=y_true[mask.to_numpy()],
                predictions=predictions[mask.to_numpy()],
                probabilities=probabilities[mask.to_numpy()],
            )
        if segment_details:
            summary[segment_name] = segment_details

    return summary


class Phase6AdvancedModelingService:
    def __init__(
        self,
        loader: CsvDatasetLoader,
        logger: logging.Logger,
        *,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self._loader = loader
        self._logger = logger
        self._test_size = test_size
        self._validation_size = validation_size
        self._random_state = random_state

    def run(self, project_paths: ProjectPaths) -> Phase6Artifacts:
        dataset = self._loader.load(project_paths.dataset_path)
        deduplicated_dataset = deduplicate_dataset(dataset)
        role_matrix = build_feature_role_matrix()
        validate_required_columns(
            deduplicated_dataset,
            role_matrix.base_input_features + (role_matrix.target_column,),
        )

        artifacts = self._ensure_phase6_directories(project_paths.phase_6_dir)
        phase5_context = self._load_json_context(project_paths.phase_5_dir / "imbalance_summary.json")
        self._logger.info("Faz 6 ileri modelleme baslatildi: %s", project_paths.dataset_path)

        # Ayrik validation bolmesi tuning ve varyant secimini train tarafinda tutar; outer test yalnizca son kontrol icindir.
        outer_train, outer_test = split_dataset_stratified(
            deduplicated_dataset,
            role_matrix.target_column,
            test_size=self._test_size,
            random_state=self._random_state,
        )
        validation_train, validation_holdout = split_dataset_stratified(
            outer_train,
            role_matrix.target_column,
            test_size=self._validation_size,
            random_state=self._random_state,
        )

        outcomes = self._run_candidate_suite(
            validation_train=validation_train,
            validation_holdout=validation_holdout,
            outer_train=outer_train,
            outer_test=outer_test,
            role_matrix=role_matrix,
        )
        validation_results = {family_key: outcome.validation_result for family_key, outcome in outcomes.items()}
        holdout_results = {family_key: outcome.holdout_result for family_key, outcome in outcomes.items() if outcome.holdout_result}
        champion_decision = choose_champion_and_challenger(validation_results)

        focus_keys = [champion_decision["champion_key"], champion_decision["challenger_key"]]
        stability_summary = {
            family_key: self._evaluate_stability(outer_train, role_matrix, outcomes[family_key]) for family_key in focus_keys
        }
        segment_summary = {
            family_key: summarize_segment_performance(outer_test, outcomes[family_key].holdout_result, min_slice_size=20)
            for family_key in focus_keys
            if outcomes[family_key].holdout_result is not None
        }

        summary = self._build_summary(
            project_paths=project_paths,
            phase5_context=phase5_context,
            deduplicated_row_count=len(deduplicated_dataset),
            outer_train=outer_train,
            outer_test=outer_test,
            validation_train=validation_train,
            validation_holdout=validation_holdout,
            outcomes=outcomes,
            champion_decision=champion_decision,
            stability_summary=stability_summary,
            segment_summary=segment_summary,
        )

        self._write_model_artifacts(artifacts.models_dir, outcomes)
        self._write_plots(
            plots_dir=artifacts.plots_dir,
            outcomes=outcomes,
            holdout_target=outer_test[role_matrix.target_column],
            champion_decision=champion_decision,
            stability_summary=stability_summary,
        )
        self._write_json_summary(artifacts.json_summary_path, summary)
        self._write_markdown_report(artifacts.markdown_report_path, summary)
        self._logger.info("Faz 6 ileri modelleme tamamlandi.")
        return artifacts

    def _ensure_phase6_directories(self, phase_6_dir: Path) -> Phase6Artifacts:
        plots_dir = phase_6_dir / "plots"
        models_dir = phase_6_dir / "models"
        for directory in (phase_6_dir, plots_dir, models_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return Phase6Artifacts(
            markdown_report_path=phase_6_dir / "advanced_modeling_report.md",
            json_summary_path=phase_6_dir / "advanced_modeling_summary.json",
            plots_dir=plots_dir,
            models_dir=models_dir,
        )

    def _run_candidate_suite(
        self,
        *,
        validation_train: pd.DataFrame,
        validation_holdout: pd.DataFrame,
        outer_train: pd.DataFrame,
        outer_test: pd.DataFrame,
        role_matrix: FeatureRoleMatrix,
    ) -> dict[str, ModelFamilyOutcome]:
        outcomes: dict[str, ModelFamilyOutcome] = {}
        for family_key in MODEL_FAMILY_LABELS:
            self._logger.info("Faz 6 tuning basladi: %s", family_key)
            outcome = self._tune_family(
                family_key=family_key,
                validation_train=validation_train,
                validation_holdout=validation_holdout,
                role_matrix=role_matrix,
            )
            holdout_result, holdout_model = self._refit_selected_variant(
                family_key=family_key,
                outcome=outcome,
                train_frame=outer_train,
                evaluation_frame=outer_test,
                role_matrix=role_matrix,
            )
            outcome.holdout_result = holdout_result
            outcome.holdout_model = holdout_model
            outcomes[family_key] = outcome
        return outcomes

    def _tune_family(
        self,
        *,
        family_key: str,
        validation_train: pd.DataFrame,
        validation_holdout: pd.DataFrame,
        role_matrix: FeatureRoleMatrix,
    ) -> ModelFamilyOutcome:
        X_train = validation_train.loc[:, role_matrix.base_input_features]
        y_train = validation_train[role_matrix.target_column]
        X_eval = validation_holdout.loc[:, role_matrix.base_input_features]
        y_eval = validation_holdout[role_matrix.target_column]

        best_params: dict[str, Any] | None = None
        best_result: BaselineModelResult | None = None
        tuning_history: list[dict[str, Any]] = []

        # Dar grid, Faz 6'yi hiperparametre patlamasina cevirmeden savunulabilir karsilastirma yapmamizi saglar.
        for index, params in enumerate(self._candidate_grid(family_key), start=1):
            try:
                result, _ = self._fit_estimator(
                    family_key=family_key,
                    params=params,
                    calibrated=False,
                    X_train=X_train,
                    y_train=y_train,
                    X_eval=X_eval,
                    y_eval=y_eval,
                    role_matrix=role_matrix,
                )
                tuning_history.append(
                    {
                        "candidate_id": f"{family_key}_{index}",
                        "params": params,
                        "metrics": result.metrics,
                        "calibration_gap": result.calibration.get("mean_absolute_gap"),
                    }
                )
                if best_result is None or rank_model_result(result) > rank_model_result(best_result):
                    best_result = result
                    best_params = params
            except DataAnalysisError as error:
                self._logger.warning("Faz 6 tuning adayi basarisiz oldu | %s | %s", family_key, error)
                tuning_history.append(
                    {
                        "candidate_id": f"{family_key}_{index}",
                        "params": params,
                        "error": str(error),
                    }
                )

        if best_params is None or best_result is None:
            raise DataAnalysisError(f"Faz 6 {family_key} ailesi icin gecerli tuning adayi bulunamadi.")

        base_validation_result, _ = self._fit_estimator(
            family_key=family_key,
            params=best_params,
            calibrated=False,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            role_matrix=role_matrix,
        )

        validation_variants = {"base": base_validation_result}
        variant_decision: dict[str, str]
        try:
            calibrated_validation_result, _ = self._fit_estimator(
                family_key=family_key,
                params=best_params,
                calibrated=True,
                X_train=X_train,
                y_train=y_train,
                X_eval=X_eval,
                y_eval=y_eval,
                role_matrix=role_matrix,
            )
            validation_variants["sigmoid_calibrated"] = calibrated_validation_result
            variant_decision = choose_model_variant(base_validation_result, calibrated_validation_result)
        except DataAnalysisError as error:
            self._logger.warning("Faz 6 kalibrasyon varyanti basarisiz oldu | %s | %s", family_key, error)
            variant_decision = {
                "selected_variant": "base",
                "rationale": f"Sigmoid calibration denemesi basarisiz oldugu icin baz varyant korundu: {error}",
            }

        selected_variant = variant_decision["selected_variant"]
        return ModelFamilyOutcome(
            family_key=family_key,
            family_label=MODEL_FAMILY_LABELS[family_key],
            best_params=best_params,
            tuning_history=tuning_history,
            selected_variant=selected_variant,
            variant_rationale=variant_decision["rationale"],
            validation_variants=validation_variants,
            validation_result=validation_variants[selected_variant],
        )

    def _refit_selected_variant(
        self,
        *,
        family_key: str,
        outcome: ModelFamilyOutcome,
        train_frame: pd.DataFrame,
        evaluation_frame: pd.DataFrame,
        role_matrix: FeatureRoleMatrix,
    ) -> tuple[BaselineModelResult, Any]:
        X_train = train_frame.loc[:, role_matrix.base_input_features]
        y_train = train_frame[role_matrix.target_column]
        X_eval = evaluation_frame.loc[:, role_matrix.base_input_features]
        y_eval = evaluation_frame[role_matrix.target_column]

        calibrated = outcome.selected_variant == "sigmoid_calibrated"
        return self._fit_estimator(
            family_key=family_key,
            params=outcome.best_params,
            calibrated=calibrated,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            role_matrix=role_matrix,
        )

    def _fit_estimator(
        self,
        *,
        family_key: str,
        params: dict[str, Any],
        calibrated: bool,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        role_matrix: FeatureRoleMatrix,
    ) -> tuple[BaselineModelResult, Any]:
        estimator = self._build_family_estimator(
            family_key=family_key,
            params=params,
            calibrated=calibrated,
            role_matrix=role_matrix,
        )

        try:
            estimator.fit(X_train, y_train)
            probabilities = estimator.predict_proba(X_eval)[:, 1]
            predictions = estimator.predict(X_eval)
        except (TypeError, ValueError) as error:
            raise DataAnalysisError(f"Faz 6 modeli egitilemedi: {family_key} | {error}") from error

        result = evaluate_binary_classifier(
            name=self._result_name(family_key, calibrated),
            family=family_key,
            y_true=y_eval,
            predictions=predictions,
            probabilities=probabilities,
            note=self._build_result_note(family_key, calibrated),
        )
        return result, estimator

    def _build_family_estimator(
        self,
        *,
        family_key: str,
        params: dict[str, Any],
        calibrated: bool,
        role_matrix: FeatureRoleMatrix,
    ) -> Any:
        if family_key == "random_forest":
            base_estimator = Pipeline(
                steps=[
                    ("preprocessing", build_tree_preprocessing_pipeline(role_matrix)),
                    (
                        "model",
                        RandomForestClassifier(
                            random_state=self._random_state,
                            n_jobs=-1,
                            **params,
                        ),
                    ),
                ]
            )
        elif family_key == "gradient_boosting":
            # Boosting tarafinda kategorik alanlari sahte ordinal etkiye hapsetmemek icin one-hot temsil korunur.
            base_estimator = Pipeline(
                steps=[
                    ("preprocessing", build_linear_preprocessing_pipeline(role_matrix)),
                    (
                        "model",
                        GradientBoostingClassifier(
                            random_state=self._random_state,
                            **params,
                        ),
                    ),
                ]
            )
        elif family_key == "hist_gradient_boosting":
            base_estimator = Pipeline(
                steps=[
                    ("preprocessing", build_linear_preprocessing_pipeline(role_matrix)),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            random_state=self._random_state,
                            early_stopping=False,
                            **params,
                        ),
                    ),
                ]
            )
        else:
            raise DataAnalysisError(f"Desteklenmeyen Faz 6 model ailesi: {family_key}")

        if not calibrated:
            return base_estimator

        return CalibratedClassifierCV(estimator=base_estimator, method="sigmoid", cv=3)

    def _candidate_grid(self, family_key: str) -> list[dict[str, Any]]:
        if family_key == "random_forest":
            return [
                {"n_estimators": 240, "max_depth": 8, "min_samples_leaf": 4, "max_features": "sqrt"},
                {"n_estimators": 320, "max_depth": 10, "min_samples_leaf": 6, "max_features": "sqrt"},
                {"n_estimators": 280, "max_depth": None, "min_samples_leaf": 8, "max_features": 0.8},
            ]
        if family_key == "gradient_boosting":
            return [
                {"n_estimators": 140, "learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 12, "subsample": 0.9},
                {"n_estimators": 180, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 8, "subsample": 0.85},
                {"n_estimators": 120, "learning_rate": 0.08, "max_depth": 2, "min_samples_leaf": 10, "subsample": 0.9},
            ]
        if family_key == "hist_gradient_boosting":
            return [
                {"max_iter": 160, "learning_rate": 0.05, "max_depth": 6, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0},
                {"max_iter": 220, "learning_rate": 0.05, "max_depth": 8, "max_leaf_nodes": 63, "min_samples_leaf": 15, "l2_regularization": 0.1},
                {"max_iter": 180, "learning_rate": 0.08, "max_depth": 6, "max_leaf_nodes": 31, "min_samples_leaf": 10, "l2_regularization": 0.2},
            ]
        raise DataAnalysisError(f"Faz 6 icin grid tanimi bulunamadi: {family_key}")

    def _result_name(self, family_key: str, calibrated: bool) -> str:
        return f"{family_key}_{'sigmoid_calibrated' if calibrated else 'base'}"

    def _build_result_note(self, family_key: str, calibrated: bool) -> str:
        base_notes = {
            "random_forest": "Bootstrap topluluk mantigi ile daha dayanikli agac ailesi elde edilmesi hedeflendi.",
            "gradient_boosting": "Kucuk ama yonlu boosting adimlari ile zayif kaliplari daha hassas yakalamak hedeflendi.",
            "hist_gradient_boosting": "Sklearn'in modern histogram tabanli boosting uygulamasi ile daha hizli ve daha duzenli bir ensemble deneyi kuruldu.",
        }
        if calibrated:
            return base_notes[family_key] + " Sigmoid calibration, politika katmanina daha guvenilir olasilik tasimak icin eklendi."
        return base_notes[family_key]

    def _evaluate_stability(
        self,
        train_frame: pd.DataFrame,
        role_matrix: FeatureRoleMatrix,
        outcome: ModelFamilyOutcome,
    ) -> dict[str, Any]:
        X_train = train_frame.loc[:, role_matrix.base_input_features]
        y_train = train_frame[role_matrix.target_column]
        estimator = self._build_family_estimator(
            family_key=outcome.family_key,
            params=outcome.best_params,
            calibrated=outcome.selected_variant == "sigmoid_calibrated",
            role_matrix=role_matrix,
        )

        scoring = {
            "pr_auc": "average_precision",
            "balanced_accuracy": "balanced_accuracy",
            "roc_auc": "roc_auc",
        }
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=self._random_state)

        try:
            cv_results = cross_validate(
                estimator,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
            )
        except ValueError as error:
            self._logger.warning("Faz 6 stabilite analizi basarisiz oldu | %s | %s", outcome.family_key, error)
            return {
                "status": "failed",
                "error": str(error),
            }

        return {
            "status": "completed",
            "cv_folds": 4,
            **summarize_cross_validation(cv_results),
        }

    def _build_summary(
        self,
        *,
        project_paths: ProjectPaths,
        phase5_context: dict[str, Any] | None,
        deduplicated_row_count: int,
        outer_train: pd.DataFrame,
        outer_test: pd.DataFrame,
        validation_train: pd.DataFrame,
        validation_holdout: pd.DataFrame,
        outcomes: dict[str, ModelFamilyOutcome],
        champion_decision: dict[str, Any],
        stability_summary: dict[str, Any],
        segment_summary: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "dataset_path": str(project_paths.dataset_path),
            "deduplicated_row_count": deduplicated_row_count,
            "library_strategy": {
                "selected_stack": "sklearn_native_ensembles",
                "rationale": "XGBoost/CatBoost gibi ek bagimliliklar ortamda bulunmadigi icin Faz 6, mevcut dependency zincirini bozmayan sklearn tabanli ensemble ailesiyle tasarlandi.",
            },
            "split_summary": {
                "outer_train_row_count": len(outer_train),
                "outer_test_row_count": len(outer_test),
                "validation_train_row_count": len(validation_train),
                "validation_holdout_row_count": len(validation_holdout),
                "outer_train_target_distribution": self._target_distribution(outer_train),
                "outer_test_target_distribution": self._target_distribution(outer_test),
                "validation_holdout_target_distribution": self._target_distribution(validation_holdout),
                "test_size": self._test_size,
                "validation_size": self._validation_size,
                "random_state": self._random_state,
            },
            "phase5_context": {
                "recommended_strategy": phase5_context.get("recommended_strategy") if phase5_context else None,
                "selected_threshold": phase5_context.get("threshold_strategy", {})
                .get("validation_selection", {})
                .get("selected_threshold") if phase5_context else None,
            },
            "candidate_families": {
                family_key: {
                    "family_label": outcome.family_label,
                    "best_params": outcome.best_params,
                    "selected_variant": outcome.selected_variant,
                    "variant_rationale": outcome.variant_rationale,
                    "validation_variants": {
                        variant_name: self._result_summary(result)
                        for variant_name, result in outcome.validation_variants.items()
                    },
                    "selected_validation_result": self._result_summary(outcome.validation_result),
                    "holdout_result": self._result_summary(outcome.holdout_result),
                    "tuning_history": outcome.tuning_history,
                }
                for family_key, outcome in outcomes.items()
            },
            "champion_selection": champion_decision,
            "stability_analysis": stability_summary,
            "segment_performance": segment_summary,
            "phase6_closeout": {
                "summary": "Faz 6'da kontrollu tuning ile uc ensemble ailesi validation tarafinda yaristirildi, her aile icin kalibrasyon varyanti kontrol edildi ve champion/challenger ikilisi holdout uzerinde sabitlendi.",
                "next_step": "Faz 7'de champion ve challenger icin SHAP, yerel aciklama ve fairness-lite analizi uretilecektir.",
            },
        }

    def _target_distribution(self, frame: pd.DataFrame) -> dict[str, float]:
        counts = frame[TARGET_COLUMN].value_counts().sort_index().to_dict()
        return {str(key): round(float(value) / len(frame), 6) for key, value in counts.items()}

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

    def _write_model_artifacts(self, models_dir: Path, outcomes: dict[str, ModelFamilyOutcome]) -> None:
        try:
            for outcome in outcomes.values():
                if outcome.holdout_model is None:
                    continue
                joblib.dump(outcome.holdout_model, models_dir / f"{outcome.validation_result.name}.joblib")
        except OSError as error:
            raise ArtifactWriteError(f"Faz 6 model artefaktlari yazilamadi: {models_dir}") from error

    def _write_plots(
        self,
        *,
        plots_dir: Path,
        outcomes: dict[str, ModelFamilyOutcome],
        holdout_target: pd.Series,
        champion_decision: dict[str, Any],
        stability_summary: dict[str, Any],
    ) -> None:
        self._plot_family_performance(plots_dir / "family_performance.png", outcomes)
        self._plot_holdout_pr_curves(
            plots_dir / "champion_vs_challenger_pr_curves.png",
            outcomes,
            holdout_target,
            champion_decision,
        )
        self._plot_stability_summary(plots_dir / "cv_stability_summary.png", stability_summary, champion_decision)

    def _plot_family_performance(self, output_path: Path, outcomes: dict[str, ModelFamilyOutcome]) -> None:
        labels = [f"{outcome.family_label}\n({outcome.selected_variant})" for outcome in outcomes.values()]
        validation_pr_auc = [outcome.validation_result.metrics["pr_auc"] or 0.0 for outcome in outcomes.values()]
        holdout_pr_auc = [outcome.holdout_result.metrics["pr_auc"] if outcome.holdout_result else 0.0 for outcome in outcomes.values()]
        validation_balanced_accuracy = [outcome.validation_result.metrics["balanced_accuracy"] for outcome in outcomes.values()]
        holdout_balanced_accuracy = [
            outcome.holdout_result.metrics["balanced_accuracy"] if outcome.holdout_result else 0.0 for outcome in outcomes.values()
        ]
        x_positions = np.arange(len(labels))
        width = 0.36

        figure, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].bar(x_positions - (width / 2), validation_pr_auc, width=width, label="validation")
        axes[0].bar(x_positions + (width / 2), holdout_pr_auc, width=width, label="holdout")
        axes[0].set_title("Faz 6 PR-AUC Karsilastirmasi")
        axes[0].set_xticks(x_positions, labels, rotation=20, ha="right")
        axes[0].set_ylim(0.0, 1.05)
        axes[0].legend()

        axes[1].bar(x_positions - (width / 2), validation_balanced_accuracy, width=width, label="validation")
        axes[1].bar(x_positions + (width / 2), holdout_balanced_accuracy, width=width, label="holdout")
        axes[1].set_title("Faz 6 Balanced Accuracy Karsilastirmasi")
        axes[1].set_xticks(x_positions, labels, rotation=20, ha="right")
        axes[1].set_ylim(0.0, 1.05)
        axes[1].legend()

        figure.tight_layout()
        self._save_figure(figure, output_path)

    def _plot_holdout_pr_curves(
        self,
        output_path: Path,
        outcomes: dict[str, ModelFamilyOutcome],
        holdout_target: pd.Series,
        champion_decision: dict[str, Any],
    ) -> None:
        figure, axis = plt.subplots(figsize=(9, 7))
        for family_key in (champion_decision["champion_key"], champion_decision["challenger_key"]):
            outcome = outcomes[family_key]
            if outcome.holdout_result is None:
                continue
            precision_values, recall_values, _ = precision_recall_curve(
                np.asarray(holdout_target, dtype=int),
                np.asarray(outcome.holdout_result.probabilities, dtype=float),
            )
            axis.plot(
                recall_values,
                precision_values,
                label=f"{outcome.family_label} | AP={outcome.holdout_result.metrics['pr_auc']}",
            )

        axis.set_title("Faz 6 Champion ve Challenger Precision-Recall Egrileri")
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.05)
        axis.legend()
        figure.tight_layout()
        self._save_figure(figure, output_path)

    def _plot_stability_summary(
        self,
        output_path: Path,
        stability_summary: dict[str, Any],
        champion_decision: dict[str, Any],
    ) -> None:
        figure, axis = plt.subplots(figsize=(9, 6))
        labels: list[str] = []
        means: list[float] = []
        stds: list[float] = []

        for family_key in (champion_decision["champion_key"], champion_decision["challenger_key"]):
            summary = stability_summary.get(family_key, {})
            if summary.get("status") != "completed":
                continue
            metric_summary = summary["metrics"].get("pr_auc")
            if metric_summary is None:
                continue
            labels.append(family_key)
            means.append(metric_summary["mean"])
            stds.append(metric_summary["std"])

        if labels:
            x_positions = np.arange(len(labels))
            axis.bar(x_positions, means, yerr=stds, capsize=6)
            axis.set_xticks(x_positions, labels)
            axis.set_ylim(0.0, 1.05)
            axis.set_ylabel("CV PR-AUC")
            axis.set_title("Faz 6 CV Stabilite Ozeti")
        else:
            axis.text(0.5, 0.5, "Stabilite ozetine uygun veri olusmadi.", ha="center", va="center")
            axis.set_axis_off()

        figure.tight_layout()
        self._save_figure(figure, output_path)

    def _save_figure(self, figure: plt.Figure, output_path: Path) -> None:
        try:
            figure.savefig(output_path, dpi=200, bbox_inches="tight")
        except OSError as error:
            raise ArtifactWriteError(f"Faz 6 grafigi kaydedilemedi: {output_path}") from error
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
            raise ArtifactWriteError(f"Faz 6 JSON ozeti yazilamadi: {output_path}") from error

    def _write_markdown_report(self, output_path: Path, summary: dict[str, Any]) -> None:
        lines = [
            "# Faz 6 Ileri Modelleme ve Hiperparametre Optimizasyonu Raporu",
            "",
            f"- Veri seti: `{summary['dataset_path']}`",
            f"- Tekillestirilmis satir sayisi: {summary['deduplicated_row_count']}",
            f"- Outer train/test: {summary['split_summary']['outer_train_row_count']} / {summary['split_summary']['outer_test_row_count']}",
            f"- Validation train/holdout: {summary['split_summary']['validation_train_row_count']} / {summary['split_summary']['validation_holdout_row_count']}",
            "",
            "## Faz 5 Baglami",
            "",
            f"- Onerilen strateji: {summary['phase5_context']['recommended_strategy']}",
            f"- Referans esik: {summary['phase5_context']['selected_threshold']}",
            f"- Kutuphane karari: {summary['library_strategy']['rationale']}",
            "",
            "## Aile Sonuclari",
            "",
        ]

        for family_key, family_summary in summary["candidate_families"].items():
            lines.append(f"### {family_key}")
            lines.append(f"- Etiket: {family_summary['family_label']}")
            lines.append(f"- En iyi parametreler: {family_summary['best_params']}")
            lines.append(f"- Secilen varyant: {family_summary['selected_variant']}")
            lines.append(f"- Varyant gerekcesi: {family_summary['variant_rationale']}")
            lines.append(f"- Validation sonuc: {family_summary['selected_validation_result']}")
            lines.append(f"- Holdout sonuc: {family_summary['holdout_result']}")
            lines.append("")

        lines.extend(
            [
                "## Champion ve Challenger",
                "",
                f"- Champion: {summary['champion_selection']['champion_model']}",
                f"- Challenger: {summary['champion_selection']['challenger_model']}",
                f"- Gerekce: {summary['champion_selection']['rationale']}",
                f"- Sirali adaylar: {summary['champion_selection']['ordered_candidates']}",
                "",
                "## Stabilite Analizi",
                "",
            ]
        )

        for family_key, stability in summary["stability_analysis"].items():
            lines.append(f"### {family_key}")
            lines.append(f"- Ozet: {stability}")
            lines.append("")

        lines.extend(["## Segment Performansi", ""])
        for family_key, segment_details in summary["segment_performance"].items():
            lines.append(f"### {family_key}")
            lines.append(f"- Segmentler: {segment_details}")
            lines.append("")

        lines.extend(
            [
                "## Faz 6 Kapanis",
                "",
                f"- Ozet: {summary['phase6_closeout']['summary']}",
                f"- Sonraki adim: {summary['phase6_closeout']['next_step']}",
            ]
        )

        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                output_file.write("\n".join(lines))
        except OSError as error:
            raise ArtifactWriteError(f"Faz 6 markdown raporu yazilamadi: {output_path}") from error