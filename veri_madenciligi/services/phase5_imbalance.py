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
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
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


@dataclass(frozen=True)
class Phase5Artifacts:
    markdown_report_path: Path
    json_summary_path: Path
    plots_dir: Path
    models_dir: Path


def profile_imbalance(y: pd.Series) -> dict[str, Any]:
    counts = y.value_counts().sort_index().to_dict()
    minority_label = min(counts, key=counts.get)
    majority_label = max(counts, key=counts.get)
    minority_count = int(counts[minority_label])
    majority_count = int(counts[majority_label])
    ratio = round(minority_count / majority_count, 6)

    if ratio >= 0.75:
        severity = "mild"
        recommendation = "Sentetik veri varsayilan secim olmamali; once reference ve class weight stratejileri denenmelidir."
    elif ratio >= 0.5:
        severity = "moderate"
        recommendation = "Class weight ana adaydir; SMOTE yalnizca kontrollu karsilastirma olarak devreye alinmalidir."
    else:
        severity = "severe"
        recommendation = "Class weight ve sentetik yeniden ornekleme birlikte degerlendirilmelidir."

    return {
        "counts": {str(key): int(value) for key, value in counts.items()},
        "minority_label": str(minority_label),
        "majority_label": str(majority_label),
        "minority_count": minority_count,
        "majority_count": majority_count,
        "minority_to_majority_ratio": ratio,
        "severity": severity,
        "recommendation": recommendation,
    }


def compute_threshold_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true_array = np.asarray(y_true, dtype=int)
    probability_array = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    if len(y_true_array) != len(probability_array):
        raise DataAnalysisError("Esik metrikleri icin hedef ve olasilik uzunluklari esit olmalidir.")
    predicted_labels = (probability_array >= threshold).astype(int)

    true_negative = int(((predicted_labels == 0) & (y_true_array == 0)).sum())
    false_positive = int(((predicted_labels == 1) & (y_true_array == 0)).sum())
    false_negative = int(((predicted_labels == 0) & (y_true_array == 1)).sum())
    true_positive = int(((predicted_labels == 1) & (y_true_array == 1)).sum())

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    specificity_denominator = true_negative + false_positive

    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    specificity = true_negative / specificity_denominator if specificity_denominator else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    balanced_accuracy = (recall + specificity) / 2.0

    return {
        "threshold": round(float(threshold), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "balanced_accuracy": round(float(balanced_accuracy), 6),
        "predicted_positive_rate": round(float(predicted_labels.mean()), 6),
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
    }


def build_decision_bands(center_threshold: float) -> dict[str, float]:
    low_threshold = round(max(0.2, center_threshold - 0.15), 6)
    high_threshold = round(min(0.85, center_threshold + 0.15), 6)

    if low_threshold >= center_threshold:
        low_threshold = round(max(0.1, center_threshold - 0.1), 6)
    if high_threshold <= center_threshold:
        high_threshold = round(min(0.9, center_threshold + 0.1), 6)

    return {
        "low_action_threshold": low_threshold,
        "binary_decision_threshold": round(float(center_threshold), 6),
        "high_confidence_no_discount_threshold": high_threshold,
    }


def choose_optimal_threshold(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        raise DataAnalysisError("Esik secimi icin bos hedef dizisi kullanilamaz.")

    candidate_thresholds = np.round(np.linspace(0.2, 0.8, 31), 3)
    threshold_rows = [compute_threshold_metrics(y_true, probabilities, threshold) for threshold in candidate_thresholds]

    feasible_rows = [row for row in threshold_rows if row["precision"] >= 0.55]
    candidate_rows = feasible_rows if feasible_rows else threshold_rows
    best_row = max(
        candidate_rows,
        key=lambda row: (row["f1"], row["recall"], row["precision"], -abs(row["threshold"] - 0.5)),
    )

    return {
        "selected_threshold": best_row["threshold"],
        "selection_logic": "Precision en az 0.55 ise F1 maksimize edildi; aksi durumda tum esikler icinde en yuksek F1 secildi.",
        "selected_metrics": best_row,
        "decision_bands": build_decision_bands(best_row["threshold"]),
        "evaluated_thresholds": threshold_rows,
    }


def choose_recommended_imbalance_strategy(results: dict[str, BaselineModelResult]) -> dict[str, str]:
    if not results:
        raise DataAnalysisError("Faz 5 tavsiye karari icin sonuc bulunamadi.")

    sorted_candidates = sorted(
        results.values(),
        key=lambda result: (
            result.metrics["pr_auc"] if result.metrics["pr_auc"] is not None else -1.0,
            result.metrics["recall"],
            result.metrics["balanced_accuracy"],
            result.metrics["precision"],
        ),
        reverse=True,
    )
    strongest = sorted_candidates[0]
    reference = results.get("logistic_reference")
    weighted_logistic = results.get("logistic_class_weight")

    if reference is not None and strongest.name != "logistic_reference":
        pr_gap = (strongest.metrics["pr_auc"] or 0.0) - (reference.metrics["pr_auc"] or 0.0)
        ba_gap = strongest.metrics["balanced_accuracy"] - reference.metrics["balanced_accuracy"]
        if pr_gap <= 0.02 and ba_gap <= 0.02:
            return {
                "recommended_strategy": "logistic_reference",
                "rationale": "Dengesizlik hafif oldugu icin sentetik veya agir class-weight stratejileri plain lojistik regresyona anlamli ustunluk kuramadi.",
            }

    if strongest.name.startswith("smote") and weighted_logistic is not None:
        pr_gap = (strongest.metrics["pr_auc"] or 0.0) - (weighted_logistic.metrics["pr_auc"] or 0.0)
        ba_gap = strongest.metrics["balanced_accuracy"] - weighted_logistic.metrics["balanced_accuracy"]
        if pr_gap <= 0.015 and ba_gap <= 0.02:
            return {
                "recommended_strategy": "logistic_class_weight",
                "rationale": "SMOTE tabanli strateji az farkla onde olsa da class-weight lojistik daha sade, daha savunulabilir ve sentetik veri riskini azaltan secenektir.",
            }

    return {
        "recommended_strategy": strongest.name,
        "rationale": "Secilen strateji validation asamasinda PR-AUC, recall ve balanced accuracy dengesinde en guclu profili urettigi icin tercih edildi.",
    }


class Phase5ImbalanceService:
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

    def run(self, project_paths: ProjectPaths) -> Phase5Artifacts:
        dataset = self._loader.load(project_paths.dataset_path)
        deduplicated_dataset = deduplicate_dataset(dataset)
        role_matrix = build_feature_role_matrix()
        validate_required_columns(
            deduplicated_dataset,
            role_matrix.base_input_features + (role_matrix.target_column,),
        )

        artifacts = self._ensure_phase5_directories(project_paths.phase_5_dir)
        self._logger.info("Faz 5 sinif dengesizligi ve esik stratejisi baslatildi: %s", project_paths.dataset_path)

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

        imbalance_profile = profile_imbalance(outer_train[role_matrix.target_column])
        validation_results, _validation_bundles = self._run_strategy_suite(
            train_frame=validation_train,
            evaluation_frame=validation_holdout,
            role_matrix=role_matrix,
            fit_context="validation",
        )
        holdout_results, holdout_bundles = self._run_strategy_suite(
            train_frame=outer_train,
            evaluation_frame=outer_test,
            role_matrix=role_matrix,
            fit_context="holdout",
        )

        recommendation = choose_recommended_imbalance_strategy(validation_results)
        selected_strategy_name = recommendation["recommended_strategy"]
        threshold_summary = self._build_threshold_summary(
            selected_strategy_name=selected_strategy_name,
            validation_result=validation_results[selected_strategy_name],
            holdout_result=holdout_results[selected_strategy_name],
            y_validation=validation_holdout[role_matrix.target_column],
            y_holdout=outer_test[role_matrix.target_column],
        )

        summary = self._build_summary(
            project_paths=project_paths,
            deduplicated_row_count=len(deduplicated_dataset),
            outer_train=outer_train,
            outer_test=outer_test,
            validation_train=validation_train,
            validation_holdout=validation_holdout,
            imbalance_profile=imbalance_profile,
            validation_results=validation_results,
            holdout_results=holdout_results,
            recommendation=recommendation,
            threshold_summary=threshold_summary,
        )

        self._write_model_artifacts(artifacts.models_dir, holdout_bundles)
        self._write_plots(
            plots_dir=artifacts.plots_dir,
            validation_results=validation_results,
            holdout_results=holdout_results,
            selected_strategy_name=selected_strategy_name,
            selected_validation_probabilities=np.asarray(validation_results[selected_strategy_name].probabilities),
            validation_target=validation_holdout[role_matrix.target_column],
            holdout_target=outer_test[role_matrix.target_column],
            threshold_summary=threshold_summary,
        )
        self._write_json_summary(artifacts.json_summary_path, summary)
        self._write_markdown_report(artifacts.markdown_report_path, summary)
        self._logger.info("Faz 5 sinif dengesizligi ve esik stratejisi tamamlandi.")
        return artifacts

    def _ensure_phase5_directories(self, phase_5_dir: Path) -> Phase5Artifacts:
        plots_dir = phase_5_dir / "plots"
        models_dir = phase_5_dir / "models"
        for directory in (phase_5_dir, plots_dir, models_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return Phase5Artifacts(
            markdown_report_path=phase_5_dir / "imbalance_report.md",
            json_summary_path=phase_5_dir / "imbalance_summary.json",
            plots_dir=plots_dir,
            models_dir=models_dir,
        )

    def _run_strategy_suite(
        self,
        *,
        train_frame: pd.DataFrame,
        evaluation_frame: pd.DataFrame,
        role_matrix: FeatureRoleMatrix,
        fit_context: str,
    ) -> tuple[dict[str, BaselineModelResult], dict[str, Any]]:
        X_train = train_frame.loc[:, role_matrix.base_input_features]
        y_train = train_frame[role_matrix.target_column]
        X_eval = evaluation_frame.loc[:, role_matrix.base_input_features]
        y_eval = evaluation_frame[role_matrix.target_column]

        results: dict[str, BaselineModelResult] = {}
        bundles: dict[str, Any] = {}

        logistic_reference = self._fit_logistic_strategy(
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            role_matrix=role_matrix,
            class_weight=None,
            name="logistic_reference",
            note="Dengesizlik mudahalesi olmadan plain lojistik referans olarak tutuldu.",
        )
        results["logistic_reference"], bundles["logistic_reference"] = logistic_reference

        logistic_class_weight = self._fit_logistic_strategy(
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            role_matrix=role_matrix,
            class_weight="balanced",
            name="logistic_class_weight",
            note="Class weight ile precision-recall dengesi iyilesiyor mu sorusunu test eder.",
        )
        results["logistic_class_weight"], bundles["logistic_class_weight"] = logistic_class_weight

        random_forest_class_weight = self._fit_random_forest_strategy(
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            role_matrix=role_matrix,
            class_weight="balanced_subsample",
            name="random_forest_class_weight",
            note="Agaç tabanli modelde class weight etkisini gormek icin kontrollu RF deneyi kuruldu.",
        )
        results["random_forest_class_weight"], bundles["random_forest_class_weight"] = random_forest_class_weight

        smote_logistic = self._fit_smote_logistic_strategy(
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            role_matrix=role_matrix,
            name="smote_logistic",
            note="One-hot sonrasi sayisal uzayda SMOTE uygulanan kontrollu lojistik deneyi.",
        )
        results["smote_logistic"], bundles["smote_logistic"] = smote_logistic

        smotenc_random_forest = self._fit_smotenc_random_forest_strategy(
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            role_matrix=role_matrix,
            name="smotenc_random_forest",
            note="Kategorik yapiyi korumak icin SMOTENC ile RF birlikte test edildi.",
        )
        results["smotenc_random_forest"], bundles["smotenc_random_forest"] = smotenc_random_forest

        self._logger.info("Faz 5 %s suite tamamlandi.", fit_context)
        return results, bundles

    def _fit_logistic_strategy(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        role_matrix: FeatureRoleMatrix,
        class_weight: str | None,
        name: str,
        note: str,
    ) -> tuple[BaselineModelResult, Any]:
        pipeline = Pipeline(
            steps=[
                ("preprocessing", build_linear_preprocessing_pipeline(role_matrix)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=self._random_state,
                        class_weight=class_weight,
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)
        result = evaluate_binary_classifier(
            name=name,
            family="linear_model",
            y_true=y_eval,
            predictions=pipeline.predict(X_eval),
            probabilities=pipeline.predict_proba(X_eval)[:, 1],
            note=note,
        )
        return result, pipeline

    def _fit_random_forest_strategy(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        role_matrix: FeatureRoleMatrix,
        class_weight: str | None,
        name: str,
        note: str,
    ) -> tuple[BaselineModelResult, Any]:
        pipeline = Pipeline(
            steps=[
                ("preprocessing", build_tree_preprocessing_pipeline(role_matrix)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=240,
                        max_depth=8,
                        min_samples_leaf=max(6, int(len(X_train) * 0.01)),
                        class_weight=class_weight,
                        random_state=self._random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)
        result = evaluate_binary_classifier(
            name=name,
            family="tree_model",
            y_true=y_eval,
            predictions=pipeline.predict(X_eval),
            probabilities=pipeline.predict_proba(X_eval)[:, 1],
            note=note,
        )
        return result, pipeline

    def _fit_smote_logistic_strategy(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        role_matrix: FeatureRoleMatrix,
        name: str,
        note: str,
    ) -> tuple[BaselineModelResult, Any]:
        preprocessing = build_linear_preprocessing_pipeline(role_matrix)
        X_train_processed = preprocessing.fit_transform(X_train)
        X_eval_processed = preprocessing.transform(X_eval)

        smote = SMOTE(
            random_state=self._random_state,
            k_neighbors=self._resolve_smote_neighbors(y_train),
        )
        X_resampled, y_resampled = smote.fit_resample(X_train_processed, y_train)

        model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=self._random_state)
        model.fit(X_resampled, y_resampled)

        result = evaluate_binary_classifier(
            name=name,
            family="linear_model",
            y_true=y_eval,
            predictions=model.predict(X_eval_processed),
            probabilities=model.predict_proba(X_eval_processed)[:, 1],
            note=note,
        )
        bundle = {
            "preprocessing": preprocessing,
            "sampler": smote,
            "model": model,
        }
        return result, bundle

    def _fit_smotenc_random_forest_strategy(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        role_matrix: FeatureRoleMatrix,
        name: str,
        note: str,
    ) -> tuple[BaselineModelResult, Any]:
        preprocessing = build_tree_preprocessing_pipeline(role_matrix)
        X_train_processed = preprocessing.fit_transform(X_train)
        X_eval_processed = preprocessing.transform(X_eval)

        categorical_start = len(role_matrix.tree_numeric_features)
        categorical_stop = categorical_start + len(role_matrix.tree_categorical_features)
        smotenc = SMOTENC(
            categorical_features=list(range(categorical_start, categorical_stop)),
            random_state=self._random_state,
            k_neighbors=self._resolve_smote_neighbors(y_train),
        )
        X_resampled, y_resampled = smotenc.fit_resample(X_train_processed, y_train)

        model = RandomForestClassifier(
            n_estimators=240,
            max_depth=8,
            min_samples_leaf=max(6, int(len(X_train) * 0.01)),
            random_state=self._random_state,
            n_jobs=-1,
        )
        model.fit(X_resampled, y_resampled)

        result = evaluate_binary_classifier(
            name=name,
            family="tree_model",
            y_true=y_eval,
            predictions=model.predict(X_eval_processed),
            probabilities=model.predict_proba(X_eval_processed)[:, 1],
            note=note,
        )
        bundle = {
            "preprocessing": preprocessing,
            "sampler": smotenc,
            "model": model,
        }
        return result, bundle

    def _resolve_smote_neighbors(self, y_train: pd.Series) -> int:
        class_counts = y_train.value_counts()
        minority_count = int(class_counts.min())
        if minority_count < 2:
            raise DataAnalysisError("SMOTE icin azinlik sinifta en az iki ornek bulunmalidir.")
        return max(1, min(5, minority_count - 1))

    def _build_threshold_summary(
        self,
        *,
        selected_strategy_name: str,
        validation_result: BaselineModelResult,
        holdout_result: BaselineModelResult,
        y_validation: pd.Series,
        y_holdout: pd.Series,
    ) -> dict[str, Any]:
        threshold_selection = choose_optimal_threshold(
            y_validation.reset_index(drop=True),
            np.asarray(validation_result.probabilities, dtype=float),
        )
        holdout_metrics = compute_threshold_metrics(
            y_holdout.reset_index(drop=True),
            np.asarray(holdout_result.probabilities, dtype=float),
            threshold_selection["selected_threshold"],
        )
        policy_band_summary = self._summarize_policy_bands(
            probabilities=np.asarray(holdout_result.probabilities, dtype=float),
            y_true=y_holdout.reset_index(drop=True),
            decision_bands=threshold_selection["decision_bands"],
        )
        return {
            "selected_strategy": selected_strategy_name,
            "validation_selection": threshold_selection,
            "holdout_metrics": holdout_metrics,
            "policy_band_summary": policy_band_summary,
            "decision_note": "Esik validation bolmesinde secildi; holdout yalnizca son etkiyi okumak icin kullanildi.",
        }

    def _summarize_policy_bands(
        self,
        *,
        probabilities: np.ndarray,
        y_true: pd.Series,
        decision_bands: dict[str, float],
    ) -> dict[str, Any]:
        frame = pd.DataFrame(
            {
                "probability": np.asarray(probabilities, dtype=float),
                "target": np.asarray(y_true, dtype=int),
            }
        )

        low_threshold = decision_bands["low_action_threshold"]
        high_threshold = decision_bands["high_confidence_no_discount_threshold"]

        def assign_band(probability: float) -> str:
            if probability < low_threshold:
                return "low_intent_holdout"
            if probability < high_threshold:
                return "mid_band_targeted_offer"
            return "high_confidence_no_discount"

        frame["policy_band"] = frame["probability"].map(assign_band)
        grouped = frame.groupby("policy_band", sort=False)

        return {
            band_name: {
                "count": int(group.shape[0]),
                "share": round(float(group.shape[0] / len(frame)), 6),
                "observed_purchase_rate": round(float(group["target"].mean()), 6),
                "average_probability": round(float(group["probability"].mean()), 6),
            }
            for band_name, group in grouped
        }

    def _build_summary(
        self,
        *,
        project_paths: ProjectPaths,
        deduplicated_row_count: int,
        outer_train: pd.DataFrame,
        outer_test: pd.DataFrame,
        validation_train: pd.DataFrame,
        validation_holdout: pd.DataFrame,
        imbalance_profile: dict[str, Any],
        validation_results: dict[str, BaselineModelResult],
        holdout_results: dict[str, BaselineModelResult],
        recommendation: dict[str, str],
        threshold_summary: dict[str, Any],
    ) -> dict[str, Any]:
        phase4_context = self._load_json_context(project_paths.phase_4_dir / "baseline_summary.json")

        return {
            "dataset_path": str(project_paths.dataset_path),
            "deduplicated_row_count": deduplicated_row_count,
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
            "phase4_context": {
                "recommended_baseline": phase4_context.get("recommended_baseline") if phase4_context else None,
                "model_results": phase4_context.get("model_results") if phase4_context else None,
            },
            "imbalance_profile": imbalance_profile,
            "metric_registry": {
                "pr_auc": "Sinif dengesi kusurlu oldugunda model kalitesini okumak icin ana metriktir.",
                "balanced_accuracy": "Her iki sinifi esit agirlikla yorumlar ve tek tarafa yigilmayi cezalandirir.",
                "precision": "Yanlis pozitif maliyetini kontrol eder.",
                "recall": "Gercek pozitifleri kacirmama kapasitesini olcer.",
                "brier_score": "Olasilik kalitesini ve esik guvenilirligini okumak icin izlenir.",
            },
            "validation_results": self._result_summary(validation_results),
            "holdout_results": self._result_summary(holdout_results),
            "recommended_strategy": recommendation,
            "threshold_strategy": threshold_summary,
            "phase5_closeout": {
                "summary": "Faz 5'te reference, class weight ve sentetik yeniden ornekleme stratejileri ayni leakage-safe split rejiminde karsilastirildi; ardindan secilen strateji icin operasyonel esik validasyon uzerinden kalibre edildi.",
                "next_step": "Faz 6'da champion adaylari daha genis hiperparametre arama ve capraz dogrulama ile derinlestirilebilir.",
            },
        }

    def _target_distribution(self, frame: pd.DataFrame) -> dict[str, float]:
        counts = frame[TARGET_COLUMN].value_counts().sort_index().to_dict()
        return {str(key): round(float(value) / len(frame), 6) for key, value in counts.items()}

    def _result_summary(self, results: dict[str, BaselineModelResult]) -> dict[str, Any]:
        return {
            name: {
                "family": result.family,
                "metrics": result.metrics,
                "calibration": result.calibration,
                "note": result.note,
            }
            for name, result in results.items()
        }

    def _write_model_artifacts(self, models_dir: Path, models: dict[str, Any]) -> None:
        try:
            for model_name, model_object in models.items():
                joblib.dump(model_object, models_dir / f"{model_name}.joblib")
        except OSError as error:
            raise ArtifactWriteError(f"Faz 5 model artefaktlari yazilamadi: {models_dir}") from error

    def _write_plots(
        self,
        *,
        plots_dir: Path,
        validation_results: dict[str, BaselineModelResult],
        holdout_results: dict[str, BaselineModelResult],
        selected_strategy_name: str,
        selected_validation_probabilities: np.ndarray,
        validation_target: pd.Series,
        holdout_target: pd.Series,
        threshold_summary: dict[str, Any],
    ) -> None:
        self._plot_strategy_pr_auc_comparison(
            output_path=plots_dir / "imbalance_strategy_pr_auc.png",
            validation_results=validation_results,
            holdout_results=holdout_results,
        )
        self._plot_holdout_pr_curves(
            output_path=plots_dir / "imbalance_holdout_pr_curves.png",
            holdout_results=holdout_results,
            holdout_target=holdout_target,
        )
        self._plot_threshold_tradeoff(
            output_path=plots_dir / "selected_strategy_threshold_tradeoff.png",
            selected_strategy_name=selected_strategy_name,
            validation_target=validation_target,
            validation_probabilities=selected_validation_probabilities,
            threshold_summary=threshold_summary,
        )

    def _plot_strategy_pr_auc_comparison(
        self,
        *,
        output_path: Path,
        validation_results: dict[str, BaselineModelResult],
        holdout_results: dict[str, BaselineModelResult],
    ) -> None:
        strategy_names = list(validation_results.keys())
        validation_scores = [validation_results[name].metrics["pr_auc"] or 0.0 for name in strategy_names]
        holdout_scores = [holdout_results[name].metrics["pr_auc"] or 0.0 for name in strategy_names]
        x_positions = np.arange(len(strategy_names))
        width = 0.36

        figure, axis = plt.subplots(figsize=(14, 7))
        axis.bar(x_positions - (width / 2), validation_scores, width=width, label="validation")
        axis.bar(x_positions + (width / 2), holdout_scores, width=width, label="holdout")
        axis.set_title("Faz 5 Strateji Bazli PR-AUC Karsilastirmasi")
        axis.set_ylabel("PR-AUC")
        axis.set_xticks(x_positions, strategy_names, rotation=20, ha="right")
        axis.set_ylim(0.0, 1.05)
        axis.legend()
        figure.tight_layout()
        self._save_figure(figure, output_path)

    def _plot_holdout_pr_curves(
        self,
        *,
        output_path: Path,
        holdout_results: dict[str, BaselineModelResult],
        holdout_target: pd.Series,
    ) -> None:
        figure, axis = plt.subplots(figsize=(9, 7))
        for strategy_name, result in holdout_results.items():
            precision_values, recall_values, _ = precision_recall_curve(
                np.asarray(holdout_target, dtype=int),
                np.asarray(result.probabilities, dtype=float),
            )
            axis.plot(recall_values, precision_values, label=f"{strategy_name} | AP={result.metrics['pr_auc']}")

        axis.set_title("Faz 5 Holdout Precision-Recall Egrileri")
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.05)
        axis.legend(fontsize=8)
        figure.tight_layout()
        self._save_figure(figure, output_path)

    def _plot_threshold_tradeoff(
        self,
        *,
        output_path: Path,
        selected_strategy_name: str,
        validation_target: pd.Series,
        validation_probabilities: np.ndarray,
        threshold_summary: dict[str, Any],
    ) -> None:
        threshold_rows = threshold_summary["validation_selection"]["evaluated_thresholds"]
        thresholds = [row["threshold"] for row in threshold_rows]
        precision_scores = [row["precision"] for row in threshold_rows]
        recall_scores = [row["recall"] for row in threshold_rows]
        f1_scores = [row["f1"] for row in threshold_rows]
        selected_threshold = threshold_summary["validation_selection"]["selected_threshold"]

        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(thresholds, precision_scores, label="precision")
        axis.plot(thresholds, recall_scores, label="recall")
        axis.plot(thresholds, f1_scores, label="f1")
        axis.axvline(selected_threshold, color="black", linestyle="--", label=f"secilen esik={selected_threshold}")
        axis.set_title(f"Faz 5 Esik Trade-off | {selected_strategy_name}")
        axis.set_xlabel("Esik")
        axis.set_ylabel("Skor")
        axis.set_ylim(0.0, 1.05)
        axis.legend()
        figure.tight_layout()
        self._save_figure(figure, output_path)

    def _save_figure(self, figure: plt.Figure, output_path: Path) -> None:
        try:
            figure.savefig(output_path, dpi=200, bbox_inches="tight")
        except OSError as error:
            raise ArtifactWriteError(f"Faz 5 grafigi kaydedilemedi: {output_path}") from error
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
            raise ArtifactWriteError(f"Faz 5 JSON ozeti yazilamadi: {output_path}") from error

    def _write_markdown_report(self, output_path: Path, summary: dict[str, Any]) -> None:
        lines = [
            "# Faz 5 Sinif Dengesizligi ve Esik Stratejisi Raporu",
            "",
            f"- Veri seti: `{summary['dataset_path']}`",
            f"- Tekillestirilmis satir sayisi: {summary['deduplicated_row_count']}",
            f"- Outer train/test: {summary['split_summary']['outer_train_row_count']} / {summary['split_summary']['outer_test_row_count']}",
            f"- Validation train/holdout: {summary['split_summary']['validation_train_row_count']} / {summary['split_summary']['validation_holdout_row_count']}",
            "",
            "## Faz 4 Baglami",
            "",
            f"- Onerilen baseline: {summary['phase4_context']['recommended_baseline']}",
            "",
            "## Dengesizlik Profili",
            "",
            f"- Sayimlar: {summary['imbalance_profile']['counts']}",
            f"- Oran: {summary['imbalance_profile']['minority_to_majority_ratio']}",
            f"- Siddet: {summary['imbalance_profile']['severity']}",
            f"- Not: {summary['imbalance_profile']['recommendation']}",
            "",
            "## Validation Sonuclari",
            "",
        ]

        for strategy_name, result in summary["validation_results"].items():
            lines.append(f"### {strategy_name}")
            lines.append(f"- Metrikler: {result['metrics']}")
            lines.append(f"- Kalibrasyon: {result['calibration']}")
            lines.append(f"- Not: {result['note']}")
            lines.append("")

        lines.extend(["## Holdout Sonuclari", ""])
        for strategy_name, result in summary["holdout_results"].items():
            lines.append(f"### {strategy_name}")
            lines.append(f"- Metrikler: {result['metrics']}")
            lines.append(f"- Kalibrasyon: {result['calibration']}")
            lines.append(f"- Not: {result['note']}")
            lines.append("")

        lines.extend(
            [
                "## Tavsiye Edilen Strateji",
                "",
                f"- Strateji: {summary['recommended_strategy']['recommended_strategy']}",
                f"- Gerekce: {summary['recommended_strategy']['rationale']}",
                "",
                "## Esik ve Politika Bantlari",
                "",
                f"- Secilen strateji: {summary['threshold_strategy']['selected_strategy']}",
                f"- Validation secimi: {summary['threshold_strategy']['validation_selection']['selected_metrics']}",
                f"- Bantlar: {summary['threshold_strategy']['validation_selection']['decision_bands']}",
                f"- Holdout etkisi: {summary['threshold_strategy']['holdout_metrics']}",
                f"- Politika bantlari: {summary['threshold_strategy']['policy_band_summary']}",
                f"- Not: {summary['threshold_strategy']['decision_note']}",
                "",
                "## Faz 5 Kapanis",
                "",
                f"- Ozet: {summary['phase5_closeout']['summary']}",
                f"- Sonraki adim: {summary['phase5_closeout']['next_step']}",
            ]
        )

        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                output_file.write("\n".join(lines))
        except OSError as error:
            raise ArtifactWriteError(f"Faz 5 markdown raporu yazilamadi: {output_path}") from error
