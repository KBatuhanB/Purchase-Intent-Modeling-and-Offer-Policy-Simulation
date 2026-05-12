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
from sklearn.base import ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ..config import TARGET_COLUMN, ProjectPaths
from ..core.exceptions import ArtifactWriteError, DataAnalysisError, SchemaValidationError
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


@dataclass(frozen=True)
class Phase4Artifacts:
    markdown_report_path: Path
    json_summary_path: Path
    plots_dir: Path
    models_dir: Path


@dataclass(frozen=True)
class BaselineModelResult:
    name: str
    family: str
    metrics: dict[str, Any]
    calibration: dict[str, Any]
    note: str
    probabilities: list[float] = field(repr=False)
    predictions: list[int] = field(repr=False)


class RuleBasedBaselineClassifier:
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "RuleBasedBaselineClassifier":
        frame = self._validate_input(X)

        # Esikler train verisinden ogrenilir; boylece kural bazli benchmark dahi test bilgisi gormez.
        self._purchase_threshold = float(
            pd.to_numeric(frame["NumberOfPurchases"], errors="coerce").median()
        )
        self._time_threshold = float(
            pd.to_numeric(frame["TimeSpentOnWebsite"], errors="coerce").median()
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        frame = self._validate_input(X)
        purchases = pd.to_numeric(frame["NumberOfPurchases"], errors="coerce").fillna(self._purchase_threshold)
        time_spent = pd.to_numeric(frame["TimeSpentOnWebsite"], errors="coerce").fillna(self._time_threshold)
        loyalty = pd.to_numeric(frame["LoyaltyProgram"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)

        purchase_signal = (purchases >= self._purchase_threshold).astype(float)
        time_signal = (time_spent >= self._time_threshold).astype(float)
        probability = (0.45 * loyalty) + (0.35 * purchase_signal) + (0.20 * time_signal)
        probability = np.clip(probability.to_numpy(dtype=float), 0.0, 1.0)

        return np.column_stack([1.0 - probability, probability])

    def export_config(self) -> dict[str, float]:
        return {
            "purchase_threshold": float(self._purchase_threshold),
            "time_threshold": float(self._time_threshold),
        }

    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise SchemaValidationError("Kural bazli baseline DataFrame girdi bekler.")
        validate_required_columns(
            X,
            ("NumberOfPurchases", "TimeSpentOnWebsite", "LoyaltyProgram"),
        )
        return X


def evaluate_binary_classifier(
    *,
    name: str,
    family: str,
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    note: str,
) -> BaselineModelResult:
    y_true_array = np.asarray(y_true, dtype=int)
    prediction_array = np.asarray(predictions, dtype=int)
    probability_array = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)

    roc_auc_value: float | None
    average_precision_value: float | None
    try:
        roc_auc_value = round(float(roc_auc_score(y_true_array, probability_array)), 6)
        average_precision_value = round(float(average_precision_score(y_true_array, probability_array)), 6)
    except ValueError:
        roc_auc_value = None
        average_precision_value = None

    confusion = confusion_matrix(y_true_array, prediction_array, labels=[0, 1])
    calibration = build_calibration_summary(y_true_array, probability_array)

    metrics = {
        "accuracy": round(float(accuracy_score(y_true_array, prediction_array)), 6),
        "precision": round(float(precision_score(y_true_array, prediction_array, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true_array, prediction_array, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true_array, prediction_array, zero_division=0)), 6),
        "roc_auc": roc_auc_value,
        "pr_auc": average_precision_value,
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true_array, prediction_array)), 6),
        "brier_score": round(float(brier_score_loss(y_true_array, probability_array)), 6),
        "confusion_matrix": confusion.astype(int).tolist(),
    }

    return BaselineModelResult(
        name=name,
        family=family,
        metrics=metrics,
        calibration=calibration,
        note=note,
        probabilities=probability_array.tolist(),
        predictions=prediction_array.tolist(),
    )


def build_calibration_summary(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    positive_rate, mean_predicted_value = calibration_curve(
        y_true,
        probabilities,
        n_bins=8,
        strategy="quantile",
    )
    mean_abs_gap = round(
        float(np.mean(np.abs(np.asarray(positive_rate) - np.asarray(mean_predicted_value)))),
        6,
    ) if len(positive_rate) else None

    return {
        "bin_true_rate": [round(float(value), 6) for value in positive_rate],
        "bin_predicted_probability": [round(float(value), 6) for value in mean_predicted_value],
        "mean_absolute_gap": mean_abs_gap,
    }


def choose_recommended_baseline(results: dict[str, BaselineModelResult]) -> dict[str, str]:
    candidate_names = ["logistic_regression", "decision_tree"]
    candidate_results = [results[name] for name in candidate_names if name in results]
    if not candidate_results:
        raise DataAnalysisError("Faz 4 tavsiye karari icin uygun baseline sonucu bulunamadi.")

    sorted_candidates = sorted(
        candidate_results,
        key=lambda result: (
            result.metrics["balanced_accuracy"],
            result.metrics["roc_auc"] if result.metrics["roc_auc"] is not None else -1.0,
            -(result.metrics["brier_score"]),
        ),
        reverse=True,
    )
    strongest = sorted_candidates[0]
    logistic_result = results.get("logistic_regression")

    if logistic_result is not None:
        balanced_gap = strongest.metrics["balanced_accuracy"] - logistic_result.metrics["balanced_accuracy"]
        roc_gap = 0.0
        if strongest.metrics["roc_auc"] is not None and logistic_result.metrics["roc_auc"] is not None:
            roc_gap = strongest.metrics["roc_auc"] - logistic_result.metrics["roc_auc"]
        brier_gap = logistic_result.metrics["brier_score"] - strongest.metrics["brier_score"]

        if strongest.name != "logistic_regression" and balanced_gap <= 0.02 and roc_gap <= 0.02 and brier_gap <= 0.02:
            return {
                "recommended_model": "logistic_regression",
                "rationale": "Karar agaci biraz daha guclu olsa da lojistik regresyon neredeyse ayni baseline performansini daha iyi savunulabilirlik ile sunuyor.",
            }

    return {
        "recommended_model": strongest.name,
        "rationale": "Secilen model baseline fazinda en guclu ayrisma ve kabul edilebilir kalibrasyon profilini sagladi.",
    }


class Phase4BaselineService:
    def __init__(
        self,
        loader: CsvDatasetLoader,
        logger: logging.Logger,
        *,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self._loader = loader
        self._logger = logger
        self._test_size = test_size
        self._random_state = random_state

    def run(self, project_paths: ProjectPaths) -> Phase4Artifacts:
        dataset = self._loader.load(project_paths.dataset_path)
        deduplicated_dataset = deduplicate_dataset(dataset)
        role_matrix = build_feature_role_matrix()
        validate_required_columns(
            deduplicated_dataset,
            role_matrix.base_input_features + (role_matrix.target_column,),
        )

        artifacts = self._ensure_phase4_directories(project_paths.phase_4_dir)
        self._logger.info("Faz 4 baseline modelleme baslatildi: %s", project_paths.dataset_path)

        train_frame, test_frame = split_dataset_stratified(
            deduplicated_dataset,
            role_matrix.target_column,
            test_size=self._test_size,
            random_state=self._random_state,
        )
        X_train = train_frame.loc[:, role_matrix.base_input_features]
        X_test = test_frame.loc[:, role_matrix.base_input_features]
        y_train = train_frame[role_matrix.target_column]
        y_test = test_frame[role_matrix.target_column]

        baseline_results, fitted_models = self._train_and_evaluate_models(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            role_matrix=role_matrix,
        )
        recommendation = choose_recommended_baseline(baseline_results)
        summary = self._build_summary(
            project_paths=project_paths,
            deduplicated_row_count=len(deduplicated_dataset),
            train_frame=train_frame,
            test_frame=test_frame,
            results=baseline_results,
            recommendation=recommendation,
        )

        self._write_model_artifacts(artifacts.models_dir, fitted_models)
        self._write_plots(artifacts.plots_dir, baseline_results)
        self._write_json_summary(artifacts.json_summary_path, summary)
        self._write_markdown_report(artifacts.markdown_report_path, summary)
        self._logger.info("Faz 4 baseline modelleme tamamlandi.")
        return artifacts

    def _ensure_phase4_directories(self, phase_4_dir: Path) -> Phase4Artifacts:
        plots_dir = phase_4_dir / "plots"
        models_dir = phase_4_dir / "models"
        for directory in (phase_4_dir, plots_dir, models_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return Phase4Artifacts(
            markdown_report_path=phase_4_dir / "baseline_report.md",
            json_summary_path=phase_4_dir / "baseline_summary.json",
            plots_dir=plots_dir,
            models_dir=models_dir,
        )

    def _train_and_evaluate_models(
        self,
        *,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        role_matrix: FeatureRoleMatrix,
    ) -> tuple[dict[str, BaselineModelResult], dict[str, Any]]:
        results: dict[str, BaselineModelResult] = {}
        fitted_models: dict[str, Any] = {}

        majority_model = DummyClassifier(strategy="prior")
        majority_model.fit(X_train, y_train)
        results["majority_baseline"] = evaluate_binary_classifier(
            name="majority_baseline",
            family="dummy",
            y_true=y_test,
            predictions=majority_model.predict(X_test),
            probabilities=majority_model.predict_proba(X_test)[:, 1],
            note="Cogunluk sinifi tahmini, gelismis modellerin gercekten deger katip katmadigini gosteren referanstir.",
        )

        random_model = DummyClassifier(strategy="stratified", random_state=self._random_state)
        random_model.fit(X_train, y_train)
        results["random_baseline"] = evaluate_binary_classifier(
            name="random_baseline",
            family="dummy",
            y_true=y_test,
            predictions=random_model.predict(X_test),
            probabilities=random_model.predict_proba(X_test)[:, 1],
            note="Rastgele dagitim referansi, ogrenilen modellerin saf sansin uzerine ne kadar ciktigini gosterir.",
        )

        rule_model = RuleBasedBaselineClassifier().fit(X_train, y_train)
        rule_probabilities = rule_model.predict_proba(X_test)[:, 1]
        results["rule_based_baseline"] = evaluate_binary_classifier(
            name="rule_based_baseline",
            family="heuristic",
            y_true=y_test,
            predictions=rule_model.predict(X_test),
            probabilities=rule_probabilities,
            note="Sadakat, satin alma gecmisi ve site suresi uzerine kurulu basit kural tabani benchmark'tir.",
        )
        fitted_models["rule_based_baseline"] = rule_model.export_config()

        logistic_pipeline = Pipeline(
            steps=[
                ("preprocessing", build_linear_preprocessing_pipeline(role_matrix)),
                (
                    "model",
                    LogisticRegression(max_iter=1500, solver="lbfgs", random_state=self._random_state),
                ),
            ]
        )
        logistic_pipeline.fit(X_train, y_train)
        results["logistic_regression"] = evaluate_binary_classifier(
            name="logistic_regression",
            family="linear_model",
            y_true=y_test,
            predictions=logistic_pipeline.predict(X_test),
            probabilities=logistic_pipeline.predict_proba(X_test)[:, 1],
            note="Aciklanabilir ve olasilik uretebilen ilk ciddi baseline modeldir.",
        )
        fitted_models["logistic_regression"] = logistic_pipeline

        decision_tree_model = Pipeline(
            steps=[
                ("preprocessing", build_tree_preprocessing_pipeline(role_matrix)),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=4,
                        min_samples_leaf=max(10, int(len(X_train) * 0.02)),
                        random_state=self._random_state,
                    ),
                ),
            ]
        )
        decision_tree_model.fit(X_train, y_train)
        results["decision_tree"] = evaluate_binary_classifier(
            name="decision_tree",
            family="tree_model",
            y_true=y_test,
            predictions=decision_tree_model.predict(X_test),
            probabilities=decision_tree_model.predict_proba(X_test)[:, 1],
            note="Dogrusal olmayan iliskileri hizli okumak icin kontrollu derinlikte tutulan agac baseline'dir.",
        )
        fitted_models["decision_tree"] = decision_tree_model

        return results, fitted_models

    def _build_summary(
        self,
        *,
        project_paths: ProjectPaths,
        deduplicated_row_count: int,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        results: dict[str, BaselineModelResult],
        recommendation: dict[str, str],
    ) -> dict[str, Any]:
        phase1_context = self._load_json_context(project_paths.phase_1_dir / "audit_summary.json")
        phase3_context = self._load_json_context(project_paths.phase_3_dir / "preprocessing_summary.json")

        calibration_candidates = []
        calibration_focus_models = {"logistic_regression", "decision_tree"}
        for result in results.values():
            if result.name not in calibration_focus_models:
                continue
            mean_gap = result.calibration.get("mean_absolute_gap")
            if mean_gap is not None and (mean_gap > 0.05 or result.metrics["brier_score"] > 0.2):
                calibration_candidates.append(result.name)

        return {
            "dataset_path": str(project_paths.dataset_path),
            "deduplicated_row_count": int(deduplicated_row_count),
            "split_summary": {
                "test_size": self._test_size,
                "random_state": self._random_state,
                "train_row_count": int(len(train_frame)),
                "test_row_count": int(len(test_frame)),
                "train_target_distribution": self._target_distribution(train_frame),
                "test_target_distribution": self._target_distribution(test_frame),
            },
            "phase1_context": {
                "final_decision": phase1_context.get("final_decision") if phase1_context else None,
                "decision_rationale": phase1_context.get("decision_rationale") if phase1_context else None,
            },
            "phase3_context": {
                "validation_summary": phase3_context.get("validation_summary") if phase3_context else None,
                "preprocessing_map": phase3_context.get("preprocessing_map") if phase3_context else None,
            },
            "metric_registry": {
                "accuracy": "Genel dogru tahmin orani; tek basina karar verdirici degildir.",
                "precision": "Yanlis kisilere mudahale etme maliyetini okumak icin izlenir.",
                "recall": "Kritik kullanicilari kacirmama kapasitesini gosterir.",
                "f1": "Precision ve recall arasindaki dengeyi ozetler.",
                "roc_auc": "Pozitif ve negatif sinif ayrisma gucunu genel olarak gosterir.",
                "pr_auc": "Ozellikle sinif dengesinin bozuldugu senaryolarda daha aciklayicidir.",
                "balanced_accuracy": "Iki sinifi esit agirlikla yorumlayan denge metrigidir.",
                "brier_score": "Olasilik kalitesini ve kalibrasyon ihtiyacini okumak icin kullanilir.",
                "confusion_matrix": "Is aksiyonlarinin sayisal dagilimini yorumlamak icin saklanir.",
            },
            "model_results": {
                name: {
                    "family": result.family,
                    "metrics": result.metrics,
                    "calibration": result.calibration,
                    "note": result.note,
                }
                for name, result in results.items()
            },
            "calibration_assessment": {
                "models_requiring_follow_up": calibration_candidates,
                "recommendation": "Kalibrasyon sonraki fazda ozellikle champion adayi modeller icin yeniden olculmelidir.",
            },
            "recommended_baseline": recommendation,
            "baseline_closeout": {
                "summary": "Majority ve random referanslar asildiktan sonra lojistik regresyon ile karar agaci ayni split uzerinde karsilastirildi; boylece Faz 6'ya gecmeden once hem aciklanabilir hem de dogrusal olmayan baseline davranisi gorunur hale geldi.",
                "next_step": "Faz 5'te class weight ve esik stratejileri bu sabit baseline metrik seti uzerinden degerlendirilecektir.",
            },
        }

    def _target_distribution(self, frame: pd.DataFrame) -> dict[str, float]:
        counts = frame[TARGET_COLUMN].value_counts().sort_index().to_dict()
        return {str(key): round(float(value) / len(frame), 6) for key, value in counts.items()}

    def _write_model_artifacts(self, models_dir: Path, models: dict[str, Any]) -> None:
        try:
            for model_name, model_object in models.items():
                if isinstance(model_object, (Pipeline, ClassifierMixin)):
                    joblib.dump(model_object, models_dir / f"{model_name}.joblib")
                else:
                    with (models_dir / f"{model_name}.json").open("w", encoding="utf-8") as output_file:
                        json.dump(to_json_safe(model_object), output_file, ensure_ascii=False, indent=2)
        except OSError as error:
            raise ArtifactWriteError(f"Faz 4 model artefaktlari yazilamadi: {models_dir}") from error

    def _write_plots(self, plots_dir: Path, results: dict[str, BaselineModelResult]) -> None:
        self._plot_metric_comparison(plots_dir / "baseline_metric_comparison.png", results)
        self._plot_calibration_curves(plots_dir / "baseline_calibration_curves.png", results)

    def _plot_metric_comparison(self, output_path: Path, results: dict[str, BaselineModelResult]) -> None:
        tracked_metrics = ["balanced_accuracy", "f1", "roc_auc", "pr_auc"]
        model_names = list(results.keys())
        x_positions = np.arange(len(model_names))
        width = 0.18

        figure, axis = plt.subplots(figsize=(14, 7))
        for offset_index, metric_name in enumerate(tracked_metrics):
            values = [
                results[model_name].metrics[metric_name] if results[model_name].metrics[metric_name] is not None else 0.0
                for model_name in model_names
            ]
            axis.bar(x_positions + (offset_index * width), values, width=width, label=metric_name)

        axis.set_title("Faz 4 Baseline Metrik Karsilastirmasi")
        axis.set_ylabel("Skor")
        axis.set_xticks(x_positions + (width * 1.5), model_names, rotation=20, ha="right")
        axis.legend()
        axis.set_ylim(0.0, 1.05)
        figure.tight_layout()
        self._save_figure(figure, output_path)

    def _plot_calibration_curves(self, output_path: Path, results: dict[str, BaselineModelResult]) -> None:
        plotted_models = ["logistic_regression", "decision_tree", "rule_based_baseline"]
        figure, axis = plt.subplots(figsize=(8, 6))
        axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="ideal")

        for model_name in plotted_models:
            result = results.get(model_name)
            if result is None:
                continue
            axis.plot(
                result.calibration["bin_predicted_probability"],
                result.calibration["bin_true_rate"],
                marker="o",
                label=model_name,
            )

        axis.set_title("Faz 4 Kalibrasyon Egrileri")
        axis.set_xlabel("Tahmin Edilen Olasilik")
        axis.set_ylabel("Gercek Pozitif Orani")
        axis.legend()
        figure.tight_layout()
        self._save_figure(figure, output_path)

    def _save_figure(self, figure: plt.Figure, output_path: Path) -> None:
        try:
            figure.savefig(output_path, dpi=200, bbox_inches="tight")
        except OSError as error:
            raise ArtifactWriteError(f"Faz 4 grafigi kaydedilemedi: {output_path}") from error
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
            raise ArtifactWriteError(f"Faz 4 JSON ozeti yazilamadi: {output_path}") from error

    def _write_markdown_report(self, output_path: Path, summary: dict[str, Any]) -> None:
        lines = [
            "# Faz 4 Baseline Modelleme Raporu",
            "",
            f"- Veri seti: `{summary['dataset_path']}`",
            f"- Tekillestirilmis satir sayisi: {summary['deduplicated_row_count']}",
            f"- Train/Test satir sayisi: {summary['split_summary']['train_row_count']} / {summary['split_summary']['test_row_count']}",
            "",
            "## Faz Baglami",
            "",
            f"- Faz 1 karari: {summary['phase1_context']['final_decision']}",
            f"- Faz 3 preprocessing ozeti bulundu mu: {summary['phase3_context']['preprocessing_map'] is not None}",
            "",
            "## Sabit Metrik Seti",
            "",
        ]
        for metric_name, description in summary["metric_registry"].items():
            lines.append(f"- {metric_name}: {description}")

        lines.extend(["", "## Model Sonuclari", ""])
        for model_name, result in summary["model_results"].items():
            lines.append(f"### {model_name}")
            lines.append(f"- Aile: {result['family']}")
            lines.append(f"- Metrikler: {result['metrics']}")
            lines.append(f"- Kalibrasyon: {result['calibration']}")
            lines.append(f"- Not: {result['note']}")
            lines.append("")

        lines.extend(
            [
                "## Kalibrasyon On Degerlendirmesi",
                "",
                f"- Takip edilmesi gereken modeller: {summary['calibration_assessment']['models_requiring_follow_up']}",
                f"- Not: {summary['calibration_assessment']['recommendation']}",
                "",
                "## Baseline Kapanis Karari",
                "",
                f"- Onerilen baseline: {summary['recommended_baseline']['recommended_model']}",
                f"- Gerekce: {summary['recommended_baseline']['rationale']}",
                f"- Faz ozeti: {summary['baseline_closeout']['summary']}",
                f"- Sonraki adim: {summary['baseline_closeout']['next_step']}",
            ]
        )

        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                output_file.write("\n".join(lines))
        except OSError as error:
            raise ArtifactWriteError(f"Faz 4 markdown raporu yazilamadi: {output_path}") from error
