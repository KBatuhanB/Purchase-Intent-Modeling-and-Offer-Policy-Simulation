from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ..config import (
    CATEGORICAL_COLUMNS,
    LEAKY_OR_ANALYSIS_ONLY_COLUMNS,
    SAFE_NUMERIC_COLUMNS,
    TARGET_COLUMN,
    ProjectPaths,
)
from ..core.exceptions import ArtifactWriteError, SchemaValidationError
from ..core.serialization import to_json_safe
from .csv_loader import CsvDatasetLoader
from .phase2_eda import deduplicate_dataset


@dataclass(frozen=True)
class FeatureRoleMatrix:
    target_column: str
    base_numeric_features: tuple[str, ...]
    base_categorical_features: tuple[str, ...]
    excluded_from_main_model: tuple[str, ...]
    analysis_only_features: tuple[str, ...]
    engineered_numeric_features: tuple[str, ...]
    engineered_categorical_features: tuple[str, ...]

    @property
    def base_input_features(self) -> tuple[str, ...]:
        return self.base_numeric_features + self.base_categorical_features

    @property
    def linear_numeric_features(self) -> tuple[str, ...]:
        return self.base_numeric_features + self.engineered_numeric_features

    @property
    def linear_categorical_features(self) -> tuple[str, ...]:
        return self.base_categorical_features + self.engineered_categorical_features

    @property
    def tree_numeric_features(self) -> tuple[str, ...]:
        return self.base_numeric_features + self.engineered_numeric_features

    @property
    def tree_categorical_features(self) -> tuple[str, ...]:
        return self.base_categorical_features + self.engineered_categorical_features


@dataclass(frozen=True)
class Phase3Artifacts:
    markdown_report_path: Path
    json_summary_path: Path
    datasets_dir: Path
    transformed_dir: Path
    models_dir: Path


def build_feature_role_matrix() -> FeatureRoleMatrix:
    return FeatureRoleMatrix(
        target_column=TARGET_COLUMN,
        base_numeric_features=SAFE_NUMERIC_COLUMNS,
        base_categorical_features=CATEGORICAL_COLUMNS,
        excluded_from_main_model=LEAKY_OR_ANALYSIS_ONLY_COLUMNS,
        analysis_only_features=LEAKY_OR_ANALYSIS_ONLY_COLUMNS,
        engineered_numeric_features=(
            "income_per_purchase_proxy",
            "loyalty_time_interaction",
            "high_time_low_history_flag",
            "non_loyal_high_time_flag",
        ),
        engineered_categorical_features=(
            "income_bucket",
            "purchase_frequency_bucket",
            "time_spent_bucket",
            "category_income_interaction",
        ),
    )


def split_dataset_stratified(
    dataset: pd.DataFrame,
    target_column: str,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_counts = dataset[target_column].value_counts(dropna=False)
    if target_counts.shape[0] < 2:
        raise SchemaValidationError("Stratified split icin hedef degisken en az iki sinif icermelidir.")
    if int(target_counts.min()) < 2:
        raise SchemaValidationError(
            "Stratified split icin her sinifta en az iki ornek bulunmalidir."
        )

    train_frame, test_frame = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset[target_column],
    )
    return train_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


def build_linear_preprocessing_pipeline(role_matrix: FeatureRoleMatrix) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, role_matrix.linear_numeric_features),
            ("categorical", categorical_pipeline, role_matrix.linear_categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            (
                "feature_engineering",
                FeatureEngineeringTransformer(role_matrix),
            ),
            (
                "preprocessor",
                preprocessor,
            ),
        ]
    )


def build_tree_preprocessing_pipeline(role_matrix: FeatureRoleMatrix) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, role_matrix.tree_numeric_features),
            ("categorical", categorical_pipeline, role_matrix.tree_categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            (
                "feature_engineering",
                FeatureEngineeringTransformer(role_matrix),
            ),
            (
                "preprocessor",
                preprocessor,
            ),
        ]
    )


def validate_required_columns(dataset: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    missing_columns = sorted(set(required_columns) - set(dataset.columns))
    if missing_columns:
        raise SchemaValidationError(
            f"Faz 3 icin gerekli sutunlar eksik: {', '.join(missing_columns)}"
        )


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, role_matrix: FeatureRoleMatrix) -> None:
        self.role_matrix = role_matrix

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureEngineeringTransformer":
        frame = self._validate_input(X)

        # Bu esikler sadece train verisinden ogrenilir; boylece test bilgisi ozellik tasarimina sizmaz.
        self._income_edges = self._build_quantile_edges(frame["AnnualIncome"])
        self._purchase_edges = self._build_quantile_edges(frame["NumberOfPurchases"])
        self._time_edges = self._build_quantile_edges(frame["TimeSpentOnWebsite"])
        self._high_time_threshold = float(pd.to_numeric(frame["TimeSpentOnWebsite"], errors="coerce").quantile(0.75))
        self._low_history_threshold = float(
            pd.to_numeric(frame["NumberOfPurchases"], errors="coerce").quantile(0.25)
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._validate_input(X)
        transformed = frame.copy(deep=True)

        transformed["income_bucket"] = self._apply_bands(
            series=pd.to_numeric(transformed["AnnualIncome"], errors="coerce"),
            edges=self._income_edges,
            prefix="income",
        )
        transformed["purchase_frequency_bucket"] = self._apply_bands(
            series=pd.to_numeric(transformed["NumberOfPurchases"], errors="coerce"),
            edges=self._purchase_edges,
            prefix="purchase_frequency",
        )
        transformed["time_spent_bucket"] = self._apply_bands(
            series=pd.to_numeric(transformed["TimeSpentOnWebsite"], errors="coerce"),
            edges=self._time_edges,
            prefix="time_spent",
        )

        numeric_purchase_count = pd.to_numeric(transformed["NumberOfPurchases"], errors="coerce").fillna(0.0)
        numeric_income = pd.to_numeric(transformed["AnnualIncome"], errors="coerce")
        numeric_time = pd.to_numeric(transformed["TimeSpentOnWebsite"], errors="coerce")
        loyalty_series = pd.to_numeric(transformed["LoyaltyProgram"], errors="coerce").fillna(0.0)

        # Oran tabanli proxy, sifira bolme hatasi uretmesin diye +1 ile korunur.
        transformed["income_per_purchase_proxy"] = numeric_income / (numeric_purchase_count + 1.0)
        transformed["loyalty_time_interaction"] = loyalty_series * numeric_time
        transformed["high_time_low_history_flag"] = (
            (numeric_time >= self._high_time_threshold) & (numeric_purchase_count <= self._low_history_threshold)
        ).astype(int)
        transformed["non_loyal_high_time_flag"] = (
            (loyalty_series == 0) & (numeric_time >= self._high_time_threshold)
        ).astype(int)
        transformed["category_income_interaction"] = (
            transformed["ProductCategory"].astype(str) + "__" + transformed["income_bucket"].fillna("unknown")
        )

        return transformed

    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise SchemaValidationError("Feature engineering adimi DataFrame girdi bekler.")
        validate_required_columns(X, self.role_matrix.base_input_features)
        return X.loc[:, self.role_matrix.base_input_features]

    def _build_quantile_edges(self, series: pd.Series) -> list[float] | None:
        numeric_series = pd.to_numeric(series, errors="coerce").dropna()
        if numeric_series.nunique() < 2:
            return None

        effective_quantiles = min(4, numeric_series.nunique())
        _, raw_edges = pd.qcut(numeric_series, q=effective_quantiles, retbins=True, duplicates="drop")
        unique_edges = sorted({float(edge) for edge in raw_edges})

        if len(unique_edges) < 2:
            return None

        unique_edges[0] = float("-inf")
        unique_edges[-1] = float("inf")
        return unique_edges

    def _apply_bands(self, series: pd.Series, edges: list[float] | None, prefix: str) -> pd.Series:
        if edges is None:
            return pd.Series([f"{prefix}_single_band"] * len(series), index=series.index, dtype="object")

        labels = [f"{prefix}_q{index}" for index in range(1, len(edges))]
        bucketed = pd.cut(series, bins=edges, labels=labels, include_lowest=True)
        return bucketed.astype(str).fillna(f"{prefix}_unknown")


class Phase3PreprocessingService:
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

    def run(self, project_paths: ProjectPaths) -> Phase3Artifacts:
        dataset = self._loader.load(project_paths.dataset_path)
        deduplicated_dataset = deduplicate_dataset(dataset)
        role_matrix = build_feature_role_matrix()

        validate_required_columns(
            deduplicated_dataset,
            role_matrix.base_input_features
            + role_matrix.excluded_from_main_model
            + (role_matrix.target_column,),
        )

        artifacts = self._ensure_phase3_directories(project_paths.phase_3_dir)
        self._logger.info("Faz 3 preprocessing baslatildi: %s", project_paths.dataset_path)

        train_frame, test_frame = self._split_dataset(deduplicated_dataset, role_matrix.target_column)
        train_features = train_frame.loc[:, role_matrix.base_input_features]
        test_features = test_frame.loc[:, role_matrix.base_input_features]
        train_target = train_frame[role_matrix.target_column]
        test_target = test_frame[role_matrix.target_column]

        linear_pipeline = self._build_linear_pipeline(role_matrix)
        tree_pipeline = self._build_tree_pipeline(role_matrix)
        linear_pipeline.fit(train_features)
        tree_pipeline.fit(train_features)

        linear_train_matrix = self._transform_pipeline(linear_pipeline, train_features)
        linear_test_matrix = self._transform_pipeline(linear_pipeline, test_features)
        tree_train_matrix = self._transform_pipeline(tree_pipeline, train_features)
        tree_test_matrix = self._transform_pipeline(tree_pipeline, test_features)

        validation_summary = self._run_validation_experiment(
            train_features=train_features,
            train_target=train_target,
            test_features=test_features,
            test_target=test_target,
            role_matrix=role_matrix,
            linear_train_matrix=linear_train_matrix,
            linear_test_matrix=linear_test_matrix,
        )

        summary = self._build_summary(
            project_paths=project_paths,
            role_matrix=role_matrix,
            deduplicated_dataset=deduplicated_dataset,
            train_frame=train_frame,
            test_frame=test_frame,
            linear_train_matrix=linear_train_matrix,
            linear_test_matrix=linear_test_matrix,
            tree_train_matrix=tree_train_matrix,
            tree_test_matrix=tree_test_matrix,
            validation_summary=validation_summary,
        )

        self._write_split_artifacts(
            artifacts=artifacts,
            train_frame=train_frame,
            test_frame=test_frame,
            train_target=train_target,
            test_target=test_target,
            linear_train_matrix=linear_train_matrix,
            linear_test_matrix=linear_test_matrix,
            tree_train_matrix=tree_train_matrix,
            tree_test_matrix=tree_test_matrix,
        )
        self._write_pipeline_artifacts(
            models_dir=artifacts.models_dir,
            linear_pipeline=linear_pipeline,
            tree_pipeline=tree_pipeline,
        )
        self._write_json_summary(artifacts.json_summary_path, summary)
        self._write_markdown_report(artifacts.markdown_report_path, summary)
        self._logger.info("Faz 3 preprocessing tamamlandi.")
        return artifacts

    def _ensure_phase3_directories(self, phase_3_dir: Path) -> Phase3Artifacts:
        datasets_dir = phase_3_dir / "datasets"
        transformed_dir = phase_3_dir / "transformed"
        models_dir = phase_3_dir / "models"

        for directory in (phase_3_dir, datasets_dir, transformed_dir, models_dir):
            directory.mkdir(parents=True, exist_ok=True)

        return Phase3Artifacts(
            markdown_report_path=phase_3_dir / "preprocessing_report.md",
            json_summary_path=phase_3_dir / "preprocessing_summary.json",
            datasets_dir=datasets_dir,
            transformed_dir=transformed_dir,
            models_dir=models_dir,
        )

    def _split_dataset(self, dataset: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        return split_dataset_stratified(
            dataset,
            target_column,
            test_size=self._test_size,
            random_state=self._random_state,
        )

    def _build_linear_pipeline(self, role_matrix: FeatureRoleMatrix) -> Pipeline:
        return build_linear_preprocessing_pipeline(role_matrix)

    def _build_tree_pipeline(self, role_matrix: FeatureRoleMatrix) -> Pipeline:
        return build_tree_preprocessing_pipeline(role_matrix)

    def _transform_pipeline(self, pipeline: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
        transformed_array = pipeline.transform(features)
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        transformed_frame = pd.DataFrame(transformed_array, columns=feature_names, index=features.index)

        if transformed_frame.isna().any().any():
            raise SchemaValidationError("Preprocessing sonrasi ozellik matrisinde beklenmeyen bos deger kaldı.")
        return transformed_frame.reset_index(drop=True)

    def _run_validation_experiment(
        self,
        *,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        test_features: pd.DataFrame,
        test_target: pd.Series,
        role_matrix: FeatureRoleMatrix,
        linear_train_matrix: pd.DataFrame,
        linear_test_matrix: pd.DataFrame,
    ) -> dict[str, Any]:
        validation_pipeline = Pipeline(
            steps=[
                (
                    "feature_engineering",
                    FeatureEngineeringTransformer(role_matrix),
                ),
                (
                    "preprocessor",
                    self._build_linear_pipeline(role_matrix).named_steps["preprocessor"],
                ),
                (
                    "model",
                    LogisticRegression(max_iter=1000, solver="lbfgs"),
                ),
            ]
        )
        validation_pipeline.fit(train_features, train_target)

        predictions = validation_pipeline.predict(test_features)
        probabilities = validation_pipeline.predict_proba(test_features)[:, 1]

        roc_auc_value: float | None
        try:
            roc_auc_value = round(float(roc_auc_score(test_target, probabilities)), 6)
        except ValueError:
            roc_auc_value = None

        return {
            "train_feature_count": int(linear_train_matrix.shape[1]),
            "test_feature_count": int(linear_test_matrix.shape[1]),
            "columns_aligned": list(linear_train_matrix.columns) == list(linear_test_matrix.columns),
            "train_has_missing": bool(linear_train_matrix.isna().any().any()),
            "test_has_missing": bool(linear_test_matrix.isna().any().any()),
            "accuracy": round(float(accuracy_score(test_target, predictions)), 6),
            "balanced_accuracy": round(float(balanced_accuracy_score(test_target, predictions)), 6),
            "f1": round(float(f1_score(test_target, predictions)), 6),
            "roc_auc": roc_auc_value,
            "note": "Bu skorlar nihai model secimi icin degil, preprocessing hattinin teknik saglamligini hizli dogrulamak icindir.",
        }

    def _build_summary(
        self,
        *,
        project_paths: ProjectPaths,
        role_matrix: FeatureRoleMatrix,
        deduplicated_dataset: pd.DataFrame,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        linear_train_matrix: pd.DataFrame,
        linear_test_matrix: pd.DataFrame,
        tree_train_matrix: pd.DataFrame,
        tree_test_matrix: pd.DataFrame,
        validation_summary: dict[str, Any],
    ) -> dict[str, Any]:
        phase1_context = self._load_json_context(project_paths.phase_1_dir / "audit_summary.json")
        phase2_context = self._load_json_context(project_paths.phase_2_dir / "eda_summary.json")

        return {
            "dataset_path": str(project_paths.dataset_path),
            "deduplicated_row_count": int(len(deduplicated_dataset)),
            "split_strategy": {
                "method": "stratified_train_test_split",
                "test_size": self._test_size,
                "random_state": self._random_state,
                "train_row_count": int(len(train_frame)),
                "test_row_count": int(len(test_frame)),
                "train_target_distribution": self._target_distribution(train_frame),
                "test_target_distribution": self._target_distribution(test_frame),
            },
            "phase1_context": phase1_context,
            "phase2_context": {
                "dropped_duplicate_count": phase2_context.get("dropped_duplicate_count") if phase2_context else None,
                "hypothesis_count": len(phase2_context.get("hypotheses", [])) if phase2_context else None,
            },
            "feature_roles": asdict(role_matrix),
            "preprocessing_map": {
                "linear_branch": {
                    "encoding": "OneHotEncoder(handle_unknown='ignore')",
                    "imputation": "median + most_frequent",
                    "scaling": "StandardScaler",
                    "train_feature_count": int(linear_train_matrix.shape[1]),
                    "test_feature_count": int(linear_test_matrix.shape[1]),
                },
                "tree_branch": {
                    "encoding": "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)",
                    "imputation": "median + most_frequent",
                    "scaling": "none",
                    "train_feature_count": int(tree_train_matrix.shape[1]),
                    "test_feature_count": int(tree_test_matrix.shape[1]),
                },
            },
            "feature_engineering_catalog": self._build_feature_engineering_catalog(),
            "outlier_policy": self._collect_outlier_policy(train_frame),
            "validation_summary": validation_summary,
        }

    def _target_distribution(self, frame: pd.DataFrame) -> dict[str, float]:
        counts = frame[TARGET_COLUMN].value_counts().sort_index().to_dict()
        return {
            str(key): round(float(value) / len(frame), 6) for key, value in counts.items()
        }

    def _build_feature_engineering_catalog(self) -> list[dict[str, str]]:
        return [
            {
                "name": "income_bucket",
                "feature_type": "categorical",
                "rationale": "Gelir etkisinin dogrusal olmama ihtimalini bucket seviyesinde modellemek icin kullanilir.",
            },
            {
                "name": "purchase_frequency_bucket",
                "feature_type": "categorical",
                "rationale": "Gecmis satin alma yogunlugunu lineer olmayan sinyal olarak yakalamak icin eklenir.",
            },
            {
                "name": "time_spent_bucket",
                "feature_type": "categorical",
                "rationale": "Site suresi etkisini risk bantlari halinde modellemek icin uretilir.",
            },
            {
                "name": "income_per_purchase_proxy",
                "feature_type": "numeric",
                "rationale": "Gelir ile satin alma gecmisi arasindaki dengeyi tek bir oransal sinyale indirger.",
            },
            {
                "name": "loyalty_time_interaction",
                "feature_type": "numeric",
                "rationale": "Sadakat uyeligi ile sitede gecirilen zaman birlikte yorumlanabilsin diye uretilir.",
            },
            {
                "name": "category_income_interaction",
                "feature_type": "categorical",
                "rationale": "Urun kategorisi ile gelir segmentini birlestirerek daha zengin segmentler olusturur.",
            },
            {
                "name": "high_time_low_history_flag",
                "feature_type": "numeric",
                "rationale": "Yuksek ilgi ama dusuk gecmis satin alma davranisini kararsiz segment olarak isaretler.",
            },
            {
                "name": "non_loyal_high_time_flag",
                "feature_type": "numeric",
                "rationale": "Sadakati olmayan ama ilgisi yuksek kullanicilari ayristirmak icin eklenir.",
            },
        ]

    def _collect_outlier_policy(self, train_frame: pd.DataFrame) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []

        for column_name in SAFE_NUMERIC_COLUMNS:
            series = pd.to_numeric(train_frame[column_name], errors="coerce")
            first_quartile = float(series.quantile(0.25))
            third_quartile = float(series.quantile(0.75))
            iqr = third_quartile - first_quartile
            lower_bound = first_quartile - (1.5 * iqr)
            upper_bound = third_quartile + (1.5 * iqr)
            outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())

            entries.append(
                {
                    "column": column_name,
                    "lower_bound": round(lower_bound, 6),
                    "upper_bound": round(upper_bound, 6),
                    "observed_outlier_count": outlier_count,
                    "policy": "Varsayilan olarak silme uygulanmadi; gerekirse Faz 4 oncesi kontrollu clipping deneyi yapilacak.",
                }
            )

        return {
            "method": "iqr_monitoring_only",
            "applied_in_pipeline": False,
            "entries": entries,
        }

    def _load_json_context(self, file_path: Path) -> dict[str, Any] | None:
        if not file_path.exists():
            return None

        try:
            with file_path.open("r", encoding="utf-8") as input_file:
                return json.load(input_file)
        except (OSError, json.JSONDecodeError):
            return None

    def _write_split_artifacts(
        self,
        *,
        artifacts: Phase3Artifacts,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        train_target: pd.Series,
        test_target: pd.Series,
        linear_train_matrix: pd.DataFrame,
        linear_test_matrix: pd.DataFrame,
        tree_train_matrix: pd.DataFrame,
        tree_test_matrix: pd.DataFrame,
    ) -> None:
        self._write_dataframe(artifacts.datasets_dir / "train_dataset.csv", train_frame)
        self._write_dataframe(artifacts.datasets_dir / "test_dataset.csv", test_frame)
        self._write_dataframe(
            artifacts.transformed_dir / "linear_train_matrix.csv",
            linear_train_matrix.assign(**{TARGET_COLUMN: train_target.reset_index(drop=True)}),
        )
        self._write_dataframe(
            artifacts.transformed_dir / "linear_test_matrix.csv",
            linear_test_matrix.assign(**{TARGET_COLUMN: test_target.reset_index(drop=True)}),
        )
        self._write_dataframe(
            artifacts.transformed_dir / "tree_train_matrix.csv",
            tree_train_matrix.assign(**{TARGET_COLUMN: train_target.reset_index(drop=True)}),
        )
        self._write_dataframe(
            artifacts.transformed_dir / "tree_test_matrix.csv",
            tree_test_matrix.assign(**{TARGET_COLUMN: test_target.reset_index(drop=True)}),
        )

    def _write_pipeline_artifacts(self, *, models_dir: Path, linear_pipeline: Pipeline, tree_pipeline: Pipeline) -> None:
        try:
            joblib.dump(linear_pipeline, models_dir / "linear_preprocessing_pipeline.joblib")
            joblib.dump(tree_pipeline, models_dir / "tree_preprocessing_pipeline.joblib")
        except OSError as error:
            raise ArtifactWriteError(f"Preprocessing pipeline artefakti yazilamadi: {models_dir}") from error

    def _write_dataframe(self, output_path: Path, dataframe: pd.DataFrame) -> None:
        try:
            dataframe.to_csv(output_path, index=False)
        except OSError as error:
            raise ArtifactWriteError(f"Veri artefakti yazilamadi: {output_path}") from error

    def _write_json_summary(self, output_path: Path, summary: dict[str, Any]) -> None:
        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                json.dump(to_json_safe(summary), output_file, ensure_ascii=False, indent=2)
        except OSError as error:
            raise ArtifactWriteError(f"Faz 3 JSON ozeti yazilamadi: {output_path}") from error

    def _write_markdown_report(self, output_path: Path, summary: dict[str, Any]) -> None:
        lines = [
            "# Faz 3 Preprocessing ve Ozellik Muhendisligi Raporu",
            "",
            f"- Veri seti: `{summary['dataset_path']}`",
            f"- Tekillestirilmis satir sayisi: {summary['deduplicated_row_count']}",
            f"- Split yontemi: {summary['split_strategy']['method']}",
            f"- Train/Test: {summary['split_strategy']['train_row_count']} / {summary['split_strategy']['test_row_count']}",
            "",
            "## Faz Baglami",
            "",
            f"- Faz 1 karari: {summary['phase1_context']['final_decision'] if summary['phase1_context'] else 'Bulunamadi'}",
            f"- Faz 2 hipotez sayisi: {summary['phase2_context']['hypothesis_count'] if summary['phase2_context'] else 'Bulunamadi'}",
            "",
            "## Ozellik Rol Matrisi",
            "",
            f"- Hedef degisken: {summary['feature_roles']['target_column']}",
            f"- Ana model disi tutulan alanlar: {summary['feature_roles']['excluded_from_main_model']}",
            f"- Ana sayisal alanlar: {summary['feature_roles']['base_numeric_features']}",
            f"- Ana kategorik alanlar: {summary['feature_roles']['base_categorical_features']}",
            "",
            "## Split Stratejisi",
            "",
            f"- Test oranı: {summary['split_strategy']['test_size']}",
            f"- Random state: {summary['split_strategy']['random_state']}",
            f"- Train hedef dagilimi: {summary['split_strategy']['train_target_distribution']}",
            f"- Test hedef dagilimi: {summary['split_strategy']['test_target_distribution']}",
            "",
            "## Preprocessing Haritasi",
            "",
            f"- Linear branch: {summary['preprocessing_map']['linear_branch']}",
            f"- Tree branch: {summary['preprocessing_map']['tree_branch']}",
            "",
            "## Feature Engineering Katalogu",
            "",
        ]

        for item in summary["feature_engineering_catalog"]:
            lines.append(
                f"- {item['name']} ({item['feature_type']}): {item['rationale']}"
            )

        lines.extend(["", "## Outlier Politikasi", ""])
        for entry in summary["outlier_policy"]["entries"]:
            lines.append(
                f"- {entry['column']}: alt={entry['lower_bound']}, ust={entry['upper_bound']}, gozlenen={entry['observed_outlier_count']}, politika={entry['policy']}"
            )

        lines.extend(
            [
                "",
                "## Kucuk Dogrulama Deneyi",
                "",
                f"- Kolon hizasi korundu mu: {summary['validation_summary']['columns_aligned']}",
                f"- Train feature sayisi: {summary['validation_summary']['train_feature_count']}",
                f"- Test feature sayisi: {summary['validation_summary']['test_feature_count']}",
                f"- Accuracy: {summary['validation_summary']['accuracy']}",
                f"- Balanced accuracy: {summary['validation_summary']['balanced_accuracy']}",
                f"- F1: {summary['validation_summary']['f1']}",
                f"- ROC-AUC: {summary['validation_summary']['roc_auc']}",
                f"- Not: {summary['validation_summary']['note']}",
            ]
        )

        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                output_file.write("\n".join(lines))
        except OSError as error:
            raise ArtifactWriteError(f"Faz 3 markdown raporu yazilamadi: {output_path}") from error
