from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN, ProjectPaths
from ..core.exceptions import ArtifactWriteError
from ..core.serialization import to_json_safe
from .csv_loader import CsvDatasetLoader

sns.set_theme(style="whitegrid")


@dataclass(frozen=True)
class Phase2EdaArtifacts:
    markdown_report_path: Path
    json_summary_path: Path
    plots_dir: Path


def deduplicate_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset.drop_duplicates().reset_index(drop=True)


def safe_quantile_bands(series: pd.Series, quantile_count: int = 4) -> pd.Series:
    non_null_series = series.dropna()
    if non_null_series.nunique() < 2:
        return pd.Series(["tek_bant"] * len(series), index=series.index, dtype="object")

    effective_q = min(quantile_count, non_null_series.nunique())
    try:
        return pd.qcut(series, q=effective_q, duplicates="drop").astype(str)
    except ValueError:
        return pd.cut(series, bins=effective_q, duplicates="drop").astype(str)


class Phase2EdaService:
    def __init__(self, loader: CsvDatasetLoader, logger: logging.Logger) -> None:
        self._loader = loader
        self._logger = logger

    def run(self, project_paths: ProjectPaths) -> Phase2EdaArtifacts:
        dataset = self._loader.load(project_paths.dataset_path)
        deduplicated_dataset = deduplicate_dataset(dataset)
        plots_dir = project_paths.phase_2_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        self._logger.info("Faz 2 EDA baslatildi: %s", project_paths.dataset_path)
        summary = self._build_summary(project_paths, dataset, deduplicated_dataset)
        self._generate_plots(deduplicated_dataset, plots_dir)

        artifacts = Phase2EdaArtifacts(
            markdown_report_path=project_paths.phase_2_dir / "eda_report.md",
            json_summary_path=project_paths.phase_2_dir / "eda_summary.json",
            plots_dir=plots_dir,
        )

        self._write_json_summary(artifacts.json_summary_path, summary)
        self._write_markdown_report(artifacts.markdown_report_path, summary)
        self._logger.info("Faz 2 EDA tamamlandi.")
        return artifacts

    def _build_summary(
        self,
        project_paths: ProjectPaths,
        raw_dataset: pd.DataFrame,
        deduplicated_dataset: pd.DataFrame,
    ) -> dict[str, Any]:
        phase1_context = self._load_phase1_context(project_paths.phase_1_dir / "audit_summary.json")
        numeric_summary = self._collect_numeric_univariate_summary(deduplicated_dataset)
        categorical_summary = self._collect_categorical_summary(deduplicated_dataset)
        numeric_target_relationships = self._collect_numeric_target_relationships(deduplicated_dataset)
        correlation_summary = self._collect_correlation_summary(deduplicated_dataset)
        segment_profiles = self._collect_segment_profiles(deduplicated_dataset)
        anomaly_summary = self._collect_anomaly_summary(deduplicated_dataset)
        hypotheses = self._build_hypotheses(
            categorical_summary=categorical_summary,
            numeric_target_relationships=numeric_target_relationships,
            segment_profiles=segment_profiles,
            anomaly_summary=anomaly_summary,
            raw_row_count=len(raw_dataset),
            deduplicated_row_count=len(deduplicated_dataset),
        )

        return {
            "dataset_path": str(project_paths.dataset_path),
            "raw_row_count": int(len(raw_dataset)),
            "deduplicated_row_count": int(len(deduplicated_dataset)),
            "dropped_duplicate_count": int(len(raw_dataset) - len(deduplicated_dataset)),
            "phase1_context": phase1_context,
            "target_summary": self._collect_target_summary(raw_dataset, deduplicated_dataset),
            "numeric_univariate_summary": numeric_summary,
            "categorical_summary": categorical_summary,
            "numeric_target_relationships": numeric_target_relationships,
            "correlation_summary": correlation_summary,
            "segment_profiles": segment_profiles,
            "anomaly_summary": anomaly_summary,
            "hypotheses": hypotheses,
        }

    def _load_phase1_context(self, audit_summary_path: Path) -> dict[str, Any] | None:
        if not audit_summary_path.exists():
            return None

        try:
            with audit_summary_path.open("r", encoding="utf-8") as input_file:
                audit_summary = json.load(input_file)
        except (OSError, json.JSONDecodeError):
            return None

        return {
            "final_decision": audit_summary.get("final_decision"),
            "decision_rationale": audit_summary.get("decision_rationale"),
            "duplicate_rate": audit_summary.get("duplicate_summary", {}).get("duplicate_rate"),
        }

    def _collect_target_summary(
        self,
        raw_dataset: pd.DataFrame,
        deduplicated_dataset: pd.DataFrame,
    ) -> dict[str, Any]:
        raw_counts = raw_dataset[TARGET_COLUMN].value_counts().sort_index().to_dict()
        deduplicated_counts = deduplicated_dataset[TARGET_COLUMN].value_counts().sort_index().to_dict()

        return {
            "raw_counts": {str(key): int(value) for key, value in raw_counts.items()},
            "raw_rates": {
                str(key): round(value / len(raw_dataset), 6)
                for key, value in raw_counts.items()
            },
            "deduplicated_counts": {
                str(key): int(value) for key, value in deduplicated_counts.items()
            },
            "deduplicated_rates": {
                str(key): round(value / len(deduplicated_dataset), 6)
                for key, value in deduplicated_counts.items()
            },
        }

    def _collect_numeric_univariate_summary(self, dataset: pd.DataFrame) -> dict[str, Any]:
        summary: dict[str, Any] = {}

        for column_name in NUMERIC_COLUMNS:
            series = pd.to_numeric(dataset[column_name], errors="coerce")
            summary[column_name] = {
                "mean": round(float(series.mean()), 6),
                "median": round(float(series.median()), 6),
                "std": round(float(series.std()), 6),
                "min": round(float(series.min()), 6),
                "max": round(float(series.max()), 6),
                "skew": round(float(series.skew()), 6),
            }

        return summary

    def _collect_categorical_summary(self, dataset: pd.DataFrame) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        relevant_columns = list(CATEGORICAL_COLUMNS) + ["DiscountsAvailed"]

        for column_name in relevant_columns:
            grouped = (
                dataset.groupby(column_name, dropna=False)[TARGET_COLUMN]
                .agg(["count", "mean"])
                .reset_index()
                .rename(columns={"count": "row_count", "mean": "purchase_rate"})
            )
            summary[column_name] = grouped.to_dict(orient="records")

        return summary

    def _collect_numeric_target_relationships(self, dataset: pd.DataFrame) -> dict[str, Any]:
        relationships: dict[str, Any] = {}

        for column_name in NUMERIC_COLUMNS:
            working_frame = pd.DataFrame(
                {
                    column_name: pd.to_numeric(dataset[column_name], errors="coerce"),
                    TARGET_COLUMN: dataset[TARGET_COLUMN],
                }
            ).dropna()
            working_frame["band"] = safe_quantile_bands(working_frame[column_name])
            grouped = (
                working_frame.groupby("band", dropna=False)[TARGET_COLUMN]
                .agg(["count", "mean"])
                .reset_index()
                .rename(columns={"count": "row_count", "mean": "purchase_rate"})
            )
            relationships[column_name] = grouped.to_dict(orient="records")

        return relationships

    def _collect_correlation_summary(self, dataset: pd.DataFrame) -> dict[str, Any]:
        correlation_frame = dataset[list(NUMERIC_COLUMNS) + [TARGET_COLUMN]].corr(numeric_only=True)
        target_correlations = correlation_frame[TARGET_COLUMN].drop(labels=[TARGET_COLUMN]).sort_values(ascending=False)
        return {
            "target_correlations": {
                key: round(float(value), 6) for key, value in target_correlations.items()
            }
        }

    def _collect_segment_profiles(self, dataset: pd.DataFrame) -> list[dict[str, Any]]:
        working_frame = dataset.copy()
        working_frame["age_band"] = safe_quantile_bands(pd.to_numeric(working_frame["Age"], errors="coerce"))
        working_frame["income_band"] = safe_quantile_bands(
            pd.to_numeric(working_frame["AnnualIncome"], errors="coerce")
        )

        grouped = (
            working_frame.groupby(["age_band", "income_band", "LoyaltyProgram"], dropna=False)[TARGET_COLUMN]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"count": "row_count", "mean": "purchase_rate"})
            .sort_values(["row_count", "purchase_rate"], ascending=[False, False])
            .head(10)
        )
        return grouped.to_dict(orient="records")

    def _collect_anomaly_summary(self, dataset: pd.DataFrame) -> dict[str, Any]:
        income_threshold = pd.to_numeric(dataset["AnnualIncome"], errors="coerce").quantile(0.95)
        time_threshold = pd.to_numeric(dataset["TimeSpentOnWebsite"], errors="coerce").quantile(0.95)

        high_income_mask = dataset["AnnualIncome"] >= income_threshold
        high_time_low_history_mask = (
            (dataset["TimeSpentOnWebsite"] >= time_threshold)
            & (dataset["NumberOfPurchases"] <= dataset["NumberOfPurchases"].quantile(0.25))
        )

        return {
            "high_income_row_count": int(high_income_mask.sum()),
            "high_income_purchase_rate": round(float(dataset.loc[high_income_mask, TARGET_COLUMN].mean()), 6),
            "high_time_low_history_row_count": int(high_time_low_history_mask.sum()),
            "high_time_low_history_purchase_rate": round(
                float(dataset.loc[high_time_low_history_mask, TARGET_COLUMN].mean()), 6
            ),
        }

    def _build_hypotheses(
        self,
        categorical_summary: dict[str, Any],
        numeric_target_relationships: dict[str, Any],
        segment_profiles: list[dict[str, Any]],
        anomaly_summary: dict[str, Any],
        raw_row_count: int,
        deduplicated_row_count: int,
    ) -> list[dict[str, str]]:
        loyalty_rates = categorical_summary["LoyaltyProgram"]
        loyalty_insight = "Sadakat programindaki musterilerin satin alma orani daha yuksek olabilir."
        if len(loyalty_rates) >= 2 and loyalty_rates[1]["purchase_rate"] < loyalty_rates[0]["purchase_rate"]:
            loyalty_insight = "Sadakat programi etkisi veride dogrusal olmayabilir; modelleme asamasinda dikkatle test edilmelidir."

        top_segment = segment_profiles[0] if segment_profiles else None
        top_segment_note = "Segment bazli anlamli hikaye bulunamadi."
        if top_segment:
            top_segment_note = (
                f"En yogun segmentlerden biri yas={top_segment['age_band']}, gelir={top_segment['income_band']}, "
                f"sadakat={top_segment['LoyaltyProgram']} kombinasyonudur."
            )

        return [
            {
                "hypothesis": "Sadakat programi uyeligi satin alma olasiligini artirabilir.",
                "why": loyalty_insight,
                "how_to_test": "LoyaltyProgram icin hedef oranlari ve SHAP onem sirasi birlikte incelenecek.",
            },
            {
                "hypothesis": "Gecmis satin alma sayisi yuksek olan musteriler daha yuksek donusum gosterir.",
                "why": "NumberOfPurchases bantlarina gore satin alma oranlari Faz 2 ozetinde uretildi.",
                "how_to_test": "Bant bazli EDA sonucu ile lojistik regresyon katsayisi ve SHAP etkisi karsilastirilacak.",
            },
            {
                "hypothesis": "Sitede gecirilen sure arttikca satin alma niyeti de artabilir.",
                "why": "TimeSpentOnWebsite quantile bantlarina gore hedef oranlari ayri cikarildi.",
                "how_to_test": "Quantile trendi, PDP benzeri grafik ve SHAP local explanation ile dogrulanacak.",
            },
            {
                "hypothesis": "Urun kategorileri arasinda belirgin donusum farklari vardir.",
                "why": "ProductCategory bazli satin alma oranlari ayri tablolanmistir.",
                "how_to_test": "Kategori dummy etkileri ve ağaç tabanli modellerde feature importance incelenecek.",
            },
            {
                "hypothesis": "Gelir seviyesi ile satin alma iliskisi dogrusal degil, bantli olabilir.",
                "why": "AnnualIncome icin quantile tabanli hedef oranlari hesaplandi.",
                "how_to_test": "Gelir bandi ozellikleri ile lineer ve ağaç tabanli modeller karsilastirilacak.",
            },
            {
                "hypothesis": "DiscountsAvailed guclu sinyal veriyor gibi gorunse de nedensel olarak guvenli olmayabilir.",
                "why": "Bu alan politika-duyarli olarak isaretlendi ve Faz 1'de yuksek sizinti riski aldi.",
                "how_to_test": "Ana model bu alan olmadan egitilecek; yalnizca ablation deneyinde ayri bakilacak.",
            },
            {
                "hypothesis": "Tekrar kayitlar temizlenmeden yapilan EDA belirli negatif sinif paternlerini abartabilir.",
                "why": f"Ham veri {raw_row_count}, tekillestirilmis veri {deduplicated_row_count} satirdir.",
                "how_to_test": "Ham ve tekillestirilmis dagilimler yan yana raporlanacak ve modelleme yalnizca tekil veriyle yapilacak.",
            },
            {
                "hypothesis": "Yuksek site suresi ama dusuk gecmis satin alma gecmisi olan kullanicilar kararsiz segmenti temsil ediyor olabilir.",
                "why": (
                    "Anomali ozetinde bu segment ayri sayildi ve donusum orani takip edildi. "
                    f"Gozlem sayisi: {anomaly_summary['high_time_low_history_row_count']}"
                ),
                "how_to_test": "Bu segmente ozel flag ozelligi ile esitk tabanli politika sonucu analiz edilecek.",
            },
            {
                "hypothesis": "Yuksek gelirli segmentlerde satin alma davranisi farkli bir karar mantigi gerektirebilir.",
                "why": (
                    "Yuksek gelirli satirlar ayri izlendi. "
                    f"Satir sayisi: {anomaly_summary['high_income_row_count']}"
                ),
                "how_to_test": "Gelir segmentleri icin slice-based evaluation uygulanacak.",
            },
            {
                "hypothesis": "En buyuk segmentlerin davranisi rapordaki ana is hikayesini tasiyacaktir.",
                "why": top_segment_note,
                "how_to_test": "Ilk 3 segment senaryo karti olarak Faz 9 prototipte kullanilacak.",
            },
        ]

    def _generate_plots(self, dataset: pd.DataFrame, plots_dir: Path) -> None:
        self._plot_target_distribution(dataset, plots_dir / "target_distribution.png")
        self._plot_numeric_distributions(dataset, plots_dir / "numeric_distributions.png")
        self._plot_categorical_purchase_rates(dataset, plots_dir / "categorical_purchase_rates.png")
        self._plot_numeric_purchase_rates(dataset, plots_dir / "numeric_purchase_rates.png")
        self._plot_correlation_heatmap(dataset, plots_dir / "correlation_heatmap.png")

    def _plot_target_distribution(self, dataset: pd.DataFrame, output_path: Path) -> None:
        figure, axis = plt.subplots(figsize=(8, 5))
        sns.countplot(data=dataset, x=TARGET_COLUMN, ax=axis, color="#1f77b4")
        axis.set_title("Deduplicate Sonrasi Hedef Dagilimi")
        axis.set_xlabel("PurchaseStatus")
        axis.set_ylabel("Kayit Sayisi")
        self._save_figure(figure, output_path)

    def _plot_numeric_distributions(self, dataset: pd.DataFrame, output_path: Path) -> None:
        figure, axes = plt.subplots(3, 2, figsize=(14, 12))
        axes_flat = axes.flatten()

        for axis, column_name in zip(axes_flat, NUMERIC_COLUMNS, strict=False):
            sns.histplot(dataset[column_name], kde=True, ax=axis, color="#1f77b4")
            axis.set_title(f"{column_name} Dagilimi")

        if len(axes_flat) > len(NUMERIC_COLUMNS):
            axes_flat[-1].axis("off")

        figure.tight_layout()
        self._save_figure(figure, output_path)

    def _plot_categorical_purchase_rates(self, dataset: pd.DataFrame, output_path: Path) -> None:
        figure, axes = plt.subplots(2, 2, figsize=(14, 10))
        plot_columns = list(CATEGORICAL_COLUMNS) + ["DiscountsAvailed"]

        for axis, column_name in zip(axes.flatten(), plot_columns, strict=False):
            grouped = (
                dataset.groupby(column_name, dropna=False)[TARGET_COLUMN]
                .mean()
                .reset_index(name="purchase_rate")
            )
            sns.barplot(data=grouped, x=column_name, y="purchase_rate", ax=axis, color="#2a9d8f")
            axis.set_title(f"{column_name} Bazli Satin Alma Orani")
            axis.set_ylabel("Purchase Rate")

        figure.tight_layout()
        self._save_figure(figure, output_path)

    def _plot_numeric_purchase_rates(self, dataset: pd.DataFrame, output_path: Path) -> None:
        figure, axes = plt.subplots(3, 2, figsize=(14, 12))
        axes_flat = axes.flatten()

        for axis, column_name in zip(axes_flat, NUMERIC_COLUMNS, strict=False):
            working_frame = pd.DataFrame(
                {
                    column_name: pd.to_numeric(dataset[column_name], errors="coerce"),
                    TARGET_COLUMN: dataset[TARGET_COLUMN],
                }
            ).dropna()
            working_frame["band"] = safe_quantile_bands(working_frame[column_name])
            grouped = (
                working_frame.groupby("band", dropna=False)[TARGET_COLUMN]
                .mean()
                .reset_index(name="purchase_rate")
            )
            sns.barplot(data=grouped, x="band", y="purchase_rate", ax=axis, color="#e76f51")
            axis.set_title(f"{column_name} Bantlarina Gore Satin Alma")
            axis.tick_params(axis="x", rotation=30)

        if len(axes_flat) > len(NUMERIC_COLUMNS):
            axes_flat[-1].axis("off")

        figure.tight_layout()
        self._save_figure(figure, output_path)

    def _plot_correlation_heatmap(self, dataset: pd.DataFrame, output_path: Path) -> None:
        figure, axis = plt.subplots(figsize=(10, 8))
        correlation_frame = dataset[list(NUMERIC_COLUMNS) + [TARGET_COLUMN]].corr(numeric_only=True)
        sns.heatmap(correlation_frame, annot=True, cmap="coolwarm", fmt=".2f", ax=axis)
        axis.set_title("Sayisal Degisken Korelasyon Isı Haritasi")
        self._save_figure(figure, output_path)

    def _save_figure(self, figure: plt.Figure, output_path: Path) -> None:
        try:
            figure.savefig(output_path, dpi=200, bbox_inches="tight")
        except OSError as error:
            raise ArtifactWriteError(f"Grafik kaydedilemedi: {output_path}") from error
        finally:
            plt.close(figure)

    def _write_json_summary(self, output_path: Path, summary: dict[str, Any]) -> None:
        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                json.dump(to_json_safe(summary), output_file, ensure_ascii=False, indent=2)
        except OSError as error:
            raise ArtifactWriteError(f"Faz 2 JSON ozeti yazilamadi: {output_path}") from error

    def _write_markdown_report(self, output_path: Path, summary: dict[str, Any]) -> None:
        lines = [
            "# Faz 2 Aciklayici Veri Analizi Raporu",
            "",
            f"- Veri seti: `{summary['dataset_path']}`",
            f"- Ham satir sayisi: {summary['raw_row_count']}",
            f"- Tekillestirilmis satir sayisi: {summary['deduplicated_row_count']}",
            f"- Dusurulen tekrar sayisi: {summary['dropped_duplicate_count']}",
            "",
        ]

        if summary["phase1_context"]:
            lines.extend(
                [
                    "## Faz 1 Baglami",
                    "",
                    f"- Faz 1 karari: {summary['phase1_context']['final_decision']}",
                    f"- Faz 1 gerekcesi: {summary['phase1_context']['decision_rationale']}",
                    f"- Faz 1 tekrar orani: {summary['phase1_context']['duplicate_rate']}",
                    "",
                ]
            )

        lines.extend(
            [
                "## Hedef Gorunumu",
                "",
                f"- Ham sinif dagilimi: {summary['target_summary']['raw_counts']}",
                f"- Tekillestirilmis sinif dagilimi: {summary['target_summary']['deduplicated_counts']}",
                f"- Tekillestirilmis sinif oranlari: {summary['target_summary']['deduplicated_rates']}",
                "",
                "## Sayisal Tek Degiskenli Ozet",
                "",
            ]
        )

        for column_name, column_summary in summary["numeric_univariate_summary"].items():
            lines.append(f"- {column_name}: {column_summary}")

        lines.extend(["", "## Kategorik Ozetler", ""])
        for column_name, rows in summary["categorical_summary"].items():
            lines.append(f"### {column_name}")
            for row in rows:
                lines.append(
                    f"- deger={row[column_name]}, kayit={row['row_count']}, satin_alma_orani={round(float(row['purchase_rate']), 6)}"
                )
            lines.append("")

        lines.extend(["## Hedef Ile Iliski Ozeti", ""])
        for column_name, rows in summary["numeric_target_relationships"].items():
            lines.append(f"### {column_name}")
            for row in rows:
                lines.append(
                    f"- bant={row['band']}, kayit={row['row_count']}, satin_alma_orani={round(float(row['purchase_rate']), 6)}"
                )
            lines.append("")

        lines.extend(["## Korelasyon Ozeti", ""])
        for column_name, correlation in summary["correlation_summary"]["target_correlations"].items():
            lines.append(f"- {column_name}: {correlation}")

        lines.extend(["", "## Segment Profilleri", ""])
        for row in summary["segment_profiles"]:
            lines.append(
                f"- yas={row['age_band']}, gelir={row['income_band']}, sadakat={row['LoyaltyProgram']}, kayit={row['row_count']}, satin_alma_orani={round(float(row['purchase_rate']), 6)}"
            )

        lines.extend(
            [
                "",
                "## Anomali Ozeti",
                "",
                f"- Yuksek gelirli satir sayisi: {summary['anomaly_summary']['high_income_row_count']}",
                f"- Yuksek gelirli satin alma orani: {summary['anomaly_summary']['high_income_purchase_rate']}",
                f"- Yuksek sure dusuk gecmis satir sayisi: {summary['anomaly_summary']['high_time_low_history_row_count']}",
                f"- Yuksek sure dusuk gecmis satin alma orani: {summary['anomaly_summary']['high_time_low_history_purchase_rate']}",
                "",
                "## Hipotez Katalogu",
                "",
            ]
        )

        for item in summary["hypotheses"]:
            lines.append(f"- Hipotez: {item['hypothesis']}")
            lines.append(f"  - Neden: {item['why']}")
            lines.append(f"  - Nasil test edilecek: {item['how_to_test']}")

        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                output_file.write("\n".join(lines))
        except OSError as error:
            raise ArtifactWriteError(f"Faz 2 markdown raporu yazilamadi: {output_path}") from error
