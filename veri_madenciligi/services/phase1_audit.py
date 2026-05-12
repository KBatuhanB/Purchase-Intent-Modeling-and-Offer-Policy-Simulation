from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import (
    CATEGORICAL_COLUMNS,
    EXPECTED_COLUMNS,
    INTEGER_LIKE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    VALID_TARGET_VALUES,
    ProjectPaths,
)
from ..core.exceptions import ArtifactWriteError, SchemaValidationError
from ..core.serialization import to_json_safe
from .csv_loader import CsvDatasetLoader


@dataclass(frozen=True)
class Phase1AuditArtifacts:
    markdown_report_path: Path
    json_summary_path: Path


def normalize_binary_like_values(series: pd.Series) -> list[Any]:
    normalized_values: set[Any] = set()

    for raw_value in series.dropna().tolist():
        if isinstance(raw_value, str):
            candidate = raw_value.strip()
        else:
            candidate = raw_value

        if candidate in {0, 0.0, "0", "0.0"}:
            normalized_values.add(0)
        elif candidate in {1, 1.0, "1", "1.0"}:
            normalized_values.add(1)
        else:
            normalized_values.add(str(candidate))

    return sorted(normalized_values, key=str)


def determine_go_decision(
    schema_issues: bool,
    target_valid: bool,
    duplicate_rate: float,
    has_high_risk_feature: bool,
) -> str:
    if schema_issues or not target_valid:
        return "No Go"
    if duplicate_rate > 0 or has_high_risk_feature:
        return "Conditional Go"
    return "Go"


class Phase1AuditService:
    def __init__(self, loader: CsvDatasetLoader, logger: logging.Logger) -> None:
        self._loader = loader
        self._logger = logger

    def run(self, project_paths: ProjectPaths) -> Phase1AuditArtifacts:
        dataset = self._loader.load(project_paths.dataset_path)
        self._logger.info("Faz 1 veri denetimi baslatildi: %s", project_paths.dataset_path)

        summary = self._build_summary(dataset=dataset, dataset_path=project_paths.dataset_path)
        artifacts = Phase1AuditArtifacts(
            markdown_report_path=project_paths.phase_1_dir / "audit_report.md",
            json_summary_path=project_paths.phase_1_dir / "audit_summary.json",
        )

        self._write_json_summary(artifacts.json_summary_path, summary)
        self._write_markdown_report(artifacts.markdown_report_path, summary)
        self._logger.info("Faz 1 veri denetimi tamamlandi.")
        return artifacts

    def _build_summary(self, dataset: pd.DataFrame, dataset_path: Path) -> dict[str, Any]:
        schema_summary = self._collect_schema_summary(dataset)
        target_summary = self._collect_target_summary(dataset)
        missing_summary = self._collect_missing_summary(dataset)
        duplicate_summary = self._collect_duplicate_summary(dataset)
        numeric_checks = self._collect_numeric_checks(dataset)
        leakage_assessment = self._collect_leakage_assessment()

        decision = determine_go_decision(
            schema_issues=bool(schema_summary["missing_columns"] or schema_summary["unexpected_columns"]),
            target_valid=target_summary["is_binary_and_complete"],
            duplicate_rate=duplicate_summary["duplicate_rate"],
            has_high_risk_feature=any(
                row["risk_level"] == "high" for row in leakage_assessment["entries"]
            ),
        )

        if not target_summary["is_binary_and_complete"]:
            raise SchemaValidationError(
                "PurchaseStatus kolonu beklenen 0/1 ikili etiket yapisini saglamiyor."
            )

        return {
            "dataset_path": str(dataset_path),
            "row_count": int(dataset.shape[0]),
            "column_count": int(dataset.shape[1]),
            "schema_summary": schema_summary,
            "target_summary": target_summary,
            "missing_summary": missing_summary,
            "duplicate_summary": duplicate_summary,
            "numeric_checks": numeric_checks,
            "leakage_assessment": leakage_assessment,
            "final_decision": decision,
            "decision_rationale": self._build_decision_rationale(decision, duplicate_summary["duplicate_rate"]),
        }

    def _collect_schema_summary(self, dataset: pd.DataFrame) -> dict[str, Any]:
        actual_columns = tuple(dataset.columns.tolist())
        missing_columns = sorted(set(EXPECTED_COLUMNS) - set(actual_columns))
        unexpected_columns = sorted(set(actual_columns) - set(EXPECTED_COLUMNS))

        column_profiles: list[dict[str, Any]] = []
        for column_name in dataset.columns:
            series = dataset[column_name]
            profile = {
                "column": column_name,
                "dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "null_count": int(series.isna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
            }

            if column_name in CATEGORICAL_COLUMNS or column_name == TARGET_COLUMN:
                preview_values = series.dropna().astype(str).unique().tolist()[:10]
                profile["preview_values"] = preview_values
            else:
                numeric_series = pd.to_numeric(series, errors="coerce")
                profile["min"] = float(numeric_series.min())
                profile["max"] = float(numeric_series.max())

            column_profiles.append(profile)

        return {
            "expected_columns": list(EXPECTED_COLUMNS),
            "actual_columns": list(actual_columns),
            "missing_columns": missing_columns,
            "unexpected_columns": unexpected_columns,
            "column_profiles": column_profiles,
        }

    def _collect_target_summary(self, dataset: pd.DataFrame) -> dict[str, Any]:
        target_series = dataset[TARGET_COLUMN]
        normalized_values = normalize_binary_like_values(target_series)
        target_counts = (
            target_series.value_counts(dropna=False)
            .sort_index()
            .rename_axis("label")
            .reset_index(name="count")
        )

        counts = {
            str(record["label"]): int(record["count"])
            for record in target_counts.to_dict(orient="records")
        }
        rates = {
            key: round(value / len(dataset), 6)
            for key, value in counts.items()
        }

        is_binary_and_complete = set(normalized_values) == VALID_TARGET_VALUES and target_series.notna().all()

        return {
            "normalized_values": normalized_values,
            "counts": counts,
            "rates": rates,
            "missing_target_count": int(target_series.isna().sum()),
            "is_binary_and_complete": is_binary_and_complete,
        }

    def _collect_missing_summary(self, dataset: pd.DataFrame) -> dict[str, Any]:
        missing_by_column = dataset.isna().sum().to_dict()
        missing_ratio_by_column = {
            column_name: round(count / len(dataset), 6)
            for column_name, count in missing_by_column.items()
        }

        return {
            "total_missing_cells": int(dataset.isna().sum().sum()),
            "rows_with_missing_values": int(dataset.isna().any(axis=1).sum()),
            "missing_by_column": {key: int(value) for key, value in missing_by_column.items()},
            "missing_ratio_by_column": missing_ratio_by_column,
            "fallback_policy": "Ana veri setinde imputasyon gerekmiyor; manuel senaryo girislerinde zorunlu alan validasyonu uygulanacak.",
        }

    def _collect_duplicate_summary(self, dataset: pd.DataFrame) -> dict[str, Any]:
        duplicate_mask = dataset.duplicated(keep=False)
        duplicate_rows = dataset[duplicate_mask]
        duplicate_row_count = int(dataset.duplicated().sum())

        preview_records = duplicate_rows.head(5).to_dict(orient="records")
        duplicate_target_counts = {}
        if not duplicate_rows.empty:
            duplicate_target_counts = {
                str(key): int(value)
                for key, value in duplicate_rows[TARGET_COLUMN].value_counts().to_dict().items()
            }

        duplicate_rate = round(duplicate_row_count / len(dataset), 6)

        return {
            "duplicate_row_count": duplicate_row_count,
            "duplicate_rate": duplicate_rate,
            "duplicate_target_counts": duplicate_target_counts,
            "preview_records": preview_records,
            "policy": "Tam tekrar satirlar train-test bolunmesinden once tekillestirilecek ve temizlenmis veri ana modelleme girdisi olacak.",
        }

    def _collect_numeric_checks(self, dataset: pd.DataFrame) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []

        for column_name in NUMERIC_COLUMNS:
            numeric_series = pd.to_numeric(dataset[column_name], errors="coerce")
            suspicious_mask = pd.Series(False, index=dataset.index)

            if column_name == "Age":
                suspicious_mask = suspicious_mask | (numeric_series < 0) | (numeric_series > 120)
            elif column_name == "AnnualIncome":
                suspicious_mask = suspicious_mask | (numeric_series < 0)
            elif column_name == "NumberOfPurchases":
                suspicious_mask = suspicious_mask | (numeric_series < 0)
            elif column_name == "TimeSpentOnWebsite":
                suspicious_mask = suspicious_mask | (numeric_series < 0)
            elif column_name == "DiscountsAvailed":
                suspicious_mask = suspicious_mask | (numeric_series < 0)

            integer_like_violation_count = 0
            if column_name in INTEGER_LIKE_COLUMNS:
                non_null_series = numeric_series.dropna()
                integer_like_violation_count = int((non_null_series % 1 != 0).sum())

            checks.append(
                {
                    "column": column_name,
                    "min": round(float(numeric_series.min()), 6),
                    "max": round(float(numeric_series.max()), 6),
                    "mean": round(float(numeric_series.mean()), 6),
                    "suspicious_value_count": int(suspicious_mask.sum()),
                    "integer_like_violation_count": integer_like_violation_count,
                }
            )

        return {"entries": checks}

    def _collect_leakage_assessment(self) -> dict[str, Any]:
        # Bu matris, hangi alanin hangi anda guvenle kullanilabilecegini erken sabitler.
        entries = [
            {
                "column": "Age",
                "availability_stage": "pre_session",
                "risk_level": "low",
                "policy": "Ana modelde guvenle kullanilabilir.",
                "rationale": "Demografik profil bilgisidir ve tahmin anindan once bilinebilir.",
            },
            {
                "column": "Gender",
                "availability_stage": "pre_session",
                "risk_level": "low",
                "policy": "Ana modelde guvenle kullanilabilir.",
                "rationale": "Demografik profil bilgisidir ve sonuc degiskeni degildir.",
            },
            {
                "column": "AnnualIncome",
                "availability_stage": "pre_session",
                "risk_level": "low",
                "policy": "Ana modelde guvenle kullanilabilir.",
                "rationale": "Musteri profili bilgisidir ve kampanya cikisina bagli degildir.",
            },
            {
                "column": "NumberOfPurchases",
                "availability_stage": "pre_session",
                "risk_level": "low",
                "policy": "Ana modelde guvenle kullanilabilir.",
                "rationale": "Gecmis davranis bilgisidir ve tahmin anindan once mevcuttur.",
            },
            {
                "column": "ProductCategory",
                "availability_stage": "session_context",
                "risk_level": "medium",
                "policy": "Ana modelde kullanilabilir ancak kategori anlami belgelenmelidir.",
                "rationale": "Urun baglami sinyali tasir; fakat kodlamanin anlami sozluk ile dogrulanmalidir.",
            },
            {
                "column": "TimeSpentOnWebsite",
                "availability_stage": "session_end_proxy",
                "risk_level": "medium",
                "policy": "Ana modelde kullanilabilir, erken mudahale modeli icin ayrica gozden gecirilmelidir.",
                "rationale": "Site suresi guclu sinyal tasir fakat oturum sonuna yakin hesaplandigi varsayilabilir.",
            },
            {
                "column": "LoyaltyProgram",
                "availability_stage": "pre_session",
                "risk_level": "medium",
                "policy": "Ana modelde kullanilabilir, ancak kampanya politikasi ile karismamasi icin belgeye not dusulmelidir.",
                "rationale": "Onceden bilinebilen musteri statusu olabilir; yine de ticari politika ile iliskisi tartisilmalidir.",
            },
            {
                "column": "DiscountsAvailed",
                "availability_stage": "post_intervention_or_leaky",
                "risk_level": "high",
                "policy": "Ana modelden varsayilan olarak dislanacak, yalnizca kontrollu karsilastirma analizinde ele alinacak.",
                "rationale": "Gecmis indirim kullanimi, mudahale sonucu veya politika degiskeni olabilir ve sahte performans uretebilir.",
            },
            {
                "column": "PurchaseStatus",
                "availability_stage": "target",
                "risk_level": "high",
                "policy": "Yalnizca hedef degisken olarak tutulur.",
                "rationale": "Modelin tahmin etmeye calistigi sonuc etiketidir.",
            },
        ]

        return {"entries": entries}

    def _build_decision_rationale(self, decision: str, duplicate_rate: float) -> str:
        if decision == "No Go":
            return "Sema veya hedef butunlugu kirik oldugu icin modelleme fazina gecilmemelidir."
        if decision == "Conditional Go":
            return (
                "Veri seti temel olarak kullanilabilir; ancak tekrar satirlar temizlenmeden ve "
                f"DiscountsAvailed ana modelden cikarilmadan ilerlenmemelidir. Tekrar oranı: {duplicate_rate:.2%}."
            )
        return "Veri seti Faz 3'e dogrudan gecis icin yeterince temiz gorunmektedir."

    def _write_json_summary(self, output_path: Path, summary: dict[str, Any]) -> None:
        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                json.dump(to_json_safe(summary), output_file, ensure_ascii=False, indent=2)
        except OSError as error:
            raise ArtifactWriteError(f"JSON ozeti yazilamadi: {output_path}") from error

    def _write_markdown_report(self, output_path: Path, summary: dict[str, Any]) -> None:
        markdown_lines = [
            "# Faz 1 Veri Denetim Raporu",
            "",
            f"- Veri seti: `{summary['dataset_path']}`",
            f"- Satir sayisi: {summary['row_count']}",
            f"- Sutun sayisi: {summary['column_count']}",
            f"- Nihai karar: **{summary['final_decision']}**",
            f"- Karar gerekcesi: {summary['decision_rationale']}",
            "",
            "## Sema Ozeti",
            "",
            f"- Eksik sutunlar: {summary['schema_summary']['missing_columns'] or 'Yok'}",
            f"- Beklenmeyen sutunlar: {summary['schema_summary']['unexpected_columns'] or 'Yok'}",
            "",
            "### Sutun Profilleri",
            "",
        ]

        for profile in summary["schema_summary"]["column_profiles"]:
            markdown_lines.append(
                f"- {profile['column']}: dtype={profile['dtype']}, bos olmayan={profile['non_null_count']}, benzersiz={profile['unique_count']}"
            )

        markdown_lines.extend(
            [
                "",
                "## Hedef Degisken Butunlugu",
                "",
                f"- Normalize etiket degerleri: {summary['target_summary']['normalized_values']}",
                f"- Eksik hedef sayisi: {summary['target_summary']['missing_target_count']}",
                f"- Sinif dagilimi: {summary['target_summary']['counts']}",
                f"- Sinif oranlari: {summary['target_summary']['rates']}",
                f"- Ikili ve tam etiket durumu: {summary['target_summary']['is_binary_and_complete']}",
                "",
                "## Eksik Veri Denetimi",
                "",
                f"- Toplam eksik hucre: {summary['missing_summary']['total_missing_cells']}",
                f"- Eksik veri iceren satir sayisi: {summary['missing_summary']['rows_with_missing_values']}",
                f"- Politika: {summary['missing_summary']['fallback_policy']}",
                "",
                "## Tekrar Kayit Denetimi",
                "",
                f"- Tekrar satir sayisi: {summary['duplicate_summary']['duplicate_row_count']}",
                f"- Tekrar orani: {summary['duplicate_summary']['duplicate_rate']}",
                f"- Tekrar satirlardaki hedef dagilimi: {summary['duplicate_summary']['duplicate_target_counts']}",
                f"- Politika: {summary['duplicate_summary']['policy']}",
                "",
                "## Sayisal Mantik Kontrolleri",
                "",
            ]
        )

        for entry in summary["numeric_checks"]["entries"]:
            markdown_lines.append(
                f"- {entry['column']}: min={entry['min']}, max={entry['max']}, ortalama={entry['mean']}, supheli_deger={entry['suspicious_value_count']}"
            )

        markdown_lines.extend(["", "## Sizinti ve Erisilebilirlik Matrisi", ""])
        for entry in summary["leakage_assessment"]["entries"]:
            markdown_lines.append(
                f"- {entry['column']}: asama={entry['availability_stage']}, risk={entry['risk_level']}, politika={entry['policy']}"
            )

        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                output_file.write("\n".join(markdown_lines))
        except OSError as error:
            raise ArtifactWriteError(f"Markdown raporu yazilamadi: {output_path}") from error

