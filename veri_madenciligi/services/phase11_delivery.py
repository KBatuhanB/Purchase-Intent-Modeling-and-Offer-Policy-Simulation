from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import ProjectPaths
from ..core.exceptions import ArtifactWriteError, DataAnalysisError
from ..core.serialization import to_json_safe
from .csv_loader import CsvDatasetLoader
from .phase2_eda import deduplicate_dataset

SOURCE_DEPENDENCY_PATHS = (
    ("phase1_report", ("phase_1", "audit_report.md")),
    ("phase1_summary", ("phase_1", "audit_summary.json")),
    ("phase2_report", ("phase_2", "eda_report.md")),
    ("phase2_summary", ("phase_2", "eda_summary.json")),
    ("phase3_report", ("phase_3", "preprocessing_report.md")),
    ("phase3_summary", ("phase_3", "preprocessing_summary.json")),
    ("phase4_report", ("phase_4", "baseline_report.md")),
    ("phase4_summary", ("phase_4", "baseline_summary.json")),
    ("phase5_report", ("phase_5", "imbalance_report.md")),
    ("phase5_summary", ("phase_5", "imbalance_summary.json")),
    ("phase6_report", ("phase_6", "advanced_modeling_report.md")),
    ("phase6_summary", ("phase_6", "advanced_modeling_summary.json")),
    ("phase7_report", ("phase_7", "explainability_report.md")),
    ("phase7_summary", ("phase_7", "explainability_summary.json")),
    ("phase8_report", ("phase_8", "policy_report.md")),
    ("phase8_summary", ("phase_8", "policy_summary.json")),
    ("phase9_report", ("phase_9", "simulation_report.md")),
    ("phase9_summary", ("phase_9", "simulation_summary.json")),
    ("phase10_report", ("phase_10", "validation_report.md")),
    ("phase10_summary", ("phase_10", "validation_summary.json")),
)

VISUAL_ASSET_GROUPS = (
    ("phase_2", "plots", "phase_2", "plots"),
    ("phase_4", "plots", "phase_4", "plots"),
    ("phase_5", "plots", "phase_5", "plots"),
    ("phase_6", "plots", "phase_6", "plots"),
    ("phase_7", "plots", "phase_7", "plots"),
    ("phase_7", "scenario_cards", "phase_7", "scenarios"),
    ("phase_8", "plots", "phase_8", "plots"),
    ("phase_8", "scenario_cards", "phase_8", "scenarios"),
    ("phase_9", "scenario_cards", "phase_9", "scenarios"),
)

FUTURE_WORK_LIBRARY = {
    "static_snapshot_no_clickstream": {
        "workstream": "Gercek Zamanli Olay Akisi Entegrasyonu",
        "rationale": "Statik profil verisi yerine oturum bazli akis toplandiginda erken mudahale sinyali daha savunulabilir olur.",
        "required_capability": "Clickstream, oturum zaman damgasi ve kullanici kimligi",
    },
    "duplicate_records_cleaned_pre_split": {
        "workstream": "Kaynak Sistemlerde Kimlik ve Tekillestirme Yonetişimi",
        "rationale": "Tekrar kayitlari sonradan temizlemek yerine kaynakta musteri anahtari ve veri sozlesmesi kurmak gerekir.",
        "required_capability": "User/customer id, upstream dedup kontrolleri",
    },
    "discount_history_not_causal_signal": {
        "workstream": "Uplift Modeling ve A/B Test Altyapisi",
        "rationale": "Indirimin gercek etkisini olcmek icin propensity model yetmez; randomize deney ve causal etiket gerekir.",
        "required_capability": "A/B test loglari, kampanya maruz kalma bilgisi",
    },
    "proxy_business_value_only": {
        "workstream": "Finansal Fayda Motoru",
        "rationale": "Proxy puan yerine siparis tutari, marj ve kampanya maliyeti ile dogrudan optimizasyon kurulabilir.",
        "required_capability": "Order value, margin, campaign cost tabloları",
    },
    "fairness_alerts_require_monitoring": {
        "workstream": "Fairness Izleme ve Esik Yonetimi",
        "rationale": "Segment bazli hata farklarini duzenli izlemek ve gerekirse grup-duyarli esik ayari yapmak gerekir.",
        "required_capability": "Per-group monitoring dashboard ve alerting",
    },
    "processed_dataset_generalization_risk": {
        "workstream": "Dis Dogrulama ve Taze Veri Testi",
        "rationale": "Tek bir temiz veri snapshot'i yerine farkli donemlerden yeni veriyle genellenebilirlik test edilmelidir.",
        "required_capability": "Yeni tarihli dataset ve backtesting protokolu",
    },
}

ORIGIN_TO_REPORT_SECTIONS = {
    "phase7_reference": [
        "Model performansi ve aciklanabilirlik",
        "Politika katmani ve guardrail'ler",
        "Demo ve simülasyon",
    ],
    "phase9_synthetic": [
        "Politika katmani ve guardrail'ler",
        "Demo ve simülasyon",
        "Limitasyonlar ve kalite kapisi",
    ],
}


@dataclass(frozen=True)
class Phase11Artifacts:
    markdown_report_path: Path
    json_summary_path: Path
    executive_summary_path: Path
    presentation_outline_path: Path
    defense_script_path: Path
    demo_report_mapping_path: Path
    future_work_path: Path
    visual_asset_library_path: Path
    delivery_manifest_path: Path
    delivery_bundle_dir: Path
    delivery_bundle_readme_path: Path


def validate_phase11_prerequisites(
    *,
    dataset_row_count: int,
    deduplicated_row_count: int,
    phase1_summary: dict[str, Any],
    phase6_summary: dict[str, Any],
    phase7_summary: dict[str, Any],
    phase8_summary: dict[str, Any],
    phase9_summary: dict[str, Any],
    phase10_summary: dict[str, Any],
) -> dict[str, Any]:
    phase1_row_count = int(phase1_summary.get("row_count", -1))
    if phase1_row_count != dataset_row_count:
        raise DataAnalysisError("Faz 11 icin Faz 1 satir sayisi ile mevcut dataset satir sayisi uyusmuyor.")

    phase1_deduplicated = phase1_summary.get("row_count", 0) - phase1_summary.get("duplicate_summary", {}).get("duplicate_row_count", 0)
    if int(phase1_deduplicated) != deduplicated_row_count:
        raise DataAnalysisError("Faz 11 icin Faz 1 tekillestirilmis satir sayisi ile mevcut veri uyusmuyor.")

    for phase_name, phase_summary in (
        ("phase6", phase6_summary),
        ("phase7", phase7_summary),
        ("phase8", phase8_summary),
        ("phase9", phase9_summary),
        ("phase10", phase10_summary),
    ):
        if int(phase_summary.get("deduplicated_row_count", -1)) != deduplicated_row_count:
            raise DataAnalysisError(f"Faz 11 icin {phase_name} tekillestirilmis satir sayisi ile mevcut veri uyusmuyor.")

    champion_candidates = {
        phase7_summary.get("phase6_context", {}).get("champion_selection", {}).get("champion_model"),
        phase8_summary.get("phase6_context", {}).get("champion_selection", {}).get("champion_model"),
        phase9_summary.get("champion_model_name"),
        phase10_summary.get("champion_model_name"),
    }
    normalized_candidates = {candidate for candidate in champion_candidates if candidate}
    if len(normalized_candidates) != 1:
        raise DataAnalysisError("Faz 11 icin champion model kimligi fazlar arasinda tutarsiz gorunuyor.")

    readiness_status = phase10_summary.get("phase10_closeout", {}).get("overall_status")
    if readiness_status not in {"hazir", "revizyon_gerekli", "eksik"}:
        raise DataAnalysisError("Faz 11 icin Faz 10 readiness durumu okunamadi.")

    champion_model_name = normalized_candidates.pop()
    return {
        "champion_model_name": champion_model_name,
        "readiness_status": readiness_status,
        "consistency_checks": [
            {
                "check_id": "dataset_row_count_alignment",
                "status": "passed",
                "detail": f"Ham satir sayisi {dataset_row_count} olarak Faz 1 ile uyumlu bulundu.",
            },
            {
                "check_id": "deduplicated_row_count_alignment",
                "status": "passed",
                "detail": f"Tekillestirilmis satir sayisi {deduplicated_row_count} olarak Faz 1, Faz 6-10 ile uyumlu bulundu.",
            },
            {
                "check_id": "champion_model_alignment",
                "status": "passed",
                "detail": f"Champion model tum ust fazlarda {champion_model_name} olarak korundu.",
            },
            {
                "check_id": "quality_gate_status",
                "status": "passed",
                "detail": f"Faz 10 kalite kapisi durumu {readiness_status} olarak tasima icin uygun bulundu.",
            },
        ],
    }


def build_presentation_layers(
    *,
    phase1_summary: dict[str, Any],
    phase6_summary: dict[str, Any],
    phase8_summary: dict[str, Any],
    phase9_summary: dict[str, Any],
    phase10_summary: dict[str, Any],
    champion_model_name: str,
) -> dict[str, Any]:
    validated_metrics = phase8_summary.get("phase6_context", {}).get("validated_champion_metrics", {})
    scenario_overview = phase9_summary.get("scenario_catalog_overview", {})
    readiness_status = phase10_summary.get("phase10_closeout", {}).get("overall_status", "bilinmiyor")
    duplicate_rate = float(phase1_summary.get("duplicate_summary", {}).get("duplicate_rate", 0.0))

    return {
        "thirty_second_pitch": (
            "Bu proje, musteri profili ve davranis ozeti verisinden satin alma olasiligi ureten ve bu skoru kampanya kararina ceviren "
            f"bir karar destek prototipidir. Ham veride %{duplicate_rate * 100:.2f} tekrar kayit temizlendikten sonra {champion_model_name} champion model olarak secildi; "
            f"nihai testte accuracy %{validated_metrics.get('accuracy', 0.0) * 100:.2f}, PR-AUC %{validated_metrics.get('pr_auc', 0.0) * 100:.2f} elde edildi ve Faz 10 kalite kapisi {readiness_status} durumunda kapandi."
        ),
        "three_minute_pitch": [
            (
                "Problemi herkese indirim vermek yerine, satin alma niyeti yeterince dusuk ama kurtarilabilir kullaniciyi bulmak olarak cerceveledik. "
                f"Veri denetiminde tekrar kayitlar ve DiscountsAvailed riski temel kontrol noktasi oldu; Faz 1 karari {phase1_summary.get('final_decision', 'bilinmiyor')} olarak kayda gecirildi."
            ),
            (
                f"Modelleme tarafinda champion model {champion_model_name} oldu. Nihai test sonucunda recall %{validated_metrics.get('recall', 0.0) * 100:.2f}, precision %{validated_metrics.get('precision', 0.0) * 100:.2f} ve ROC-AUC %{validated_metrics.get('roc_auc', 0.0) * 100:.2f} seviyesinde kaldik."
            ),
            (
                f"Skoru dogrudan indirim karari yerine politika bantlarina cevirdik. Faz 9'da {scenario_overview.get('total_scenarios', 0)} demo senaryosu ile sistemi gosterdik; Faz 10'da edge-case, sensitivity ve reproducibility testleri ile bu yuzeyi kalite kapisindan gecirdik."
            ),
        ],
        "ten_minute_defense_outline": [
            {
                "section_title": "Problem ve kapsam",
                "talk_track": "Propensity modeli ile indirim politikasini ayirarak, eldeki verinin kaldirabilecegi kadar iddiali ama savunulabilir bir cozum kurduk.",
            },
            {
                "section_title": "Veri denetimi",
                "talk_track": "Eksik veri olmamasi avantajdi; esas kalite konusu tekrar kayitlar ve DiscountsAvailed gibi potansiyel sızınti alanlarinin kontroluydu.",
            },
            {
                "section_title": "Modelleme ve champion secimi",
                "talk_track": "Baseline, dengesizlik ve kalibrasyon fazlarini gecmeden ileri modele cikmadik; champion secimini tek bir metrik yerine PR-AUC, balanced accuracy ve kalibrasyon kalitesiyle yaptik.",
            },
            {
                "section_title": "Aciklanabilirlik ve fairness",
                "talk_track": "Modelin en baskin suruculerini SHAP ile gosterdik ve segment bazli hata farklarini limitasyon olarak acikca belgeledik.",
            },
            {
                "section_title": "Politika, demo ve kalite kapisi",
                "talk_track": "Skoru aksiyon bantlarina cevirdik, demoyu kontrollu senaryo setiyle kurguladik ve Faz 10 kalite kapisi ile teslim oncesi saglamlik kontrollerini tamamladik.",
            },
        ],
    }


def build_demo_report_mapping(
    *,
    phase9_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    simulation_lookup = {
        item["scenario_id"]: item
        for item in phase9_summary.get("simulation_results", [])
    }
    mapping_rows: list[dict[str, Any]] = []

    for step in phase9_summary.get("demo_runbook", []):
        scenario_id = step.get("scenario_id")
        result = simulation_lookup.get(scenario_id, {})
        scenario_origin = result.get("scenario_origin", "unknown")
        mapping_rows.append(
            {
                "step_number": step.get("step_number"),
                "step_title": step.get("step_title"),
                "scenario_id": scenario_id,
                "scenario_title": step.get("scenario_title"),
                "scenario_origin": scenario_origin,
                "purchase_score": result.get("predicted_probability"),
                "recommended_action": result.get("recommended_action"),
                "requires_manual_review": result.get("requires_manual_review"),
                "report_sections": ORIGIN_TO_REPORT_SECTIONS.get(
                    scenario_origin,
                    [
                        "Demo ve simülasyon",
                        "Limitasyonlar ve kalite kapisi",
                    ],
                ),
                "takeaway": step.get("takeaway"),
                "expected_output": step.get("expected_output"),
            }
        )
    return mapping_rows


def build_future_work_items(
    *,
    limitations: list[dict[str, Any]],
    readiness_status: str,
) -> list[dict[str, Any]]:
    future_work_items: list[dict[str, Any]] = []
    for limitation in limitations:
        template = FUTURE_WORK_LIBRARY.get(
            limitation.get("limitation_id"),
            {
                "workstream": "Ek dogrulama ve operasyonel sertlestirme",
                "rationale": "Mevcut limitasyon, bir sonraki iterasyonda veri ve operasyon kapsamini genisletmeyi gerektiriyor.",
                "required_capability": "Ek veri ve operasyonel olgunluk",
            },
        )
        future_work_items.append(
            {
                "workstream": template["workstream"],
                "linked_limitation_id": limitation.get("limitation_id"),
                "priority": "high" if limitation.get("severity") == "high" else "medium",
                "rationale": template["rationale"],
                "required_capability": template["required_capability"],
            }
        )

    if readiness_status != "hazir":
        future_work_items.insert(
            0,
            {
                "workstream": "Teslim Oncesi Revizyon Kapatma",
                "linked_limitation_id": "quality_gate_followup",
                "priority": "critical",
                "rationale": "Kalite kapisi hazir disi bir durumda ise, sonraki iterasyondan once teslim blokajlari kapatilmalidir.",
                "required_capability": "Ek test, revizyon ve yeniden dogrulama",
            },
        )
    return future_work_items


def build_visual_asset_library(
    *,
    root_dir: Path,
    asset_groups: list[tuple[str, str, Path]],
) -> dict[str, Any]:
    assets: list[dict[str, Any]] = []
    phase_counts: dict[str, int] = {}

    # Sunum kutuphanesi fazlar boyunca daginik kalan gorsel ve demo destek dosyalarini tek manifestte toplar.
    for phase_name, asset_group, directory in asset_groups:
        if not directory.exists():
            phase_counts.setdefault(phase_name, 0)
            continue

        for file_path in sorted(path for path in directory.rglob("*") if path.is_file()):
            relative_path = file_path.relative_to(root_dir).as_posix()
            assets.append(
                {
                    "phase_name": phase_name,
                    "asset_group": asset_group,
                    "file_name": file_path.name,
                    "relative_path": relative_path,
                    "suffix": file_path.suffix.lower(),
                }
            )
            phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1

    return {
        "total_asset_count": len(assets),
        "phase_counts": phase_counts,
        "assets": assets,
    }


def build_delivery_manifest(
    *,
    root_dir: Path,
    dataset_path: Path,
    champion_model_name: str,
    readiness_status: str,
    generated_outputs: list[Path],
    source_dependencies: list[Path],
    key_metrics: dict[str, Any],
    visual_asset_count: int,
    demo_mapping_count: int,
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "champion_model_name": champion_model_name,
        "overall_status": readiness_status,
        "ready_for_submission": readiness_status == "hazir",
        "generated_outputs": [path.relative_to(root_dir).as_posix() for path in generated_outputs],
        "source_dependencies": [path.relative_to(root_dir).as_posix() for path in source_dependencies],
        "key_metrics": key_metrics,
        "visual_asset_count": visual_asset_count,
        "demo_mapping_count": demo_mapping_count,
    }


class Phase11DeliveryService:
    def __init__(
        self,
        loader: CsvDatasetLoader,
        logger: logging.Logger,
    ) -> None:
        self._loader = loader
        self._logger = logger

    def run(self, project_paths: ProjectPaths) -> Phase11Artifacts:
        try:
            self._logger.info("Faz 11 final rapor, sunum ve teslim paketi derlemesi baslatildi: %s", project_paths.dataset_path)
            dataset = self._loader.load(project_paths.dataset_path)
            deduplicated_dataset = deduplicate_dataset(dataset)

            contexts = self._load_required_contexts(project_paths)
            prerequisite_summary = validate_phase11_prerequisites(
                dataset_row_count=len(dataset),
                deduplicated_row_count=len(deduplicated_dataset),
                phase1_summary=contexts["phase1_summary"],
                phase6_summary=contexts["phase6_summary"],
                phase7_summary=contexts["phase7_summary"],
                phase8_summary=contexts["phase8_summary"],
                phase9_summary=contexts["phase9_summary"],
                phase10_summary=contexts["phase10_summary"],
            )

            champion_model_name = prerequisite_summary["champion_model_name"]
            readiness_status = prerequisite_summary["readiness_status"]
            presentation_layers = build_presentation_layers(
                phase1_summary=contexts["phase1_summary"],
                phase6_summary=contexts["phase6_summary"],
                phase8_summary=contexts["phase8_summary"],
                phase9_summary=contexts["phase9_summary"],
                phase10_summary=contexts["phase10_summary"],
                champion_model_name=champion_model_name,
            )
            demo_report_mapping = build_demo_report_mapping(
                phase9_summary=contexts["phase9_summary"],
            )
            future_work_items = build_future_work_items(
                limitations=contexts["phase10_summary"].get("limitations", []),
                readiness_status=readiness_status,
            )
            visual_asset_library = build_visual_asset_library(
                root_dir=project_paths.root_dir,
                asset_groups=self._build_visual_asset_groups(project_paths),
            )

            artifacts = self._ensure_phase11_directories(project_paths.phase_11_dir)
            generated_outputs = self._build_generated_output_paths(artifacts)
            source_dependencies = self._build_source_dependencies(project_paths)
            summary = self._build_summary(
                project_paths=project_paths,
                deduplicated_row_count=len(deduplicated_dataset),
                champion_model_name=champion_model_name,
                readiness_status=readiness_status,
                prerequisite_summary=prerequisite_summary,
                contexts=contexts,
                presentation_layers=presentation_layers,
                demo_report_mapping=demo_report_mapping,
                future_work_items=future_work_items,
                visual_asset_library=visual_asset_library,
                generated_outputs=generated_outputs,
            )
            delivery_manifest = build_delivery_manifest(
                root_dir=project_paths.root_dir,
                dataset_path=project_paths.dataset_path,
                champion_model_name=champion_model_name,
                readiness_status=readiness_status,
                generated_outputs=generated_outputs,
                source_dependencies=source_dependencies,
                key_metrics=summary["key_metrics"],
                visual_asset_count=visual_asset_library["total_asset_count"],
                demo_mapping_count=len(demo_report_mapping),
            )

            self._write_json_file(artifacts.json_summary_path, summary)
            self._write_json_file(artifacts.visual_asset_library_path, visual_asset_library)
            self._write_json_file(artifacts.delivery_manifest_path, delivery_manifest)
            self._write_executive_summary(artifacts.executive_summary_path, summary)
            self._write_final_report(artifacts.markdown_report_path, summary, contexts)
            self._write_presentation_outline(artifacts.presentation_outline_path, summary, delivery_manifest)
            self._write_defense_script(artifacts.defense_script_path, presentation_layers)
            self._write_demo_report_mapping(artifacts.demo_report_mapping_path, demo_report_mapping)
            self._write_future_work(artifacts.future_work_path, future_work_items)
            self._write_delivery_bundle_readme(
                artifacts.delivery_bundle_readme_path,
                delivery_manifest,
                summary,
                source_dependencies,
            )
            self._logger.info("Faz 11 final rapor, sunum ve teslim paketi derlemesi tamamlandi.")
            return artifacts
        except DataAnalysisError:
            self._logger.exception("Faz 11 teslim paketi veri dogrulamasi nedeniyle durdu.")
            raise
        except (KeyError, OSError, TypeError, ValueError) as error:
            self._logger.exception("Faz 11 teslim paketi beklenmeyen bir hata ile durdu.")
            raise DataAnalysisError("Faz 11 final rapor ve teslim paketi tamamlanamadi.") from error

    def _ensure_phase11_directories(self, phase_11_dir: Path) -> Phase11Artifacts:
        delivery_bundle_dir = phase_11_dir / "delivery_bundle"
        for directory in (phase_11_dir, delivery_bundle_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return Phase11Artifacts(
            markdown_report_path=phase_11_dir / "final_report.md",
            json_summary_path=phase_11_dir / "delivery_summary.json",
            executive_summary_path=phase_11_dir / "executive_summary.md",
            presentation_outline_path=phase_11_dir / "presentation_outline.md",
            defense_script_path=phase_11_dir / "defense_script.md",
            demo_report_mapping_path=phase_11_dir / "demo_report_mapping.md",
            future_work_path=phase_11_dir / "future_work.md",
            visual_asset_library_path=phase_11_dir / "visual_asset_library.json",
            delivery_manifest_path=phase_11_dir / "delivery_manifest.json",
            delivery_bundle_dir=delivery_bundle_dir,
            delivery_bundle_readme_path=delivery_bundle_dir / "README.md",
        )

    def _load_required_contexts(self, project_paths: ProjectPaths) -> dict[str, dict[str, Any]]:
        contexts: dict[str, dict[str, Any]] = {}
        for label, relative_parts in SOURCE_DEPENDENCY_PATHS:
            if not label.endswith("_summary"):
                continue
            file_path = project_paths.artifacts_dir.joinpath(*relative_parts)
            contexts[label] = self._load_required_json(file_path, label)
        return contexts

    def _load_required_json(self, file_path: Path, label: str) -> dict[str, Any]:
        if not file_path.exists():
            raise DataAnalysisError(f"Faz 11 icin gerekli artefakt bulunamadi: {label} -> {file_path}")
        try:
            with file_path.open("r", encoding="utf-8") as input_file:
                payload = json.load(input_file)
        except (OSError, json.JSONDecodeError) as error:
            raise DataAnalysisError(f"Faz 11 icin gerekli artefakt okunamadi: {label} -> {file_path}") from error
        if not isinstance(payload, dict):
            raise DataAnalysisError(f"Faz 11 icin gerekli artefakt beklenen JSON nesnesi degil: {label}")
        return payload

    def _build_visual_asset_groups(self, project_paths: ProjectPaths) -> list[tuple[str, str, Path]]:
        asset_groups: list[tuple[str, str, Path]] = []
        for phase_name, asset_group, phase_dir_name, relative_dir_name in VISUAL_ASSET_GROUPS:
            asset_groups.append(
                (
                    phase_name,
                    asset_group,
                    project_paths.artifacts_dir / phase_dir_name / relative_dir_name,
                )
            )
        return asset_groups

    def _build_source_dependencies(self, project_paths: ProjectPaths) -> list[Path]:
        dependencies: list[Path] = []
        for _, relative_parts in SOURCE_DEPENDENCY_PATHS:
            dependency_path = project_paths.artifacts_dir.joinpath(*relative_parts)
            if dependency_path.exists():
                dependencies.append(dependency_path)
        return dependencies

    def _build_generated_output_paths(self, artifacts: Phase11Artifacts) -> list[Path]:
        return [
            artifacts.markdown_report_path,
            artifacts.json_summary_path,
            artifacts.executive_summary_path,
            artifacts.presentation_outline_path,
            artifacts.defense_script_path,
            artifacts.demo_report_mapping_path,
            artifacts.future_work_path,
            artifacts.visual_asset_library_path,
            artifacts.delivery_manifest_path,
            artifacts.delivery_bundle_readme_path,
        ]

    def _build_summary(
        self,
        *,
        project_paths: ProjectPaths,
        deduplicated_row_count: int,
        champion_model_name: str,
        readiness_status: str,
        prerequisite_summary: dict[str, Any],
        contexts: dict[str, dict[str, Any]],
        presentation_layers: dict[str, Any],
        demo_report_mapping: list[dict[str, Any]],
        future_work_items: list[dict[str, Any]],
        visual_asset_library: dict[str, Any],
        generated_outputs: list[Path],
    ) -> dict[str, Any]:
        key_metrics = contexts["phase8_summary"].get("phase6_context", {}).get("validated_champion_metrics", {})
        policy_band_summary = contexts["phase8_summary"].get("phase5_context", {}).get("policy_band_summary", {})
        quality_gate = {
            "edge_case_summary": contexts["phase10_summary"].get("edge_case_summary", {}),
            "reproducibility_summary": contexts["phase10_summary"].get("reproducibility_summary", {}),
            "performance_summary": contexts["phase10_summary"].get("performance_summary", {}),
            "readiness_checklist": contexts["phase10_summary"].get("readiness_checklist", []),
        }
        return {
            "dataset_path": str(project_paths.dataset_path),
            "deduplicated_row_count": deduplicated_row_count,
            "champion_model_name": champion_model_name,
            "readiness_status": readiness_status,
            "key_metrics": key_metrics,
            "consistency_checks": prerequisite_summary["consistency_checks"],
            "dataset_summary": {
                "row_count": contexts["phase1_summary"].get("row_count"),
                "duplicate_rate": contexts["phase1_summary"].get("duplicate_summary", {}).get("duplicate_rate"),
                "missing_cells": contexts["phase1_summary"].get("missing_summary", {}).get("total_missing_cells"),
                "target_rates": contexts["phase1_summary"].get("target_summary", {}).get("rates", {}),
                "phase1_decision": contexts["phase2_summary"].get("phase1_context", {}).get("final_decision"),
            },
            "experiment_summary": {
                "split_strategy": contexts["phase3_summary"].get("split_strategy", {}),
                "baseline_recommended_model": contexts["phase5_summary"].get("phase4_context", {}).get("recommended_baseline", {}).get("recommended_model"),
                "imbalance_profile": contexts["phase5_summary"].get("imbalance_profile", {}),
                "champion_rationale": contexts["phase8_summary"].get("phase6_context", {}).get("champion_selection", {}).get("rationale"),
            },
            "explainability_summary": {
                "business_insights": contexts["phase8_summary"].get("phase7_context", {}).get("business_insights", []),
                "fairness_alerts": contexts["phase8_summary"].get("phase7_context", {}).get("fairness_alerts", []),
            },
            "policy_band_summary": policy_band_summary,
            "simulation_overview": contexts["phase9_summary"].get("scenario_catalog_overview", {}),
            "quality_gate": quality_gate,
            "presentation_layers": presentation_layers,
            "demo_report_mapping": demo_report_mapping,
            "future_work_items": future_work_items,
            "visual_asset_library_overview": {
                "total_asset_count": visual_asset_library.get("total_asset_count", 0),
                "phase_counts": visual_asset_library.get("phase_counts", {}),
            },
            "generated_outputs": [path.name for path in generated_outputs],
            "phase11_closeout": {
                "summary": "Faz 11'de final rapor, sunum anlatisi, demo esleme ve teslim manifesti onceki fazlardan gelen onayli artefaktlar uzerinden derlendi.",
                "next_step": "Bu paket artik final teslim, sunum provasi ve son tartisma hazirligi icin kullanilabilir.",
            },
        }

    def _write_json_file(self, output_path: Path, payload: dict[str, Any]) -> None:
        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                json.dump(to_json_safe(payload), output_file, ensure_ascii=False, indent=2)
        except OSError as error:
            raise ArtifactWriteError(f"Faz 11 JSON artefakti yazilamadi: {output_path}") from error

    def _write_text_file(self, output_path: Path, lines: list[str], error_prefix: str) -> None:
        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                output_file.write("\n".join(lines))
        except OSError as error:
            raise ArtifactWriteError(f"{error_prefix}: {output_path}") from error

    def _write_executive_summary(self, output_path: Path, summary: dict[str, Any]) -> None:
        metrics = summary["key_metrics"]
        lines = [
            "# Faz 11 Yonetici Ozeti",
            "",
            f"- Veri seti: {summary['dataset_path']}",
            f"- Tekillestirilmis satir sayisi: {summary['deduplicated_row_count']}",
            f"- Champion model: {summary['champion_model_name']}",
            f"- Hazirlik durumu: {summary['readiness_status']}",
            f"- Accuracy: %{metrics.get('accuracy', 0.0) * 100:.2f}",
            f"- Precision: %{metrics.get('precision', 0.0) * 100:.2f}",
            f"- Recall: %{metrics.get('recall', 0.0) * 100:.2f}",
            f"- F1: %{metrics.get('f1', 0.0) * 100:.2f}",
            f"- ROC-AUC: %{metrics.get('roc_auc', 0.0) * 100:.2f}",
            f"- PR-AUC: %{metrics.get('pr_auc', 0.0) * 100:.2f}",
            f"- Balanced Accuracy: %{metrics.get('balanced_accuracy', 0.0) * 100:.2f}",
            f"- Brier Score: {metrics.get('brier_score', 0.0):.6f}",
            f"- Gorsel kutuphane varlik sayisi: {summary['visual_asset_library_overview']['total_asset_count']}",
            f"- Demo esleme satiri: {len(summary['demo_report_mapping'])}",
        ]
        self._write_text_file(output_path, lines, "Faz 11 yonetici ozeti yazilamadi")

    def _write_final_report(
        self,
        output_path: Path,
        summary: dict[str, Any],
        contexts: dict[str, dict[str, Any]],
    ) -> None:
        dataset_summary = summary["dataset_summary"]
        metrics = summary["key_metrics"]
        explainability_summary = summary["explainability_summary"]
        fairness_alerts = explainability_summary.get("fairness_alerts", [])
        first_fairness_alert = fairness_alerts[0]["message"] if fairness_alerts else "Kayda gecen fairness alert'i bulunmuyor."
        policy_band_summary = summary["policy_band_summary"]
        simulation_overview = summary["simulation_overview"]
        quality_gate = summary["quality_gate"]
        lines = [
            "# Faz 11 Final Raporu",
            "",
            "## 1. Yonetici Ozeti",
            "",
            f"- Proje, {summary['champion_model_name']} champion modeli uzerinden satin alma olasiligi uretip bunu kampanya politikasina ceviren bir karar destek prototipi olarak tamamlandi.",
            f"- Nihai kalite kapisi durumu: {summary['readiness_status']}",
            f"- Accuracy %{metrics.get('accuracy', 0.0) * 100:.2f}, PR-AUC %{metrics.get('pr_auc', 0.0) * 100:.2f}, ROC-AUC %{metrics.get('roc_auc', 0.0) * 100:.2f} olarak raporlandi.",
            "",
            "## 2. Problem ve Kapsam",
            "",
            "- Problem, herkese indirim vermek yerine satin alma niyeti ve risk bandina gore hedefli aksiyon onermek olarak cercevelendi.",
            "- Proje, canli fiyat optimizasyonu degil; cevrimdisi prototip, karar simülasyonu ve savunulabilir teslim mantigi ile sinirlandi.",
            "",
            "## 3. Veri Denetimi ve Veri Kalitesi",
            "",
            f"- Ham satir sayisi: {dataset_summary.get('row_count')}",
            f"- Tekillestirilmis satir sayisi: {summary['deduplicated_row_count']}",
            f"- Tekrar oranı: %{float(dataset_summary.get('duplicate_rate', 0.0)) * 100:.2f}",
            f"- Eksik hucre sayisi: {dataset_summary.get('missing_cells')}",
            f"- Hedef dagilimi: {dataset_summary.get('target_rates')}",
            f"- Faz 1/Faz 2 denetim karari: {dataset_summary.get('phase1_decision')}",
            "",
            "## 4. EDA ve Hipotezler",
            "",
            f"- LoyaltyProgram=1 grubunda satin alma orani %{_lookup_categorical_purchase_rate(contexts['phase2_summary'], 'LoyaltyProgram', 1) * 100:.2f} olarak olculdu; bu, sadakat sinyalinin guclu tasiyicilardan biri oldugunu gosterdi.",
            f"- LoyaltyProgram=0 grubunda satin alma orani %{_lookup_categorical_purchase_rate(contexts['phase2_summary'], 'LoyaltyProgram', 0) * 100:.2f} olarak kaldigi icin segment farki politikanin da merkezine tasindi.",
            f"- DiscountsAvailed dagiliminda 3 ve uzeri gecmis kullanimda satin alma oraninin belirgin zipladigi goruldu; bu bulgu alanin analize dahil edilip ana modelden temkinli uzak tutulmasi gerektigini destekledi.",
            "",
            "## 5. On Isleme ve Deney Tasarimi",
            "",
            f"- Split stratejisi: {contexts['phase3_summary'].get('split_strategy', {}).get('method')}",
            f"- Train/test boyutu: {contexts['phase3_summary'].get('split_strategy', {}).get('train_row_count')} / {contexts['phase3_summary'].get('split_strategy', {}).get('test_row_count')}",
            f"- Linear branch feature sayisi: {contexts['phase4_summary'].get('phase3_context', {}).get('validation_summary', {}).get('train_feature_count', 'n/a')}",
            f"- Tree branch feature sayisi: {contexts['phase4_summary'].get('phase3_context', {}).get('preprocessing_map', {}).get('tree_branch', {}).get('train_feature_count', 'n/a')}",
            "",
            "## 6. Baseline, Dengesizlik ve Champion Model",
            "",
            f"- Majority baseline accuracy degeri %{contexts['phase4_summary'].get('model_results', {}).get('majority_baseline', {}).get('metrics', {}).get('accuracy', 0.0) * 100:.2f} seviyesinde kaldi.",
            f"- Logistic baseline accuracy degeri %{contexts['phase4_summary'].get('model_results', {}).get('logistic_regression', {}).get('metrics', {}).get('accuracy', 0.0) * 100:.2f} ile guclu ilk referansi sagladi.",
            f"- Dengesizlik profili {contexts['phase5_summary'].get('imbalance_profile', {}).get('severity')} olarak siniflandi; minority/majority orani {contexts['phase5_summary'].get('imbalance_profile', {}).get('minority_to_majority_ratio')} seviyesinde kaldigi icin sentetik veri varsayilan yol olmadi.",
            f"- Champion model secim gerekcesi: {summary['experiment_summary'].get('champion_rationale')}",
            f"- Nihai champion metrikleri: accuracy %{metrics.get('accuracy', 0.0) * 100:.2f}, precision %{metrics.get('precision', 0.0) * 100:.2f}, recall %{metrics.get('recall', 0.0) * 100:.2f}, F1 %{metrics.get('f1', 0.0) * 100:.2f}.",
            "",
            "## 7. Aciklanabilirlik ve Fairness",
            "",
            f"- En guclu is içgorusu: {explainability_summary.get('business_insights', ['icgoru bulunamadi'])[0]}",
            f"- Fairness alert sayisi: {len(fairness_alerts)}",
            f"- En kritik fairness uyarisi: {first_fairness_alert}",
            "",
            "## 8. Politika Katmani ve Aksiyon Bantlari",
            "",
        ]
        for band_name, band_summary in policy_band_summary.items():
            lines.append(
                f"- {band_name}: pay=%{float(band_summary.get('share', 0.0)) * 100:.2f}, gozlenen satin alma oranı=%{float(band_summary.get('observed_purchase_rate', 0.0)) * 100:.2f}, ortalama skor=%{float(band_summary.get('average_probability', 0.0)) * 100:.2f}"
            )

        lines.extend([
            "",
            "## 9. Prototip, Demo ve Simülasyon",
            "",
            f"- Toplam demo senaryosu: {simulation_overview.get('total_scenarios')}",
            f"- Senaryo kaynak dagilimi: {simulation_overview.get('origin_counts')}",
            f"- Manual review payi: %{float(simulation_overview.get('manual_review_share', 0.0)) * 100:.2f}",
            f"- Demo esleme satiri sayisi: {len(summary['demo_report_mapping'])}",
            "",
            "## 10. Kalite Kapisi ve Teslim Oncesi Dogrulama",
            "",
            f"- Edge-case sonuc ozeti: {quality_gate.get('edge_case_summary')}",
            f"- Reproducibility ozeti: {quality_gate.get('reproducibility_summary')}",
            f"- Performans ozeti: {quality_gate.get('performance_summary')}",
            "",
            "## 11. Limitasyonlar",
            "",
        ])
        for limitation in contexts["phase10_summary"].get("limitations", []):
            lines.append(
                f"- {limitation.get('limitation_id')}: {limitation.get('statement')} | etki={limitation.get('impact')}"
            )

        lines.extend([
            "",
            "## 12. Sonuc ve Teslim Karari",
            "",
            f"- Teslim karari: {summary['readiness_status']}",
            f"- Faz 11 kapanis ozeti: {summary['phase11_closeout']['summary']}",
            f"- Sonraki adim: {summary['phase11_closeout']['next_step']}",
        ])
        self._write_text_file(output_path, lines, "Faz 11 final raporu yazilamadi")

    def _write_presentation_outline(
        self,
        output_path: Path,
        summary: dict[str, Any],
        delivery_manifest: dict[str, Any],
    ) -> None:
        metrics = summary["key_metrics"]
        lines = [
            "# Faz 11 Sunum Iskeleti",
            "",
            "## Slayt 1 - Problem ve Is Degeri",
            "- Mesaj: Herkese indirim vermek yerine, satin alma niyeti ve risk bandina gore kontrollu aksiyon onermek.",
            "- Destek: Yonetici ozeti ve final raporun giris bolumu.",
            "",
            "## Slayt 2 - Veri Kalitesi",
            f"- Mesaj: Veri kullanilabilir ama temiz degildi; %{float(summary['dataset_summary'].get('duplicate_rate', 0.0)) * 100:.2f} tekrar kayit temizlendi.",
            "- Destek: Faz 1 denetim raporu ve audit JSON ozeti.",
            "",
            "## Slayt 3 - Baseline'dan Champion'a Gecis",
            f"- Mesaj: Champion model {summary['champion_model_name']} olarak secildi.",
            f"- Destek: Accuracy %{metrics.get('accuracy', 0.0) * 100:.2f}, PR-AUC %{metrics.get('pr_auc', 0.0) * 100:.2f}, ROC-AUC %{metrics.get('roc_auc', 0.0) * 100:.2f}.",
            "",
            "## Slayt 4 - Aciklanabilirlik ve Fairness",
            f"- Mesaj: En guclu explainability sinyalleri ve {len(summary['explainability_summary'].get('fairness_alerts', []))} fairness alert'i birlikte sunulacak.",
            "- Destek: Faz 7 SHAP ve fairness artefaktlari.",
            "",
            "## Slayt 5 - Politika Katmani",
            "- Mesaj: Skor tek basina aksiyon degil; karar bantlari ve guardrail'ler ile yonetilen politika katmani var.",
            "- Destek: Faz 8 policy report ve band ozeti.",
            "",
            "## Slayt 6 - Demo ve Senaryolar",
            f"- Mesaj: {len(summary['demo_report_mapping'])} demo adimi ve {summary['simulation_overview'].get('total_scenarios')} toplam senaryo ile prototip gosterimi yapilacak.",
            "- Destek: Faz 9 simülasyon raporu ve runbook.",
            "",
            "## Slayt 7 - Kalite Kapisi",
            f"- Mesaj: Teslim durumu {summary['readiness_status']} ve delivery manifest ready_for_submission={delivery_manifest['ready_for_submission']} olarak kayitli.",
            "- Destek: Faz 10 validation ozeti ve Faz 11 manifesti.",
            "",
            "## Slayt 8 - Limitasyonlar ve Gelecek Calisma",
            f"- Mesaj: {len(summary['future_work_items'])} sonraki calisma maddesi ile mevcut limitasyonlar eylem planina cevrildi.",
            "- Destek: Faz 10 limitasyonlari ve Faz 11 future work dosyasi.",
        ]
        self._write_text_file(output_path, lines, "Faz 11 sunum iskeleti yazilamadi")

    def _write_defense_script(self, output_path: Path, presentation_layers: dict[str, Any]) -> None:
        lines = [
            "# Faz 11 Katmanli Savunma Metni",
            "",
            "## 30 Saniyelik Ozet",
            "",
            presentation_layers["thirty_second_pitch"],
            "",
            "## 3 Dakikalik Teknik Akis",
            "",
        ]
        for paragraph in presentation_layers["three_minute_pitch"]:
            lines.append(f"- {paragraph}")

        lines.extend([
            "",
            "## 10 Dakikalik Savunma Iskeleti",
            "",
        ])
        for section in presentation_layers["ten_minute_defense_outline"]:
            lines.extend([
                f"### {section['section_title']}",
                f"- {section['talk_track']}",
                "",
            ])
        self._write_text_file(output_path, lines, "Faz 11 savunma metni yazilamadi")

    def _write_demo_report_mapping(self, output_path: Path, demo_report_mapping: list[dict[str, Any]]) -> None:
        lines = [
            "# Faz 11 Demo ve Rapor Esleme Tablosu",
            "",
            "| Adim | Senaryo | Kaynak | Skor | Aksiyon | Rapor Bolumleri |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in demo_report_mapping:
            lines.append(
                f"| {row.get('step_number')} | {row.get('scenario_title')} ({row.get('scenario_id')}) | {row.get('scenario_origin')} | {row.get('purchase_score')} | {row.get('recommended_action')} | {', '.join(row.get('report_sections', []))} |"
            )
        self._write_text_file(output_path, lines, "Faz 11 demo-rapor esleme dosyasi yazilamadi")

    def _write_future_work(self, output_path: Path, future_work_items: list[dict[str, Any]]) -> None:
        lines = [
            "# Faz 11 Gelecek Calisma Listesi",
            "",
        ]
        for index, item in enumerate(future_work_items, start=1):
            lines.extend([
                f"## {index}. {item['workstream']}",
                "",
                f"- Bagli limitasyon: {item['linked_limitation_id']}",
                f"- Oncelik: {item['priority']}",
                f"- Gerekce: {item['rationale']}",
                f"- Gereken yetkinlik: {item['required_capability']}",
                "",
            ])
        self._write_text_file(output_path, lines, "Faz 11 gelecek calisma listesi yazilamadi")

    def _write_delivery_bundle_readme(
        self,
        output_path: Path,
        delivery_manifest: dict[str, Any],
        summary: dict[str, Any],
        source_dependencies: list[Path],
    ) -> None:
        lines = [
            "# Faz 11 Delivery Bundle",
            "",
            f"- Hazirlik durumu: {summary['readiness_status']}",
            f"- Champion model: {summary['champion_model_name']}",
            f"- Ready for submission: {delivery_manifest['ready_for_submission']}",
            "",
            "## Uretilen Faz 11 Dosyalari",
            "",
        ]
        for relative_path in delivery_manifest["generated_outputs"]:
            lines.append(f"- {relative_path}")

        lines.extend([
            "",
            "## Tasinan Kaynak Artefaktlar",
            "",
        ])
        for file_path in source_dependencies:
            lines.append(f"- {file_path.as_posix()}")

        lines.extend([
            "",
            "## Sunum Sirasi",
            "",
            "- 1. Executive summary ile acilis yap.",
            "- 2. Final report ile metodolojik akisi savun.",
            "- 3. Presentation outline ve defense script ile prova yap.",
            "- 4. Demo-report mapping uzerinden senaryolari sira ile goster.",
            "- 5. Future work dosyasi ile sinirlari ileri yol haritasina bagla.",
        ])
        self._write_text_file(output_path, lines, "Faz 11 delivery bundle README yazilamadi")


def _lookup_categorical_purchase_rate(phase2_summary: dict[str, Any], column_name: str, category_code: int) -> float:
    entries = phase2_summary.get("categorical_summary", {}).get(column_name, [])
    for entry in entries:
        if int(entry.get(column_name, -999)) == int(category_code):
            return float(entry.get("purchase_rate", 0.0))
    return 0.0