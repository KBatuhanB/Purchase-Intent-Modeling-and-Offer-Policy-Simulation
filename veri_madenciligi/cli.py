from __future__ import annotations

import argparse
from pathlib import Path

from .config import ProjectPaths
from .core.exceptions import DataAnalysisError
from .core.logging_utils import get_application_logger
from .services.csv_loader import CsvDatasetLoader
from .services.phase1_audit import Phase1AuditService
from .services.phase2_eda import Phase2EdaService
from .services.phase3_preprocessing import Phase3PreprocessingService
from .services.phase4_baseline import Phase4BaselineService
from .services.phase5_imbalance import Phase5ImbalanceService
from .services.phase6_advanced_modeling import Phase6AdvancedModelingService
from .services.phase7_explainability import Phase7ExplainabilityService
from .services.phase8_policy import Phase8PolicyService
from .services.phase9_simulation import Phase9SimulationService
from .services.phase10_validation import Phase10ValidationService
from .services.phase11_delivery import Phase11DeliveryService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Veri denetimi, EDA ve preprocessing artefaktlari ureten komut satiri araci."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    phase_1_parser = subparsers.add_parser(
        "phase1",
        help="Faz 1 veri denetimi ve butunluk dogrulamasini calistirir.",
    )
    phase_1_parser.add_argument(
        "--dataset",
        required=True,
        help="Analiz edilecek CSV dosyasinin yolu.",
    )

    phase_2_parser = subparsers.add_parser(
        "phase2",
        help="Faz 2 aciklayici veri analizi ve hipotez uretimini calistirir.",
    )
    phase_2_parser.add_argument(
        "--dataset",
        required=True,
        help="Analiz edilecek CSV dosyasinin yolu.",
    )

    phase_3_parser = subparsers.add_parser(
        "phase3",
        help="Faz 3 on isleme, split ve ozellik muhendisligi tasarimini calistirir.",
    )
    phase_3_parser.add_argument(
        "--dataset",
        required=True,
        help="Analiz edilecek CSV dosyasinin yolu.",
    )

    phase_4_parser = subparsers.add_parser(
        "phase4",
        help="Faz 4 baseline modelleme ve kalibrasyon on degerlendirmesini calistirir.",
    )
    phase_4_parser.add_argument(
        "--dataset",
        required=True,
        help="Analiz edilecek CSV dosyasinin yolu.",
    )
    phase_4_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Train/test bolmesindeki test veri orani. Varsayilan deger 0.2'dir.",
    )
    phase_4_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Tekrarlanabilirlik icin kullanilacak rastgelelik tohumu.",
    )

    phase_5_parser = subparsers.add_parser(
        "phase5",
        help="Faz 5 sinif dengesizligi ve esik stratejisi deneylerini calistirir.",
    )
    phase_5_parser.add_argument(
        "--dataset",
        required=True,
        help="Analiz edilecek CSV dosyasinin yolu.",
    )
    phase_5_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Outer test bolmesi orani. Varsayilan deger 0.2'dir.",
    )
    phase_5_parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Train icindeki validation bolmesi orani. Varsayilan deger 0.2'dir.",
    )
    phase_5_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Tekrarlanabilirlik icin kullanilacak rastgelelik tohumu.",
    )

    phase_6_parser = subparsers.add_parser(
        "phase6",
        help="Faz 6 ileri modelleme ve hiperparametre optimizasyonunu calistirir.",
    )
    phase_6_parser.add_argument(
        "--dataset",
        required=True,
        help="Analiz edilecek CSV dosyasinin yolu.",
    )
    phase_6_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Outer test bolmesi orani. Varsayilan deger 0.2'dir.",
    )
    phase_6_parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Train icindeki validation bolmesi orani. Varsayilan deger 0.2'dir.",
    )
    phase_6_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Tekrarlanabilirlik icin kullanilacak rastgelelik tohumu.",
    )

    phase_7_parser = subparsers.add_parser(
        "phase7",
        help="Faz 7 aciklanabilirlik ve guvenilirlik analizini calistirir.",
    )
    phase_7_parser.add_argument(
        "--dataset",
        required=True,
        help="Analiz edilecek CSV dosyasinin yolu.",
    )

    phase_8_parser = subparsers.add_parser(
        "phase8",
        help="Faz 8 politika, aksiyon ve guardrail katmanini calistirir.",
    )
    phase_8_parser.add_argument(
        "--dataset",
        required=True,
        help="Analiz edilecek CSV dosyasinin yolu.",
    )

    phase_9_parser = subparsers.add_parser(
        "phase9",
        help="Faz 9 prototip, demo senaryolari ve simülasyon katmanini calistirir.",
    )
    phase_9_parser.add_argument(
        "--dataset",
        required=True,
        help="Analiz edilecek CSV dosyasinin yolu.",
    )

    phase_10_parser = subparsers.add_parser(
        "phase10",
        help="Faz 10 dogrulama, stres testi ve teslim oncesi kalite kapisini calistirir.",
    )
    phase_10_parser.add_argument(
        "--dataset",
        required=True,
        help="Analiz edilecek CSV dosyasinin yolu.",
    )

    phase_11_parser = subparsers.add_parser(
        "phase11",
        help="Faz 11 final rapor, sunum notlari ve teslim paketini derler.",
    )
    phase_11_parser.add_argument(
        "--dataset",
        required=True,
        help="Analiz edilecek CSV dosyasinin yolu.",
    )

    return parser


def resolve_csv_path(raw_path: str) -> Path:
    # Girdi yolu, komut satirindan geldigi icin erken asamada dogrulanir.
    candidate = Path(raw_path).expanduser().resolve()
    if candidate.suffix.lower() != ".csv":
        raise ValueError("Analiz araci yalnizca .csv uzantili dosyalarla calisir.")
    return candidate


def run_phase_1(dataset_path: Path) -> int:
    project_paths = ProjectPaths.from_dataset_path(dataset_path)
    project_paths.ensure_output_directories()

    logger = get_application_logger(project_paths.logs_dir)
    service = Phase1AuditService(loader=CsvDatasetLoader(), logger=logger)
    artifacts = service.run(project_paths)

    print(f"Faz 1 raporu olusturuldu: {artifacts.markdown_report_path}")
    print(f"Faz 1 JSON ozeti olusturuldu: {artifacts.json_summary_path}")
    return 0


def run_phase_2(dataset_path: Path) -> int:
    project_paths = ProjectPaths.from_dataset_path(dataset_path)
    project_paths.ensure_output_directories()

    logger = get_application_logger(project_paths.logs_dir)
    service = Phase2EdaService(loader=CsvDatasetLoader(), logger=logger)
    artifacts = service.run(project_paths)

    print(f"Faz 2 raporu olusturuldu: {artifacts.markdown_report_path}")
    print(f"Faz 2 JSON ozeti olusturuldu: {artifacts.json_summary_path}")
    print(f"Faz 2 grafik klasoru: {artifacts.plots_dir}")
    return 0


def run_phase_3(dataset_path: Path) -> int:
    project_paths = ProjectPaths.from_dataset_path(dataset_path)
    project_paths.ensure_output_directories()

    logger = get_application_logger(project_paths.logs_dir)
    service = Phase3PreprocessingService(loader=CsvDatasetLoader(), logger=logger)
    artifacts = service.run(project_paths)

    print(f"Faz 3 raporu olusturuldu: {artifacts.markdown_report_path}")
    print(f"Faz 3 JSON ozeti olusturuldu: {artifacts.json_summary_path}")
    print(f"Faz 3 veri artefakt klasoru: {artifacts.datasets_dir}")
    print(f"Faz 3 pipeline klasoru: {artifacts.models_dir}")
    return 0


def run_phase_4(dataset_path: Path, test_size: float, random_state: int) -> int:
    project_paths = ProjectPaths.from_dataset_path(dataset_path)
    project_paths.ensure_output_directories()

    logger = get_application_logger(project_paths.logs_dir)
    service = Phase4BaselineService(
        loader=CsvDatasetLoader(),
        logger=logger,
        test_size=test_size,
        random_state=random_state,
    )
    artifacts = service.run(project_paths)

    print(f"Faz 4 raporu olusturuldu: {artifacts.markdown_report_path}")
    print(f"Faz 4 JSON ozeti olusturuldu: {artifacts.json_summary_path}")
    print(f"Faz 4 grafik klasoru: {artifacts.plots_dir}")
    print(f"Faz 4 model klasoru: {artifacts.models_dir}")
    return 0


def run_phase_5(dataset_path: Path, test_size: float, validation_size: float, random_state: int) -> int:
    project_paths = ProjectPaths.from_dataset_path(dataset_path)
    project_paths.ensure_output_directories()

    logger = get_application_logger(project_paths.logs_dir)
    service = Phase5ImbalanceService(
        loader=CsvDatasetLoader(),
        logger=logger,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
    )
    artifacts = service.run(project_paths)

    print(f"Faz 5 raporu olusturuldu: {artifacts.markdown_report_path}")
    print(f"Faz 5 JSON ozeti olusturuldu: {artifacts.json_summary_path}")
    print(f"Faz 5 grafik klasoru: {artifacts.plots_dir}")
    print(f"Faz 5 model klasoru: {artifacts.models_dir}")
    return 0


def run_phase_6(dataset_path: Path, test_size: float, validation_size: float, random_state: int) -> int:
    project_paths = ProjectPaths.from_dataset_path(dataset_path)
    project_paths.ensure_output_directories()

    logger = get_application_logger(project_paths.logs_dir)
    service = Phase6AdvancedModelingService(
        loader=CsvDatasetLoader(),
        logger=logger,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
    )
    artifacts = service.run(project_paths)

    print(f"Faz 6 raporu olusturuldu: {artifacts.markdown_report_path}")
    print(f"Faz 6 JSON ozeti olusturuldu: {artifacts.json_summary_path}")
    print(f"Faz 6 grafik klasoru: {artifacts.plots_dir}")
    print(f"Faz 6 model klasoru: {artifacts.models_dir}")
    return 0


def run_phase_7(dataset_path: Path) -> int:
    project_paths = ProjectPaths.from_dataset_path(dataset_path)
    project_paths.ensure_output_directories()

    logger = get_application_logger(project_paths.logs_dir)
    service = Phase7ExplainabilityService(
        loader=CsvDatasetLoader(),
        logger=logger,
    )
    artifacts = service.run(project_paths)

    print(f"Faz 7 raporu olusturuldu: {artifacts.markdown_report_path}")
    print(f"Faz 7 JSON ozeti olusturuldu: {artifacts.json_summary_path}")
    print(f"Faz 7 grafik klasoru: {artifacts.plots_dir}")
    print(f"Faz 7 senaryo klasoru: {artifacts.scenarios_dir}")
    return 0


def run_phase_8(dataset_path: Path) -> int:
    project_paths = ProjectPaths.from_dataset_path(dataset_path)
    project_paths.ensure_output_directories()

    logger = get_application_logger(project_paths.logs_dir)
    service = Phase8PolicyService(
        loader=CsvDatasetLoader(),
        logger=logger,
    )
    artifacts = service.run(project_paths)

    print(f"Faz 8 raporu olusturuldu: {artifacts.markdown_report_path}")
    print(f"Faz 8 JSON ozeti olusturuldu: {artifacts.json_summary_path}")
    print(f"Faz 8 grafik klasoru: {artifacts.plots_dir}")
    print(f"Faz 8 senaryo klasoru: {artifacts.scenarios_dir}")
    return 0


def run_phase_9(dataset_path: Path) -> int:
    project_paths = ProjectPaths.from_dataset_path(dataset_path)
    project_paths.ensure_output_directories()

    logger = get_application_logger(project_paths.logs_dir)
    service = Phase9SimulationService(
        loader=CsvDatasetLoader(),
        logger=logger,
    )
    artifacts = service.run(project_paths)

    print(f"Faz 9 raporu olusturuldu: {artifacts.markdown_report_path}")
    print(f"Faz 9 JSON ozeti olusturuldu: {artifacts.json_summary_path}")
    print(f"Faz 9 input schema dosyasi: {artifacts.input_schema_path}")
    print(f"Faz 9 simülasyon logu: {artifacts.simulation_log_path}")
    print(f"Faz 9 demo runbook: {artifacts.demo_runbook_path}")
    print(f"Faz 9 senaryo klasoru: {artifacts.scenarios_dir}")
    return 0


def run_phase_10(dataset_path: Path) -> int:
    project_paths = ProjectPaths.from_dataset_path(dataset_path)
    project_paths.ensure_output_directories()

    logger = get_application_logger(project_paths.logs_dir)
    service = Phase10ValidationService(
        loader=CsvDatasetLoader(),
        logger=logger,
    )
    artifacts = service.run(project_paths)

    print(f"Faz 10 raporu olusturuldu: {artifacts.markdown_report_path}")
    print(f"Faz 10 JSON ozeti olusturuldu: {artifacts.json_summary_path}")
    print(f"Faz 10 edge-case matrisi: {artifacts.edge_case_matrix_path}")
    print(f"Faz 10 sensitivity analizi: {artifacts.sensitivity_analysis_path}")
    print(f"Faz 10 performans benchmark: {artifacts.performance_benchmark_path}")
    print(f"Faz 10 reproducibility notu: {artifacts.reproducibility_note_path}")
    print(f"Faz 10 limitasyon listesi: {artifacts.limitations_path}")
    print(f"Faz 10 readiness checklist: {artifacts.readiness_checklist_path}")
    return 0


def run_phase_11(dataset_path: Path) -> int:
    project_paths = ProjectPaths.from_dataset_path(dataset_path)
    project_paths.ensure_output_directories()

    logger = get_application_logger(project_paths.logs_dir)
    service = Phase11DeliveryService(
        loader=CsvDatasetLoader(),
        logger=logger,
    )
    artifacts = service.run(project_paths)

    print(f"Faz 11 final raporu: {artifacts.markdown_report_path}")
    print(f"Faz 11 JSON ozeti: {artifacts.json_summary_path}")
    print(f"Faz 11 yonetici ozeti: {artifacts.executive_summary_path}")
    print(f"Faz 11 sunum iskeleti: {artifacts.presentation_outline_path}")
    print(f"Faz 11 savunma metni: {artifacts.defense_script_path}")
    print(f"Faz 11 demo-rapor esleme: {artifacts.demo_report_mapping_path}")
    print(f"Faz 11 gelecek calisma listesi: {artifacts.future_work_path}")
    print(f"Faz 11 gorsel kutuphane manifesti: {artifacts.visual_asset_library_path}")
    print(f"Faz 11 delivery manifesti: {artifacts.delivery_manifest_path}")
    print(f"Faz 11 delivery bundle klasoru: {artifacts.delivery_bundle_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        dataset_path = resolve_csv_path(args.dataset)
        if args.command == "phase1":
            return run_phase_1(dataset_path)
        if args.command == "phase2":
            return run_phase_2(dataset_path)
        if args.command == "phase3":
            return run_phase_3(dataset_path)
        if args.command == "phase4":
            if not 0.05 <= args.test_size <= 0.4:
                raise ValueError("Test boyutu 0.05 ile 0.4 arasinda olmalidir.")
            if args.random_state < 0:
                raise ValueError("Random state negatif olamaz.")
            return run_phase_4(dataset_path, test_size=args.test_size, random_state=args.random_state)
        if args.command == "phase5":
            if not 0.05 <= args.test_size <= 0.4:
                raise ValueError("Test boyutu 0.05 ile 0.4 arasinda olmalidir.")
            if not 0.1 <= args.validation_size <= 0.4:
                raise ValueError("Validation boyutu 0.1 ile 0.4 arasinda olmalidir.")
            if args.random_state < 0:
                raise ValueError("Random state negatif olamaz.")
            return run_phase_5(
                dataset_path,
                test_size=args.test_size,
                validation_size=args.validation_size,
                random_state=args.random_state,
            )
        if args.command == "phase6":
            if not 0.05 <= args.test_size <= 0.4:
                raise ValueError("Test boyutu 0.05 ile 0.4 arasinda olmalidir.")
            if not 0.1 <= args.validation_size <= 0.4:
                raise ValueError("Validation boyutu 0.1 ile 0.4 arasinda olmalidir.")
            if args.random_state < 0:
                raise ValueError("Random state negatif olamaz.")
            return run_phase_6(
                dataset_path,
                test_size=args.test_size,
                validation_size=args.validation_size,
                random_state=args.random_state,
            )
        if args.command == "phase7":
            return run_phase_7(dataset_path)
        if args.command == "phase8":
            return run_phase_8(dataset_path)
        if args.command == "phase9":
            return run_phase_9(dataset_path)
        if args.command == "phase10":
            return run_phase_10(dataset_path)
        if args.command == "phase11":
            return run_phase_11(dataset_path)
    except ValueError as error:
        parser.exit(status=2, message=f"Gecersiz girdi: {error}\n")
    except DataAnalysisError as error:
        parser.exit(status=1, message=f"Analiz hatasi: {error}\n")

    parser.exit(status=2, message="Desteklenmeyen komut alindi.\n")
    return 2
