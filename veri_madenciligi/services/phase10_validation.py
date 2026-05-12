from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..config import ProjectPaths
from ..core.exceptions import ArtifactWriteError, DataAnalysisError
from ..core.serialization import to_json_safe
from .csv_loader import CsvDatasetLoader
from .phase2_eda import deduplicate_dataset
from .phase3_preprocessing import build_feature_role_matrix
from .phase9_simulation import (
    SIMULATION_INPUT_COLUMNS,
    build_input_validation_rules,
    sanitize_simulation_input,
    simulate_customer_profile,
)

# Bu esitler canli API SLA'i degil, insan tempolu demo ve egitim ortami icin kabul edilebilir bekleme butcesini temsil eder.
DEMO_SINGLE_SCENARIO_BUDGET_MS = 1500.0
DEMO_BATCH_BUDGET_MS = 15000.0
REPRODUCIBILITY_RERUNS = 2
SENSITIVITY_PRIORITY_SCENARIOS = (
    "boundary_case",
    "non_loyal_high_time",
    "high_confidence_true_positive",
    "synthetic_loyal_discount_fatigue",
)


@dataclass(frozen=True)
class Phase10Artifacts:
    markdown_report_path: Path
    json_summary_path: Path
    edge_case_matrix_path: Path
    sensitivity_analysis_path: Path
    performance_benchmark_path: Path
    reproducibility_note_path: Path
    limitations_path: Path
    readiness_checklist_path: Path


def extract_phase9_scenarios(phase9_summary: dict[str, Any]) -> list[dict[str, Any]]:
    simulation_results = phase9_summary.get("simulation_results", [])
    if not simulation_results:
        raise DataAnalysisError("Faz 10 icin Faz 9 simülasyon sonuclari bulunamadi.")

    scenarios: list[dict[str, Any]] = []
    for result in simulation_results:
        scenarios.append(
            {
                "scenario_id": result["scenario_id"],
                "title": result["title"],
                "scenario_origin": result["scenario_origin"],
                "selection_reason": result.get("selection_reason"),
                "input_payload": result["simulation_input"],
            }
        )
    return scenarios


def build_edge_case_catalog(validation_rules: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    base_payload = {
        column: validation_rules[column]["example_value"]
        for column in SIMULATION_INPUT_COLUMNS
    }

    # Bu katalog, sistemin sadece iyi veride degil bozuk ve sinir degerlerde de kontrollu davrandigini gostermek icin kurulur.
    return [
        {
            "case_id": "valid_min_boundary_profile",
            "title": "Minimum Sinirda Gecerli Profil",
            "expected_valid": True,
            "payload": {
                **base_payload,
                "Age": validation_rules["Age"]["min_value"],
                "NumberOfPurchases": validation_rules["NumberOfPurchases"]["min_value"],
                "TimeSpentOnWebsite": validation_rules["TimeSpentOnWebsite"]["min_value"],
                "DiscountsAvailed": validation_rules["DiscountsAvailed"]["min_value"],
            },
            "rationale": "Sistemin alt sinirdaki gecerli profilleri sessizce reddetmedigini dogrular.",
        },
        {
            "case_id": "valid_max_boundary_profile",
            "title": "Maksimum Sinirda Gecerli Profil",
            "expected_valid": True,
            "payload": {
                **base_payload,
                "Age": validation_rules["Age"]["max_value"],
                "AnnualIncome": validation_rules["AnnualIncome"]["max_value"],
                "NumberOfPurchases": validation_rules["NumberOfPurchases"]["max_value"],
                "TimeSpentOnWebsite": validation_rules["TimeSpentOnWebsite"]["max_value"],
                "DiscountsAvailed": validation_rules["DiscountsAvailed"]["max_value"],
            },
            "rationale": "Ust sinirlarda input kirpilmadan ya da bozulmadan islenmeli.",
        },
        {
            "case_id": "valid_zero_history_high_attention",
            "title": "Sifir Gecmis Ama Yuksek Dikkat",
            "expected_valid": True,
            "payload": {
                **base_payload,
                "NumberOfPurchases": validation_rules["NumberOfPurchases"]["min_value"],
                "TimeSpentOnWebsite": round(float(validation_rules["TimeSpentOnWebsite"]["max_value"]) * 0.95, 6),
                "DiscountsAvailed": validation_rules["DiscountsAvailed"]["min_value"],
            },
            "rationale": "Planin ozellikle andigi ilgi yuksek ama gecmisi zayif vakalar simülasyonda gecerli olmali.",
        },
        {
            "case_id": "invalid_age_below_min",
            "title": "Min Yasin Altinda Kullanici",
            "expected_valid": False,
            "payload": {
                **base_payload,
                "Age": int(validation_rules["Age"]["min_value"]) - 1,
            },
            "rationale": "Yas alt sinir ihlalinde sistemin kontrollu validation hatasi vermesi beklenir.",
        },
        {
            "case_id": "invalid_income_above_max",
            "title": "Gelir Ust Sinir Asimi",
            "expected_valid": False,
            "payload": {
                **base_payload,
                "AnnualIncome": round(float(validation_rules["AnnualIncome"]["max_value"]) + 5000.0, 6),
            },
            "rationale": "Aykiri ve aralik disi gelir girdileri sessizce kabul edilmemeli.",
        },
        {
            "case_id": "invalid_unknown_category_code",
            "title": "Bilinmeyen Kategori Kodu",
            "expected_valid": False,
            "payload": {
                **base_payload,
                "ProductCategory": int(max(validation_rules["ProductCategory"]["allowed_values"])) + 1,
            },
            "rationale": "Kategori kodlari whitelisting ile korunmali; yeni veya bozuk kodlar reddedilmeli.",
        },
        {
            "case_id": "invalid_loyalty_code",
            "title": "Uyumsuz Loyalty Kodu",
            "expected_valid": False,
            "payload": {
                **base_payload,
                "LoyaltyProgram": 2,
            },
            "rationale": "Ikili sadakat alani 0 ve 1 disinda deger almamali.",
        },
        {
            "case_id": "invalid_time_above_max",
            "title": "Asiri Uzun Site Suresi",
            "expected_valid": False,
            "payload": {
                **base_payload,
                "TimeSpentOnWebsite": round(float(validation_rules["TimeSpentOnWebsite"]["max_value"]) + 3.0, 6),
            },
            "rationale": "Plan geregi asiri uzun site suresi kontrollu olarak ele alinmali ve validation seviyesinde durdurulmali.",
        },
        {
            "case_id": "invalid_negative_discount_history",
            "title": "Negatif Indirim Gecmisi",
            "expected_valid": False,
            "payload": {
                **base_payload,
                "DiscountsAvailed": -1,
            },
            "rationale": "Negatif kampanya gecmisi mantiksal olarak gecersizdir ve zincirleme hatalari onlemek icin erken reddedilmelidir.",
        },
    ]


def evaluate_edge_case_matrix(
    *,
    edge_cases: list[dict[str, Any]],
    champion_model: Any,
    role_matrix: Any,
    validation_rules: dict[str, dict[str, Any]],
    decision_bands: dict[str, float],
    reference_policy_lookup: dict[str, dict[str, Any]] | None = None,
    global_driver_hints: list[str] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for case in edge_cases:
        try:
            simulated = simulate_customer_profile(
                scenario={
                    "scenario_id": case["case_id"],
                    "title": case["title"],
                    "scenario_origin": "phase10_edge_case",
                    "selection_reason": case["rationale"],
                    "input_payload": case["payload"],
                },
                champion_model=champion_model,
                role_matrix=role_matrix,
                validation_rules=validation_rules,
                decision_bands=decision_bands,
                reference_policy_lookup=reference_policy_lookup,
                global_driver_hints=global_driver_hints,
            )
            results.append(
                {
                    "case_id": case["case_id"],
                    "title": case["title"],
                    "expected_valid": case["expected_valid"],
                    "observed_status": "passed" if case["expected_valid"] else "failed_unexpected_accept",
                    "recommended_action": simulated["recommended_action"],
                    "predicted_probability": simulated["predicted_probability"],
                    "requires_manual_review": simulated["requires_manual_review"],
                    "error_message": None,
                    "rationale": case["rationale"],
                }
            )
        except DataAnalysisError as error:
            results.append(
                {
                    "case_id": case["case_id"],
                    "title": case["title"],
                    "expected_valid": case["expected_valid"],
                    "observed_status": "passed" if not case["expected_valid"] else "failed_unexpected_reject",
                    "recommended_action": None,
                    "predicted_probability": None,
                    "requires_manual_review": None,
                    "error_message": str(error),
                    "rationale": case["rationale"],
                }
            )
    return results


def build_sensitivity_variants(
    base_input: dict[str, int | float],
    validation_rules: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    seen_payloads: set[tuple[tuple[str, int | float], ...]] = set()

    def register_variant(variant_id: str, field_name: str, new_value: int | float, note: str) -> None:
        payload = dict(base_input)
        payload[field_name] = new_value
        payload = sanitize_simulation_input(payload, validation_rules)
        payload_key = tuple(sorted(payload.items()))
        if payload_key in seen_payloads or payload == base_input:
            return
        seen_payloads.add(payload_key)
        variants.append(
            {
                "variant_id": variant_id,
                "field_name": field_name,
                "note": note,
                "input_payload": payload,
            }
        )

    # Bu perturbasyonlar, modelin kritik girdilerde ufak oynamalara karsi ne kadar hassas davrandigini gostermek icin sinirli tutulur.
    if "TimeSpentOnWebsite" in base_input:
        time_step = max(1.0, (float(validation_rules["TimeSpentOnWebsite"]["max_value"]) - float(validation_rules["TimeSpentOnWebsite"]["min_value"])) * 0.05)
        register_variant(
            "time_spent_up",
            "TimeSpentOnWebsite",
            _bounded_numeric_update(base_input["TimeSpentOnWebsite"], validation_rules["TimeSpentOnWebsite"], time_step),
            "Site suresi sinirli miktarda artirildi.",
        )
        register_variant(
            "time_spent_down",
            "TimeSpentOnWebsite",
            _bounded_numeric_update(base_input["TimeSpentOnWebsite"], validation_rules["TimeSpentOnWebsite"], -time_step),
            "Site suresi sinirli miktarda azaltildi.",
        )

    if "NumberOfPurchases" in base_input:
        register_variant(
            "purchase_count_up",
            "NumberOfPurchases",
            _bounded_numeric_update(base_input["NumberOfPurchases"], validation_rules["NumberOfPurchases"], 1),
            "Gecmis satin alma sayisi bir artis ile test edildi.",
        )
        register_variant(
            "purchase_count_down",
            "NumberOfPurchases",
            _bounded_numeric_update(base_input["NumberOfPurchases"], validation_rules["NumberOfPurchases"], -1),
            "Gecmis satin alma sayisi bir azalis ile test edildi.",
        )

    if "AnnualIncome" in base_input:
        income_step = (float(validation_rules["AnnualIncome"]["max_value"]) - float(validation_rules["AnnualIncome"]["min_value"])) * 0.05
        register_variant(
            "income_up",
            "AnnualIncome",
            _bounded_numeric_update(base_input["AnnualIncome"], validation_rules["AnnualIncome"], income_step),
            "Gelir seviyesi ufak bir artisla tekrar denendi.",
        )
        register_variant(
            "income_down",
            "AnnualIncome",
            _bounded_numeric_update(base_input["AnnualIncome"], validation_rules["AnnualIncome"], -income_step),
            "Gelir seviyesi ufak bir azalisla tekrar denendi.",
        )

    if "LoyaltyProgram" in base_input and len(validation_rules["LoyaltyProgram"]["allowed_values"]) > 1:
        register_variant(
            "loyalty_toggle",
            "LoyaltyProgram",
            _toggle_allowed_value(int(base_input["LoyaltyProgram"]), validation_rules["LoyaltyProgram"]["allowed_values"]),
            "Sadakat programi uyeligi tersine cevrilerek karar yuzeyi yeniden okundu.",
        )

    return variants


def evaluate_sensitivity_suite(
    *,
    baseline_scenarios: list[dict[str, Any]],
    baseline_results_lookup: dict[str, dict[str, Any]],
    champion_model: Any,
    role_matrix: Any,
    validation_rules: dict[str, dict[str, Any]],
    decision_bands: dict[str, float],
    reference_policy_lookup: dict[str, dict[str, Any]] | None = None,
    global_driver_hints: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected_scenarios = _select_sensitivity_scenarios(baseline_scenarios)
    sensitivity_results: list[dict[str, Any]] = []

    for scenario in selected_scenarios:
        baseline_result = baseline_results_lookup[scenario["scenario_id"]]
        variants = build_sensitivity_variants(scenario["input_payload"], validation_rules)
        variant_results: list[dict[str, Any]] = []

        for variant in variants:
            simulated_variant = simulate_customer_profile(
                scenario={
                    "scenario_id": f"{scenario['scenario_id']}::{variant['variant_id']}",
                    "title": scenario["title"],
                    "scenario_origin": "phase10_sensitivity_variant",
                    "selection_reason": variant["note"],
                    "input_payload": variant["input_payload"],
                },
                champion_model=champion_model,
                role_matrix=role_matrix,
                validation_rules=validation_rules,
                decision_bands=decision_bands,
                reference_policy_lookup=reference_policy_lookup,
                global_driver_hints=global_driver_hints,
            )
            probability_delta = round(
                abs(float(simulated_variant["predicted_probability"]) - float(baseline_result["predicted_probability"])),
                6,
            )
            variant_results.append(
                {
                    "variant_id": variant["variant_id"],
                    "field_name": variant["field_name"],
                    "note": variant["note"],
                    "predicted_probability": simulated_variant["predicted_probability"],
                    "score_band": simulated_variant["score_band"],
                    "recommended_action": simulated_variant["recommended_action"],
                    "requires_manual_review": simulated_variant["requires_manual_review"],
                    "probability_delta": probability_delta,
                    "action_changed": simulated_variant["recommended_action"] != baseline_result["recommended_action"],
                    "band_changed": simulated_variant["score_band"] != baseline_result["score_band"],
                    "manual_review_changed": simulated_variant["requires_manual_review"] != baseline_result["requires_manual_review"],
                }
            )

        max_delta = round(max((item["probability_delta"] for item in variant_results), default=0.0), 6)
        action_change_count = int(sum(1 for item in variant_results if item["action_changed"]))
        band_change_count = int(sum(1 for item in variant_results if item["band_changed"]))
        manual_review_change_count = int(sum(1 for item in variant_results if item["manual_review_changed"]))
        sensitivity_results.append(
            {
                "scenario_id": scenario["scenario_id"],
                "title": scenario["title"],
                "baseline_probability": baseline_result["predicted_probability"],
                "baseline_action": baseline_result["recommended_action"],
                "baseline_band": baseline_result["score_band"],
                "max_probability_delta": max_delta,
                "action_change_count": action_change_count,
                "band_change_count": band_change_count,
                "manual_review_change_count": manual_review_change_count,
                "stability_assessment": "stable" if action_change_count == 0 and band_change_count == 0 else "monitor",
                "variants": variant_results,
            }
        )

    return sensitivity_results


def benchmark_runtime_surface(
    *,
    baseline_scenarios: list[dict[str, Any]],
    champion_model: Any,
    role_matrix: Any,
    validation_rules: dict[str, dict[str, Any]],
    decision_bands: dict[str, float],
    reference_policy_lookup: dict[str, dict[str, Any]] | None = None,
    global_driver_hints: list[str] | None = None,
) -> dict[str, Any]:
    if not baseline_scenarios:
        raise DataAnalysisError("Performans benchmark'i icin en az bir senaryo gereklidir.")

    single_scenario = baseline_scenarios[0]
    single_iterations = 20
    batch_iterations = 5

    single_start = time.perf_counter()
    for _ in range(single_iterations):
        simulate_customer_profile(
            scenario=single_scenario,
            champion_model=champion_model,
            role_matrix=role_matrix,
            validation_rules=validation_rules,
            decision_bands=decision_bands,
            reference_policy_lookup=reference_policy_lookup,
            global_driver_hints=global_driver_hints,
        )
    single_duration_ms = ((time.perf_counter() - single_start) / single_iterations) * 1000.0

    batch_start = time.perf_counter()
    for _ in range(batch_iterations):
        for scenario in baseline_scenarios:
            simulate_customer_profile(
                scenario=scenario,
                champion_model=champion_model,
                role_matrix=role_matrix,
                validation_rules=validation_rules,
                decision_bands=decision_bands,
                reference_policy_lookup=reference_policy_lookup,
                global_driver_hints=global_driver_hints,
            )
    batch_duration_ms = ((time.perf_counter() - batch_start) / batch_iterations) * 1000.0

    return {
        "single_scenario_avg_ms": round(single_duration_ms, 6),
        "batch_avg_ms": round(batch_duration_ms, 6),
        "scenario_count": len(baseline_scenarios),
        "throughput_scenarios_per_second": round((len(baseline_scenarios) / (batch_duration_ms / 1000.0)), 6) if batch_duration_ms > 0 else None,
        "meets_demo_budget": bool(single_duration_ms <= DEMO_SINGLE_SCENARIO_BUDGET_MS and batch_duration_ms <= DEMO_BATCH_BUDGET_MS),
        "budget_thresholds_ms": {
            "single_scenario": DEMO_SINGLE_SCENARIO_BUDGET_MS,
            "batch": DEMO_BATCH_BUDGET_MS,
        },
        "note": "SHAP hesaplamasi Faz 7'de onceden uretildigi icin Faz 10 benchmark'i demo yuzeyindeki validasyon ve skor akisini olcer.",
    }


def evaluate_reproducibility_surface(
    *,
    baseline_scenarios: list[dict[str, Any]],
    baseline_results_lookup: dict[str, dict[str, Any]],
    champion_model: Any,
    role_matrix: Any,
    validation_rules: dict[str, dict[str, Any]],
    decision_bands: dict[str, float],
    reruns: int,
    reference_policy_lookup: dict[str, dict[str, Any]] | None = None,
    global_driver_hints: list[str] | None = None,
) -> dict[str, Any]:
    run_details: list[dict[str, Any]] = []
    max_probability_delta = 0.0
    action_mismatch_count = 0
    band_mismatch_count = 0
    manual_review_mismatch_count = 0

    for run_index in range(1, reruns + 1):
        scenario_details: list[dict[str, Any]] = []
        for scenario in baseline_scenarios:
            rerun_result = simulate_customer_profile(
                scenario=scenario,
                champion_model=champion_model,
                role_matrix=role_matrix,
                validation_rules=validation_rules,
                decision_bands=decision_bands,
                reference_policy_lookup=reference_policy_lookup,
                global_driver_hints=global_driver_hints,
            )
            baseline_result = baseline_results_lookup[scenario["scenario_id"]]
            probability_delta = round(
                abs(float(rerun_result["predicted_probability"]) - float(baseline_result["predicted_probability"])),
                6,
            )
            action_match = rerun_result["recommended_action"] == baseline_result["recommended_action"]
            band_match = rerun_result["score_band"] == baseline_result["score_band"]
            manual_review_match = rerun_result["requires_manual_review"] == baseline_result["requires_manual_review"]

            max_probability_delta = max(max_probability_delta, probability_delta)
            action_mismatch_count += int(not action_match)
            band_mismatch_count += int(not band_match)
            manual_review_mismatch_count += int(not manual_review_match)
            scenario_details.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "probability_delta": probability_delta,
                    "action_match": action_match,
                    "band_match": band_match,
                    "manual_review_match": manual_review_match,
                }
            )
        run_details.append(
            {
                "run_index": run_index,
                "scenario_details": scenario_details,
            }
        )

    return {
        "runs_evaluated": reruns,
        "scenario_count": len(baseline_scenarios),
        "max_probability_delta": round(max_probability_delta, 6),
        "action_mismatch_count": int(action_mismatch_count),
        "band_mismatch_count": int(band_mismatch_count),
        "manual_review_mismatch_count": int(manual_review_mismatch_count),
        "deterministic": bool(
            max_probability_delta <= 1e-6
            and action_mismatch_count == 0
            and band_mismatch_count == 0
            and manual_review_mismatch_count == 0
        ),
        "run_details": run_details,
    }


def build_limitations_catalog(
    *,
    raw_row_count: int,
    deduplicated_row_count: int,
    phase7_summary: dict[str, Any],
    phase8_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    duplicate_count = int(raw_row_count - deduplicated_row_count)
    fairness_alerts = phase8_summary.get("phase7_context", {}).get("fairness_alerts", [])
    strongest_alert = fairness_alerts[0]["message"] if fairness_alerts else "Faz 7 fairness-lite tarafinda dikkat gerektiren alert yok."

    return [
        {
            "limitation_id": "static_snapshot_no_clickstream",
            "severity": "medium",
            "statement": "Veri seti statik musteri ozeti seviyesinde; gercek zamanli olay akisi veya clickstream sirasini tasimiyor.",
            "impact": "Canli oturum ortasi tahmin veya zaman bagimli davranis modeli savunulamaz.",
            "mitigation": "Sunum dili çevrimdışı prototip ve karar simülasyonu olarak tutuldu.",
        },
        {
            "limitation_id": "duplicate_records_cleaned_pre_split",
            "severity": "medium",
            "statement": f"Ham veri ile tekillestirilmis veri arasinda {duplicate_count} satir farki var; bu tekrarlar split oncesi temizlendi.",
            "impact": "Tekrarlar temizlenmezse sahte performans olusabilirdi.",
            "mitigation": "Tum modelleme ve simülasyon asamalari deduplicated veri ustunden yurutuldu.",
        },
        {
            "limitation_id": "discount_history_not_causal_signal",
            "severity": "high",
            "statement": "DiscountsAvailed alanı nedensel uplift sinyali olarak degil, yalnizca guardrail ve analiz girdisi olarak kullanildi.",
            "impact": "Gercek kampanya etkisini dogrudan ispatlamak mumkun degil.",
            "mitigation": "Politika katmani bu alanla model egitmek yerine aksiyon siddetini frenlemek icin kullaniyor.",
        },
        {
            "limitation_id": "proxy_business_value_only",
            "severity": "medium",
            "statement": "Politika faydasi gercek gelir, marj ve kampanya maliyeti yerine proxy islev uzerinden hesaplandi.",
            "impact": "Ekonomik getiri gercek finansal optimizasyon gibi sunulamaz.",
            "mitigation": "Tum raporlar proxy fayda ifadesini acikca etiketliyor ve Faz 9/10 bunu ayni sekilde koruyor.",
        },
        {
            "limitation_id": "fairness_alerts_require_monitoring",
            "severity": "high" if fairness_alerts else "low",
            "statement": strongest_alert,
            "impact": "Belirli segmentlerde hata dagilimi esit olmayabilir ve sunumda dikkatle yorumlanmalidir.",
            "mitigation": "Faz 8 guardrail'leri LoyaltyProgram aksiyon dagilimini izliyor; Faz 10 checklist'i bu riski acik kayit altina aliyor.",
        },
        {
            "limitation_id": "processed_dataset_generalization_risk",
            "severity": "medium",
            "statement": "Veri seti temiz ve kodlanmis gorunumu nedeniyle gercek saha dagilimlarini tam temsil etmiyor olabilir.",
            "impact": "Genellenebilirlik sinirlari final teslimde mutlaka belirtilmelidir.",
            "mitigation": "Reproducibility ve sensitivity raporlari sistemin tutarliligini gosterir, ancak saha genellenebilirligini garanti etmez.",
        },
    ]


def build_readiness_checklist(
    *,
    edge_case_results: list[dict[str, Any]],
    sensitivity_results: list[dict[str, Any]],
    performance_summary: dict[str, Any],
    reproducibility_summary: dict[str, Any],
    limitations: list[dict[str, Any]],
    phase8_summary: dict[str, Any],
    phase9_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    edge_case_failures = [item for item in edge_case_results if item["observed_status"] != "passed"]
    fairness_alert_count = len(phase8_summary.get("phase7_context", {}).get("fairness_alerts", []))

    return [
        {
            "check_id": "phase9_prototype_assets",
            "status": "hazir" if phase9_summary.get("simulation_results") else "eksik",
            "detail": "Phase 9 simülasyon sonuçları ve demo runbook artefaktlari mevcut.",
        },
        {
            "check_id": "edge_case_controls",
            "status": "hazir" if not edge_case_failures else "revizyon_gerekli",
            "detail": f"Edge-case matrisi {len(edge_case_results)} vaka uzerinde kosuldu; beklenmeyen sonuc sayisi {len(edge_case_failures)}.",
        },
        {
            "check_id": "sensitivity_analysis",
            "status": "hazir" if sensitivity_results else "eksik",
            "detail": f"Sensitivity analizi {len(sensitivity_results)} referans senaryo uzerinde tamamlandi.",
        },
        {
            "check_id": "reproducibility",
            "status": "hazir" if reproducibility_summary["deterministic"] else "revizyon_gerekli",
            "detail": f"Tekrar kosumlarda max skor farki {reproducibility_summary['max_probability_delta']} olarak olculdu.",
        },
        {
            "check_id": "performance_budget",
            "status": "hazir" if performance_summary["meets_demo_budget"] else "revizyon_gerekli",
            "detail": (
                f"Tek senaryo ortalama {performance_summary['single_scenario_avg_ms']} ms, batch {performance_summary['batch_avg_ms']} ms."
            ),
        },
        {
            "check_id": "limitations_documented",
            "status": "hazir" if limitations else "eksik",
            "detail": f"Toplam {len(limitations)} limitasyon kaydi final teslim icin listelendi.",
        },
        {
            "check_id": "fairness_followup_documented",
            "status": "hazir",
            "detail": f"Faz 7/Faz 8 fairness alert sayisi {fairness_alert_count}; guardrail ve limitasyon belgelerinde acikca kayda alindi.",
        },
    ]


class Phase10ValidationService:
    def __init__(
        self,
        loader: CsvDatasetLoader,
        logger: logging.Logger,
    ) -> None:
        self._loader = loader
        self._logger = logger

    def run(self, project_paths: ProjectPaths) -> Phase10Artifacts:
        try:
            self._logger.info("Faz 10 dogrulama ve kalite kapisi baslatildi: %s", project_paths.dataset_path)
            dataset = self._loader.load(project_paths.dataset_path)
            deduplicated_dataset = deduplicate_dataset(dataset)

            phase7_summary = self._load_json_context(project_paths.phase_7_dir / "explainability_summary.json")
            phase8_summary = self._load_json_context(project_paths.phase_8_dir / "policy_summary.json")
            phase9_summary = self._load_json_context(project_paths.phase_9_dir / "simulation_summary.json")
            if phase7_summary is None or phase8_summary is None or phase9_summary is None:
                raise DataAnalysisError("Faz 10 icin Faz 7, Faz 8 ve Faz 9 artefaktlari eksiksiz bulunmalidir.")

            current_validation_rules = build_input_validation_rules(deduplicated_dataset)
            stored_validation_rules = phase9_summary.get("input_validation_rules")
            if stored_validation_rules != to_json_safe(current_validation_rules):
                raise DataAnalysisError("Faz 10 icin Faz 9 girdi sozlesmesi ile mevcut veri kurallari uyusmuyor.")

            decision_bands = phase9_summary.get("decision_bands")
            if decision_bands != phase8_summary.get("phase5_context", {}).get("decision_bands"):
                raise DataAnalysisError("Faz 10 icin Faz 8 ve Faz 9 karar bantlari uyusmuyor.")

            role_matrix = build_feature_role_matrix()
            champion_model_name = phase9_summary["champion_model_name"]
            champion_model = self._load_champion_model(project_paths.phase_6_dir / "models" / f"{champion_model_name}.joblib")
            baseline_scenarios = extract_phase9_scenarios(phase9_summary)
            baseline_results_lookup = {
                item["scenario_id"]: item
                for item in phase9_summary["simulation_results"]
            }
            champion_key = phase7_summary.get("phase7_closeout", {}).get("champion_key")
            global_driver_hints = [
                item["feature_name"]
                for item in phase7_summary.get("model_explanations", {}).get(champion_key, {}).get("shap_global_summary", [])[:3]
            ]
            reference_policy_lookup = {
                item["scenario_id"]: item
                for item in phase8_summary.get("scenario_policy_validation", [])
            }

            edge_case_results = evaluate_edge_case_matrix(
                edge_cases=build_edge_case_catalog(current_validation_rules),
                champion_model=champion_model,
                role_matrix=role_matrix,
                validation_rules=current_validation_rules,
                decision_bands=decision_bands,
                reference_policy_lookup=reference_policy_lookup,
                global_driver_hints=global_driver_hints,
            )
            sensitivity_results = evaluate_sensitivity_suite(
                baseline_scenarios=baseline_scenarios,
                baseline_results_lookup=baseline_results_lookup,
                champion_model=champion_model,
                role_matrix=role_matrix,
                validation_rules=current_validation_rules,
                decision_bands=decision_bands,
                reference_policy_lookup=reference_policy_lookup,
                global_driver_hints=global_driver_hints,
            )
            performance_summary = benchmark_runtime_surface(
                baseline_scenarios=baseline_scenarios,
                champion_model=champion_model,
                role_matrix=role_matrix,
                validation_rules=current_validation_rules,
                decision_bands=decision_bands,
                reference_policy_lookup=reference_policy_lookup,
                global_driver_hints=global_driver_hints,
            )
            reproducibility_summary = evaluate_reproducibility_surface(
                baseline_scenarios=baseline_scenarios,
                baseline_results_lookup=baseline_results_lookup,
                champion_model=champion_model,
                role_matrix=role_matrix,
                validation_rules=current_validation_rules,
                decision_bands=decision_bands,
                reruns=REPRODUCIBILITY_RERUNS,
                reference_policy_lookup=reference_policy_lookup,
                global_driver_hints=global_driver_hints,
            )
            limitations = build_limitations_catalog(
                raw_row_count=len(dataset),
                deduplicated_row_count=len(deduplicated_dataset),
                phase7_summary=phase7_summary,
                phase8_summary=phase8_summary,
            )
            readiness_checklist = build_readiness_checklist(
                edge_case_results=edge_case_results,
                sensitivity_results=sensitivity_results,
                performance_summary=performance_summary,
                reproducibility_summary=reproducibility_summary,
                limitations=limitations,
                phase8_summary=phase8_summary,
                phase9_summary=phase9_summary,
            )

            artifacts = self._ensure_phase10_directories(project_paths.phase_10_dir)
            summary = self._build_summary(
                dataset_path=project_paths.dataset_path,
                deduplicated_row_count=len(deduplicated_dataset),
                champion_model_name=champion_model_name,
                edge_case_results=edge_case_results,
                sensitivity_results=sensitivity_results,
                performance_summary=performance_summary,
                reproducibility_summary=reproducibility_summary,
                limitations=limitations,
                readiness_checklist=readiness_checklist,
            )

            self._write_json_file(artifacts.json_summary_path, summary)
            self._write_edge_case_matrix(artifacts.edge_case_matrix_path, edge_case_results)
            self._write_json_file(artifacts.sensitivity_analysis_path, {"sensitivity_results": sensitivity_results})
            self._write_json_file(artifacts.performance_benchmark_path, performance_summary)
            self._write_reproducibility_note(artifacts.reproducibility_note_path, reproducibility_summary)
            self._write_limitations(artifacts.limitations_path, limitations)
            self._write_readiness_checklist(artifacts.readiness_checklist_path, readiness_checklist)
            self._write_markdown_report(artifacts.markdown_report_path, summary)
            self._logger.info("Faz 10 dogrulama ve kalite kapisi tamamlandi.")
            return artifacts
        except DataAnalysisError:
            self._logger.exception("Faz 10 dogrulama ve kalite kapisi veri dogrulamasi nedeniyle durdu.")
            raise
        except (KeyError, OSError, TypeError, ValueError) as error:
            self._logger.exception("Faz 10 dogrulama ve kalite kapisi beklenmeyen bir hata ile durdu.")
            raise DataAnalysisError("Faz 10 dogrulama ve kalite kapisi tamamlanamadi.") from error

    def _ensure_phase10_directories(self, phase_10_dir: Path) -> Phase10Artifacts:
        phase_10_dir.mkdir(parents=True, exist_ok=True)
        return Phase10Artifacts(
            markdown_report_path=phase_10_dir / "validation_report.md",
            json_summary_path=phase_10_dir / "validation_summary.json",
            edge_case_matrix_path=phase_10_dir / "edge_case_matrix.csv",
            sensitivity_analysis_path=phase_10_dir / "sensitivity_analysis.json",
            performance_benchmark_path=phase_10_dir / "performance_benchmark.json",
            reproducibility_note_path=phase_10_dir / "reproducibility_note.md",
            limitations_path=phase_10_dir / "limitations.md",
            readiness_checklist_path=phase_10_dir / "readiness_checklist.md",
        )

    def _load_json_context(self, file_path: Path) -> dict[str, Any] | None:
        if not file_path.exists():
            return None
        try:
            with file_path.open("r", encoding="utf-8") as input_file:
                return json.load(input_file)
        except (OSError, json.JSONDecodeError) as error:
            self._logger.warning("Faz 10 baglam dosyasi okunamadi: %s | %s", file_path, error)
            return None

    def _load_champion_model(self, model_path: Path) -> Any:
        if not model_path.exists():
            raise DataAnalysisError(f"Faz 10 icin champion model artefakti bulunamadi: {model_path}")
        try:
            return joblib.load(model_path)
        except (OSError, ValueError) as error:
            raise DataAnalysisError(f"Faz 10 icin champion model artefakti yuklenemedi: {model_path}") from error

    def _build_summary(
        self,
        *,
        dataset_path: Path,
        deduplicated_row_count: int,
        champion_model_name: str,
        edge_case_results: list[dict[str, Any]],
        sensitivity_results: list[dict[str, Any]],
        performance_summary: dict[str, Any],
        reproducibility_summary: dict[str, Any],
        limitations: list[dict[str, Any]],
        readiness_checklist: list[dict[str, Any]],
    ) -> dict[str, Any]:
        edge_case_summary = {
            "total_cases": len(edge_case_results),
            "passed_cases": int(sum(1 for item in edge_case_results if item["observed_status"] == "passed")),
            "failed_cases": int(sum(1 for item in edge_case_results if item["observed_status"] != "passed")),
        }
        readiness_statuses = [item["status"] for item in readiness_checklist]
        overall_status = "hazir"
        if any(status == "revizyon_gerekli" for status in readiness_statuses):
            overall_status = "revizyon_gerekli"
        elif any(status == "eksik" for status in readiness_statuses):
            overall_status = "eksik"

        return {
            "dataset_path": str(dataset_path),
            "deduplicated_row_count": deduplicated_row_count,
            "champion_model_name": champion_model_name,
            "edge_case_summary": edge_case_summary,
            "edge_case_results": edge_case_results,
            "sensitivity_analysis": sensitivity_results,
            "performance_summary": performance_summary,
            "reproducibility_summary": reproducibility_summary,
            "limitations": limitations,
            "readiness_checklist": readiness_checklist,
            "phase10_closeout": {
                "overall_status": overall_status,
                "summary": "Faz 10'da edge-case, sensitivity, performans ve tekrar uretilebilirlik kontrolleri ayni prototip yuzeyi uzerinde tamamlandi.",
                "next_step": "Faz 11'de bu kalite kapisi sonucunu rapor, sunum ve teslim paketine dogrudan yansitmak gerekir.",
            },
        }

    def _write_json_file(self, output_path: Path, payload: dict[str, Any]) -> None:
        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                json.dump(to_json_safe(payload), output_file, ensure_ascii=False, indent=2)
        except OSError as error:
            raise ArtifactWriteError(f"Faz 10 JSON artefakti yazilamadi: {output_path}") from error

    def _write_edge_case_matrix(self, output_path: Path, edge_case_results: list[dict[str, Any]]) -> None:
        try:
            pd.DataFrame(edge_case_results).to_csv(output_path, index=False, encoding="utf-8")
        except OSError as error:
            raise ArtifactWriteError(f"Faz 10 edge-case matrisi yazilamadi: {output_path}") from error

    def _write_reproducibility_note(self, output_path: Path, reproducibility_summary: dict[str, Any]) -> None:
        lines = [
            "# Faz 10 Reproducibility Note",
            "",
            f"- Rerun sayisi: {reproducibility_summary['runs_evaluated']}",
            f"- Senaryo sayisi: {reproducibility_summary['scenario_count']}",
            f"- Max skor farki: {reproducibility_summary['max_probability_delta']}",
            f"- Action mismatch sayisi: {reproducibility_summary['action_mismatch_count']}",
            f"- Band mismatch sayisi: {reproducibility_summary['band_mismatch_count']}",
            f"- Manual review mismatch sayisi: {reproducibility_summary['manual_review_mismatch_count']}",
            f"- Deterministik sonuc: {reproducibility_summary['deterministic']}",
        ]
        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                output_file.write("\n".join(lines))
        except OSError as error:
            raise ArtifactWriteError(f"Faz 10 reproducibility notu yazilamadi: {output_path}") from error

    def _write_limitations(self, output_path: Path, limitations: list[dict[str, Any]]) -> None:
        lines = [
            "# Faz 10 Limitasyon Listesi",
            "",
        ]
        for limitation in limitations:
            lines.extend(
                [
                    f"## {limitation['limitation_id']}",
                    "",
                    f"- Seviye: {limitation['severity']}",
                    f"- Tespit: {limitation['statement']}",
                    f"- Etki: {limitation['impact']}",
                    f"- Azaltma yaklasimi: {limitation['mitigation']}",
                    "",
                ]
            )
        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                output_file.write("\n".join(lines))
        except OSError as error:
            raise ArtifactWriteError(f"Faz 10 limitasyon listesi yazilamadi: {output_path}") from error

    def _write_readiness_checklist(self, output_path: Path, readiness_checklist: list[dict[str, Any]]) -> None:
        lines = [
            "# Faz 10 Readiness Checklist",
            "",
        ]
        for entry in readiness_checklist:
            lines.append(f"- {entry['check_id']}: durum={entry['status']} | detay={entry['detail']}")
        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                output_file.write("\n".join(lines))
        except OSError as error:
            raise ArtifactWriteError(f"Faz 10 readiness checklist yazilamadi: {output_path}") from error

    def _write_markdown_report(self, output_path: Path, summary: dict[str, Any]) -> None:
        lines = [
            "# Faz 10 Dogrulama ve Kalite Kapisi Raporu",
            "",
            f"- Veri seti: `{summary['dataset_path']}`",
            f"- Tekillestirilmis satir sayisi: {summary['deduplicated_row_count']}",
            f"- Champion model: {summary['champion_model_name']}",
            "",
            "## Edge Case Ozeti",
            "",
            f"- {summary['edge_case_summary']}",
            "",
            "## Sensitivity Analizi",
            "",
        ]
        for entry in summary["sensitivity_analysis"]:
            lines.append(
                f"- {entry['scenario_id']}: max_delta={entry['max_probability_delta']}, action_changes={entry['action_change_count']}, band_changes={entry['band_change_count']}, durum={entry['stability_assessment']}"
            )

        lines.extend([
            "",
            "## Performans",
            "",
            f"- {summary['performance_summary']}",
            "",
            "## Reproducibility",
            "",
            f"- {summary['reproducibility_summary']}",
            "",
            "## Limitasyonlar",
            "",
        ])
        for limitation in summary["limitations"]:
            lines.append(f"- {limitation['limitation_id']}: {limitation['statement']}")

        lines.extend([
            "",
            "## Readiness Checklist",
            "",
        ])
        for entry in summary["readiness_checklist"]:
            lines.append(f"- {entry['check_id']}: durum={entry['status']} | detay={entry['detail']}")

        lines.extend([
            "",
            "## Faz 10 Kapanis",
            "",
            f"- Genel durum: {summary['phase10_closeout']['overall_status']}",
            f"- Ozet: {summary['phase10_closeout']['summary']}",
            f"- Sonraki adim: {summary['phase10_closeout']['next_step']}",
        ])
        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                output_file.write("\n".join(lines))
        except OSError as error:
            raise ArtifactWriteError(f"Faz 10 markdown raporu yazilamadi: {output_path}") from error


def _bounded_numeric_update(current_value: int | float, rule: dict[str, Any], delta: float) -> int | float:
    min_value = float(rule["min_value"])
    max_value = float(rule["max_value"])
    candidate_value = float(np.clip(float(current_value) + delta, min_value, max_value))
    if rule["kind"] == "integer_range":
        return int(round(candidate_value))
    return round(candidate_value, 6)


def _toggle_allowed_value(current_value: int, allowed_values: list[int]) -> int:
    normalized_values = list(sorted(int(value) for value in allowed_values))
    if current_value not in normalized_values:
        return normalized_values[0]
    current_index = normalized_values.index(current_value)
    return normalized_values[(current_index + 1) % len(normalized_values)]


def _select_sensitivity_scenarios(baseline_scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenario_lookup = {scenario["scenario_id"]: scenario for scenario in baseline_scenarios}
    selected: list[dict[str, Any]] = []
    for scenario_id in SENSITIVITY_PRIORITY_SCENARIOS:
        scenario = scenario_lookup.get(scenario_id)
        if scenario is not None:
            selected.append(scenario)

    if len(selected) < 4:
        for scenario in baseline_scenarios:
            if scenario["scenario_id"] in {item["scenario_id"] for item in selected}:
                continue
            selected.append(scenario)
            if len(selected) == 4:
                break
    return selected