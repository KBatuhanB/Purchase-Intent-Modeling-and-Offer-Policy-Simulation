import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const currentFilePath = fileURLToPath(import.meta.url);
const frontendRoot = resolve(dirname(currentFilePath), "..");
const repoRoot = resolve(frontendRoot, "..");

function readJson(relativePath) {
  const absolutePath = resolve(repoRoot, relativePath);
  const rawContent = readFileSync(absolutePath, "utf-8");
  return JSON.parse(rawContent);
}

function floorToStep(value, step) {
  return Math.floor(value / step) * step;
}

function ceilToStep(value, step) {
  return Math.ceil(value / step) * step;
}

function roundToStep(value, step) {
  return Math.round(value / step) * step;
}

function buildUiFriendlySchema(schema) {
  const excludedFields = new Set(["DiscountsAvailed"]);
  const purposeOverrides = {
    Age: "Müşterinin yaşı; segment ve davranış bağlamını destekler.",
    Gender: "Müşterinin cinsiyeti (0: Erkek, 1: Kadın)",
    AnnualIncome: "Müşterinin yıllık gelir seviyesi.",
    NumberOfPurchases: "Müşterinin geçmiş satın alma sayısı.",
    ProductCategory: "Satın alınan ürünün kategorisi (0: Elektronik, 1: Giyim, 2: Ev Yaşam, 3: Güzellik, 4: Spor)",
    TimeSpentOnWebsite: "Müşterinin sitede geçirdiği süre.",
    LoyaltyProgram: "Müşterinin sadakat programına üyeliği (0: Hayır, 1: Evet)",
  };
  const roundingSteps = {
    AnnualIncome: 1000,
    TimeSpentOnWebsite: 1,
  };

  return Object.fromEntries(
    Object.entries(schema)
      .filter(([fieldName]) => !excludedFields.has(fieldName))
      .map(([fieldName, rule]) => {
      const nextRule = { ...rule };

      if (purposeOverrides[fieldName]) {
        nextRule.purpose = purposeOverrides[fieldName];
      }

      if (rule.kind === "float_range") {
        const roundingStep = roundingSteps[fieldName] ?? 1;
        nextRule.min_value = floorToStep(rule.min_value, roundingStep);
        nextRule.max_value = ceilToStep(rule.max_value, roundingStep);
        nextRule.example_value = roundToStep(rule.example_value, roundingStep);
      }

        return [fieldName, nextRule];
      }),
  );
}

function localizeActionCatalog(actionCatalog) {
  const localizedContent = {
    holdout_low_cost_nurture: {
      title: "Düşük maliyetli hatırlatma",
      description: "Düşük olasılıklı müşteriye indirim yerine hafif bir hatırlatma uygulanır.",
    },
    targeted_standard_discount: {
      title: "Standart hedefli indirim",
      description: "Kararsız müşteride dönüşümü artırmak için kontrollü indirim uygulanır.",
    },
    targeted_light_discount: {
      title: "Hafif indirim veya paket",
      description: "Satın almaya yakın müşteride marjı koruyan hafif teklif uygulanır.",
    },
    protect_margin_no_discount: {
      title: "Marjı koru, indirim verme",
      description: "Müşteri zaten satın almaya yakın olduğu için gereksiz indirim uygulanmaz.",
    },
    manual_review_discount_cap: {
      title: "Manuel inceleme ve indirim sınırı",
      description: "Belirsiz vakalarda otomatik karar durdurulur ve karar manuel incelemeye bırakılır.",
    },
  };

  return actionCatalog.map((actionItem) => {
    const localizedAction = localizedContent[actionItem.action_key];
    if (!localizedAction) {
      return actionItem;
    }

    return {
      ...actionItem,
      title: localizedAction.title,
      description: localizedAction.description,
    };
  });
}

function buildFakeProductTheme() {
  return {
    marketplaceName: "PazarNova",
    productTitle: "Nova X17 512 GB Ada Çayı",
    brand: "Nova Mobile",
    priceTry: 89999,
    rating: 4.8,
    reviewCount: 16,
    monthlyPaymentLabel: "3 ay 31.796 TL'den başlayan ödeme seçenekleri",
    heroBadges: ["Hızlı Teslimat", "Sigortaya Uygun", "Demo Sahnesi"],
    sellerCards: [
      {
        sellerName: "PazarNova",
        sellerScore: 9.2,
        badge: "Hızlı Satıcı",
        priceTry: 89999,
      },
      {
        sellerName: "Mediatel",
        sellerScore: 9.1,
        badge: "Kargo Bedava",
        priceTry: 90999,
      },
    ],
    galleryLabels: ["Arka Görünüm", "Ön Görünüm", "Kamera", "Kenar Detayı", "Kutu"],
    accentColor: "#f27a1a",
  };
}

// Frontend icindeki veri kopyasi bilincli olarak build-time uretilir; boylece form kisitlari ve politika bantlari Python artefaktlari ile ayni kaynaktan gelir.
function buildProjectContext() {
  const inputSchema = buildUiFriendlySchema(readJson("artifacts/phase_9/input_schema.json"));
  const policySummary = readJson("artifacts/phase_8/policy_summary.json");
  const simulationSummary = readJson("artifacts/phase_9/simulation_summary.json");
  const validationSummary = readJson("artifacts/phase_10/validation_summary.json");
  const deliverySummary = readJson("artifacts/phase_11/delivery_summary.json");

  return {
    schema: inputSchema,
    policy: {
      championModelName: deliverySummary.champion_model_name,
      readinessStatus: deliverySummary.readiness_status,
      validatedMetrics: deliverySummary.key_metrics,
      decisionBands: policySummary.phase5_context.decision_bands,
      policyBandSummary: policySummary.phase5_context.policy_band_summary,
      actionCatalog: localizeActionCatalog(policySummary.action_catalog),
      businessInsights: policySummary.phase7_context.business_insights,
      fairnessAlerts: policySummary.phase7_context.fairness_alerts,
      simulationOverview: simulationSummary.scenario_catalog_overview,
      actionDistribution: simulationSummary.action_distribution,
      validationOverview: {
        edgeCaseSummary: validationSummary.edge_case_summary,
        reproducibilitySummary: validationSummary.reproducibility_summary,
        performanceSummary: validationSummary.performance_summary,
      },
    },
    ui: {
      fakeProduct: buildFakeProductTheme(),
    },
  };
}

const outputDirectory = resolve(frontendRoot, "src", "data", "generated");
const outputFilePath = resolve(outputDirectory, "project-context.json");
const projectContext = buildProjectContext();

mkdirSync(outputDirectory, { recursive: true });
writeFileSync(outputFilePath, JSON.stringify(projectContext, null, 2), "utf-8");
console.log(`Frontend data synced to ${outputFilePath}`);
