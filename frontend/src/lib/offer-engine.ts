import { getActionSpec, projectContext } from "./project-context";
import type { FieldRule, OfferResult, RangeFieldRule, UserInputFieldName, UserInputValues } from "../types";

type ValidationErrors = Partial<Record<keyof UserInputValues, string>>;

function normalize(value: number, min: number, max: number): number {
  if (max <= min) {
    return 0;
  }
  return (value - min) / (max - min);
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

function clampProbability(value: number): number {
  return Number(Math.min(0.985, Math.max(0.015, value)).toFixed(6));
}

function toPurchaseScorePercent(probability: number): number {
  return Math.max(0, Math.min(100, Math.round(probability * 100)));
}

export function calculateDiscountRateFromScore(purchaseScorePercent: number): number {
  const boundedScore = Math.max(0, Math.min(100, Math.round(purchaseScorePercent)));
  if (boundedScore >= 75) {
    return 0;
  }

  const discountPercent = Math.ceil((75 - boundedScore) / 2);
  return Number((discountPercent / 100).toFixed(2));
}

function formatConstraintValue(value: number): string {
  if (Number.isInteger(value)) {
    return String(value);
  }
  return String(Number(value.toFixed(1)));
}

function rangeStep(fieldName: UserInputFieldName, rule: FieldRule): number {
  if (fieldName === "AnnualIncome") {
    return 1000;
  }
  if (fieldName === "TimeSpentOnWebsite") {
    return 1;
  }
  if (rule.kind === "float_range" && Number.isInteger(rule.min_value) && Number.isInteger(rule.max_value)) {
    return 1;
  }
  return rule.kind === "float_range" ? 0.001 : 1;
}

export function buildInitialValues(): UserInputValues {
  const schema = projectContext.schema;
  return {
    Age: schema.Age.example_value,
    Gender: schema.Gender.example_value,
    AnnualIncome: schema.AnnualIncome.example_value,
    NumberOfPurchases: schema.NumberOfPurchases.example_value,
    ProductCategory: schema.ProductCategory.example_value,
    TimeSpentOnWebsite: schema.TimeSpentOnWebsite.example_value,
    LoyaltyProgram: schema.LoyaltyProgram.example_value,
  };
}

export function validateInputValues(values: UserInputValues): ValidationErrors {
  const errors: ValidationErrors = {};

  (Object.entries(projectContext.schema) as Array<[UserInputFieldName, FieldRule]>).forEach(([fieldName, rule]) => {
    const rawValue = values[fieldName];

    if (!Number.isFinite(rawValue)) {
      errors[fieldName] = "Gecerli bir sayisal deger girin.";
      return;
    }

    if (rule.kind === "integer_range") {
      if (!Number.isInteger(rawValue)) {
        errors[fieldName] = "Bu alan tam sayi olmali.";
        return;
      }
      if (rawValue < rule.min_value || rawValue > rule.max_value) {
        errors[fieldName] = `Deger ${formatConstraintValue(rule.min_value)} ile ${formatConstraintValue(rule.max_value)} arasinda olmali.`;
      }
      return;
    }

    if (rule.kind === "float_range") {
      if (rawValue < rule.min_value || rawValue > rule.max_value) {
        errors[fieldName] = `Deger ${formatConstraintValue(rule.min_value)} ile ${formatConstraintValue(rule.max_value)} arasinda olmali.`;
      }
      return;
    }

    if (rule.kind === "categorical_integer" && !rule.allowed_values.includes(rawValue)) {
      errors[fieldName] = `Izin verilen degerler: ${rule.allowed_values.join(", ")}`;
    }
  });

  return errors;
}

export function getFieldStep(fieldName: UserInputFieldName): number {
  const rule = projectContext.schema[fieldName];
  return rule.kind === "categorical_integer" ? 1 : rangeStep(fieldName, rule);
}

export function evaluateOffer(values: UserInputValues): OfferResult {
  const ageRule = getRangeRule("Age");
  const incomeRule = getRangeRule("AnnualIncome");
  const purchaseRule = getRangeRule("NumberOfPurchases");
  const timeRule = getRangeRule("TimeSpentOnWebsite");
  const decisionBands = projectContext.policy.decisionBands;

  const ageMomentum = 1 - normalize(values.Age, ageRule.min_value, ageRule.max_value);
  const incomeValue = normalize(values.AnnualIncome, incomeRule.min_value, incomeRule.max_value);
  const purchaseValue = normalize(
    values.NumberOfPurchases,
    purchaseRule.min_value,
    purchaseRule.max_value,
  );
  const timeValue = normalize(
    values.TimeSpentOnWebsite,
    timeRule.min_value,
    timeRule.max_value,
  );

  const categoryWeightMap: Record<number, number> = {
    0: -0.05,
    1: 0.03,
    2: 0.01,
    3: 0.05,
    4: 0,
  };

  const contributions = {
    LoyaltyProgram: values.LoyaltyProgram === 1 ? 1.55 : 0,
    TimeSpentOnWebsite: timeValue * 1.1,
    NumberOfPurchases: purchaseValue * 1.05,
    AnnualIncome: incomeValue * 0.55,
    Age: ageMomentum * 0.45,
    ProductCategory: categoryWeightMap[values.ProductCategory] ?? 0,
    Gender: values.Gender === 1 ? 0.01 : -0.01,
  };

  let rawScore =
    -2.45 +
    contributions.LoyaltyProgram +
    contributions.TimeSpentOnWebsite +
    contributions.NumberOfPurchases +
    contributions.AnnualIncome +
    contributions.Age +
    contributions.ProductCategory +
    contributions.Gender;

  // Bu ek kurallar, Phase 7 ve Phase 10'da gozlenen davranis yuzeyini frontend demosunda daha inandirici bir sekilde taklit eder.
  if (timeValue > 0.72 && ageMomentum > 0.8) {
    rawScore += 0.9;
  }
  if (timeValue > 0.72 && purchaseValue < 0.2) {
    rawScore += 0.14;
  }
  if (values.LoyaltyProgram === 1 && purchaseValue > 0.7) {
    rawScore += 0.22;
  }

  const probability = clampProbability(sigmoid(rawScore));
  const purchaseScorePercent = toPurchaseScorePercent(probability);
  const predictedLabel = probability >= decisionBands.binary_decision_threshold ? 1 : 0;

  let riskBand: OfferResult["riskBand"];
  let actionKey: string;
  if (probability < decisionBands.low_action_threshold) {
    riskBand = "low_intent_holdout";
    actionKey = "holdout_low_cost_nurture";
  } else if (probability < decisionBands.binary_decision_threshold) {
    riskBand = "lower_targeting_band";
    actionKey = "targeted_standard_discount";
  } else if (probability < decisionBands.high_confidence_no_discount_threshold) {
    riskBand = "upper_targeting_band";
    actionKey = "targeted_light_discount";
  } else {
    riskBand = "high_confidence_no_discount";
    actionKey = "protect_margin_no_discount";
  }

  const guardrailFlags: string[] = [];
  let requiresManualReview = false;

  const thresholdDistance = Math.min(
    Math.abs(probability - decisionBands.binary_decision_threshold),
    Math.abs(probability - decisionBands.high_confidence_no_discount_threshold),
  );
  if (thresholdDistance <= 0.025) {
    requiresManualReview = true;
    guardrailFlags.push("Karar skoru esik yakininda; manuel kontrol onerilir.");
  }

  if (requiresManualReview) {
    actionKey = "manual_review_discount_cap";
  }

  const actionSpec = getActionSpec(actionKey);
  if (!actionSpec) {
    throw new Error(`Action spec not found: ${actionKey}`);
  }

  const automaticDiscountRate = calculateDiscountRateFromScore(purchaseScorePercent);
  const applyAutomaticDiscount = automaticDiscountRate > 0;
  const discountedPriceTry = Math.round(projectContext.ui.fakeProduct.priceTry * (1 - automaticDiscountRate));

  const topDriverHints = Object.entries(contributions)
    .sort(([, leftValue], [, rightValue]) => Math.abs(rightValue) - Math.abs(leftValue))
    .slice(0, 3)
    .map(([featureName]) => featureName);

  const summary = buildSummary({
    purchaseScorePercent,
    probability,
    riskBand,
    actionSpec,
    requiresManualReview,
    automaticDiscountRate,
    guardrailFlags,
  });

  return {
    probability,
    predictedLabel,
    riskBand,
    actionKey: actionSpec.action_key,
    actionTitle: actionSpec.title,
    automaticDiscountRate,
    discountedPriceTry,
    requiresManualReview,
    guardrailFlags,
    topDriverHints,
    applyAutomaticDiscount,
    summary,
    actionDescription: actionSpec.description,
  };
}

function buildSummary(input: {
  purchaseScorePercent: number;
  probability: number;
  riskBand: OfferResult["riskBand"];
  actionSpec: NonNullable<ReturnType<typeof getActionSpec>>;
  requiresManualReview: boolean;
  automaticDiscountRate: number;
  guardrailFlags: string[];
}): string {
  if (input.requiresManualReview && input.automaticDiscountRate > 0) {
    return `Satın alma skoru %${input.purchaseScorePercent} seviyesine düştüğü için %${Math.round(input.automaticDiscountRate * 100)} indirim önerildi; ancak güvenlik kuralı nedeniyle önce manuel kontrol gerekli.`;
  }

  if (input.requiresManualReview) {
    return `Skor eşik değerine yakın olduğu için karar manuel kontrolde tutuldu.`;
  }

  if (input.automaticDiscountRate > 0) {
    return `Satın alma skoru %${input.purchaseScorePercent} seviyesinde olduğu için %${Math.round(input.automaticDiscountRate * 100)} otomatik indirim uygulandı.`;
  }

  if (input.guardrailFlags.length > 0) {
    return `${input.actionSpec.title} seçildi. Ek güvenlik notu: ${input.guardrailFlags.join(" ")}`;
  }

  return `Satın alma skoru %${input.purchaseScorePercent} olduğu için indirim uygulanmadı.`;
}

function getRangeRule(fieldName: UserInputFieldName): RangeFieldRule {
  const rule = projectContext.schema[fieldName];
  if (rule.kind === "categorical_integer") {
    throw new Error(`Range rule expected for ${fieldName}`);
  }
  return rule;
}
