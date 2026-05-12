export type UserInputFieldName =
  | "Age"
  | "Gender"
  | "AnnualIncome"
  | "NumberOfPurchases"
  | "ProductCategory"
  | "TimeSpentOnWebsite"
  | "LoyaltyProgram";

export interface RangeFieldRule {
  required: boolean;
  purpose: string;
  kind: "integer_range" | "float_range";
  min_value: number;
  max_value: number;
  example_value: number;
}

export interface CategoricalFieldRule {
  required: boolean;
  purpose: string;
  kind: "categorical_integer";
  allowed_values: number[];
  example_value: number;
}

export type FieldRule = RangeFieldRule | CategoricalFieldRule;

export interface ActionSpec {
  action_key: string;
  title: string;
  description: string;
  default_discount_rate: number;
  expected_uplift: number;
  contact_cost: number;
  primary_channel: string;
  target_bands: string[];
}

export interface PolicyBandSummary {
  count: number;
  share: number;
  observed_purchase_rate: number;
  average_probability: number;
}

export interface ValidatedMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  pr_auc: number;
  balanced_accuracy: number;
  brier_score: number;
  confusion_matrix: number[][];
}

export interface FakeProductTheme {
  marketplaceName: string;
  productTitle: string;
  brand: string;
  priceTry: number;
  rating: number;
  reviewCount: number;
  monthlyPaymentLabel: string;
  heroBadges: string[];
  sellerCards: Array<{
    sellerName: string;
    sellerScore: number;
    badge: string;
    priceTry: number;
  }>;
  galleryLabels: string[];
  accentColor: string;
}

export interface ProjectContext {
  schema: Record<UserInputFieldName, FieldRule>;
  policy: {
    championModelName: string;
    readinessStatus: string;
    validatedMetrics: ValidatedMetrics;
    decisionBands: {
      low_action_threshold: number;
      binary_decision_threshold: number;
      high_confidence_no_discount_threshold: number;
    };
    policyBandSummary: Record<string, PolicyBandSummary>;
    actionCatalog: ActionSpec[];
    businessInsights: string[];
    fairnessAlerts: Array<{
      group_name: string;
      group_value: string;
      metric_name: string;
      gap_value: number;
      message: string;
    }>;
    simulationOverview: {
      total_scenarios: number;
      origin_counts: Record<string, number>;
      manual_review_share: number;
    };
    actionDistribution: Record<string, number>;
    validationOverview: {
      edgeCaseSummary: {
        total_cases: number;
        passed_cases: number;
        failed_cases: number;
      };
      reproducibilitySummary: {
        deterministic: boolean;
        max_probability_delta: number;
      };
      performanceSummary: {
        meets_demo_budget: boolean;
        single_scenario_avg_ms: number;
        batch_avg_ms: number;
      };
    };
  };
  ui: {
    fakeProduct: FakeProductTheme;
  };
}

export type UserInputValues = Record<UserInputFieldName, number>;

export interface OfferResult {
  probability: number;
  predictedLabel: 0 | 1;
  riskBand: "low_intent_holdout" | "lower_targeting_band" | "upper_targeting_band" | "high_confidence_no_discount";
  actionKey: string;
  actionTitle: string;
  automaticDiscountRate: number;
  discountedPriceTry: number;
  requiresManualReview: boolean;
  guardrailFlags: string[];
  topDriverHints: string[];
  applyAutomaticDiscount: boolean;
  summary: string;
  actionDescription: string;
}
