import { describe, expect, it } from "vitest";

import { buildInitialValues, calculateDiscountRateFromScore, evaluateOffer, validateInputValues } from "./offer-engine";

describe("offer engine", () => {
  it("maps purchase score to discount with the new 75-threshold rule", () => {
    expect(calculateDiscountRateFromScore(75)).toBe(0);
    expect(calculateDiscountRateFromScore(74)).toBe(0.01);
    expect(calculateDiscountRateFromScore(73)).toBe(0.01);
    expect(calculateDiscountRateFromScore(71)).toBe(0.02);
    expect(calculateDiscountRateFromScore(0)).toBe(0.38);
  });

  it("returns no discount for a strong loyalty profile", () => {
    const result = evaluateOffer({
      Age: 28,
      Gender: 1,
      AnnualIncome: 72663.35,
      NumberOfPurchases: 17,
      ProductCategory: 1,
      TimeSpentOnWebsite: 58.9,
      LoyaltyProgram: 1,
    });

    expect(result.riskBand).toBe("high_confidence_no_discount");
    expect(result.applyAutomaticDiscount).toBe(false);
    expect(result.automaticDiscountRate).toBe(0);
  });

  it("returns a discount for a medium-risk shopper", () => {
    const result = evaluateOffer({
      Age: 35,
      Gender: 0,
      AnnualIncome: 120000,
      NumberOfPurchases: 3,
      ProductCategory: 2,
      TimeSpentOnWebsite: 48,
      LoyaltyProgram: 0,
    });

    expect(["lower_targeting_band", "upper_targeting_band"]).toContain(result.riskBand);
    expect(result.actionKey).not.toBe("protect_margin_no_discount");
    expect(result.applyAutomaticDiscount).toBe(true);
    expect(result.automaticDiscountRate).toBe(calculateDiscountRateFromScore(Math.round(result.probability * 100)));
  });

  it("validates out-of-range values defensively", () => {
    const errors = validateInputValues({
      ...buildInitialValues(),
      Age: 12,
    });

    expect(errors.Age).toBeTruthy();
  });

  it("does not expose DiscountsAvailed in frontend defaults", () => {
    expect("DiscountsAvailed" in buildInitialValues()).toBe(false);
  });
});
