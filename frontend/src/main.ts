import "./styles.css";

import { formatPercent, formatTry } from "./lib/formatters";
import { buildInitialValues, evaluateOffer, getFieldStep, validateInputValues } from "./lib/offer-engine";
import { orderedFieldNames, projectContext } from "./lib/project-context";
import type { CategoricalFieldRule, FieldRule, OfferResult, RangeFieldRule, UserInputFieldName, UserInputValues } from "./types";

type Screen = "capture" | "offer";

interface AppState {
  screen: Screen;
  values: UserInputValues;
  errors: Partial<Record<keyof UserInputValues, string>>;
  result: OfferResult | null;
}

interface UserProfilePreset {
  key: "low" | "medium" | "high";
  title: string;
  description: string;
  values: UserInputValues;
}

const rootElement = document.querySelector<HTMLDivElement>("#app");

if (!rootElement) {
  throw new Error("App root not found.");
}

const appRoot: HTMLDivElement = rootElement;

const appState: AppState = {
  screen: "capture",
  values: buildInitialValues(),
  errors: {},
  result: null,
};

const userProfilePresets: UserProfilePreset[] = [
  {
    key: "low",
    title: "Düşük satın alma ihtimali",
    description: "Kısa ziyaretli, düşük etkileşimli örnek profil.",
    values: {
      Age: 62,
      Gender: 0,
      AnnualIncome: 25000,
      NumberOfPurchases: 0,
      ProductCategory: 0,
      TimeSpentOnWebsite: 3,
      LoyaltyProgram: 0,
    },
  },
  {
    key: "medium",
    title: "Orta satın alma ihtimali",
    description: "Hedeflenebilir ama kararsız müşteri örneği.",
    values: {
      Age: 35,
      Gender: 0,
      AnnualIncome: 120000,
      NumberOfPurchases: 3,
      ProductCategory: 2,
      TimeSpentOnWebsite: 48,
      LoyaltyProgram: 0,
    },
  },
  {
    key: "high",
    title: "Yüksek satın alma ihtimali",
    description: "Güçlü sinyalli, genelde indirimsiz alınabilir profil.",
    values: {
      Age: 28,
      Gender: 1,
      AnnualIncome: 72663.35,
      NumberOfPurchases: 17,
      ProductCategory: 1,
      TimeSpentOnWebsite: 58.9,
      LoyaltyProgram: 1,
    },
  },
];

function renderApp() {
  appRoot.innerHTML = appState.screen === "capture" ? renderCaptureScreen() : renderOfferScreen();

  if (appState.screen === "capture") {
    bindCaptureScreen();
    return;
  }

  bindOfferScreen();
}

function renderCaptureScreen(): string {
  return `
    <main class="app-shell">
      <section class="capture-shell">
        <section class="form-panel glass-panel">
          <div class="panel-heading">
            <div>
              <p class="section-kicker">Veri Girişi</p>
              <h2>Müşteri profilini doldur</h2>
            </div>
            <div class="panel-badge-stack">
              <span class="eyebrow-pill">${orderedFieldNames.length} alan</span>
              <span class="eyebrow-pill muted-pill">Yuvarlanmış aralıklar</span>
            </div>
          </div>

          <section class="preset-section">
            <div class="preset-section__header">
              <div>
                <span class="section-kicker">Hazır Profiller</span>
                <h3>Tek tıkla örnek doldur</h3>
              </div>
              <p class="preset-section__copy">Bir profil seç, form otomatik dolsun.</p>
            </div>
            <div class="preset-grid">
              ${userProfilePresets.map((preset) => renderPresetCard(preset)).join("")}
            </div>
          </section>

          <form id="customer-form" class="field-grid" novalidate>
            ${orderedFieldNames.map((fieldName) => renderFieldCard(fieldName, appState.values[fieldName], appState.errors[fieldName])).join("")}
            <div class="form-actions">
              <div class="form-hint-stack">
                <span class="hint-label">Kalite Kapısı</span>
                <strong>${formatReadinessStatus(projectContext.policy.readinessStatus)}</strong>
                <small>Tekrarlanabilirlik ${projectContext.policy.validationOverview.reproducibilitySummary.deterministic ? "hazır" : "izlenmeli"}</small>
              </div>
              <button type="submit" class="primary-button">Teklifi Göster</button>
            </div>
          </form>
        </section>
      </section>
    </main>
  `;
}

function renderPresetCard(preset: UserProfilePreset): string {
  const evaluation = evaluateOffer(preset.values);
  const isActive = isSameValues(appState.values, preset.values);

  return `
    <button
      type="button"
      class="preset-card ${isActive ? "preset-card--active" : ""}"
      data-preset-key="${preset.key}"
      aria-pressed="${isActive ? "true" : "false"}"
    >
      <span class="preset-card__topline">
        <span class="preset-card__title">${preset.title}</span>
        <span class="preset-card__probability">${formatPercent(evaluation.probability, 1)}</span>
      </span>
      <span class="preset-card__description">${preset.description}</span>
      <span class="preset-card__footer">
        <span>${formatRiskBandLabel(evaluation.riskBand)}</span>
        <span>${evaluation.actionTitle}</span>
      </span>
    </button>
  `;
}

function renderFieldCard(fieldName: UserInputFieldName, fieldValue: number, errorMessage?: string): string {
  const rule = projectContext.schema[fieldName];
  const helperText = renderConstraintText(rule);

  return `
    <label class="field-card ${errorMessage ? "field-card--error" : ""}">
      <span class="field-card__topline">
        <span class="field-name">${getFieldLabel(fieldName)}</span>
        <span class="field-badge">${rule.kind === "categorical_integer" ? "Kodlu alan" : "Sayısal alan"}</span>
      </span>
      <span class="field-purpose">${rule.purpose}</span>
      ${rule.kind === "categorical_integer" ? renderSelect(fieldName, rule, fieldValue) : renderNumberInput(fieldName, rule, fieldValue)}
      <span class="field-meta">${helperText}</span>
      ${errorMessage ? `<span class="field-error">${errorMessage}</span>` : `<span class="field-example">Örnek değer: ${formatDisplayValue(rule.example_value)}</span>`}
    </label>
  `;
}

function renderSelect(fieldName: UserInputFieldName, rule: CategoricalFieldRule, fieldValue: number): string {
  return `
    <select name="${fieldName}" class="field-input" aria-label="${getFieldLabel(fieldName)}">
      ${rule.allowed_values
        .map((allowedValue) => {
          const isSelected = allowedValue === fieldValue ? "selected" : "";
          return `<option value="${allowedValue}" ${isSelected}>${describeCode(fieldName, allowedValue)}</option>`;
        })
        .join("")}
    </select>
  `;
}

function renderNumberInput(
  fieldName: UserInputFieldName,
  rule: RangeFieldRule,
  fieldValue: number,
): string {
  return `
    <input
      class="field-input"
      type="number"
      name="${fieldName}"
      value="${fieldValue}"
      min="${rule.min_value}"
      max="${rule.max_value}"
      step="${getFieldStep(fieldName)}"
      inputmode="decimal"
      aria-label="${getFieldLabel(fieldName)}"
    />
  `;
}

function renderOfferScreen(): string {
  const fakeProduct = projectContext.ui.fakeProduct;
  const result = appState.result;
  if (!result) {
    return "";
  }

  const originalPrice = fakeProduct.priceTry;
  const popupStateClass = result.requiresManualReview
    ? "offer-popup--manual"
    : result.applyAutomaticDiscount
      ? "offer-popup--active"
      : "offer-popup--neutral";

  return `
    <main class="market-scene">
      <div class="market-backdrop">
        <header class="market-header glass-strip">
          <div class="market-brand">${fakeProduct.marketplaceName}</div>
          <div class="market-search">ürün, kategori veya marka ara</div>
          <nav class="market-nav">
            <span>Elektronik</span>
            <span>Moda</span>
            <span>Ev Yaşam</span>
            <span>Fırsatlar</span>
          </nav>
        </header>

        <section class="product-scene">
          <div class="product-gallery glass-panel">
            <div class="gallery-badges">
              ${fakeProduct.heroBadges.map((badge) => `<span class="eyebrow-pill">${badge}</span>`).join("")}
            </div>
            <div class="phone-stage">
              <button class="gallery-arrow" type="button" aria-label="Önizleme geri" disabled>&larr;</button>
              <div class="phone-stage__device phone-stage__device--hero">
                <div class="phone-stage__camera"></div>
                <div class="phone-stage__reflection"></div>
              </div>
              <button class="gallery-arrow" type="button" aria-label="Önizleme ileri" disabled>&rarr;</button>
            </div>
            <div class="thumb-row">
              ${fakeProduct.galleryLabels.map((label, index) => `<span class="thumb-chip ${index === 0 ? "thumb-chip--active" : ""}">${label}</span>`).join("")}
            </div>
          </div>

          <div class="product-summary glass-panel">
            <div class="crumbs">PazarNova &gt; Elektronik &gt; Telefon &gt; Premium Cihaz</div>
            <div class="summary-headline">
              <div>
                <p class="brand-label">${fakeProduct.brand}</p>
                <h1>${fakeProduct.productTitle}</h1>
              </div>
              <div class="rating-bubble">${fakeProduct.rating} / 5</div>
            </div>
            <div class="rating-row">
              <span>${"★".repeat(5)}</span>
              <span>${fakeProduct.reviewCount} değerlendirme</span>
              <span>Demo ürün sahnesi</span>
            </div>
            <div class="price-block">
              <span class="price-block__current">${formatTry(originalPrice)} TL</span>
              <span class="price-block__subline">${fakeProduct.monthlyPaymentLabel}</span>
            </div>
            <div class="option-grid">
              <div class="option-box option-box--active">Renk: Ada Çayı</div>
              <div class="option-box option-box--active">Depolama: 512 GB</div>
            </div>
            <div class="button-row">
              <button class="secondary-button" type="button" disabled>Şimdi Al</button>
              <button class="primary-button primary-button--wide" type="button" disabled>Sepete Ekle</button>
            </div>
            <div class="delivery-card">
              <strong>Bugün kargo sahnesi</strong>
              <p>Bu ekran dekoratif bir e-ticaret sayfasıdır. Asıl karar açılır pencerede verilir.</p>
            </div>
          </div>

          <aside class="seller-column">
            ${fakeProduct.sellerCards.map((card) => `
              <article class="seller-card glass-panel glass-panel--dense">
                <div class="seller-card__topline">
                  <strong>${card.sellerName}</strong>
                  <span class="seller-score">${card.sellerScore}</span>
                </div>
                <span class="seller-badge">${card.badge}</span>
                <div class="seller-price">${formatTry(card.priceTry)} TL</div>
              </article>
            `).join("")}
          </aside>
        </section>
      </div>

      <div class="offer-overlay"></div>
      <section class="offer-popup ${popupStateClass}" aria-live="polite">
        <div class="offer-popup__header">
          <div>
            <span class="offer-status-chip">${result.requiresManualReview ? "Manuel kontrol" : result.applyAutomaticDiscount ? "İndirim aktif" : "İndirim uygulanmıyor"}</span>
            <h2>Kişisel teklif sonucu hazır</h2>
          </div>
          <button class="ghost-button" id="retry-button" type="button">Tekrar Dene</button>
        </div>

        <div class="offer-popup__hero">
          <div class="score-block">
            <span class="score-label">Satın alma skoru</span>
            <strong>${formatPercent(result.probability, 1)}</strong>
            <div class="score-meter">
              <span style="width:${Math.max(6, result.probability * 100)}%"></span>
            </div>
          </div>

          <div class="discount-block">
            <span class="score-label">Karar</span>
            <strong>${result.applyAutomaticDiscount ? `${Math.round(result.automaticDiscountRate * 100)}% indirim` : result.requiresManualReview ? "Otomatik indirim yok" : "%0 indirim"}</strong>
            <p>${result.actionTitle}</p>
          </div>
        </div>

        <div class="price-comparison">
          <div>
            <span>Liste fiyatı</span>
            <strong>${formatTry(originalPrice)} TL</strong>
          </div>
          <div>
            <span>Teklif sonucu</span>
            <strong>${formatTry(result.discountedPriceTry)} TL</strong>
          </div>
        </div>

        <p class="offer-summary">${result.summary}</p>

        <footer class="popup-footer">
          <div>
            <span class="footer-label">Model</span>
            <strong>${projectContext.policy.championModelName}</strong>
          </div>
          <div>
            <span class="footer-label">Hazırlık</span>
            <strong>${formatReadinessStatus(projectContext.policy.readinessStatus)}</strong>
          </div>
          <div>
            <span class="footer-label">Demo bütçesi</span>
            <strong>${projectContext.policy.validationOverview.performanceSummary.meets_demo_budget ? "Karşılandı" : "İzlenmeli"}</strong>
          </div>
        </footer>
      </section>
    </main>
  `;
}

function bindCaptureScreen() {
  const formElement = document.querySelector<HTMLFormElement>("#customer-form");
  if (!formElement) {
    return;
  }

  document.querySelectorAll<HTMLButtonElement>("[data-preset-key]").forEach((buttonElement) => {
    buttonElement.addEventListener("click", () => {
      const presetKey = buttonElement.dataset.presetKey;
      const preset = userProfilePresets.find((item) => item.key === presetKey);
      if (!preset) {
        return;
      }

      appState.values = { ...preset.values };
      appState.errors = {};
      renderApp();
    });
  });

  formElement.addEventListener("submit", (event) => {
    event.preventDefault();
    const formData = new FormData(formElement);
    const nextValues = orderedFieldNames.reduce((accumulator, fieldName) => {
      const rawValue = formData.get(fieldName);
      accumulator[fieldName] = Number(rawValue);
      return accumulator;
    }, {} as UserInputValues);

    const validationErrors = validateInputValues(nextValues);
    appState.values = nextValues;
    appState.errors = validationErrors;

    if (Object.keys(validationErrors).length > 0) {
      renderApp();
      return;
    }

    appState.result = evaluateOffer(nextValues);
    appState.screen = "offer";
    appState.errors = {};
    renderApp();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

function bindOfferScreen() {
  const retryButton = document.querySelector<HTMLButtonElement>("#retry-button");
  retryButton?.addEventListener("click", () => {
    appState.screen = "capture";
    appState.result = null;
    renderApp();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

function renderConstraintText(rule: FieldRule): string {
  if (rule.kind === "categorical_integer") {
    return `İzin verilen kodlar: ${rule.allowed_values.join(", ")}`;
  }

  return `En az ${formatDisplayValue(rule.min_value)} / En çok ${formatDisplayValue(rule.max_value)}`;
}

function describeCode(fieldName: UserInputFieldName, code: number): string {
  if (fieldName === "Gender") {
    return code === 0 ? "0: Erkek" : "1: Kadın";
  }
  if (fieldName === "LoyaltyProgram") {
    return code === 1 ? "1: Evet" : "0: Hayır";
  }
  if (fieldName === "ProductCategory") {
    const categoryLabels: Record<number, string> = {
      0: "0: Elektronik",
      1: "1: Giyim",
      2: "2: Ev Yaşam",
      3: "3: Güzellik",
      4: "4: Spor",
    };
    return categoryLabels[code] ?? `Kategori ${code}`;
  }
  return `${getFieldLabel(fieldName)} ${code}`;
}

function formatDisplayValue(value: number): string {
  if (Number.isInteger(value)) {
    return String(value);
  }
  return String(Number(value.toFixed(1)));
}

function isSameValues(leftValues: UserInputValues, rightValues: UserInputValues): boolean {
  return orderedFieldNames.every((fieldName) => leftValues[fieldName] === rightValues[fieldName]);
}

function getFieldLabel(fieldName: UserInputFieldName): string {
  const fieldLabels: Record<UserInputFieldName, string> = {
    Age: "Yaş",
    Gender: "Cinsiyet",
    AnnualIncome: "Yıllık gelir",
    NumberOfPurchases: "Satın alma sayısı",
    ProductCategory: "Ürün kategorisi",
    TimeSpentOnWebsite: "Sitede geçirilen süre",
    LoyaltyProgram: "Sadakat programı",
  };
  return fieldLabels[fieldName];
}

function formatRiskBandLabel(riskBand: OfferResult["riskBand"]): string {
  const labels: Record<OfferResult["riskBand"], string> = {
    low_intent_holdout: "Düşük olasılık",
    lower_targeting_band: "Alt hedefleme bandı",
    upper_targeting_band: "Üst hedefleme bandı",
    high_confidence_no_discount: "Yüksek güven bandı",
  };
  return labels[riskBand];
}

function formatReadinessStatus(readinessStatus: string): string {
  return readinessStatus === "hazir" ? "Hazır" : "İzlenmeli";
}

renderApp();
