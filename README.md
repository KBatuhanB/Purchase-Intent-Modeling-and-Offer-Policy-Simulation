# Purchase Intent Modeling and Offer Policy Simulation

This repository is a portfolio-ready machine learning project that predicts purchase likelihood from customer-level e-commerce data and converts model output into an explainable offer policy.

It started as an academic data mining project, but the implementation was pushed far beyond a typical course submission: the repository includes a reproducible 11-phase Python pipeline, generated artifacts for every phase, validation and quality gates, explainability analysis, a policy layer with guardrails, and a polished TypeScript demo UI.

The result is not just a classifier. It is an end-to-end decision-support prototype.

## Why This Project Is Worth Reviewing

- It treats data quality, duplicate handling, and leakage prevention as first-class engineering problems.
- It optimizes for calibrated, decision-usable probabilities instead of chasing accuracy alone.
- It separates model prediction from business action using a policy layer with explicit decision bands and guardrails.
- It includes explainability, fairness-lite monitoring, simulation, validation, and delivery packaging.
- It ships with both backend artifacts and a frontend demo instead of stopping at notebooks or static charts.

## Project Snapshot

| Item | Value |
| --- | --- |
| Problem | Predict purchase likelihood and recommend a controlled offer action |
| Dataset | `customer_purchase_data.csv` |
| Final working dataset | 1,388 deduplicated rows |
| Champion model | `random_forest_sigmoid_calibrated` |
| Accuracy | 73.74% |
| Precision | 71.76% |
| Recall | 72.31% |
| F1 | 72.03% |
| ROC-AUC | 85.60% |
| PR-AUC | 83.35% |
| Balanced Accuracy | 73.65% |
| Brier Score | 0.154752 |
| Delivery status | Ready |
| Visual and scenario assets | 50 |

## What The System Actually Does

1. Audits the raw dataset for schema issues, duplicates, leakage risk, and target integrity.
2. Runs exploratory data analysis and generates plots to understand signal quality.
3. Builds preprocessing pipelines and feature engineering layers for both linear and tree-based models.
4. Establishes baselines before moving to stronger ensemble families.
5. Evaluates imbalance handling and chooses thresholds based on decision quality, not default cutoffs.
6. Selects a calibrated champion model and a challenger model.
7. Explains predictions with SHAP, permutation importance, and fairness-lite group checks.
8. Converts model scores into policy bands and offer actions with guardrails.
9. Exposes the logic through simulation scenarios and a demo-ready input/output contract.
10. Stress-tests the system with edge cases, sensitivity checks, reproducibility checks, and performance checks.
11. Packages the outputs into reports, demo assets, presentation materials, and a delivery manifest.

## Architecture

```text
customer_purchase_data.csv
        |
        v
Phase 1: Data audit and go/no-go checks
        |
        v
Phase 2: EDA and hypothesis generation
        |
        v
Phase 3: Preprocessing and feature engineering
        |
        v
Phase 4-6: Baselines, imbalance strategy, advanced modeling
        |
        v
Phase 7: Explainability and fairness-lite analysis
        |
        v
Phase 8: Policy and guardrail layer
        |
        v
Phase 9-10: Simulation, validation, and quality gate
        |
        v
Phase 11: Delivery bundle and presentation assets
        |
        +--> artifacts/ (reports, plots, JSON summaries, model files)
        |
        +--> frontend/scripts/sync-phase-artifacts.mjs
                  |
                  v
             TypeScript demo UI
```

## Repository Structure

```text
.
|- veri_madenciligi/          # Python package with CLI, config, core utilities, and phase services
|- tests/                     # Phase-focused backend tests
|- artifacts/                 # Generated reports, plots, JSON summaries, models, scenarios, delivery bundle
|- frontend/                  # Vite + TypeScript demo application
|- customer_purchase_data.csv # Input dataset
|- PLAN.md                    # Original implementation roadmap
|- README.md                  # Main GitHub-facing project overview
```

## Tech Stack

- Python 3.14
- pandas and NumPy for data handling
- scikit-learn for preprocessing, modeling, calibration, and metrics
- imbalanced-learn for SMOTE and SMOTENC experiments
- matplotlib and seaborn for plots
- SHAP for model explainability
- joblib for serialized pipelines and models
- TypeScript and Vite for the demo frontend
- Vitest for frontend tests

## Notable Engineering Decisions

- `DiscountsAvailed` was deliberately excluded from the main model because it behaves like a risky post-treatment or leakage-prone signal. It is only used later as a guardrail and monitoring field.
- Exact duplicate rows were removed before splitting the dataset, which reduced the working data from 1,500 to 1,388 rows and prevented inflated evaluation results.
- The project did not force SMOTE just because it was available. Class imbalance was measured first, then intervention strategies were compared empirically.
- Model selection prioritized calibrated probabilities and decision usefulness. That is why the final choice was a sigmoid-calibrated Random Forest, not simply the model with the flashiest single metric.
- The frontend does not pretend to be live inference. It uses synced backend artifact context and a TypeScript offer engine to present an honest, demo-friendly product surface.

## Running The Backend

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

### 3. Run the phase pipeline

Each phase is exposed through the CLI.

```bash
python -m veri_madenciligi phase1 --dataset customer_purchase_data.csv
python -m veri_madenciligi phase2 --dataset customer_purchase_data.csv
python -m veri_madenciligi phase3 --dataset customer_purchase_data.csv
python -m veri_madenciligi phase4 --dataset customer_purchase_data.csv
python -m veri_madenciligi phase5 --dataset customer_purchase_data.csv
python -m veri_madenciligi phase6 --dataset customer_purchase_data.csv
python -m veri_madenciligi phase7 --dataset customer_purchase_data.csv
python -m veri_madenciligi phase8 --dataset customer_purchase_data.csv
python -m veri_madenciligi phase9 --dataset customer_purchase_data.csv
python -m veri_madenciligi phase10 --dataset customer_purchase_data.csv
python -m veri_madenciligi phase11 --dataset customer_purchase_data.csv
```

Generated outputs will appear under `artifacts/phase_1` through `artifacts/phase_11`.

## Running The Frontend Demo

The frontend is a two-screen TypeScript demo that turns the backend artifacts into a recruiter-friendly product walkthrough.

```bash
cd frontend
npm install
npm run dev
```

The following commands are also available:

```bash
npm run test
npm run build
```

`npm run sync-data` reads the generated backend artifacts and refreshes `frontend/src/data/generated/project-context.json`. It runs automatically before `dev`, `build`, and `test`.

## What This Repository Demonstrates

- End-to-end ownership of a machine learning project from data audit to delivery packaging.
- Reproducible CLI-oriented project structure instead of ad hoc notebook-only work.
- Careful handling of leakage risk, duplicate contamination, and threshold strategy.
- Balanced model evaluation using PR-AUC, calibration quality, and downstream policy implications.
- Explainability and monitoring awareness through SHAP, scenario cards, and fairness-lite analysis.
- Product thinking through a separate action-policy layer and a polished demo interface.

## Limitations And Honest Scope

- This is an offline decision-support prototype, not a live production inference service.
- The dataset is a static snapshot rather than a real event stream or clickstream log.
- Offer impact is policy-driven, not causally estimated through uplift modeling or randomized experiments.
- Proxy business value is used instead of real margin and campaign cost tables.
- Fairness analysis is intentionally lightweight and should be expanded before production deployment.

## Future Extensions

- Integrate clickstream or session-level behavioral data.
- Add causal uplift modeling and A/B test instrumentation.
- Introduce real business value inputs such as margin and campaign cost.
- Add monitoring dashboards and group-specific threshold management.
- Validate on fresh data snapshots to measure generalization drift.

## If You Are Reviewing This As A Hiring Team

This repository is best read as evidence of how I think through messy, real-world ML work: I do not stop at training a model. I audit the data, question suspicious features, build repeatable artifacts, validate edge cases, expose limitations honestly, and package the result in a way that non-technical stakeholders can actually review.

If you want the shortest path through the repo, start with the executive summary, then inspect the phase reports, and finish with the frontend demo.