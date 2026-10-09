# Trustworthy Credit Risk Scoring

[![CI](https://github.com/INKiE1911/trustworthy-credit-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/INKiE1911/trustworthy-credit-risk/actions/workflows/ci.yml)

An end-to-end loan-default model on the **Home Credit Default Risk** data (307,511 real loan
applications, 7 linked tables), built to be accurate *and* honest: every number comes with a
confidence interval, and the system explains each score. PRML course project, IIT Dharwad (2026).

> **Status:** work in progress (Step 16 of 18). The models are final: the 20% test split was
> opened once, in Step 15, and nothing changed after it.

![The demo app](reports/figures/app_screenshot.png)

## Final results (test split, 61,502 applicants, used once)

| Model | Test ROC-AUC | 95% CI | 5-fold CV |
|---|---|---|---|
| **Tuned LightGBM (final model)** | **0.790** | 0.784–0.797 | 0.790 |
| LightGBM, reweighed for gender fairness | 0.790 | 0.784–0.797 | |
| EBM (glass-box model) | 0.781 | 0.774–0.788 | 0.776 |
| Logistic regression | 0.778 | 0.771–0.785 | 0.776 |
| WoE scorecard (30 features, points table) | 0.757 | 0.751–0.765 | 0.755 |

Test matches CV for every model. LightGBM ranks significantly better than all three reference
models (paired bootstrap, Holm-corrected). On the approve/decline decisions it beats only
logistic regression significantly. On test, the money rule earns **13.9% more** than approving
everyone, and the conformal layer auto-approves **9.8%** of defaulters (the promise was 10%).

## Model comparison (5-fold cross-validation, train split)

| Model | 5-fold CV ROC-AUC | 95% CI |
|---|---|---|
| **Tuned LightGBM (final model)** | **0.790** | 0.787–0.793 |
| LightGBM, default settings | 0.787 | |
| XGBoost | 0.785 | |
| HistGradientBoosting | 0.783 | |
| EBM (glass-box model) | 0.776 | |
| Logistic regression | 0.776 | 0.772–0.779 |
| PyTorch MLP with category embeddings | 0.772 | 0.768–0.775 |
| WoE scorecard (30 features, points table) | 0.755 | 0.751–0.758 |
| Decision tree | 0.725 | 0.721–0.730 |
| LightGBM on the main table only | 0.761 | 0.758–0.765 |
| Always "no default" | 0.500 | |

14 models were compared on the same folds with the same leak-free rules. Rows from the model
ladder used the features from before a small fix (it moved LightGBM by +0.001).

## What I found

- **History matters:** the six loan-history tables add **+0.025 ROC-AUC** over the main
  application table (an ablation, one table at a time).
- **Boosting wins, the library doesn't matter:** LightGBM, XGBoost and HistGradientBoosting
  are within 0.003 of each other. A glass-box EBM is only 0.01 behind; a bank-style points
  scorecard costs 0.031.
- **Class imbalance "fixes" hurt:** class weights, undersampling and SMOTE all *lowered*
  ROC-AUC. Weights and undersampling pushed the average predicted probability from 8% to
  27–34%. SMOTE kept honest probabilities for a surprising reason: every synthetic row had an
  "in-between" value no real person has, and a model could spot them perfectly (ROC-AUC 1.000).
- **Tuning helped a little:** Optuna, random and grid search at the same budget ended close;
  the tuned model gained **+0.003** (significant in a paired bootstrap).
- **Neural networks lose on this table data:** the MLP is 0.018 behind LightGBM, and a blend
  of the two is *worse* than LightGBM alone (their predictions correlate at 0.89).
- **From scratch:** seven classic algorithms (logistic regression by Newton's method with a
  Laplace Bayesian version, Naive Bayes, PCA, Fisher LDA, K-means++, GMM with EM, an MLP with
  backprop) written in NumPy match scikit-learn; K-means and EM from the same start give
  identical answers, and the MLP gradient check error is 2e-7.
- **Already calibrated; a money rule beats a fixed cut-off:** Platt and isotonic scaling did not
  improve the probabilities (average 8.03% vs 8.07% actual). Approving when
  p < margin / (margin + loss) (0.167 for a 10% margin and 50% loss) earns **13% more** than
  approving everyone on the validation split, within 0.3% of the best threshold found by
  search. Within gender it is off: men are under-predicted (9.3% vs 10.4%) and women
  over-predicted (7.4% vs 6.9%), a problem for the fairness step.
- **A guarantee instead of a guess:** a class-conditional (Mondrian) conformal layer approves,
  refers or declines with a promise: at most 10% of real defaulters auto-approved and at most
  10% of good customers auto-declined (9.3% and 9.9% on held-out data). The price: 47% of
  applicants go to a human. One shared cut-off would have auto-approved **82% of defaulters**.
- **Explanations, checked:** half the model's evidence comes from loan history (53% of
  |SHAP|). LIME agrees with SHAP on only 2 to 4 of the top 5 reasons and changes between runs,
  so the app uses exact TreeSHAP. For declined applicants, **borrowing less** gets 78% out of
  the decline zone (median 35% less); a higher income works for only 21%.
- **Fairness, measured and traded off:** good customers are wrongly declined at 12.6% for men
  vs 8.3% for women, and 18% under 30 vs 3% at 60+. Gender is not an input, yet the other
  features predict it with ROC-AUC 0.91. With different base rates no rule can be fair in every
  sense at once (shown with the real numbers). Reweighing cuts the gender gap from 4.3 to 1.7
  points for 0.1% of profit, at the cost of calibration within gender. See the
  [model card](reports/model_card.md) and [datasheet](reports/datasheet.md).
- **Clusters describe, they don't score:** applicants form one blob (best silhouette 0.19), and
  "the default rate of my cluster" ranks risk at ROC-AUC 0.50–0.55 for K-means, GMM, Ward and
  HDBSCAN. Clustering the declined applicants by their **SHAP values** works better: three risk
  personas (low external scores, a maxed-out credit card, borderline), each with its own story.
- **The test split agreed with everything:** AUC, calibration, profit, the conformal coverage
  (0.902 for both classes) and the fairness gaps all repeat their validation values. One thing
  got worse: by age, the approval ratio is 0.796, just under the "80% rule". More data would
  still help LightGBM (its learning curve is still rising) but not logistic regression.
- **Drift is watched, not guessed:** a PSI monitor stays at ≤ 0.002 between our own splits.
  For the later Kaggle test file the score is stable (0.008), but one key input shifted hard:
  loans got shorter (annuity ÷ loan amount, PSI 0.997). In a simulation, lowering everyone's
  external scores by 20% trips the score alarm (0.254) and cuts auto-approvals from 44% to 25%.
- **Bugs caught by checking the data:** counting split payments row by row said 67% of
  installments were underpaid (per installment it was 0%), and a "365243" date code had
  quietly emptied two features.

## The demo app

The model uses 240 features, mostly from loan history, so nobody can type them in. The app
picks an applicant and has three tabs:

- **Score + why:** the default probability and the strongest reasons (TreeSHAP values).
- **Decision:** approve, refer to a person, or decline (the conformal layer), the money rule's
  suggestion for referred cases, and the reasons as sentences.
- **What-if:** change 8 key fields (income, loan amount, annuity, age, years employed, three
  external scores) in the sidebar, or let the model find the smallest smaller-loan or
  higher-income change that reaches the next better decision.

Features are rebuilt with exactly the training code. Locally the applicants are **real
applicants from the Kaggle test file** (no outcome is known). The Docker image uses **made-up
applicants** instead, because the competition rules forbid sharing the data.

```bash
make app     # Streamlit page at http://localhost:8501 (builds the demo data on first run)
make api     # FastAPI service at http://localhost:8000 (interactive docs at /docs)
```

```bash
curl "localhost:8000/applicants?limit=3"                       # some demo applicant ids
curl -X POST localhost:8000/score -H "Content-Type: application/json" \
     -d '{"applicant_id": <an id from above>, "changes": {"income": 250000}}'
curl localhost:8000/applicants/<id>/counterfactuals            # smallest change to a better decision
```

The response has the probability, how it compares with the average applicant (8.1%), the
decision, the reason codes and the top 4 reasons pushing the risk up and down. Input is validated (for example, external scores
must be between 0 and 1). The probabilities are calibrated: Platt and isotonic scaling did not
improve them (Step 10).

## How it is built

```
7 CSV tables -> Parquet -> 240 features (one row per applicant, leak-free aggregates)
             -> frozen splits (train 60 / valid 10 / conformal 10 / test 20, 5 CV folds)
             -> one CV runner for every model (bootstrap CIs, paired tests, MLflow)
             -> model ladder -> experiments -> tuned LightGBM
             -> calibration check + money threshold -> conformal approve / refer / decline
             -> SHAP reasons, counterfactuals, fairness audit -> API + app (Docker) + PSI monitor
```

**Still to come:** the report and the slides.

## Run it yourself

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

The data is not included (Kaggle competition rules). Accept the rules on the competition page,
then:

```bash
make data && make parquet && make splits && make features
```

The notebooks in `notebooks/` reproduce every number (`06_experiments.ipynb` trains and saves
the final model to `models/`). Then `make app`.

| Command | What it does |
|---|---|
| `make test` | Run the tests |
| `make lint` | Check the code |
| `make mlflow` | Open the experiment dashboard |
| `make demo-data` | Rebuild the app's demo applicants (real, local only) |
| `make docker` | Build the image (final model + made-up applicants) |
| `docker compose up` | Run the app (port 8501) and the API (port 8000) together |

## Project layout

```
src/creditrisk/   data/  features/  models/  evaluation/  scratch/  serving/
                  decision/  explain/  fairness/  monitoring/
app/              the Streamlit page
demo/             made-up demo applicants (safe to share)
notebooks/        01-14, one per step, each ending with key facts and findings
tests/            unit tests, including every scratch algorithm vs scikit-learn
reports/          result tables, figures, model card, datasheet
```

**Tech:** Python, pandas, scikit-learn, LightGBM, XGBoost, PyTorch, Optuna, MLflow, FastAPI,
Pydantic, Streamlit, Docker, LIME, Fairlearn.
