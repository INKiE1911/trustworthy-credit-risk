# Model card: loan-default model (tuned LightGBM)

Following Mitchell et al., *Model Cards for Model Reporting* (2019). Numbers are from the
notebooks. The test split (61,502 applicants) was used once, in Step 15; the model is frozen. The data is
described in [`datasheet.md`](datasheet.md).

## Model details

- **What:** a gradient-boosted tree model (LightGBM, 1,871 trees of 17 leaves, tuned with
  Optuna) that gives the probability that a loan applicant will have payment difficulties.
- **Inputs:** 240 features built from the application and six loan-history tables. **Gender
  is not an input.** Age is.
- **Decision layer:** the probability is turned into a decision two ways:
  - **money rule:** approve if p < margin / (margin + loss) = 0.1667 (margin 10%, loss 50%);
  - **conformal layer:** approve if p < 0.0355, decline if p > 0.1655, otherwise refer to a
    person.
- **Explanations:** TreeSHAP reasons per applicant, plus counterfactuals for actionable
  changes (borrow less, higher income).
- **Built by:** a PRML course project, IIT Dharwad, 2026. Files:
  `models/lightgbm_final.{joblib,txt,json}`, `models/decision.json`.

## Intended use

- **For:** teaching and demonstrating trustworthy credit scoring: calibrated risk, decisions
  with error guarantees, explanations and fairness checks.
- **Not for:** real lending decisions, other lenders or countries, or deciding without human
  review. It has not been validated outside this one dataset and period.

## Factors

- **Groups audited:** gender (F/M) and age band (<30, 30–45, 45–60, 60+).
- **Not audited:** race, religion, nationality and disability. The data has no such columns,
  so nothing can be said about them.

## Metrics

- **Ranking:** ROC-AUC with a bootstrap 95% CI.
- **Probabilities:** Brier score, ECE, and mean predicted vs actual.
- **Decisions:** profit, approval rate, the share of good customers declined, and the share of
  defaulters declined.
- **Fairness:** equal opportunity gap (main), demographic parity difference, equalized odds
  difference, and disparate impact ratio.

## Training and evaluation data

- **Train:** 60% split, 184,503 applicants, with 5-fold CV for model choice.
- **Validation:** 10% (30,751), for calibration, the money rule and fairness.
- **Conformal:** 10% (30,751), for the conformal cut-offs only.
- **Test:** 20% (61,502), used once in Step 15 for the numbers below.

## Quantitative analysis

| | Test (Step 15) | Before (CV / validation) |
|---|---|---|
| ROC-AUC | **0.790** (95% CI 0.784–0.797) | 0.790 (CV), 0.793 (validation) |
| PR-AUC / KS / Brier | 0.295 / 0.439 / 0.0653 | |
| Mean predicted vs actual default rate | 7.99% vs 8.07% (ECE 0.0018) | 8.03% vs 8.07% |
| Money rule | approves 87.5%; profit +13.9% vs approving everyone | 87.4%; +13.1% |
| Conformal (α = 10%) | 9.8% of defaulters auto-approved, 9.8% of good customers auto-declined; 46.9% referred | 9.3%, 9.9%; 47.1% |
| vs EBM / logistic regression / scorecard | +0.009 / +0.013 / +0.033 ROC-AUC (all significant) | |

**By group (validation, money rule; test in Step 15 repeats these within about 1.5 points):**

| Group | Default rate | Mean predicted | ROC-AUC | Approved | Good customers declined |
|---|---|---|---|---|---|
| Women | 6.9% | 7.4% | 0.793 | 89.4% | 8.3% |
| Men | 10.4% | 9.3% | 0.784 | 83.7% | 12.6% |
| Under 30 | 11.7% | 11.5% | 0.765 | 77.8% | 18.0% |
| 30–45 | 9.2% | 8.9% | 0.797 | 85.2% | 11.5% |
| 45–60 | 6.5% | 6.7% | 0.788 | 91.2% | 6.7% |
| 60+ | 4.5% | 4.7% | 0.738 | 96.3% | 3.1% |

- **Gender:** equal opportunity gap 4.3 points (against men); disparate impact ratio 0.94.
- **Age:** equal opportunity gap 14.9 points (against under-30s); disparate impact ratio 0.81
  on validation, **0.796 on test**: just under the 0.8 line.
- **Conformal:** the guarantee holds per class overall, not per group. 23% of defaulters aged
  60+ are auto-approved, and 18% of good customers under 30 are auto-declined.
- **Mitigation (Step 13):** reweighing narrows the gender gap to 1.7 points for −0.1% profit.
  The cost is calibration within gender: men are predicted 8.6% vs 10.4% actual.
  `ExponentiatedGradient` reaches the same gap for −1.1% profit. Per-group thresholds can
  close it fully, but they need gender when deciding. On test (Step 15) reweighing gives the same
  ROC-AUC (0.790) and cuts the gender gap from 4.6 to 2.0 points for −0.55% profit. **The app keeps
  the final model**, because its conformal cut-offs are the only ones checked on test.

## Ethical considerations

- **Base rates differ** (men 10.4% vs women 6.9%), so no rule can be calibrated within both
  groups and have equal error rates (Chouldechova 2017; Kleinberg et al. 2016). Choosing which
  fairness to give up is a policy decision; this project picks equal opportunity for good
  customers.
- **Proxies:** without gender, the other features still predict it with ROC-AUC 0.91
  (occupation, employer type, car ownership). Leaving the column out is not enough on its own,
  which is why the audit uses gender.
- **External scores** are the strongest inputs, and their origin is unknown. Any bias in them
  is inherited.
- **Selection bias:** outcomes exist only for loans that were approved in the past. Rejected
  applicants are missing from the data.
- **Human review:** the refer zone keeps a person in the loop for uncertain cases. Every
  decision comes with its reasons and with what would change it.
- **Privacy:**
  - the data stays outside the repository (`data/` is ignored), and only aggregate tables and
    figures are committed;
  - the API validates every input (Pydantic ranges), stores nothing and logs no request
    bodies. Uvicorn's access log does record `/applicants/{id}` paths, so turn it off or move
    the ID into the request body before any real use;
  - a model trained on personal data can leak membership (membership-inference attacks).
    Tree ensembles with 426+ applicants per leaf limit this, but it is not tested.

## Caveats and recommendations

- One lender, one period (data released 2018). Recalibrate and re-audit before any reuse.
  Watch for drift (the Kaggle test file already differs: PSI 0.21 for contract type).
- The profit numbers rely on assumed margin and loss rates. Step 10 shows how the threshold
  moves with them.
- Gender-aware thresholds and group-conditional conformal cut-offs would close the remaining
  gaps, but they need gender when deciding. Check the law first.
