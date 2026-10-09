# Datasheet: Home Credit Default Risk, as used in this project

Following Gebru et al., *Datasheets for Datasets* (2021). It describes the data as this project
uses it. The data itself belongs to Home Credit and is not in this repository.

## Motivation

- **Why it was made:** a Kaggle competition (2018) by Home Credit, a lender to people with
  little or no credit history, to predict which applicants will have trouble repaying.
- **Who made it:** Home Credit Group. Anonymised and released through Kaggle.
- **Why this project uses it:** it is real, large (307,511 labelled loans) and relational
  (7 tables). Its people have thin credit files, so fairness and explanations matter.

## Composition

- **Instances:** one row per loan application (`SK_ID_CURR`) in `application_train`
  (307,511 rows with an outcome) and `application_test` (48,744 rows without). Six history
  tables add earlier credit bureau loans, monthly bureau statuses, earlier Home Credit
  applications, instalment payments, POS/cash balances and credit card balances.
- **Label:** `TARGET` = 1 if the client had payment difficulties (more than X days late on at
  least one of the first Y instalments; Home Credit does not publish X and Y). 8.07% of
  applications are positive.
- **Sensitive attributes:** gender (`CODE_GENDER`, 4 rows "XNA", dropped) and age (from
  `DAYS_BIRTH`). Family status, number of children, education, housing and occupation are also
  personal. There is no race, religion or nationality column.
- **Base rates differ by group:** men default at 10.14% vs women at 7.00%; applicants under 30
  at 11.44% vs 4.92% at 60 and over. About two thirds of applicants are women.
- **Errors and codes:** `DAYS_EMPLOYED = 365243` (18% of rows, almost all pensioners) is a
  "missing" code, not a date; income is very skewed (median 147,150, one applicant at
  117,000,000); 18 columns are almost constant; some old loans in the payment tables have no row
  in `previous_application`.
- **External scores:** `EXT_SOURCE_1/2/3` are normalised scores from unnamed outside sources.
  They are the strongest predictors, and nothing is known about how they were made, including
  whether they themselves treat groups differently.
- **Confidential data:** IDs are anonymised; there are no names or addresses. Combinations of
  exact age, income and region could still narrow down a person.

## Collection process

- Applications to Home Credit, probably in Eastern Europe and Asia (amounts are in an
  unnamed currency). Dates are stored relative to the application (negative days).
- **Only approved loans have outcomes.** Rejected applicants never show whether they would have
  repaid, so the training data is a biased sample of all applicants (the "reject inference"
  problem; see Step 14).
- The Kaggle test file is a different sample: 0.9% revolving loans vs 9.5% in the training file
  (PSI 0.21 for contract type).

## Preprocessing in this project

- CSV → Parquet with smaller dtypes; text columns → categories (`make parquet`).
- One row per applicant: every history table is aggregated by `SK_ID_CURR` into 240 features
  plus gender, which is kept only for fairness checks (`make features`).
- **Frozen splits** of the labelled rows, stratified by `TARGET`: train 60% (184,503, 5 CV
  folds), validation 10% (30,751), conformal 10% (30,751), test 20% (61,502, locked until the
  final evaluation). The 48,744 unlabelled applications are used only for the demo app and as
  drift examples.

## Uses

- **Used for:** a course project on trustworthy credit scoring (prediction, calibration,
  conformal decisions, explanations, fairness).
- **Should not be used for:** real lending decisions; inferring anything about individuals;
  conclusions about groups beyond this lender's applicants. The patterns are those of one
  lender's past approvals.

## Distribution and maintenance

- Kaggle competition data under the competition's rules: download it yourself
  (`make data`); it must not be redistributed, so it is not in this repository.
- Not maintained by this project. The data is a snapshot (released 2018) and will not be
  updated.
