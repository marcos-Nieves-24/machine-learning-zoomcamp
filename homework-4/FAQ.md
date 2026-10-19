# Homework 4 FAQ — Evaluation Metrics for Classification (2026 lead-scoring)

Companion to `homework-4.ipynb` and the official task:
`cohorts/2026/homework/04-evaluation/homework.md`.
Dataset: `course_lead_scoring_2026.csv`, target `converted`.
Module notes: `04-evaluation` (accuracy, confusion table, precision/recall, ROC, AUC, cross-validation).

## Answers (verified)

| Q | Answer |
|---|--------|
| Q1 — highest single-variable AUC | `lead_score` (0.788) |
| Q2 — val AUC (`liblinear`, C=1.0) | 0.732 |
| Q3 — precision/recall intersection | 0.63 |
| Q4 — max F1 threshold | 0.41 |
| Q5 — 5-fold std | 0.007 |
| Q6 — best C | 0.001 (mean 0.749 ± 0.010) |

## Q1: ROC AUC as single-variable importance

Score each numerical column with `roc_auc_score(train['converted'], train[col])`
on the **train** split. If AUC < 0.5, negate the column and take
`max(auc, 1 - auc)` — a below-0.5 AUC just means negative correlation.

Measured: `annual_income` 0.608, `number_of_courses_viewed` 0.723,
`interaction_count` 0.765, `lead_score` 0.788. Winner: `lead_score`.

⚠️ Gotcha: exclude `converted` itself from the loop — scoring the target
against itself gives AUC 1.000 and answers nothing.

## Q2: train with DictVectorizer + liblinear

`fit_transform` on train dicts, `transform` on val/test dicts, then
`LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)`.
Score with `predict_proba(val)[:, 1]` → AUC **0.732** on val.

⚠️ Gotcha: `accuracy_score` on `predict` is a different metric — the question
asks for AUC, which needs probabilities, not hard labels.

## Q3: precision/recall intersection

Sweep thresholds `np.arange(0.0, 1.0, 0.01)`, compute precision and recall per
threshold (`zero_division=0`), skip thresholds where both are 0, pick the
smallest `|P − R|`. Result: **0.63** (P≈0.733, R≈0.725).

Background (`04-precision-recall`): precision = TP/(TP+FP) over predicted
positives; recall = TP/(TP+FN) over actual positives. One rises, the other
falls — the crossing is the trade-off point.

## Q4: max F1

`F1 = 2·P·R / (P+R)` (0 when P+R == 0) over the same threshold grid.
Maximum **F1≈0.758 at 0.41**.

## Q5: 5-fold CV std

`KFold(n_splits=5, shuffle=True, random_state=1)` over `df_full_train`
(train+val merged, test untouched). Per fold: fresh `DictVectorizer`
(`fit` on the train fold only), `LogisticRegression(solver='liblinear',
C=1.0, max_iter=1000)`, AUC on the val fold.

Fold AUCs ≈ [0.731, 0.730, 0.736, 0.741, 0.721] → **std 0.007**
(`np.std` default, ddof=0).

⚠️ Gotcha (costly): the solver matters. `LogisticRegression(max_iter=1000)`
defaults to `lbfgs`, which throws `ConvergenceWarning` here and gives
std **0.013** — a wrong option. The homework explicitly requires
`solver='liblinear'`.

## Q6: tune C

Same 5-fold setup, `C in [0.000001, 0.001, 1]`:

- `C=1e-06`: 0.617 ± 0.019 (underfits — regularization crushes the weights)
- `C=0.001`: **0.749 ± 0.010** ← best mean
- `C=1`: 0.732 ± 0.007

Tie-break (lowest std, then smallest C) never triggers — `0.001` wins on mean
outright. Background (`07-cross-validation`): mean = average performance,
std = stability across folds; tiny `C` = strong regularization = underfit.

## Data-prep notes

- `fillna('NA')` for categoricals, `fillna(0.0)` for numericals. Constants, so
  doing it before the split leaks nothing (nothing is estimated from data).
- Exact splits: `test_size=0.2` then `test_size=0.25` on the remainder,
  both `random_state=1` → 60/20/20.
- Golden rule for CV: the vectorizer/model only ever see the train fold;
  the val fold is transformed/predicted, never fitted.
