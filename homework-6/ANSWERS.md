# HW6 2026 — Answers (verified with `hw06.py`)

Dataset: `car_fuel_efficiency_2026.csv` (pinned 2026 release, 10,000 rows).
Prep: fillna(0), 60/20/20 split (`random_state=1`), DictVectorizer.

| # | Answer | Evidence |
|---|--------|----------|
| Q1 | `model_year` | stump (depth=1), split at 1997.5 |
| Q2 | **1.837** | RF(10) val RMSE = 1.8332 |
| Q3 | **150** | lowest RMSE (3 decimals): 10→1.833, 50→1.774, 100→1.770, 150→1.769 |
| Q4 | **10** | mean RMSE: d10=1.7544, d15=1.7842, d20=1.7868, d25=1.7863 |
| Q5 | `vehicle_weight` | importance 0.2033 > horsepower 0.0783 (model_year ranks 1st at 0.2961 but is not among the options) |
| Q6 | **0.1** | val RMSE 1.7248 vs 1.8265 with eta=0.3 |

To do: submit at https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw06 (deadline Nov 2, 23:00) + Learning in Public post.
