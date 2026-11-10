### Course

machine-learning-zoomcamp

### Question

Why do my Homework 2 (2026) answers not match any option — my RMSE is ~0.46 instead of 0.236 or 22.10?

### Answer

**Fix first — switch to the pinned 2026 dataset and parameters:**

1. **Download the correct CSV** (not the old car-pricing dump):
   ```bash
   wget https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv
   ```

2. **Keep only the 5 homework columns** — drop the wide car-pricing schema (`Make`, `Model`, `MSRP`, etc.):
   ```python
   cols = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']
   df = df[cols]
   ```

3. **Do NOT apply `np.log1p` to the target.** The 2026 homework predicts `fuel_efficiency_mpg` on its raw MPG scale. The obsolete notebook does `y = np.log1p(df[col].values)` (target `MSRP`) which compresses the scale and yields RMSEs ~0.46 on the log scale vs. ~22.10 on the MPG scale — no option will match if you keep the log transform.

4. **Use the exact 60/20/20 split from the homework** (seed 42 for Q3–Q4; seeds 0–9 for Q5; seed 9 + `r=0.001` for Q6):
   ```python
   import pandas as pd
   import numpy as np

   df = pd.read_csv('car_fuel_efficiency_2026.csv')
   df = df[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']]

   # NO log1p — use raw mpg
   # target = df['fuel_efficiency_mpg'].values

   n = len(df)
   n_val = int(n * 0.2)
   n_test = int(n * 0.2)
   n_train = n - n_val - n_test

   np.random.seed(42)
   idx = np.arange(n)
   np.random.shuffle(idx)

   df_train = df.iloc[idx[:n_train]]
   df_val   = df.iloc[idx[n_train:n_train+n_val]]
   df_test  = df.iloc[idx[n_train+n_val:]]
   ```
   For Q5 repeat the same block for each seed in `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` (then `np.std` of the 10 RMSEs). For Q6 use `np.random.seed(9)`, combine `train+val`, train with `r=0.001`, evaluate on `test`.

**Why this happens:**

The local `homework-2.ipynb` in this repo is an **anti-example** for the 2026 cohort. It still loads `data/car-pricing/data.csv` (line ~196: `pd.read_csv("data/car-pricing/data.csv")`), applies `np.log1p` to `MSRP` (lines ~241–248: `y_train = np.log1p(df_train[colname].values)`), and splits with `np.random.seed(2)` (line ~228). That notebook targets log-MSRP on the old pricing dataset — its RMSE ~0.46 is correct *for that* setup, but it will never match the 2026 options (`0.236 / 2.236 / 22.10 / 221.0` on raw MPG). Use `data/car-pricing/data.csv` + `log1p` + `seed 2` only as what *not* to do.

Canonical homework (pinned 2026 release, seeds 42 / 0-9 / 9 with `r=0.001`, no `log1p`):

https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/02-regression/homework.md

Dataset columns: `engine_displacement`, `horsepower`, `vehicle_weight`, `model_year`, `fuel_efficiency_mpg` — no `log1p`.

Related anti-example refs: `homework-2.ipynb:196` (`read_csv`), `homework-2.ipynb:228` (`seed 2`), `homework-2.ipynb:241,228`/`246-248` (`log1p`). Fix is to replace all three with the snippet above.

### Checklist

- [x] I have searched existing FAQs and this question is not already answered
- [x] The answer provides accurate, helpful information
- [x] I have included any relevant code examples or links
