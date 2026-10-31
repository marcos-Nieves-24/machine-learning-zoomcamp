"""HW6 Zoomcamp 2026 — Decision Trees and Ensemble Learning.

Procedure from the statement (2026 cohort, pinned dataset):
- fillna(0), 60/20/20 split with random_state=1, DictVectorizer(sparse=True)
- Q1: stump (max_depth=1) -> splitting feature
- Q2: RF(10) -> validation RMSE
- Q3: grid n_estimators [10, 50, 100, 150] -> lowest RMSE (3 decimals)
- Q4: best max_depth in [10,15,20,25] by mean RMSE over the grid
- Q5: importance with RF(10, depth=20)
- Q6: XGBoost eta 0.3 vs 0.1, 100 rounds
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from xgboost import DMatrix, train as xgb_train

df = pd.read_csv("car_fuel_efficiency_2026.csv").fillna(0)
y = df["fuel_efficiency_mpg"].to_numpy()
X = df.drop(columns=["fuel_efficiency_mpg"])

X_tr_full, X_te, y_tr_full, y_te = train_test_split(
    X, y, test_size=0.2, random_state=1)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_tr_full, y_tr_full, test_size=0.25, random_state=1)  # 60/20/20

dv = DictVectorizer(sparse=True)
X_tr_m = dv.fit_transform(X_tr.to_dict(orient="records"))
X_va_m = dv.transform(X_va.to_dict(orient="records"))
print("features:", list(dv.feature_names_))

# Q1: stump
stump = DecisionTreeRegressor(max_depth=1, random_state=1)
stump.fit(X_tr_m, y_tr)
feat = dv.feature_names_[stump.tree_.feature[0]]
print(f"Q1 split feature: {feat} (threshold={stump.tree_.threshold[0]:.3f})")

# Q2: RF baseline
rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_tr_m, y_tr)
print(f"Q2 RF(10) val RMSE: {root_mean_squared_error(y_va, rf.predict(X_va_m)):.4f}")

# Q3 (2026): grid [10, 50, 100, 150] -> lowest validation RMSE (3 decimals)
print("Q3 n_est -> val RMSE:")
best3 = None
for n in [10, 50, 100, 150]:
    m = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    m.fit(X_tr_m, y_tr)
    rmse = root_mean_squared_error(y_va, m.predict(X_va_m))
    print(f"  {n:3d} {rmse:.4f} (3dec={round(rmse, 3)})")
    if best3 is None or round(rmse, 3) < best3[1]:
        best3 = (n, round(rmse, 3))
print(f"Q3 lowest RMSE at n_estimators={best3[0]}")

# Q4: best max_depth by mean RMSE
print("Q4 depth -> mean RMSE:")
best = None
for d in [10, 15, 20, 25]:
    rmses = []
    for n in [10, 50, 100, 150]:
        m = RandomForestRegressor(n_estimators=n, max_depth=d,
                                  random_state=1, n_jobs=-1)
        m.fit(X_tr_m, y_tr)
        rmses.append(root_mean_squared_error(y_va, m.predict(X_va_m)))
    mean_rmse = np.mean(rmses)
    print(f"  depth={d}: mean={mean_rmse:.4f}")
    if best is None or mean_rmse < best[1]:
        best = (d, mean_rmse)
print(f"Q4 best max_depth={best[0]}")

# Q5: importance
rf5 = RandomForestRegressor(n_estimators=10, max_depth=20,
                            random_state=1, n_jobs=-1)
rf5.fit(X_tr_m, y_tr)
imp = sorted(zip(dv.feature_names_, rf5.feature_importances_),
             key=lambda t: -t[1])[:6]
print("Q5 top importances:", [(f, round(v, 4)) for f, v in imp])

# Q6: XGBoost eta
dtr, dva = DMatrix(X_tr_m, label=y_tr), DMatrix(X_va_m, label=y_va)
for eta in (0.3, 0.1):
    params = {"eta": eta, "max_depth": 6, "min_child_weight": 1,
              "objective": "reg:squarederror", "nthread": 8,
              "seed": 1, "verbosity": 0}
    booster = xgb_train(params, dtr, num_boost_round=100,
                        evals=[(dva, "val")], verbose_eval=False)
    pred = booster.predict(dva)
    print(f"Q6 eta={eta}: val RMSE={root_mean_squared_error(y_va, pred):.4f}")
