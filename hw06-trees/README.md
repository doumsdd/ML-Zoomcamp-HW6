# Homework 6 — Decision Trees & XGBoost Solution

## Dataset Preparation

**Source:** [Car Fuel Efficiency 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv)

- **Shape:** 10,000 rows x 11 columns
- **Target:** `fuel_efficiency_mpg` (car fuel efficiency in MPG)
- **Features:** 10 features including `model_year`, `origin`, `fuel_type`, `vehicle_weight`, `horsepower`, `acceleration`, `engine_displacement`, etc.

**Preprocessing steps (common for all questions):**
1. Fill missing values with zeros
2. Train/val/test split:
   ```python
   df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
   df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
   ```
3. Split sizes: **Train: 6,000 | Val: 2,000 | Test: 2,000**
4. Transform using `DictVectorizer(sparse=True)` — produces 16 vectorized features

---

## Question 1: Decision Tree Regressor (`max_depth=1`)

### Thinking Process

We trained a decision tree with only one split (`max_depth=1`). The tree picks the single best feature to split on at the root node, choosing the one that maximizes impurity reduction (MSE reduction). We inspect `tree_.feature[0]` to find which feature was selected and map its vectorized name back to the original column name.

```python
dt_model = DecisionTreeRegressor(max_depth=1, random_state=1)
dt_model.fit(X_train, y_train)
root_feature_idx = dt_model.tree_.feature[0]  # index of splitting feature
```

### Results
- Root feature (vectorized): `model_year`
- Root threshold: 1997.50 (splits cars into model year <= 1997 or > 1997)

### Answer: `model_year`

---

## Question 2: Random Forest RMSE (`n_estimators=10`)

### Thinking Process

Trained a random forest regressor with 10 trees and computed RMSE on the validation set.

```python
rf_q2 = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf_q2.fit(X_train, y_train)
y_pred_q2 = rf_q2.predict(X_val)
rmse_q2 = np.sqrt(mean_squared_error(y_val, y_pred_q2))
```

### Results
- Computed RMSE: **1.8332**
- Closest option among [0.1837, 1.837, 18.37, 183.7]: **1.837**

### Answer: **1.837**

---

## Question 3: Tune `n_estimators`

### Thinking Process

Tried four different numbers of trees `[10, 50, 100, 150]` and measured validation RMSE for each. More trees generally help reduce overfitting by averaging more predictions. Rounded to 3 decimal places to pick the best.

```python
for n_est in [10, 50, 100, 150]:
    rf = RandomForestRegressor(n_estimators=n_est, random_state=1, n_jobs=-1)
    rmse = np.sqrt(mean_squared_error(y_val, rf.predict(X_val)))
```

### Results

| `n_estimators` | Validation RMSE |
|:-:|:-:|
| 10 | 1.833 |
| 50 | 1.774 |
| 100 | 1.770 |
| **150** | **1.769** |

Best: **150** (lowest RMSE = 1.769) — consistent trend that more trees improve performance.

### Answer: **150**

---

## Question 4: Grid Search on `max_depth` x `n_estimators`

### Thinking Process

Grid search combining 4 `max_depth` values with 4 `n_estimators` values (16 combinations). For each `max_depth`, averaged RMSE across all `n_estimators` values. Picks the depth with lowest mean RMSE.

```python
for depth in [10, 15, 20, 25]:
    rmses_for_depth = []
    for n_est in [10, 50, 100, 150]:
        rf = RandomForestRegressor(n_estimators=n_est, max_depth=depth, random_state=1)
        mse = mean_squared_error(y_val, rf.predict(X_val))
        rmses_for_depth.append(np.sqrt(mse))
    mean_rmse = np.mean(rmses_for_depth)
```

### Results

| `max_depth` | Mean RMSE |
|:-:|:-:|
| **10** | **1.754** |
| 15 | 1.784 |
| 20 | 1.787 |
| 25 | 1.786 |

Surprising finding: simpler trees performed better! Deeper trees likely overfitted despite having more trees to average. The optimal combination was `max_depth=10, n_estimators=150` (RMSE=1.744).

### Answer: **10**

---

## Question 5: Feature Importance

### Thinking Process

Random forests store feature importance as the total normalized reduction in impurity contributed by each feature across all trees. We trained a model (`n_estimators=10, max_depth=20`) and extracted `feature_importances_`. We mapped top features back to original names and found which of the 4 given options was most important.

```python
rf_q5 = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1)
rf_q5.fit(X_train, y_train)
importances = rf_q5.feature_importances_
```

### Top Features

| Rank | Feature | Importance |
|:-:|:-:|:-:|
| 1 | `model_year` | 0.2961 |
| 2 | `origin` | 0.1469 |
| 3 | **`vehicle_weight`** | **0.1194** |
| 4 | `horsepower` | 0.1108 |
| 5 | `drivetrain` | 0.0836 |
| 6 | `acceleration` | 0.0591 |
| 7 | `engine_displacement` | 0.0536 |

Among the 4 options — `vehicle_weight`, `horsepower`, `acceleration`, `engine_displacement` — **`vehicle_weight`** has the highest importance (0.1194). Makes intuitive sense: heavier cars need more energy to move, reducing fuel efficiency.

### Answer: **`vehicle_weight`**

---

## Question 6: XGBoost — Tune `eta` (Learning Rate)

### Thinking Process

XGBoost uses `eta` (learning rate) to shrink each tree's contribution. Smaller learning rates make more gradual improvements but require more rounds. We trained two models for 100 rounds each and compared validation RMSE.

```python
dtrain = DMatrix(X_train, label=y_train)
dval = DMatrix(X_val, label=y_val)
params = {'max_depth': 6, 'min_child_weight': 1, 'objective': 'reg:squarederror',
          'seed': 1, 'verbosity': 1}
# Two models: eta=0.3 and eta=0.1
```

### Results

| Model | `eta` | Validation RMSE |
|:-:|:-:|:-:|
| A | 0.3 | 1.8265 |
| **B** | **0.1** | **1.7248** |

The smaller learning rate (eta=0.1) achieved significantly better RMSE (~5% improvement). It started higher (RMSE 2.78 at round 0) but steadily improved to reach 1.72. The eta=0.3 model converged faster but plateaued around 1.83 due to overshooting.

This demonstrates a key XGBoost principle: **smaller learning rates with more rounds generally yield better results**, but you need enough boosting rounds to let the slow learner converge.

### Answer: **0.1**

---

## Final Answers Summary

| Question | Answer |
|:-:|:-:|
| **Q1** | `model_year` |
| **Q2** | `1.837` |
| **Q3** | `150` |
| **Q4** | `10` |
| **Q5** | `vehicle_weight` |
| **Q6** | `0.1` |

---

## Key Takeaways

1. **Start simple**: A depth-1 tree chose `model_year` as the most informative single feature
2. **More trees help**: Increasing `n_estimators` consistently reduced RF RMSE
3. **Simpler ensembles win**: `max_depth=10` outperformed deeper trees on average (deeper trees overfitted)
4. **Feature importance aligns with intuition**: Weight is a major factor in fuel efficiency
5. **XGBoost benefits from slower learning**: Smaller `eta` (0.1 vs 0.3) gave ~5% better RMSE
6. **Data leak prevention**: Always drop the target column before vectorizing to avoid accidentally using it as a feature

---

## Files

| File | Purpose |
|------|---------|
| `car_fuel_efficiency_2026.csv` | Dataset (10,000 rows) |
| `solution.py` | Python script solving all 6 questions |
| `README.md` | This file — explanations and answers |

