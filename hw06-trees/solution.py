import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import json

# ============================================================
# DATA PREPARATION (Common for all questions)
# ============================================================
print("=" * 70)
print("LOADING & PREPARING DATA")
print("=" * 70)

df = pd.read_csv("car_fuel_efficiency_2026.csv")
print(f"Original shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Fill missing values with zeros
df = df.fillna(0)

# Convert DataFrame to list of dicts (required by DictVectorizer)
feature_cols = [col for col in df.columns if col != 'fuel_efficiency_mpg']
print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

# Exact split as specified
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

print(f"\nTrain: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

# DictVectorizer with sparse=True
vectorizer = DictVectorizer(sparse=True)

# Drop target column BEFORE vectorization to prevent data leak
X_train = vectorizer.fit_transform(df_train.drop(columns=['fuel_efficiency_mpg']).to_dict('records'))
X_val = vectorizer.transform(df_val.drop(columns=['fuel_efficiency_mpg']).to_dict('records'))
X_test = vectorizer.transform(df_test.drop(columns=['fuel_efficiency_mpg']).to_dict('records'))

y_train = df_train['fuel_efficiency_mpg'].values
y_val = df_val['fuel_efficiency_mpg'].values
y_test = df_test['fuel_efficiency_mpg'].values

# Get feature names after vectorization
feature_names = vectorizer.get_feature_names_out()
print(f"Vectorized feature names ({len(feature_names)}): {list(feature_names)}")

# Build a mapping from vectorized names back to original column names
original_numeric_cols = ['vehicle_weight', 'horsepower', 'acceleration', 
                         'engine_displacement', 'model_year', 'cylinders', 'displacement']
categorical_cols = {'origin': ['USA', 'Europe', 'Asia'], 'fuel_type': ['gas', 'diesel']}

def map_vec_name_to_original(vec_name):
    """Map a vectorized feature name back to its original column name."""
    vec_name_clean = vec_name.replace(' ', '_')
    # Check if it's a one-hot feature (e.g., origin_USA, fuel_type_gas)
    for cat_col, categories in categorical_cols.items():
        for cat in categories:
            pattern = f"{cat_col}={cat}"
            if pattern.replace(' ', '_') in vec_name_clean or vec_name_clean == pattern.replace(' ', '_'):
                return cat_col
    # Check if it's an original numeric column
    base_name = vec_name_clean.split('=')[-1] if '=' in vec_name_clean else vec_name_clean
    for orig_col in original_numeric_cols:
        if orig_col in vec_name_clean:
            return orig_col
    return vec_name_clean

# ============================================================
# QUESTION 1: Decision Tree Regressor (max_depth=1)
# ============================================================
print("\n" + "=" * 70)
print("QUESTION 1: Decision Tree (max_depth=1)")
print("=" * 70)

dt_model = DecisionTreeRegressor(max_depth=1, random_state=1)
dt_model.fit(X_train, y_train)

# The root split feature index
root_feature_idx = dt_model.tree_.feature[0]
if root_feature_idx == -1:
    print("No split found at root!")
else:
    root_feature_name = feature_names[root_feature_idx]
    original_feature = map_vec_name_to_original(root_feature_name)
    print(f"Root feature (vectorized): {root_feature_name}")
    print(f"Root feature (original): {original_feature}")
    print(f"Root threshold: {dt_model.tree_.threshold[0]:.2f}")
    
    q1_answer = original_feature
    
result_q1 = {
    "question": 1,
    "model": "DecisionTreeRegressor(max_depth=1)",
    "root_feature_vectorized": root_feature_name,
    "root_feature_original": original_feature,
    "root_threshold": float(dt_model.tree_.threshold[0]),
    "answer": q1_answer
}
print(json.dumps(result_q1, indent=2))

# ============================================================
# QUESTION 2: Random Forest RMSE (n_estimators=10)
# ============================================================
print("\n" + "=" * 70)
print("QUESTION 2: Random Forest (n_estimators=10)")
print("=" * 70)

rf_q2 = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf_q2.fit(X_train, y_train)
y_pred_q2 = rf_q2.predict(X_val)
rmse_q2 = np.sqrt(mean_squared_error(y_val, y_pred_q2))
print(f"RMSE on validation: {rmse_q2:.4f}")

q2_options = [0.1837, 1.837, 18.37, 183.7]
q2_closest = min(q2_options, key=lambda x: abs(x - rmse_q2))
print(f"Closest option: {q2_closest}")

result_q2 = {
    "question": 2,
    "model": "RandomForestRegressor(n_estimators=10, random_state=1)",
    "rmse_val": round(rmse_q2, 4),
    "closest_option": q2_closest,
    "answer": q2_closest
}
print(json.dumps(result_q2, indent=2))

# ============================================================
# QUESTION 3: Tune n_estimators
# ============================================================
print("\n" + "=" * 70)
print("QUESTION 3: Tune n_estimators")
print("=" * 70)

n_estimators_values = [10, 50, 100, 150]
rmse_by_n_est = {}

for n_est in n_estimators_values:
    rf = RandomForestRegressor(n_estimators=n_est, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_by_n_est[n_est] = round(rmse, 3)
    print(f"  n_estimators={n_est:>3} -> RMSE = {rmse:.4f}")

best_n_est = min(rmse_by_n_est, key=rmse_by_n_est.get)
print(f"\nBest n_estimators: {best_n_est} (RMSE = {rmse_by_n_est[best_n_est]:.3f})")

result_q3 = {
    "question": 3,
    "results": rmse_by_n_est,
    "best_n_estimators": best_n_est,
    "answer": best_n_est
}
print(json.dumps(result_q3, indent=2))

# ============================================================
# QUESTION 4: Grid Search max_depth x n_estimators
# ============================================================
print("\n" + "=" * 70)
print("QUESTION 4: Grid Search max_depth x n_estimators")
print("=" * 70)

max_depth_values = [10, 15, 20, 25]
n_est_grid = [10, 50, 100, 150]

mean_rmse_by_depth = {}

for depth in max_depth_values:
    rmses_for_depth = []
    for n_est in n_est_grid:
        rf = RandomForestRegressor(n_estimators=n_est, max_depth=depth, random_state=1, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmses_for_depth.append(rmse)
        print(f"  max_depth={depth:>2}, n_estimators={n_est:>3} -> RMSE = {rmse:.4f}")
    
    mean_rmse = np.mean(rmses_for_depth)
    mean_rmse_by_depth[depth] = round(mean_rmse, 3)
    print(f"  >>> Mean RMSE for max_depth={depth}: {mean_rmse:.4f}")

best_depth = min(mean_rmse_by_depth, key=mean_rmse_by_depth.get)
print(f"\nBest max_depth: {best_depth} (mean RMSE = {mean_rmse_by_depth[best_depth]:.3f})")

result_q4 = {
    "question": 4,
    "results": mean_rmse_by_depth,
    "best_max_depth": best_depth,
    "answer": best_depth
}
print(json.dumps(result_q4, indent=2))

# ============================================================
# QUESTION 5: Feature Importance
# ============================================================
print("\n" + "=" * 70)
print("QUESTION 5: Feature Importance")
print("=" * 70)

rf_q5 = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
rf_q5.fit(X_train, y_train)

importances = rf_q5.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("Top 15 most important features:")
top_prints = []
for _, row in importance_df.head(15).iterrows():
    orig_name = map_vec_name_to_original(row['feature'])
    line = f"  {row['feature']:<40} -> {orig_name:<20} importance: {row['importance']:.4f}"
    print(line)
    top_prints.append((row['feature'], orig_name, row['importance']))

# Find the most important among the 4 options given
q5_options = ['vehicle_weight', 'horsepower', 'acceleration', 'engine_displacement']
most_important_among_options = None
max_imp = 0

for vec_name, orig_name, imp in top_prints:
    if orig_name in q5_options:
        if imp > max_imp:
            max_imp = imp
            most_important_among_options = orig_name

print(f"\nMost important among options: {most_important_among_options} (importance: {max_imp:.4f})")

result_q5 = {
    "question": 5,
    "model": "RF(n_estimators=10, max_depth=20, random_state=1)",
    "top_features": top_prints[:10],
    "most_important_among_options": most_important_among_options,
    "answer": most_important_among_options
}
print(json.dumps(result_q5, indent=2))

# ============================================================
# QUESTION 6: XGBoost - Tune eta
# ============================================================
print("\n" + "=" * 70)
print("QUESTION 6: XGBoost - Tune eta")
print("=" * 70)

import xgboost as xgb
from xgboost import DMatrix

# Create DMatrix
dtrain = DMatrix(X_train, label=y_train)
dval = DMatrix(X_val, label=y_val)
watchlist = [(dval, 'validation')]

# Base parameters
base_params = {
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

# Test eta=0.3
print("\nTraining with eta=0.3...")
params_eta03 = {**base_params, 'eta': 0.3}
model_eta03 = xgb.train(params_eta03, dtrain, num_boost_round=100, evals=watchlist)
preds_eta03 = model_eta03.predict(dval)
rmse_eta03 = np.sqrt(mean_squared_error(y_val, preds_eta03))
print(f"  RMSE with eta=0.3: {rmse_eta03:.4f}")

# Test eta=0.1
print("\nTraining with eta=0.1...")
params_eta01 = {**base_params, 'eta': 0.1}
model_eta01 = xgb.train(params_eta01, dtrain, num_boost_round=100, evals=watchlist)
preds_eta01 = model_eta01.predict(dval)
rmse_eta01 = np.sqrt(mean_squared_error(y_val, preds_eta01))
print(f"  RMSE with eta=0.1: {rmse_eta01:.4f}")

if rmse_eta01 < rmse_eta03:
    best_eta = 0.1
elif rmse_eta03 < rmse_eta01:
    best_eta = 0.3
else:
    best_eta = "Both give equal value"

print(f"\nBest eta: {best_eta}")

result_q6 = {
    "question": 6,
    "model": "XGBoost(num_rounds=100, max_depth=6)",
    "rmse_eta03": round(rmse_eta03, 4),
    "rmse_eta01": round(rmse_eta01, 4),
    "answer": best_eta
}
print(json.dumps(result_q6, indent=2))

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL ANSWERS SUMMARY")
print("=" * 70)

all_answers = [result_q1, result_q2, result_q3, result_q4, result_q5, result_q6]

for ans in all_answers:
    print(f"Q{ans['question']}: {ans['answer']}")

# Save answers to JSON file for easy reference
with open("answers.json", "w") as f:
    json.dump({
        "answers": all_answers,
        "timestamp": pd.Timestamp.now().isoformat()
    }, f, indent=2)
print("\nAnswers saved to answers.json")
