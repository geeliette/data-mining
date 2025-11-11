# random_forest_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data 
data = pd.read_csv("trip_speed_profile_data.csv")
print(f"Loaded {len(data)} trips\n")

# Select features
target = "fuel_consumed_ml"
ignore_cols = ["file", target]
features = [c for c in data.columns if c not in ignore_cols]

print("Using features:")
print(features, "\n")

X = data[features].values
y = data[target].values

# Train/test split 
test_size = 0.2 if len(data) >= 200 else 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Model setup 
rf = RandomForestRegressor(
    n_estimators=200,       # number of trees
    max_depth=10,           # limit tree depth to prevent overfitting
    min_samples_split=25,   # default split rule
    min_samples_leaf=15,    # minimum samples per leaf
    max_features=0.6,       
    random_state=42,
    n_jobs=-1               # use all CPU cores
)

# Train 
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("="*70)
print("Model C: Random Forest")
print("="*70)
print(f"Samples: {len(data)} trips")
print(f"Features: {len(features)}\n")

print("Performance:")
print(f"  MAE:  {mae:.2f} ml")
print(f"  RMSE: {rmse:.2f} ml")
print(f"  R²:   {r2:.4f}")

# Feature importance 
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\nTop 10 Feature Importances:")
print(importances.head(10).to_string())