import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
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
gb = GradientBoostingRegressor(
    n_estimators=500,       # Number of boosting stages
    learning_rate=0.05,     # Shrinks contribution of each tree
    max_depth=4,            # Shallow trees (typical for boosting)
    min_samples_split=25,   # Minimum samples to split node
    min_samples_leaf=12,    # Minimum samples in leaf
    subsample=0.8,          # Fraction of samples for each tree
    max_features=0.7,       # Fraction of features to consider
    validation_fraction=0.1,    # Use 10% of training for validation
    n_iter_no_change=20,        # Stop if no improvement for 20 iterations
    random_state=42,
    verbose=0               # Set to 1 to see training progress
)

# Train 
gb.fit(X_train, y_train)

# Predict
y_pred = gb.predict(X_test)

# Evaluate 
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("="*70)
print("Model D: Gradient Boosting")
print("="*70)
print(f"Samples: {len(data)} trips")
print(f"Features: {len(features)}\n")

print("Performance:")
print(f"  MAE:  {mae:.2f} ml")
print(f"  RMSE: {rmse:.2f} ml")
print(f"  R²:   {r2:.4f}")

# Feature importance
importances = pd.Series(gb.feature_importances_, index=features).sort_values(ascending=False)
print("\nTop 10 Feature Importances:")
print(importances.head(10).to_string())