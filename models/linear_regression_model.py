# linear_regression_speed_profile.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load preprocessed data
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
test_size = 0.2 if len(data) >= 200 else 0.3 # 20% of trips for testing (or 30% if dataset small)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("="*70)
print("Model A: Linear Regression")
print("="*70)
print(f"Samples: {len(data)} trips")

print("\nCoefficients:")
for f, coef in zip(features, lr.coef_):
    print(f"  {f:25s}: {coef:+.6f}")
print(f"  {'Intercept':25s}: {lr.intercept_:+.6f}")

print("\nPerformance:")
print(f"  MAE:  {mae:.2f} ml")
print(f"  RMSE: {rmse:.2f} ml")
print(f"  R²:   {r2:.4f}")
