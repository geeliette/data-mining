import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load preprocessed trip-level data
data = pd.read_csv("trip_aggregated_data.csv")
print(f"Loaded {len(data)} trips\n")

# Linear Regression
print("\n" + "="*70)
print("Model: Predict Fuel Consumed (ml) per Trip")
print("="*70)

features = ["avg_speed_kmh", "avg_speed_squared", "distance_km"]
X = data[features].values
y = data["fuel_consumed_ml"].values

# train/test split
test_size = 0.2 if len(data) >= 50 else 0.3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# train model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test) # predict on test data

# evaluate
mae  = mean_absolute_error(y_test, y_pred) # average absolute error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\nFeatures: {', '.join(features)}")
print(f"Samples: {len(data)} trips")

print("\nCoefficients:")
for i, feat in enumerate(features):
    print(f"  {feat:25s}: {lr.coef_[i]:+.6f}")
print(f"  {'Intercept':25s}: {lr.intercept_:+.6f}")

print("\nPerformance:")
print(f"  MAE:  {mae:.2f} ml")
print(f"  RMSE: {rmse:.2f} ml")
print(f"  R²:   {r2:.4f}")
