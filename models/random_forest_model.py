import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load preprocessed trip-level data
data = pd.read_csv("trip_speed_profile_data.csv")
print(f"Loaded {len(data)} trips\n")

# Features
feature_cols = ["avg_speed_kmh", "distance_km"]
X = data[feature_cols].values
y = data["fuel_consumed_ml"].values

# Train/test split
test_size = 0.2 if len(data) >= 200 else 0.3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

print("="*70)
print("RANDOM FOREST REGRESSION")
print("="*70)

# Random Forest parameters
# n_estimators: number of trees (more = better but slower)
# max_depth: maximum depth of trees (prevent overfitting)
# min_samples_split: minimum samples required to split node
# random_state: for reproducibility

model = RandomForestRegressor(
    n_estimators=100,      # 100 trees in the forest
    max_depth=8,           # REDUCED: shallower trees to prevent overfitting
    min_samples_split=20,  # INCREASED: require more samples to split
    min_samples_leaf=10,   # INCREASED: require more samples in leaves
    max_features=0.7,      # Only consider 70% of features per split
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)

print(f"Hyperparameters:")
print(f"  n_estimators:      {model.n_estimators}")
print(f"  max_depth:         {model.max_depth}")
print(f"  min_samples_split: {model.min_samples_split}")
print(f"  min_samples_leaf:  {model.min_samples_leaf}\n")

# Train model
print("Training Random Forest...")
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Metrics
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("Performance:")
print(f"                Train        Test")
print(f"  MAE:      {mae_train:8.2f} ml  {mae_test:8.2f} ml")
print(f"  RMSE:     {rmse_train:8.2f} ml  {rmse_test:8.2f} ml")
print(f"  R²:       {r2_train:8.4f}     {r2_test:8.4f}")

# Feature importance
print("\nFeature Importance:")
importances = model.feature_importances_
for name, importance in zip(feature_cols, importances):
    print(f"  {name:25s}: {importance:.4f}")

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                             scoring='r2', n_jobs=-1)
print(f"\nCross-Validation R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Check for overfitting
overfit = r2_train - r2_test
if overfit > 0.1:
    print(f"\n⚠️  Warning: Overfitting detected (ΔR² = {overfit:.4f})")
    print("   Consider: reducing max_depth or increasing min_samples_split")
elif overfit > 0.05:
    print(f"\n⚠️  Slight overfitting (ΔR² = {overfit:.4f})")
else:
    print(f"\n✅ Good generalization (ΔR² = {overfit:.4f})")