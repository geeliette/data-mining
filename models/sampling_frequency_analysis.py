# sampling_frequency_final.py
import pandas as pd
import numpy as np
import glob, os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

clean_dir = "../cleaned_munic_files"
files = glob.glob(os.path.join(clean_dir, "*_cleaned.csv"))

# Only test feasible sampling rates
SCENARIOS = {
    "Baseline (5s)": 1,
    "10s sampling": 2,
    "20s sampling": 4,
}

def pick_time_column(df):
    for c in ["time", "received_at"]:
        if c in df.columns:
            return c
    raise ValueError("No timestamp column")

def segment_trips(df, tcol, max_gap_minutes=10):
    dt_minutes = df[tcol].diff().dt.total_seconds() / 60
    trip_breaks = (dt_minutes > max_gap_minutes) | dt_minutes.isna()
    return trip_breaks.cumsum()

def extract_features_downsampled(downsample_factor):
    trip_rows = []
    
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        if "speed_kmh" not in df.columns or "fuel_value" not in df.columns:
            continue
        
        try:
            tcol = pick_time_column(df)
        except:
            continue
        
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
        df = df.dropna(subset=[tcol, "speed_kmh", "fuel_value"])
        df = df.sort_values(tcol).reset_index(drop=True)
        
        if len(df) < 10:
            continue
        
        if df["fuel_value"].max() <= 100:
            continue
        
        df["trip_id"] = segment_trips(df, tcol, max_gap_minutes=10)
        
        for trip_id, trip_df in df.groupby("trip_id", sort=False):
            if len(trip_df) < 10:
                continue
            
            # Check fuel on full trip
            fuel_total = trip_df["fuel_value"].diff().clip(lower=0).sum()
            if fuel_total < 10 or fuel_total > 20000:
                continue
            
            # Downsample
            trip_df = trip_df.reset_index(drop=True)
            trip_df_down = trip_df.iloc[::downsample_factor, :].copy()
            
            if len(trip_df_down) < 3:
                continue
            
            # Recalculate features
            dt_s = trip_df_down[tcol].diff().dt.total_seconds().fillna(0)
            dt_s = dt_s.mask(dt_s < 0, 0)
            
            duration_sec = dt_s.sum()
            if duration_sec == 0:
                continue
            
            speed = trip_df_down["speed_kmh"].values
            w = dt_s.values
            w_sum = w.sum()
            
            if w_sum == 0:
                w_sum = len(speed)
                w = np.ones(len(speed))
            
            avg_speed = np.average(speed, weights=w)
            speed_median = np.median(speed)
            try:
                var_w = np.average((speed - avg_speed)**2, weights=w)
                speed_std = np.sqrt(max(var_w, 0))
            except:
                speed_std = np.std(speed)
            
            max_speed = np.max(speed)
            min_speed = np.min(speed)
            
            dv = np.diff(speed, prepend=speed[0])
            dt_safe = np.where(w == 0, 1, w)
            accel = dv / dt_safe
            accel = np.where(np.isfinite(accel), accel, 0)
            
            pos_mask = accel > 0
            neg_mask = accel < 0
            
            avg_accel = np.average(accel[pos_mask], weights=w[pos_mask]) if pos_mask.any() else 0
            avg_decel = abs(np.average(accel[neg_mask], weights=w[neg_mask])) if neg_mask.any() else 0
            
            num_accel = int((accel > 0.5).sum())
            num_decel = int((accel < -0.5).sum())
            
            trip_rows.append({
                "avg_speed_kmh": float(avg_speed),
                "max_speed_kmh": float(max_speed),
                "min_speed_kmh": float(min_speed),
                "speed_std": float(speed_std),
                "speed_median": float(speed_median),
                "avg_speed_squared": float(avg_speed ** 2),
                "avg_acceleration": float(avg_accel),
                "avg_deceleration": float(avg_decel),
                "num_accelerations": num_accel,
                "num_decelerations": num_decel,
                "duration_min": duration_sec / 60.0,
                "fuel_consumed_ml": float(fuel_total),
            })
    
    return pd.DataFrame(trip_rows)

def evaluate_models(data):
    """Evaluate all 4 models."""
    if len(data) < 50:
        return None
    
    features = ["avg_speed_kmh", "max_speed_kmh", "min_speed_kmh", "speed_std", 
                "speed_median", "avg_speed_squared", "avg_acceleration", 
                "avg_deceleration", "num_accelerations", "num_decelerations", "duration_min"]
    target = "fuel_consumed_ml"
    
    X = data[features].values
    y = data[target].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    # Linear
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results['Linear_R2'] = r2_score(y_test, y_pred)
    results['Linear_MAE'] = mean_absolute_error(y_test, y_pred)
    
    # Polynomial (skip if too few samples)
    if len(X_train) >= 100:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        pr = LinearRegression()
        pr.fit(X_train_poly, y_train)
        y_pred = pr.predict(X_test_poly)
        results['Polynomial_R2'] = r2_score(y_test, y_pred)
        results['Polynomial_MAE'] = mean_absolute_error(y_test, y_pred)
    else:
        results['Polynomial_R2'] = np.nan
        results['Polynomial_MAE'] = np.nan
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                min_samples_split=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results['RF_R2'] = r2_score(y_test, y_pred)
    results['RF_MAE'] = mean_absolute_error(y_test, y_pred)
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, 
                                     max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    results['GB_R2'] = r2_score(y_test, y_pred)
    results['GB_MAE'] = mean_absolute_error(y_test, y_pred)
    
    return results

# Main Analysis
print("="*70)
print("SAMPLING FREQUENCY SENSITIVITY ANALYSIS")
print("="*70)
print("Testing 3 sampling frequencies: 5s, 10s, 20s\n")

all_results = []
baseline_r2 = None

for name, factor in SCENARIOS.items():
    print(f"{name} (factor={factor})...")
    data = extract_features_downsampled(factor)
    print(f"  Trips extracted: {len(data)}")
    
    if len(data) < 50:
        print(f"  ⚠️  Insufficient data, skipping\n")
        continue
    
    scores = evaluate_models(data)
    if scores:
        result = {
            'Scenario': name,
            'Interval_s': 5 * factor,
            'Bandwidth_%': round(100/factor, 1),
            'Trips': len(data),
        }
        result.update(scores)
        all_results.append(result)
        
        print(f"  R² scores:")
        print(f"    Linear:           {scores['Linear_R2']:.4f}")
        if not np.isnan(scores['Polynomial_R2']):
            print(f"    Polynomial:       {scores['Polynomial_R2']:.4f}")
        print(f"    Random Forest:    {scores['RF_R2']:.4f}")
        print(f"    Gradient Boosting: {scores['GB_R2']:.4f}")
        print()
        
        if factor == 1:
            baseline_r2 = max(scores['Linear_R2'], scores['GB_R2'], scores['RF_R2'])

# Results Table
print("="*70)
print("RESULTS SUMMARY")
print("="*70)

df = pd.DataFrame(all_results)

# Calculate best R2 per scenario
df['Best_R2'] = df[['Linear_R2', 'RF_R2', 'GB_R2']].max(axis=1)

if baseline_r2:
    df['Degradation'] = baseline_r2 - df['Best_R2']
    df['Degradation_%'] = (df['Degradation'] / baseline_r2 * 100).round(1)

print("\nPerformance vs Bandwidth Trade-off:\n")
display_cols = ['Scenario', 'Interval_s', 'Bandwidth_%', 'Trips', 'Best_R2', 'Degradation_%']
print(df[display_cols].to_string(index=False))

print("\n" + "="*70)
print("DETAILED MODEL COMPARISON")
print("="*70)
detail_cols = ['Scenario', 'Linear_R2', 'Polynomial_R2', 'RF_R2', 'GB_R2']
print("\n", df[detail_cols].to_string(index=False))

# Analysis
print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

for _, row in df.iterrows():
    if row['Interval_s'] == 5:
        continue
    
    print(f"\n{row['Scenario']}:")
    print(f"  Bandwidth reduction: {100 - row['Bandwidth_%']:.0f}%")
    print(f"  Best R²: {row['Best_R2']:.4f}")
    print(f"  Performance loss: {row['Degradation_%']:.1f}%")
    
    if row['Degradation_%'] < 10:
        recommendation = "RECOMMENDED: Excellent trade-off"
    elif row['Degradation_%'] < 20:
        recommendation = "ACCEPTABLE: Moderate degradation"
    else:
        recommendation = "NOT RECOMMENDED: High degradation"
    
    print(f"  Assessment: {recommendation}")