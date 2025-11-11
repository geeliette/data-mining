import pandas as pd
import numpy as np
import glob, os

clean_dir = "../cleaned_munic_files"

# Trip-based aggregation
files = glob.glob(os.path.join(clean_dir, "*_cleaned.csv"))
print(f"Found {len(files)} cleaned files\n")

def pick_time_column(df):
    for c in ["time", "received_at"]:
        if c in df.columns:
            return c
    raise ValueError("No timestamp column")

# Split data into trips: a new trip starts when time gap > 10 mins
def segment_trips(df, tcol, max_gap_minutes=10):
    dt_minutes = df[tcol].diff().dt.total_seconds() / 60
    trip_breaks = (dt_minutes > max_gap_minutes) | dt_minutes.isna()  # NaN for first row
    return trip_breaks.cumsum()

trip_rows = []

for f in files:
    df = pd.read_csv(f, low_memory=False)
    if "speed_kmh" not in df.columns or "fuel_value" not in df.columns:
        continue

    # Time prep
    try:
        tcol = pick_time_column(df)
    except:
        continue

    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")  # convert to datetime
    df = df.dropna(subset=[tcol, "speed_kmh", "fuel_value"])  # drop rows with missing key cols
    df = df.sort_values(tcol).reset_index(drop=True)  # sort by time

    if len(df) < 10:
        continue

    # Only keep ml-based fuel data (exclude % streams)
    if df["fuel_value"].max() <= 100:
        continue

    # Segment into trips; add trip id column
    df["trip_id"] = segment_trips(df, tcol, max_gap_minutes=10)

    for trip_id, trip_df in df.groupby("trip_id", sort=False):
        if len(trip_df) < 5:
            continue

        # Time deltas (s)
        dt_s = trip_df[tcol].diff().dt.total_seconds().fillna(0) # seconds since previous sample
        # Clamp negative deltas (clock issues) to 0
        dt_s = dt_s.mask(dt_s < 0, 0)

        duration_sec = float(dt_s.sum())
        if duration_sec < 30 or duration_sec > 7200:
            continue

        # Fuel consumption (target; mL)
        fuel_inc = trip_df["fuel_value"].diff().clip(lower=0).fillna(0)
        fuel_consumed = float(fuel_inc.sum())
        if fuel_consumed < 10 or fuel_consumed > 20000:
            continue

        # SPEED PROFILE FEATURES
        speed = trip_df["speed_kmh"].astype(float).values

        # Time weights for irregular sampling (first sample has dt=0; safe for weighted sums)
        w = dt_s.values # time spent at each speed before next reading
        w_sum = w.sum() # duration of trip
        if w_sum <= 0:
            continue

        # Time-weighted basic stats
        avg_speed = float(np.average(speed, weights=w))
        speed_median = float(np.median(speed))
        try:
            mean_w = avg_speed
            var_w = np.average((speed - mean_w) ** 2, weights=w)
            speed_std = float(np.sqrt(max(var_w, 0)))
        except Exception:
            speed_std = float(np.std(speed))
        max_speed = float(np.max(speed))
        min_speed = float(np.min(speed))

        # Accel / decel using discrete derivative (km/h per second)
        dv = np.diff(speed, prepend=speed[0]) # diff between consecutive speeds
        # Guard against 0 dt to avoid division by zero
        dt_s_safe = np.where(w == 0, np.nan, w)
        accel_kmh_per_s = dv / dt_s_safe
        # Replace inf/NaN from 0 dt with 0 accel
        accel_kmh_per_s = np.where(np.isfinite(accel_kmh_per_s), accel_kmh_per_s, 0.0)

        # Time-weighted mean positive/negative accel (take only + / - parts)
        pos_mask = accel_kmh_per_s > 0
        neg_mask = accel_kmh_per_s < 0
        avg_acceleration = (
            float(np.average(accel_kmh_per_s[pos_mask], weights=w[pos_mask]))
            if pos_mask.any() and w[pos_mask].sum() > 0 else 0.0
        )
        avg_deceleration = abs(
            float(np.average(accel_kmh_per_s[neg_mask], weights=w[neg_mask]))
            if neg_mask.any() and w[neg_mask].sum() > 0 else 0.0
        )

        # Count acceleration/deceleration “events” by threshold on accel magnitude
        accel_thr = 0.5  # km/h per second
        num_accelerations = int((accel_kmh_per_s >  accel_thr).sum())
        num_decelerations = int((accel_kmh_per_s < -accel_thr).sum())

        # Store features (no distance in inputs)
        trip_rows.append({
            # Basic speed statistics (time-weighted)
            "avg_speed_kmh": avg_speed,
            "max_speed_kmh": max_speed,
            "min_speed_kmh": min_speed,
            "speed_std": speed_std,
            "speed_median": speed_median,

            # Polynomial term (for polynomial regression)
            "avg_speed_squared": avg_speed ** 2,

            # Driving behavior
            "avg_acceleration": avg_acceleration,
            "avg_deceleration": avg_deceleration,
            "num_accelerations": num_accelerations,
            "num_decelerations": num_decelerations,

            # Trip duration (time-based)
            "duration_min": duration_sec / 60.0,

            # Target variable
            "fuel_consumed_ml": fuel_consumed,

            # Metadata (reference only; not for modeling)
            "file": os.path.basename(f),
        })

# === Combine all trips ===
data = pd.DataFrame(trip_rows)
print(f"Total trips extracted: {len(data)}")

if len(data) == 0:
    print("No valid trips found — check data format.")
    raise SystemExit(0)

print("\n" + "="*70)
print("SPEED PROFILE FEATURES SUMMARY")
print("="*70)

key_features = [
    "avg_speed_kmh", "speed_std",
    "avg_acceleration", "avg_deceleration",
    "duration_min", "fuel_consumed_ml"
]
for col in key_features:
    if col in data.columns:
        s = data[col]
        print(f"\n{col}:")
        print(f"  Mean: {s.mean():.2f}")
        print(f"  Std:  {s.std():.2f}")
        print(f"  Min:  {s.min():.2f}")
        print(f"  Max:  {s.max():.2f}")

# === Export aggregated data (speed profile only) ===
out_path = "trip_speed_profile_data.csv"
data.to_csv(out_path, index=False)
print("\n" + "="*70)
print(f"Speed profile data exported to: {out_path}")
print("="*70)
