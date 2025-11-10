import pandas as pd
import numpy as np
import glob, os

# === CONFIG ===
clean_dir = "../cleaned_munic_files"

# Trip-based aggregation
files = glob.glob(os.path.join(clean_dir, "*_cleaned.csv"))
print(f"Found {len(files)} cleaned files\n")

def pick_time_column(df):
    for c in ["time", "received_at"]:
        if c in df.columns:
            return c
    raise ValueError("No timestamp column")

# split data into trips: a new trip starts when time gap >10 mins
def segment_trips(df, tcol, max_gap_minutes=10):
    dt_minutes = df[tcol].diff().dt.total_seconds() / 60
    trip_breaks = (dt_minutes > max_gap_minutes) | dt_minutes.isna() # na for first row
    return trip_breaks.cumsum()

trip_rows = []

for f in files:
    df = pd.read_csv(f, low_memory=False)
    if "speed_kmh" not in df.columns or "fuel_value" not in df.columns:
        continue
    
    # time prep
    try:
        tcol = pick_time_column(df)
    except:
        continue

    df[tcol] = pd.to_datetime(df[tcol], errors="coerce") # convert to datetime
    df = df.dropna(subset=[tcol, "speed_kmh", "fuel_value"]) # drop rows where key columns r missing
    df = df.sort_values(tcol).reset_index(drop=True) # sort by time

    if len(df) < 10:
        continue

    # only keep ml-based fuel data
    if df["fuel_value"].max() <= 100:
        continue

    # segment into trips; add trip id column
    df["trip_id"] = segment_trips(df, tcol, max_gap_minutes=10)

    for trip_id, trip_df in df.groupby("trip_id"):
        if len(trip_df) < 5:
            continue

        # duration & distance
        dt_s = trip_df[tcol].diff().dt.total_seconds().fillna(0) # array of time diff btwn consecutive readings
        dt_h = dt_s / 3600.0 # convert seconds to hours
        duration_sec = dt_s.sum() # total trip duration in secs
        if duration_sec < 30 or duration_sec > 7200:
            continue

        # total distance travelled
        distance_km = (trip_df["speed_kmh"] * dt_h).sum() 
        if distance_km < 0.1:
            continue

        avg_speed = distance_km / dt_h.sum()

        # fuel consumption 
        fuel_inc = trip_df["fuel_value"].diff().clip(lower=0).fillna(0)
        fuel_consumed = fuel_inc.sum() # total fuel consumed over trip
        if fuel_consumed < 10:
            continue
        if fuel_consumed > 20000: # 20 L in ml
            continue

        trip_rows.append({
            "avg_speed_kmh": avg_speed,
            "avg_speed_squared": avg_speed ** 2,
            "distance_km": distance_km,
            "fuel_consumed_ml": fuel_consumed,
            "file": os.path.basename(f)
        })

# combine all trips
data = pd.DataFrame(trip_rows)
print(f"Total trips extracted: {len(data)}")

if len(data) == 0:
    print("No valid trips found — check data format.")
    raise SystemExit(0)

print("\nData summary:")
print(data.describe().round(2))

# === Export aggregated data ===
data.to_csv("trip_aggregated_data.csv", index=False)
print(f"\nTrip-level data exported to: trip_aggregated_data.csv")
