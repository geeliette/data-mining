import pandas as pd
import glob
import os

# === CONFIGURATION ===
# Folder containing your Munic CSV files (the raw/originals)
input_path = "./fuel_data"  # <-- adjust if needed
output_path = os.path.join(os.path.dirname(input_path), "cleaned_munic_files")
os.makedirs(output_path, exist_ok=True)

# === PROCESSING ===
csv_files = glob.glob(os.path.join(input_path, "*.csv"))

for file in csv_files:
    print(f"Processing: {file}")
    try:
        df = pd.read_csv(file, low_memory=False)

        # --- SPEED CLEANING (GPS first, then OBD) ---
        gps_col = "TRACKS.MUNIC.GPS_SPEED (km/h)"
        obd_spd_col = "TRACKS.MUNIC.MDI_OBD_SPEED (km/h)"

        if gps_col in df.columns and obd_spd_col in df.columns:
            df["speed_kmh"] = pd.to_numeric(df[gps_col], errors="coerce").fillna(
                pd.to_numeric(df[obd_spd_col], errors="coerce")
            )
        elif gps_col in df.columns:
            df["speed_kmh"] = pd.to_numeric(df[gps_col], errors="coerce")
        elif obd_spd_col in df.columns:
            df["speed_kmh"] = pd.to_numeric(df[obd_spd_col], errors="coerce")
        else:
            df["speed_kmh"] = pd.NA  # will be dropped later

        # --- FUEL CLEANING (OBD_FUEL ml, then Dashboard1 %, then Dashboard2 %) ---
        fuel_main = "TRACKS.MUNIC.MDI_OBD_FUEL (ml)"
        dash1 = "TRACKS.MUNIC.MDI_DASHBOARD_FUEL_LEVEL (%)"
        dash2 = "TRACKS.MUNIC.MDI_CC_DASHBOARD_FUEL_LEVEL_PERCENT (%)"

        parts = []
        if fuel_main in df.columns:
            parts.append(pd.to_numeric(df[fuel_main], errors="coerce"))
        if dash1 in df.columns:
            parts.append(pd.to_numeric(df[dash1], errors="coerce"))
        if dash2 in df.columns:
            parts.append(pd.to_numeric(df[dash2], errors="coerce"))

        if parts:
            fuel = parts[0]
            for p in parts[1:]:
                fuel = fuel.where(fuel.notna(), p)
            df["fuel_value"] = fuel
        else:
            df["fuel_value"] = pd.NA  # will be dropped later

        # --- DROP rows where either speed or fuel is missing ---
        df_cleaned = df.dropna(subset=["speed_kmh", "fuel_value"]).reset_index(drop=True)

        # --- REMOVE only the original raw columns D, E, F ---
        cols_to_drop = [
            "TRACKS.MUNIC.GPS_SPEED (km/h)",
            "TRACKS.MUNIC.MDI_OBD_FUEL (ml)",
            "TRACKS.MUNIC.MDI_OBD_SPEED (km/h)"
        ]
        df_out = df_cleaned.drop(columns=[c for c in cols_to_drop if c in df_cleaned.columns])

        # --- SAVE CLEANED FILE ---
        output_file = os.path.join(
            output_path, os.path.basename(file).replace(".csv", "_cleaned.csv")
        )
        df_out.to_csv(output_file, index=False)

        print(f"✅ Saved cleaned file: {output_file}  | rows: {len(df_out)}")

    except Exception as e:
        print(f"Error processing {file}: {e}")

print("\nAll files cleaned (D,E,F removed) and saved to:", output_path)

