from pathlib import Path
import pandas as pd
import numpy as np

PUBLIC_COLUMNS = ["datetime_et", "storm_id", "site", "Turbidity", "EC", "DO", "Temp", "Radar"]

def load_manuscript_events(data_dir="data/manuscript_storm_events"):
    """Load the four manuscript event-window CSV files into the all_storms structure."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("storm_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No storm_*.csv files found in {data_dir}. "
            "Run scripts/extract_manuscript_storm_events.py privately first."
        )

    all_storms = {}
    for f in files:
        df = pd.read_csv(f, parse_dates=["datetime_et"])
        missing = [c for c in PUBLIC_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{f.name} is missing required columns: {missing}")
        storm_id = int(df["storm_id"].iloc[0])
        site = str(df["site"].iloc[0])
        df = df.set_index("datetime_et").sort_index()
        all_storms[storm_id] = {
            "metadata": {"storm_id": storm_id, "site": site, "source_file": f.name},
            "data": df[["Turbidity","EC","DO","Temp","Radar","storm_id","site"]].copy()
        }
    return all_storms

def get_reference_signal(all_storms, storm_id, parameter, dropna=True):
    s = all_storms[storm_id]["data"][parameter]
    if dropna:
        s = s.dropna()
    return s.to_numpy(dtype=float)

def get_reference_series(all_storms, storm_id, parameter, dropna=True):
    s = all_storms[storm_id]["data"][parameter].astype(float)
    return s.dropna() if dropna else s
