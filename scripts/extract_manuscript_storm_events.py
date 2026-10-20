"""PRIVATE PREPARATION STEP

Run this script on your own computer, where the complete monitoring files exist.
It extracts ONLY the four event windows used in the manuscript.

The full monitoring archive is never copied to the public repository.
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------
# EDIT ONLY THESE TWO PATHS ON YOUR COMPUTER
# ---------------------------------------------------------------------
MS2226_FILE = Path(r"PATH_TO_PRIVATE_MS2226_FULL_FILE.csv")
MS3359_FILE = Path(r"PATH_TO_PRIVATE_MS3359_FULL_FILE.csv")

OUTPUT_DIR = Path("data/manuscript_storm_events")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATETIME_COLUMN = "datetime_et"

# Original source columns -> public standardized column names
COLUMN_MAP = {
    "Turbidity": "Turbidity",
    "SC": "EC",
    "DO": "DO",
    "Temperature": "Temp",
    "Radar data": "Radar",
}

EVENTS = {
    1: {"site": "MS2226", "file": MS2226_FILE,
        "start": "2022-03-08 06:00", "end": "2022-03-13 12:00"},
    2: {"site": "MS2226", "file": MS2226_FILE,
        "start": "2024-03-01 00:00", "end": "2024-03-13 00:00"},
    3: {"site": "MS3359", "file": MS3359_FILE,
        "start": "2023-02-09 00:00", "end": "2023-02-15 00:00"},
    4: {"site": "MS2226", "file": MS2226_FILE,
        "start": "2024-02-10 00:00", "end": "2024-02-18 00:00"},
}

def read_source(path):
    df = pd.read_csv(path)
    if DATETIME_COLUMN not in df.columns:
        raise ValueError(f"{DATETIME_COLUMN!r} not found in {path}")
    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(f"Missing source columns in {path.name}: {missing}")
    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN])
    return df.sort_values(DATETIME_COLUMN)

cache = {}
for storm_id, event in EVENTS.items():
    source = event["file"]
    if source not in cache:
        cache[source] = read_source(source)

    df = cache[source]
    start, end = pd.Timestamp(event["start"]), pd.Timestamp(event["end"])
    sub = df.loc[df[DATETIME_COLUMN].between(start, end),
                 [DATETIME_COLUMN, *COLUMN_MAP.keys()]].copy()
    sub = sub.rename(columns=COLUMN_MAP)
    sub.insert(1, "storm_id", storm_id)
    sub.insert(2, "site", event["site"])

    outfile = OUTPUT_DIR / f"storm_{storm_id:02d}.csv"
    sub.to_csv(outfile, index=False)
    print(f"Saved {outfile}: {len(sub):,} rows")

print("\nOnly the four manuscript event windows were exported.")
