"""Temporal aggregation helpers."""
import numpy as np
import pandas as pd

def resample_all_storms_dict(
    all_storms,
    freq,
    parameter_cols=None
):
    """
    Resample each storm dataframe when datetime_et is the index.

    all_storms is expected to be a dictionary/list-like object where:
        all_storms[i]["data"] is a dataframe
    and datetime_et is the index.
    """

    if parameter_cols is None:
        parameter_cols = ["Turbidity", "EC", "DO", "Temp", "Radar"]

    resampled_storms = {}

    for storm_key, storm_obj in all_storms.items():

        df = storm_obj["data"].copy()

        # Make sure datetime_et index is datetime
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Keep metadata
        storm_id = df["storm_id"].iloc[0] if "storm_id" in df.columns else storm_key
        site = df["site"].iloc[0] if "site" in df.columns else None

        # Resample sensor variables
        df_resampled = df[parameter_cols].resample(freq).mean()

        # Drop rows where all sensor values are missing
        df_resampled = df_resampled.dropna(how="all")

        # Add metadata back
        df_resampled["storm_id"] = storm_id
        df_resampled["site"] = site

        # Keep index name
        df_resampled.index.name = "datetime_et"

        # Store back using same structure
        new_obj = storm_obj.copy()
        new_obj["data"] = df_resampled

        resampled_storms[storm_key] = new_obj

    return resampled_storms

def freq_to_minutes(freq_label):
    """
    Convert sampling frequency label to approximate minutes.
    """

    if freq_label == "11min_original":
        return 11.5
    elif freq_label == "30min":
        return 30.0
    elif freq_label == "1h":
        return 60.0
    else:
        raise ValueError(f"Unknown frequency label: {freq_label}")

def block_steps_from_duration(freq_label, severity):
    """
    Convert target block duration to number of observations
    for each sampling frequency.
    """

    dt_min = freq_to_minutes(freq_label)

    if severity == "moderate":
        # Original: about 44–84 minutes
        min_duration = 44
        max_duration = 84

    elif severity == "severe":
        # Original: about 99–180 minutes
        min_duration = 99
        max_duration = 180

    else:
        raise ValueError(f"Unknown severity: {severity}")

    min_steps = max(1, int(round(min_duration / dt_min)))
    max_steps = max(min_steps, int(round(max_duration / dt_min)))

    return min_steps, max_steps

def apply_block_missing_frequency_aware(
    x,
    severity,
    rng,
    freq_label,
    moderate_n_blocks=(1, 4),
    severe_n_blocks=(5, 10)
):
    """
    Apply block missingness using block lengths adjusted for sampling frequency.
    """

    assert severity in ["moderate", "severe"]

    x_deg = x.copy().astype(float)
    n = len(x_deg)

    mask = np.zeros(n, dtype=bool)

    if n < 3:
        return x_deg, mask

    # Number of blocks
    if severity == "moderate":
        n_blocks = rng.integers(
            moderate_n_blocks[0],
            moderate_n_blocks[1] + 1
        )
    else:
        n_blocks = rng.integers(
            severe_n_blocks[0],
            severe_n_blocks[1] + 1
        )

    # Block length in number of observations
    min_len, max_len = block_steps_from_duration(freq_label, severity)

    max_len = min(max_len, n)
    min_len = min(min_len, max_len)

    for _ in range(n_blocks):

        block_len = int(rng.integers(min_len, max_len + 1))

        if block_len >= n:
            start = 0
        else:
            start = int(rng.integers(0, n - block_len + 1))

        end = start + block_len

        x_deg[start:end] = np.nan
        mask[start:end] = True

    return x_deg, mask
