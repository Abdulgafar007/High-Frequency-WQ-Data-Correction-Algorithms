"""Hydrologic-phase classification used in the manuscript."""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

BASELINE_QUANTILE = 0.20
ONSET_FRACTION = 0.04
END_FRACTION = 0.025
RETURN_HOURS = 3.0
MERGE_GAP_HOURS = 2.0
MIN_EPISODE_HOURS = 1.0
MIN_PEAK_DISTANCE_HOURS = 0.75
PEAK_PROMINENCE_FRACTION = 0.15
PEAK_WINDOW_FRACTION = 0.75
SMOOTH_HOURS = 0.75

def estimate_dt_minutes(index):
    if isinstance(index, pd.DatetimeIndex) and len(index) > 2:
        dt = np.diff(index.asi8) / 60e9
        dt = dt[np.isfinite(dt) & (dt > 0)]

        if len(dt):
            return float(np.median(dt))

    # Your records are approximately 11–12 min
    return 11.5

def fill_short_false_runs(mask, max_gap):
    mask = np.asarray(mask, dtype=bool).copy()
    n = len(mask)
    i = 0

    while i < n:
        if mask[i]:
            i += 1
            continue

        start = i

        while i < n and not mask[i]:
            i += 1

        end = i
        run_length = end - start

        left_active = start > 0 and mask[start - 1]
        right_active = end < n and mask[end]

        if left_active and right_active and run_length <= max_gap:
            mask[start:end] = True

    return mask

def remove_short_true_runs(mask, min_length):
    mask = np.asarray(mask, dtype=bool).copy()
    n = len(mask)
    i = 0

    while i < n:
        if not mask[i]:
            i += 1
            continue

        start = i

        while i < n and mask[i]:
            i += 1

        end = i

        if (end - start) < min_length:
            mask[start:end] = False

    return mask

def label_episodes(mask):
    episode_id = np.zeros(len(mask), dtype=int)
    episode = 0
    i = 0

    while i < len(mask):
        if not mask[i]:
            i += 1
            continue

        episode += 1
        start = i

        while i < len(mask) and mask[i]:
            i += 1

        episode_id[start:i] = episode

    return episode_id

def classify_storm_phases(storm_df, radar_column="Radar"):

    radar = pd.to_numeric(storm_df[radar_column], errors="coerce").copy()

    # Same radar cleaning logic used previously
    radar[(np.isclose(radar, 0, atol=1.0)) | (radar > 5)] = np.nan

    dt_min = estimate_dt_minutes(storm_df.index)

    # Interpolate only very short radar gaps for classification
    radar_work = radar.interpolate(limit=2, limit_direction="both")

    # --------------------------------------------------------
    # Invert:
    # lower raw radar reading = higher water level
    # --------------------------------------------------------

    inverted = radar_work.max() - radar_work

    # Robust normalization
    valid = inverted.dropna()

    q02 = valid.quantile(0.02)
    q98 = valid.quantile(0.98)

    if not np.isfinite(q98 - q02) or (q98 - q02) <= 0:
        q02 = valid.min()
        q98 = valid.max()

    stage = ((inverted - q02) / (q98 - q02)).clip(0, 1)

    # --------------------------------------------------------
    # Smooth stage only for phase detection
    # --------------------------------------------------------

    smooth_points = max(3, int(round(SMOOTH_HOURS * 60 / dt_min)))

    if smooth_points % 2 == 0:
        smooth_points += 1

    smooth = stage.rolling(
        smooth_points,
        center=True,
        min_periods=1
    ).median()

    # --------------------------------------------------------
    # Estimate baseline and active-response threshold
    # --------------------------------------------------------

# --------------------------------------------------------
# Estimate baseline and response range
# --------------------------------------------------------

    baseline = smooth.quantile(BASELINE_QUANTILE)
    upper = smooth.quantile(0.95)

    response_range = upper - baseline

    # Separate thresholds for event onset and recession/end
    onset_threshold = baseline + ONSET_FRACTION * response_range
    end_threshold = baseline + END_FRACTION * response_range

    # Number of consecutive observations required near baseline
    # before an active event is considered finished
    return_points = max(1, int(round(RETURN_HOURS * 60 / dt_min)))

    smooth_values = smooth.to_numpy(dtype=float)
    active = np.zeros(len(smooth_values), dtype=bool)

    in_event = False
    below_end_count = 0

    for i, value in enumerate(smooth_values):

        if not np.isfinite(value):
            # If stage is missing, retain the current state rather than
            # abruptly terminating an event.
            if in_event:
                active[i] = True
            continue

        # ----------------------------------------------------
        # EVENT HAS NOT STARTED
        # ----------------------------------------------------
        if not in_event:

            if value >= onset_threshold:
                in_event = True
                below_end_count = 0
                active[i] = True

        # ----------------------------------------------------
        # EVENT IS ACTIVE
        # ----------------------------------------------------
        else:

            active[i] = True

            if value <= end_threshold:
                below_end_count += 1
            else:
                below_end_count = 0

            # End only after a sustained return toward baseline
            if below_end_count >= return_points:

                end_start = i - return_points + 1

                # The near-baseline observations used to establish
                # the return are classified back as baseline.
                active[end_start:i + 1] = False

                in_event = False
                below_end_count = 0

    # --------------------------------------------------------
    # Convert time settings to number of observations
    # --------------------------------------------------------

    merge_gap_points = max(1, int(round(MERGE_GAP_HOURS * 60 / dt_min)))
    min_episode_points = max(2, int(round(MIN_EPISODE_HOURS * 60 / dt_min)))
    min_peak_distance = max(1, int(round(MIN_PEAK_DISTANCE_HOURS * 60 / dt_min)))

    # --------------------------------------------------------
    # Prevent inter-peak valleys from becoming baseline
    # --------------------------------------------------------

    active = fill_short_false_runs(active, merge_gap_points)
    active = remove_short_true_runs(active, min_episode_points)

    episode_id = label_episodes(active)

    # --------------------------------------------------------
    # Initial three-class assignment
    # --------------------------------------------------------

    phase = np.full(len(stage), "Baseline", dtype=object)
    phase[active] = "Storm limb"

    peak_mask = np.zeros(len(stage), dtype=bool)
    peak_indices = []

    # ========================================================
    # DETECT MULTIPLE PEAKS WITHIN EACH ACTIVE EPISODE
    # ========================================================

    for episode in range(1, episode_id.max() + 1):

        idx = np.where(episode_id == episode)[0]

        if len(idx) < 3:
            continue

        segment = smooth.iloc[idx].to_numpy()

        episode_baseline = max(baseline, np.nanmin(segment))
        episode_peak = np.nanmax(segment)
        episode_amplitude = episode_peak - episode_baseline

        if not np.isfinite(episode_amplitude) or episode_amplitude <= 0:
            continue

        prominence = PEAK_PROMINENCE_FRACTION * episode_amplitude

        local_peaks, properties = find_peaks(
            segment,
            prominence=prominence,
            distance=min_peak_distance
        )

        # If no prominent peak is found, retain the main episode maximum
        if len(local_peaks) == 0:
            local_peaks = np.array([int(np.nanargmax(segment))])

        for local_peak in local_peaks:

            global_peak = idx[local_peak]
            peak_indices.append(global_peak)

            peak_value = smooth.iloc[global_peak]

            # ------------------------------------------------
            # Dynamic peak window
            #
            # Rather than ± fixed observations, identify the
            # upper portion of each individual peak.
            # ------------------------------------------------

            peak_threshold = (
                episode_baseline
                + PEAK_WINDOW_FRACTION
                * (peak_value - episode_baseline)
            )

            left = local_peak
            right = local_peak

            while left > 0 and segment[left - 1] >= peak_threshold:
                left -= 1

            while right < len(segment) - 1 and segment[right + 1] >= peak_threshold:
                right += 1

            peak_global_idx = idx[left:right + 1]
            peak_mask[peak_global_idx] = True

    phase[peak_mask] = "Peak window"

    # ========================================================
    # RETURN CLASSIFICATION
    # ========================================================

    result = pd.DataFrame({
        "Radar_raw": radar,
        "Stage_normalized": stage,
        "Stage_smoothed": smooth,
        "Active_response": active,
        "Episode_ID": episode_id,
        "Phase": phase
    }, index=storm_df.index)

    info = {
        "dt_minutes": dt_min,
        "baseline": baseline,
        "onset_threshold": onset_threshold,
        "end_threshold": end_threshold,
        "n_episodes": int(episode_id.max()),
        "n_peaks": len(peak_indices),
        "peak_indices": peak_indices
    }

    return result, info

def classify_all_storms(all_storms):
    out = {}
    for storm_id, item in all_storms.items():
        classified, info = classify_storm_phases(item["data"], radar_column="Radar")
        out[storm_id] = {"data": classified, "info": info}
    return out
