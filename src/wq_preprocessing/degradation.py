"""Controlled degradation scenarios used in the manuscript workflow.

Important: the field-realistic composite fault here is the five-component version:
gain, offset, first-order lag, gradual drift, and storm-dependent heteroskedastic noise.
The later exploratory spike/saturation variant from the development notebook is NOT used.
"""
import numpy as np

def apply_random_missing(x: np.ndarray, severity: str, rng: np.random.Generator):
    assert severity in ("moderate", "severe"), f"Unknown severity: {severity!r}"
    x_deg = x.copy().astype(float)
    n = len(x_deg)
    missing_rate = 0.25 if severity == "moderate" else 0.50
    n_missing = max(1, int(round(missing_rate * n)))
    idx = rng.choice(n, size=n_missing, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    x_deg[mask] = np.nan
    return x_deg, mask

def apply_block_missing(x: np.ndarray, severity: str, rng: np.random.Generator):
    assert severity in ("moderate", "severe"), f"Unknown severity: {severity!r}"
    x_deg = x.copy().astype(float)
    n = len(x_deg)
    mask = np.zeros(n, dtype=bool)
    if severity == "moderate":
        n_blocks = rng.integers(1, 5)      
        len_low, len_high = 4, 9             
    else:
        n_blocks = rng.integers(5, 11)       
        len_low, len_high = 9, 16          
    for _ in range(n_blocks):
        block_length = int(rng.integers(len_low, len_high))
        block_length = min(block_length, n)
        max_start = n - block_length
        if max_start < 0:
            continue
        start = int(rng.integers(0, max_start + 1))
        mask[start:start + block_length] = True
    x_deg[mask] = np.nan
    return x_deg, mask

def apply_additive_noise(
    x: np.ndarray,
    severity: str,
    rng: np.random.Generator,
    noise_fraction: float = 0.30,
    spike_fraction: float = 0.03
):
    """
    Additive noise scenario with:
    1. Gaussian perturbation on a fraction of points
    2. Random spikes/dips on a smaller fraction of points
    """

    assert severity in ("moderate", "severe"), f"Unknown severity: {severity!r}"

    x_deg = x.copy().astype(float)
    n = len(x_deg)
    sigma_x = np.nanstd(x_deg)

    if not np.isfinite(sigma_x) or sigma_x == 0:
        raise ValueError("Signal is constant or invalid. Noise amplitude would be zero.")

    if n == 0:
        raise ValueError("Empty signal.")

    # -------------------------------
    # Noise severity settings
    # -------------------------------
    if severity == "moderate":
        noise_factor = 0.10
        spike_low = 1.5
        spike_high = 2.0
    else:
        noise_factor = 0.20
        spike_low = 1.5
        spike_high = 3.5

    # -------------------------------
    # Gaussian corruption
    # -------------------------------
    n_corrupted = max(1, int(round(noise_fraction * n)))
    n_corrupted = min(n_corrupted, n)

    corrupted_idx = rng.choice(n, size=n_corrupted, replace=False)

    mask = np.zeros(n, dtype=bool)
    mask[corrupted_idx] = True

    noise = rng.normal(
        loc=0.0,
        scale=noise_factor * sigma_x,
        size=n_corrupted
    )

    x_deg[corrupted_idx] += noise

    # -------------------------------
    # Random spikes/dips
    # -------------------------------
    n_spikes = max(1, int(round(spike_fraction * n)))
    n_spikes = min(n_spikes, n)

    spike_idx = rng.choice(n, size=n_spikes, replace=False)

    spike_sign = rng.choice([-1.0, 1.0], size=n_spikes)

    spike_magnitude = spike_sign * rng.uniform(
        spike_low * sigma_x,
        spike_high * sigma_x,
        size=n_spikes
    )

    x_deg[spike_idx] += spike_magnitude

    mask[spike_idx] = True

    return x_deg, mask

def first_order_lag(x: np.ndarray, lag_strength: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + lag_strength * (x[i] - y[i - 1])
    return y

def apply_field_realistic_fault(x: np.ndarray, severity: str, rng: np.random.Generator):
    assert severity in ("moderate", "severe"), f"Unknown severity: {severity!r}"
    x = x.copy().astype(float)
    n = len(x)
    t = np.linspace(0, 1, n)
    sigma_x = np.nanstd(x)
    range_x = np.nanmax(x) - np.nanmin(x)
    if range_x == 0:
        if sigma_x > 0:
            range_x = sigma_x
        else:
            raise ValueError("Signal is constant (range = 0, std = 0). Degradation scaling is undefined.")
    if severity == "moderate":
        alpha = rng.uniform(0.98, 1.02)
        offset = rng.uniform(-0.02, 0.02) * range_x
        lag_strength = rng.uniform(0.50, 0.80)
        drift_total = rng.uniform(0.02, 0.05) * range_x
        storm_noise_factor = 0.05
    else:
        alpha = rng.uniform(0.95, 1.05)
        offset = rng.uniform(-0.05, 0.05) * range_x
        lag_strength = rng.uniform(0.20, 0.50)
        drift_total = rng.uniform(0.05, 0.10) * range_x
        storm_noise_factor = 0.10
    x_lag = first_order_lag(x, lag_strength)
    k = rng.uniform(2.0, 6.0)
    drift = drift_total * (1 - np.exp(-k * t))
    diff = np.abs(np.diff(x))
    if len(diff) == 0 or not np.any(np.isfinite(diff)):
        storm_activity = np.zeros(n)
    else:
        diff_padded = np.concatenate([[diff[0]], diff])
        diff_padded = (diff_padded + np.concatenate([diff, [diff[-1]]])) / 2

        max_val = np.nanmax(diff_padded)

        if np.isfinite(max_val) and max_val > 0:
            storm_activity = diff_padded / max_val
        else:
            storm_activity = np.zeros_like(diff_padded)
    sigma_t = storm_noise_factor * sigma_x * (1 + 2 * storm_activity)
    storm_noise = rng.normal(0.0, sigma_t, size=n)
    x_deg = alpha * x_lag + offset + drift + storm_noise
    mask = np.ones(n, dtype=bool)
    return x_deg, mask

DEGRADATION_FUNCTIONS = {
    "random_missing": apply_random_missing,
    "block_missing": apply_block_missing,
    "additive_noise": apply_additive_noise,
    "field_realistic_fault": apply_field_realistic_fault,
}
