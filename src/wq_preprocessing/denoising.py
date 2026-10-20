"""Denoising/correction algorithms."""
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from .gap_filling import linear_fill, pchip_fill, kalman_local_level, ssa_denoise, match_length

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

try:
    import pywt
    HAS_PYWT = True
except Exception:
    HAS_PYWT = False

class MissingDependencyError(RuntimeError):
    pass

def ewma_denoise(x, span=None, **params):
    x = np.asarray(x, dtype=float)
    if span is None:
        span = max(3, min(5, len(x) // 5))
    return pd.Series(x).ewm(span=int(span), adjust=False).mean().to_numpy(dtype=float)

def median_denoise(x, size=3, **params):
    size = int(size)
    if size % 2 == 0:
        size += 1
    return median_filter(np.asarray(x, dtype=float), size=size, mode="nearest")

def odd_int(value, min_value=3):
    value = int(round(value))
    if value % 2 == 0:
        value += 1
    return max(min_value, value)

def savgol_denoise(x, window_length=None, polyorder=2, **params):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return x.copy()
    if window_length is None:
        window_length = min(11, max(5, odd_int(0.07 * n, min_value=5)))
    window_length = odd_int(window_length, min_value=3)
    window_length = min(window_length, n if n % 2 == 1 else n - 1)
    if window_length < 3:
        return x.copy()
    polyorder = int(polyorder)
    polyorder = min(polyorder, window_length - 1)
    return savgol_filter(x, window_length=window_length, polyorder=polyorder)

def loess_denoise(x, frac=0.03, it=0, **params):
    """
    Holdout-calibrated fixed-rule LOESS denoising.

    Calibration-selected setting:
        frac = 0.03
        it   = 0

    frac controls the local smoothing span.
    Smaller frac preserves more local storm-event variability.
    """
    x = np.asarray(x, dtype=float)

    if not HAS_STATSMODELS:
        return savgol_denoise(x)

    n = len(x)
    if n < 3:
        return x.copy()

    t = _t(x)
    valid = np.isfinite(x)

    if valid.sum() < 3:
        return x.copy()

    smoothed = lowess(
        x[valid],
        t[valid],
        frac=float(frac),
        it=int(it),
        return_sorted=False
    )

    if valid.all():
        return np.asarray(smoothed, dtype=float)

    # If input has NaNs, interpolate the smoothed valid values back to full length
    return np.interp(t, t[valid], smoothed)

def wavelet_denoise(x, wavelet="db4", level=None, threshold_multiplier=1.0, **params):
    if not HAS_PYWT:
        raise MissingDependencyError("PyWavelets is required for Wavelet. Install with: pip install PyWavelets")
    x = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(x)):
        x = pchip_fill(x)

    n = len(x)
    max_level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    if max_level < 1:
        return x.copy()
    if level is None:
        level = min(3, max_level)
    level = int(max(1, min(level, max_level)))

    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail)) / 0.6745 if len(detail) else 0
    threshold = float(threshold_multiplier) * sigma * np.sqrt(2 * np.log(n))

    new_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        new_coeffs.append(pywt.threshold(c, threshold, mode="soft"))

    recon = pywt.waverec(new_coeffs, wavelet=wavelet)
    return recon[:n]

def ceemdan_denoise(
    x,
    noise_width=0.05,
    trials=50,
    discard_first_n=1
):
    import numpy as np
    import pandas as pd
    from PyEMD.CEEMDAN import CEEMDAN as CEEMDANClass
    from PyEMD.EMD import EMD as EMDClass

    x = np.asarray(x, dtype=float)
    n = len(x)

    x_filled = (
        pd.Series(x)
        .interpolate(limit_direction="both")
        .bfill()
        .ffill()
        .to_numpy(dtype=float)
    )

    if n < 5 or np.nanstd(x_filled) == 0:
        return x_filled.copy()

    ceemdan = CEEMDANClass(
        trials=int(trials),
        epsilon=float(noise_width),
        ext_EMD=EMDClass()
    )

    components = ceemdan.ceemdan(x_filled)
    components = np.asarray(components, dtype=float)

    if components.ndim == 1:
        return match_length(components, n)

    discard_first_n = int(discard_first_n)

    if discard_first_n <= 0:
        x_hat = np.sum(components, axis=0)
    elif discard_first_n >= components.shape[0]:
        x_hat = x_filled.copy()
    else:
        x_hat = np.sum(components[discard_first_n:, :], axis=0)

    return match_length(x_hat, n)

def vmd_denoise(
    x,
    K=4,
    alpha=2000,
    corr_threshold=0.10
):
    """
    VMD denoising wrapper with length protection.
    """

    import numpy as np
    import pandas as pd

    try:
        from vmdpy import VMD
    except Exception as exc:
        raise MissingDependencyError(
            "VMD requires vmdpy. Install with: pip install vmdpy"
        ) from exc

    x = np.asarray(x, dtype=float)
    n_original = len(x)

    # Fill NaNs before VMD
    x_filled = (
        pd.Series(x)
        .interpolate(limit_direction="both")
        .bfill()
        .ffill()
        .to_numpy(dtype=float)
    )

    if n_original < 5 or np.nanstd(x_filled) == 0:
        return x_filled.copy()

    # VMD can be sensitive to odd signal lengths.
    # If odd, temporarily remove last point and restore length later.
    if len(x_filled) % 2 != 0:
        x_vmd = x_filled[:-1]
    else:
        x_vmd = x_filled.copy()

    alpha = float(alpha)
    tau = 0.0
    K = int(K)
    DC = 0
    init = 1
    tol = 1e-7

    u, u_hat, omega = VMD(
        x_vmd,
        alpha,
        tau,
        K,
        DC,
        init,
        tol
    )

    u = np.asarray(u, dtype=float)

    selected_modes = []

    for i in range(u.shape[0]):
        mode = match_length(u[i, :], len(x_vmd))

        if np.std(mode) == 0 or np.std(x_vmd) == 0:
            corr = 0
        else:
            corr = np.corrcoef(x_vmd, mode)[0, 1]

        if np.isfinite(corr) and abs(corr) >= corr_threshold:
            selected_modes.append(mode)

    if len(selected_modes) == 0:
        x_hat = x_vmd.copy()
    else:
        x_hat = np.sum(selected_modes, axis=0)

    # Restore to original length
    x_hat = match_length(x_hat, n_original)

    return x_hat

DENOISE_ALGORITHMS = {
    "Median_Filter": median_denoise,
    "LOESS": loess_denoise,
    "SSA": ssa_denoise,
    "Savitzky_Golay": savgol_denoise,
    "Kalman_Filter": kalman_local_level,
    "Wavelet": wavelet_denoise,
    "VMD": vmd_denoise,
    "EWMA": ewma_denoise,
    "CEEMDAN": ceemdan_denoise,
}
