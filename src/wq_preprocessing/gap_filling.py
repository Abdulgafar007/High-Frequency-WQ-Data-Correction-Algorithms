"""Gap-filling algorithms."""
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, UnivariateSpline

def match_length(x_hat, target_length):
    """
    Force algorithm output to be 1D and the same length as the reference signal.
    Handles algorithms such as VMD/CEEMDAN that may return modes or shifted lengths.
    """

    import numpy as np

    x_hat = np.asarray(x_hat, dtype=float)

    # If output is 2D, assume it is modes × time and reconstruct by summing modes
    if x_hat.ndim == 2:
        x_hat = np.sum(x_hat, axis=0)

    x_hat = np.ravel(x_hat)

    if len(x_hat) == 0:
        return np.full(target_length, np.nan)

    # Trim if too long
    if len(x_hat) > target_length:
        x_hat = x_hat[:target_length]

    # Pad if too short
    elif len(x_hat) < target_length:
        pad_width = target_length - len(x_hat)
        x_hat = np.pad(x_hat, (0, pad_width), mode="edge")

    return x_hat

def _t(x):
    return np.arange(len(x), dtype=float)

def _valid_xy(x):
    t = _t(x)
    valid = np.isfinite(x)
    return t[valid], x[valid]

def linear_fill(x, **params):
    return pd.Series(x).interpolate("linear", limit_direction="both").to_numpy(dtype=float)

def cubic_spline_fill(x, smoothing=0.0, **params):
    x = np.asarray(x, dtype=float)
    t = _t(x)
    tv, xv = _valid_xy(x)
    if len(xv) < 4:
        return linear_fill(x)
    try:
        return UnivariateSpline(tv, xv, k=3, s=smoothing)(t).astype(float)
    except Exception:
        return linear_fill(x)

def pchip_fill(x, **params):
    x = np.asarray(x, dtype=float)
    t = _t(x)
    tv, xv = _valid_xy(x)
    if len(xv) < 2:
        return linear_fill(x)
    try:
        return PchipInterpolator(tv, xv, extrapolate=True)(t).astype(float)
    except Exception:
        return linear_fill(x)

def kalman_local_level(x, q_over_r=0.1, obs_var=None, **params):
    """
    Local-level Kalman smoother.
    Handles missing observations. q_over_r controls responsiveness.
    """
    y = np.asarray(x, dtype=float)
    n = len(y)
    valid = np.isfinite(y)
    if valid.sum() == 0:
        return y.copy()
    if valid.sum() == 1:
        return np.full(n, y[valid][0], dtype=float)

    diffs = np.diff(y[valid])
    if obs_var is None:
        obs_var = np.nanvar(diffs) if len(diffs) > 1 else np.nanvar(y[valid])
    if not np.isfinite(obs_var) or obs_var <= 0:
        obs_var = max(np.nanvar(y[valid]), 1e-6)

    R = obs_var
    Q = max(float(q_over_r) * R, 1e-12)

    x_f = np.zeros(n)
    P_f = np.zeros(n)
    x_pred = np.zeros(n)
    P_pred = np.zeros(n)

    first = y[valid][0]
    x_prev = first
    P_prev = R * 10

    for i in range(n):
        # predict
        xp = x_prev
        Pp = P_prev + Q
        x_pred[i] = xp
        P_pred[i] = Pp

        if valid[i]:
            K = Pp / (Pp + R)
            x_curr = xp + K * (y[i] - xp)
            P_curr = (1 - K) * Pp
        else:
            x_curr = xp
            P_curr = Pp

        x_f[i] = x_curr
        P_f[i] = P_curr
        x_prev, P_prev = x_curr, P_curr

    # RTS smoother
    x_s = x_f.copy()
    P_s = P_f.copy()
    for i in range(n - 2, -1, -1):
        C = P_f[i] / P_pred[i + 1] if P_pred[i + 1] > 0 else 0
        x_s[i] = x_f[i] + C * (x_s[i + 1] - x_pred[i + 1])
        P_s[i] = P_f[i] + C * C * (P_s[i + 1] - P_pred[i + 1])

    return x_s

def get_ssa_window_length(n, window_rule="n_20"):
    """
    Convert calibrated SSA window rule to an integer window length.

    Selected calibration rule:
        window_rule = "n_20"  ->  L = n // 20

    The window is constrained to remain valid for SSA.
    """
    n = int(n)

    if n < 6:
        return max(2, n - 1)

    if window_rule == "n_20":
        L = max(4, n // 20)
    elif window_rule == "n_10":
        L = max(4, n // 10)
    elif window_rule == "n_8":
        L = max(4, n // 8)
    elif window_rule == "n_5":
        L = max(4, n // 5)
    elif window_rule == "fixed_60":
        L = 60
    else:
        raise ValueError(f"Unknown SSA window_rule: {window_rule}")

    # SSA requires 2 <= L <= n//2 in your implementation
    L = int(max(2, min(L, n // 2)))

    return L

def _ssa_reconstruct(x, window_length=None, n_components=4, energy_threshold=None):
    """
    SSA reconstruction using fixed retained components.

    Calibration-selected setting:
        n_components = 4

    energy_threshold is kept only for backward compatibility,
    but the calibrated fixed rule uses n_components=4.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    if n < 6:
        return x.copy()

    if window_length is None:
        window_length = get_ssa_window_length(n, window_rule="n_20")

    L = int(max(2, min(window_length, n // 2)))
    K = n - L + 1

    if K < 2:
        return x.copy()

    # Build trajectory matrix
    X = np.column_stack([x[i:i + L] for i in range(K)])

    # SVD
    U, s, VT = np.linalg.svd(X, full_matrices=False)

    # Holdout-calibrated retained components
    if n_components is None:
        # Fallback only if explicitly requested
        if energy_threshold is None:
            energy_threshold = 0.90
        energy = np.cumsum(s**2) / np.sum(s**2)
        r = int(np.searchsorted(energy, energy_threshold) + 1)
        r = min(r, 5, len(s))
    else:
        r = min(max(1, int(n_components)), len(s))

    # Reconstruct selected SSA components
    Xr = (U[:, :r] * s[:r]) @ VT[:r, :]

    # Diagonal averaging
    recon = np.zeros(n)
    counts = np.zeros(n)

    for i in range(L):
        for j in range(K):
            recon[i + j] += Xr[i, j]
            counts[i + j] += 1

    return recon / counts

def ssa_denoise(x, window_rule="n_20", n_components=4, **params):
    """
    Holdout-calibrated fixed-rule SSA denoising.

    Calibration-selected setting:
        window_rule  = "n_20"
        n_components = 4
    """
    x = np.asarray(x, dtype=float)

    # Fill NaNs if present
    x0 = linear_fill(x) if np.any(~np.isfinite(x)) else x.copy()

    n = len(x0)
    if n < 6:
        return x0.copy()

    L = get_ssa_window_length(n, window_rule=window_rule)

    return _ssa_reconstruct(
        x0,
        window_length=L,
        n_components=n_components
    )

def ssa_gap_fill(
    x,
    window_rule="n_20",
    n_components=4,
    max_iter=50,
    tol=1e-4,
    **params
):
    """
    Holdout-calibrated fixed-rule SSA gap filling.

    Uses PCHIP as the initial fill, then iteratively applies
    calibrated SSA reconstruction only to the missing locations.
    """
    x = np.asarray(x, dtype=float)
    missing = ~np.isfinite(x)

    if not np.any(missing):
        return ssa_denoise(
            x,
            window_rule=window_rule,
            n_components=n_components
        )

    current = pchip_fill(x)

    for _ in range(int(max_iter)):
        previous = current.copy()

        recon = ssa_denoise(
            current,
            window_rule=window_rule,
            n_components=n_components
        )

        current[missing] = recon[missing]

        denom = np.linalg.norm(previous)
        rel_change = (
            np.linalg.norm(current - previous) / denom
            if denom > 0 else np.inf
        )

        if rel_change < tol:
            break

    return current

def local_moving_mean_gapfill(

    x,

    n_each_side=2
):

    x = np.asarray(
        x,
        dtype=float
    )


    y = x.copy()


    observed_idx = np.flatnonzero(

        np.isfinite(
            x
        )
    )


    missing_idx = np.flatnonzero(

        ~np.isfinite(
            x
        )
    )


    if len(
        observed_idx
    ) == 0:

        return y


    for idx in missing_idx:


        left = observed_idx[

            observed_idx
            <
            idx

        ][
            -n_each_side:
        ]


        right = observed_idx[

            observed_idx
            >
            idx

        ][
            :n_each_side
        ]


        neighbors = np.concatenate(
            [
                left,
                right
            ]
        )


        if len(
            neighbors
        ) > 0:


            y[idx] = np.mean(

                x[
                    neighbors
                ]
            )


        else:


            y[idx] = np.nanmean(
                x
            )


    return y

def delta_h_weighted_gapfill(

    x,

    stage,

    linear_function
):

    x = np.asarray(
        x,
        dtype=float
    )


    stage = np.asarray(
        stage,
        dtype=float
    )


    if len(x) != len(stage):

        raise ValueError(

            "WQ and stage vectors "
            "have different lengths."
        )


    # Linear interpolation baseline/fallback
    y = np.asarray(

        match_length(

            linear_function(
                x
            ),

            len(x)
        ),

        dtype=float
    )


    observed_mask = np.isfinite(
        x
    )


    y[
        observed_mask
    ] = x[
        observed_mask
    ]


    missing_idx = np.flatnonzero(

        ~observed_mask
    )


    if len(
        missing_idx
    ) == 0:

        return y


    split_points = (

        np.where(

            np.diff(
                missing_idx
            )
            >
            1

        )[0]

        + 1
    )


    gap_groups = np.split(

        missing_idx,

        split_points
    )


    finite_stage = stage[

        np.isfinite(
            stage
        )
    ]


    stage_scale = (

        np.nanstd(
            finite_stage
        )

        if len(
            finite_stage
        ) > 1

        else 1.0
    )


    eps = max(

        1e-12,

        1e-8
        *
        max(
            stage_scale,
            1.0
        )
    )


    for gap in gap_groups:


        if len(
            gap
        ) == 0:

            continue


        left = (
            gap[0]
            - 1
        )


        right = (
            gap[-1]
            + 1
        )


        if (

            left < 0

            or

            right >= len(x)

        ):

            continue


        if (

            not np.isfinite(
                x[left]
            )

            or

            not np.isfinite(
                x[right]
            )

        ):

            continue


        H_left = stage[
            left
        ]


        H_right = stage[
            right
        ]


        if (

            not np.isfinite(
                H_left
            )

            or

            not np.isfinite(
                H_right
            )

        ):

            continue


        delta_H = (

            H_right
            -
            H_left
        )


        if abs(
            delta_H
        ) <= eps:

            continue


        H_gap = stage[
            gap
        ]


        alpha = (

            H_gap
            -
            H_left

        ) / delta_H


        usable = (

            np.isfinite(
                alpha
            )

            &

            (
                alpha >= 0
            )

            &

            (
                alpha <= 1
            )
        )


        if not np.any(
            usable
        ):

            continue


        usable_idx = gap[
            usable
        ]


        usable_alpha = alpha[
            usable
        ]


        y[
            usable_idx
        ] = (

            x[left]

            +

            usable_alpha

            *
            (
                x[right]
                -
                x[left]
            )
        )


    return y

def hpwi_gapfill(

    x,

    stage,

    linear_function
):

    x = np.asarray(
        x,
        dtype=float
    )


    stage = np.asarray(
        stage,
        dtype=float
    )


    if len(x) != len(stage):

        raise ValueError(

            "WQ and stage vectors "
            "have different lengths."
        )


    y = np.asarray(

        match_length(

            linear_function(
                x
            ),

            len(x)
        ),

        dtype=float
    )


    observed_mask = np.isfinite(
        x
    )


    y[
        observed_mask
    ] = x[
        observed_mask
    ]


    missing_idx = np.flatnonzero(

        ~observed_mask
    )


    if len(
        missing_idx
    ) == 0:

        return y


    split_points = (

        np.where(

            np.diff(
                missing_idx
            )
            >
            1

        )[0]

        + 1
    )


    gap_groups = np.split(

        missing_idx,

        split_points
    )


    finite_stage = stage[

        np.isfinite(
            stage
        )
    ]


    stage_scale = (

        np.nanstd(
            finite_stage
        )

        if len(
            finite_stage
        ) > 1

        else 1.0
    )


    eps = max(

        1e-12,

        1e-8
        *
        max(
            stage_scale,
            1.0
        )
    )


    for gap in gap_groups:


        if len(
            gap
        ) == 0:

            continue


        left = (
            gap[0]
            - 1
        )


        right = (
            gap[-1]
            + 1
        )


        if (

            left < 0

            or

            right >= len(x)

        ):

            continue


        if (

            not np.isfinite(
                x[left]
            )

            or

            not np.isfinite(
                x[right]
            )

        ):

            continue


        H_segment = stage[

            left:
            right + 1
        ]


        if not np.all(

            np.isfinite(
                H_segment
            )
        ):

            continue


        # Absolute hydrograph movement
        dH = np.abs(

            np.diff(
                H_segment
            )
        )


        total_path = np.sum(
            dH
        )


        if (

            not np.isfinite(
                total_path
            )

            or

            total_path <= eps

        ):

            continue


        cumulative_path = np.cumsum(
            dH
        )


        local_positions = (

            gap
            -
            left
            -
            1
        )


        D_to_missing = cumulative_path[

            local_positions
        ]


        alpha = (

            D_to_missing

            /

            total_path
        )


        alpha = np.clip(

            alpha,

            0.0,

            1.0
        )


        y[
            gap
        ] = (

            x[left]

            +

            alpha

            *
            (
                x[right]
                -
                x[left]
            )
        )


    return y

BASE_GAP_ALGORITHMS = {
    "Linear_Interpolation": linear_fill,
    "PCHIP": pchip_fill,
    "Cubic_Spline": cubic_spline_fill,
    "Kalman_Filter": kalman_local_level,
    "SSA_Gap_Filling": ssa_gap_fill,
}

def expanded_gap_algorithms(stage):
    stage = np.asarray(stage, dtype=float)
    pool = dict(BASE_GAP_ALGORITHMS)
    pool["Local_Moving_Mean"] = lambda x: local_moving_mean_gapfill(x, n_each_side=2)
    pool["Delta_H_Weighted_Interpolation"] = (
        lambda x: delta_h_weighted_gapfill(x, stage, linear_fill)
    )
    pool["HPWI"] = lambda x: hpwi_gapfill(x, stage, linear_fill)
    return pool
