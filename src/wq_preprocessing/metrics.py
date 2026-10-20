"""Performance metrics and evaluation domains."""
import numpy as np
import pandas as pd

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def nse(y_true, y_pred):
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return float(1 - np.sum((y_true - y_pred) ** 2) / denom)

def pbias(y_true, y_pred):
    denom = np.sum(y_true)
    if denom == 0:
        return np.nan
    return float(100 * np.sum(y_pred - y_true) / denom)

def pearson_r(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def kge(y_true, y_pred):
    r = pearson_r(y_true, y_pred)
    if np.isnan(r):
        return np.nan
    alpha = np.std(y_pred) / np.std(y_true) if np.std(y_true) != 0 else np.nan
    beta = np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) != 0 else np.nan
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))

def peak_magnitude_error(y_true, y_pred):
    true_peak = np.nanmax(y_true)
    pred_peak = np.nanmax(y_pred)
    if true_peak == 0:
        return np.nan
    return float(100 * (pred_peak - true_peak) / true_peak)

def time_to_peak_error(y_true, y_pred):
    return int(np.nanargmax(y_pred) - np.nanargmax(y_true))

def variance_ratio(y_true, y_pred):
    denom = np.nanvar(y_true)
    if denom == 0:
        return np.nan
    return float(np.nanvar(y_pred) / denom)

def shannon_entropy(x, bin_rule="sqrt", min_bins=5, max_bins=100):
    """
    Compute histogram-based Shannon entropy with bin number
    determined relative to signal length.

    Parameters
    ----------
    x : array-like
        Input signal.
    bin_rule : str
        "sqrt"     -> ceil(sqrt(n))
        "sturges"  -> ceil(log2(n) + 1)
        "rice"     -> ceil(2 * n^(1/3))
    min_bins : int
        Minimum allowed number of bins.
    max_bins : int
        Maximum allowed number of bins.

    Returns
    -------
    float
        Shannon entropy.
    """

    import numpy as np

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)

    if n == 0:
        return np.nan

    if np.nanstd(x) == 0:
        return 0.0
        
    bins = int(np.ceil(np.sqrt(n)))
    bins = max(min_bins, min(bins, max_bins))

    bins = max(min_bins, min(bins, max_bins))

    hist, _ = np.histogram(x, bins=bins, density=False)

    p = hist / np.sum(hist)
    p = p[p > 0]

    if len(p) == 0:
        return np.nan

    return -np.sum(p * np.log(p))

def sample_entropy(x, m=2, r=None):
    """
    Compute sample entropy.

    m = embedding dimension
    r = tolerance, commonly 0.2 * std(x)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)

    if n <= m + 2:
        return np.nan

    std_x = np.std(x)

    if r is None:
        r = 0.2 * std_x

    if not np.isfinite(r) or r == 0:
        return np.nan

    def _phi(order):
        count = 0
        total = 0

        for i in range(n - order):
            template = x[i:i + order]

            for j in range(i + 1, n - order + 1):
                compare = x[j:j + order]

                if np.max(np.abs(template - compare)) <= r:
                    count += 1

                total += 1

        if total == 0:
            return np.nan

        return count / total

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    if phi_m is None or phi_m1 is None:
        return np.nan

    if phi_m <= 0 or phi_m1 <= 0:
        return np.nan

    return -np.log(phi_m1 / phi_m)

def usgs_style_tolerance(parameter, reference_values):
    """
    Parameter-specific operational tolerance inspired by USGS shift criteria.

    This is used as a practical benchmark, not as a formal USGS compliance test.
    """
    ref = np.asarray(reference_values, dtype=float)

    parameter = str(parameter).lower()

    if parameter in ["ec", "specific conductance", "specific_conductance", "conductivity"]:
        # Greater of ±5 µS/cm or ±3% of measured value
        return np.maximum(5.0, 0.03 * np.abs(ref))

    elif parameter in ["do", "dissolved oxygen", "dissolved_oxygen"]:
        # ±0.3 mg/L
        return np.full_like(ref, 0.3, dtype=float)

    elif parameter in ["turbidity", "ntu"]:
        # Greater of ±2 NTU or ±5% of measured value
        return np.maximum(2.0, 0.05 * np.abs(ref))

    elif parameter in ["temp", "temperature", "water temperature"]:
        # Confirm your exact USGS threshold before final manuscript.
        # If you do not want to use temperature tolerance yet, return NaN instead.
        return np.full_like(ref, 0.2, dtype=float)

    else:
        return np.full_like(ref, np.nan, dtype=float)

def usgs_accuracy_rating(parameter, y_true, y_pred):
    """
    Classify pointwise absolute errors using USGS-style
    continuous water-quality accuracy ratings.

    Returns rating labels:
    Excellent, Good, Fair, Poor
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    abs_error = np.abs(y_pred - y_true)
    p = str(parameter).strip().lower()

    ratings = np.full(len(y_true), "Poor", dtype=object)
    
    # Prevent division by zero for percentage-based criteria
    eps = 1e-12
    safe_ref = np.maximum(np.abs(y_true), eps)
    

    if p in ["temp", "temperature", "water temperature"]:
        ratings[abs_error <= 0.2] = "Excellent"
        ratings[(abs_error > 0.2) & (abs_error <= 0.5)] = "Good"
        ratings[(abs_error > 0.5) & (abs_error <= 0.8)] = "Fair"
        ratings[abs_error > 0.8] = "Poor"

    elif p in ["ec", "specific conductance", "specific_conductance", "conductivity"]:
        percent_error = 100 * abs_error / np.abs(y_true)

        ratings[percent_error <= 3] = "Excellent"
        ratings[(percent_error > 3) & (percent_error <= 10)] = "Good"
        ratings[(percent_error > 10) & (percent_error <= 15)] = "Fair"
        ratings[percent_error > 15] = "Poor"

    elif p in ["do", "dissolved oxygen", "dissolved_oxygen"]:
        excellent_threshold = np.maximum(0.3, 0.05 * safe_ref)
        good_threshold = np.maximum(0.5, 0.10 * safe_ref)
        fair_threshold = np.maximum(0.8, 0.15 * safe_ref)

        ratings[abs_error <= excellent_threshold] = "Excellent"
        ratings[(abs_error > excellent_threshold) & (abs_error <= good_threshold)] = "Good"
        ratings[(abs_error > good_threshold) & (abs_error <= fair_threshold)] = "Fair"
        ratings[abs_error > fair_threshold] = "Poor"

    elif p in ["turbidity", "ntu"]:
        percent_error = 100 * abs_error / np.abs(y_true)

        excellent_threshold = np.maximum(0.5, 0.05 * np.abs(y_true))
        good_threshold = np.maximum(1.0, 0.10 * np.abs(y_true))
        fair_threshold = np.maximum(1.5, 0.15 * np.abs(y_true))

        ratings[abs_error <= excellent_threshold] = "Excellent"
        ratings[(abs_error > excellent_threshold) & (abs_error <= good_threshold)] = "Good"
        ratings[(abs_error > good_threshold) & (abs_error <= fair_threshold)] = "Fair"
        ratings[abs_error > fair_threshold] = "Poor"

    else:
        ratings[:] = None

    return ratings

def evaluation_slices(
    scenario: str,
    x_ref: np.ndarray,
    x_hat: np.ndarray,
    corruption_mask: np.ndarray
):
    """
    Return evaluation domains.

    For missing-data scenarios:
        - evaluate full signal
        - evaluate only artificially corrupted/missing locations

    For denoising scenarios:
        - evaluate full signal
    """

    x_ref = np.asarray(x_ref, dtype=float)
    x_hat = np.asarray(x_hat, dtype=float)

    valid = np.isfinite(x_ref) & np.isfinite(x_hat)

    yield "full", x_ref[valid], x_hat[valid]

    if scenario in ["random_missing", "block_missing"]:
        mask = corruption_mask.astype(bool)
        valid_missing = valid & mask

        if np.sum(valid_missing) > 0:
            yield "corrupted_only", x_ref[valid_missing], x_hat[valid_missing]

def compute_pointwise_error_metrics(y_true, y_pred, parameter):
    """
    Computes pointwise residual and tolerance-based metrics for one algorithm run.

    y_true: reference signal
    y_pred: corrected/reconstructed signal
    parameter: water-quality parameter name
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "n_eval_points": 0,
            "Bias": np.nan,
            "MAE_SD": np.nan,
            "Median_AE": np.nan,
            "P90_AE": np.nan,
            "P95_AE": np.nan,
            "Max_AE": np.nan,
            "USGS_Tolerance_Pass_Rate": np.nan,
            "USGS_Tolerance_Fail_Rate": np.nan,
            "Mean_Normalized_AE": np.nan,
            "Median_Normalized_AE": np.nan,
            "P95_Normalized_AE": np.nan,
            "Peak_Timing_Error": np.nan,
            "Peak_Magnitude_Error": np.nan,
            "Abs_Peak_Magnitude_Error": np.nan
        }

    residual = y_pred - y_true
    abs_error = np.abs(residual)

    tolerance = usgs_style_tolerance(parameter, y_true)
    valid_tol = np.isfinite(tolerance) & (tolerance > 0)

    if valid_tol.any():
        normalized_ae = abs_error[valid_tol] / tolerance[valid_tol]
        within_tol = abs_error[valid_tol] <= tolerance[valid_tol]

        pass_rate = 100 * np.mean(within_tol)
        fail_rate = 100 - pass_rate
        mean_norm_ae = np.mean(normalized_ae)
        median_norm_ae = np.median(normalized_ae)
        p95_norm_ae = np.percentile(normalized_ae, 95)
    else:
        pass_rate = np.nan
        fail_rate = np.nan
        mean_norm_ae = np.nan
        median_norm_ae = np.nan
        p95_norm_ae = np.nan

    # Peak timing and magnitude
    true_peak_idx = int(np.nanargmax(y_true))
    pred_peak_idx = int(np.nanargmax(y_pred))

    peak_timing_error = pred_peak_idx - true_peak_idx
    peak_magnitude_error = np.nanmax(y_pred) - np.nanmax(y_true)

    return {
        "n_eval_points": len(y_true),

        # Signed and absolute residual summaries
        "Bias": np.mean(residual),
        "MAE_SD": np.std(abs_error, ddof=1) if len(abs_error) > 1 else 0.0,
        "Median_AE": np.median(abs_error),
        "P90_AE": np.percentile(abs_error, 90),
        "P95_AE": np.percentile(abs_error, 95),
        "Max_AE": np.max(abs_error),

        # Practical tolerance metrics
        "USGS_Tolerance_Pass_Rate": pass_rate,
        "USGS_Tolerance_Fail_Rate": fail_rate,
        "Mean_Normalized_AE": mean_norm_ae,
        "Median_Normalized_AE": median_norm_ae,
        "P95_Normalized_AE": p95_norm_ae,

        # Peak behavior
        "Peak_Timing_Error": peak_timing_error,
        "Peak_Magnitude_Error": peak_magnitude_error,
        "Abs_Peak_Magnitude_Error": abs(peak_magnitude_error)
    }

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    parameter: str = None
) -> Dict[str, float]:
    """
    Compute reconstruction, signal-preservation, and practical residual metrics.

    If parameter is provided, USGS-style tolerance metrics are also computed.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "NSE": np.nan,
            "KGE": np.nan,
            "Pearson_r": np.nan,
            "PBIAS": np.nan,
            "Peak_Error": np.nan,
            "Abs_Peak_Error": np.nan,
            "Peak_Timing_Error": np.nan,
            "Variance_Ratio": np.nan,
            "VR_Error": np.nan,
            "Shannon_Entropy_Ref": np.nan,
            "Shannon_Entropy_Pred": np.nan,
            "Shannon_Entropy_Diff": np.nan,

            # New pointwise residual metrics
            "Bias": np.nan,
            "MAE_SD": np.nan,
            "Median_AE": np.nan,
            "P90_AE": np.nan,
            "P95_AE": np.nan,
            "Max_AE": np.nan,

            # New tolerance-based metrics
            "USGS_Tolerance_Pass_Rate": np.nan,
            "USGS_Tolerance_Fail_Rate": np.nan,
            "Mean_Normalized_AE": np.nan,
            "Median_Normalized_AE": np.nan,
            "P95_Normalized_AE": np.nan,
            "n_eval_points": 0,
        }

    error = y_pred - y_true
    abs_error = np.abs(error)
    
    if parameter is not None:
        ratings = usgs_accuracy_rating(parameter, y_true, y_pred)

        valid_ratings = pd.Series(ratings).dropna()

        if len(valid_ratings) > 0:
            excellent_rate = 100 * np.mean(valid_ratings == "Excellent")
            good_rate = 100 * np.mean(valid_ratings == "Good")
            fair_rate = 100 * np.mean(valid_ratings == "Fair")
            poor_rate = 100 * np.mean(valid_ratings == "Poor")

            good_or_better_rate = excellent_rate + good_rate
            fair_or_better_rate = excellent_rate + good_rate + fair_rate

            score_map = {
                "Excellent": 1,
                "Good": 2,
                "Fair": 3,
                "Poor": 4
            }

            mean_accuracy_rating = valid_ratings.map(score_map).mean()
        else:
            excellent_rate = np.nan
            good_rate = np.nan
            fair_rate = np.nan
            poor_rate = np.nan
            good_or_better_rate = np.nan
            fair_or_better_rate = np.nan
            mean_accuracy_rating = np.nan
    else:
        excellent_rate = np.nan
        good_rate = np.nan
        fair_rate = np.nan
        poor_rate = np.nan
        good_or_better_rate = np.nan
        fair_or_better_rate = np.nan
        mean_accuracy_rating = np.nan

    mae = np.mean(abs_error)
    rmse = np.sqrt(np.mean(error ** 2))

    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = 1 - np.sum(error ** 2) / denominator if denominator != 0 else np.nan

    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        r = np.nan

    alpha = np.std(y_pred) / np.std(y_true) if np.std(y_true) != 0 else np.nan
    beta = np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) != 0 else np.nan

    if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta):
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    else:
        kge = np.nan

    pbias = 100 * np.sum(y_pred - y_true) / np.sum(y_true) if np.sum(y_true) != 0 else np.nan

    peak_error = np.max(y_pred) - np.max(y_true)
    abs_peak_error = abs(peak_error)

    true_peak_idx = int(np.nanargmax(y_true))
    pred_peak_idx = int(np.nanargmax(y_pred))
    peak_timing_error = pred_peak_idx - true_peak_idx

    variance_ratio = np.var(y_pred) / np.var(y_true) if np.var(y_true) != 0 else np.nan
    vr_error = abs(variance_ratio - 1) if np.isfinite(variance_ratio) else np.nan

    h_ref = shannon_entropy(y_true)
    h_pred = shannon_entropy(y_pred)

    if np.isfinite(h_ref) and np.isfinite(h_pred):
        h_diff = abs(h_pred - h_ref)
    else:
        h_diff = np.nan

    # New residual-distribution metrics
    bias = np.mean(error)
    mae_sd = np.std(abs_error, ddof=1) if len(abs_error) > 1 else 0.0
    median_ae = np.median(abs_error)
    p90_ae = np.percentile(abs_error, 90)
    p95_ae = np.percentile(abs_error, 95)
    max_ae = np.max(abs_error)

    # USGS-style tolerance metrics
    if parameter is not None:
        tolerance = usgs_style_tolerance(parameter, y_true)
        valid_tol = np.isfinite(tolerance) & (tolerance > 0)

        if np.any(valid_tol):
            normalized_ae = abs_error[valid_tol] / tolerance[valid_tol]
            within_tolerance = abs_error[valid_tol] <= tolerance[valid_tol]

            usgs_pass_rate = 100 * np.mean(within_tolerance)
            usgs_fail_rate = 100 - usgs_pass_rate
            mean_normalized_ae = np.mean(normalized_ae)
            median_normalized_ae = np.median(normalized_ae)
            p95_normalized_ae = np.percentile(normalized_ae, 95)
        else:
            usgs_pass_rate = np.nan
            usgs_fail_rate = np.nan
            mean_normalized_ae = np.nan
            median_normalized_ae = np.nan
            p95_normalized_ae = np.nan
    else:
        usgs_pass_rate = np.nan
        usgs_fail_rate = np.nan
        mean_normalized_ae = np.nan
        median_normalized_ae = np.nan
        p95_normalized_ae = np.nan

    return {
        # Existing metrics
        "MAE": mae,
        "RMSE": rmse,
        "NSE": nse,
        "KGE": kge,
        "Pearson_r": r,
        "PBIAS": pbias,
        "Peak_Error": peak_error,
        "Abs_Peak_Error": abs_peak_error,
        "Peak_Timing_Error": peak_timing_error,
        "Variance_Ratio": variance_ratio,
        "VR_Error": vr_error,
        "Shannon_Entropy_Ref": h_ref,
        "Shannon_Entropy_Pred": h_pred,
        "Shannon_Entropy_Diff": h_diff,

        # New pointwise residual metrics
        "Bias": bias,
        "MAE_SD": mae_sd,
        "Median_AE": median_ae,
        "P90_AE": p90_ae,
        "P95_AE": p95_ae,
        "Max_AE": max_ae,

        # New practical tolerance metrics
        "USGS_Tolerance_Pass_Rate": usgs_pass_rate,
        "USGS_Tolerance_Fail_Rate": usgs_fail_rate,
        "Mean_Normalized_AE": mean_normalized_ae,
        "Median_Normalized_AE": median_normalized_ae,
        "P95_Normalized_AE": p95_normalized_ae,

        "n_eval_points": len(y_true),
        
        "USGS_Excellent_Rate": excellent_rate,
        "USGS_Good_Rate": good_rate,
        "USGS_Fair_Rate": fair_rate,
        "USGS_Poor_Rate": poor_rate,
        "USGS_Good_or_Better_Rate": good_or_better_rate,
        "USGS_Fair_or_Better_Rate": fair_or_better_rate,
        "USGS_Mean_Accuracy_Rating": mean_accuracy_rating,
    }

def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize metric results by parameter, storm, scenario, severity,
    algorithm, and evaluation domain.
    """

    metric_cols = [
        "MAE",
        "RMSE",
        "NSE",
        "KGE",
        "Pearson_r",
        "PBIAS",
        "Peak_Error",
        "Variance_Ratio",
        "Shannon_Entropy_Ref",
        "Shannon_Entropy_Pred",
        "Shannon_Entropy_Diff",
    ]

    group_cols = [
        "experiment_mode",
        "parameter",
        "storm_id",
        "scenario",
        "severity",
        "algorithm",
        "evaluation_domain",
    ]

    summary = (
        raw
        .groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    return summary
