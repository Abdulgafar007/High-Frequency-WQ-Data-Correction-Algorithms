"""Primary fixed-rule experiment runner.

This version uses:
- 8 gap-filling algorithms for random/block missingness;
- 9 denoising/correction algorithms for additive/field-realistic faults;
- Stage_smoothed from the manuscript hydrologic-phase classification for
  Delta-H weighted interpolation and HPWI.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .degradation import DEGRADATION_FUNCTIONS
from .gap_filling import (
    BASE_GAP_ALGORITHMS,
    match_length,
    local_moving_mean_gapfill,
    delta_h_weighted_gapfill,
    hpwi_gapfill,
)
from .denoising import DENOISE_ALGORITHMS
from .metrics import evaluation_slices, compute_metrics
from .hydrologic_phase import classify_storm_phases


def _storm_dataframe(all_storms, storm_id):
    """Return the event-window dataframe for one storm."""
    if storm_id not in all_storms:
        raise KeyError(f"Storm {storm_id} was not found in all_storms.")

    obj = all_storms[storm_id]

    if isinstance(obj, pd.DataFrame):
        return obj.copy()

    if isinstance(obj, dict) and isinstance(obj.get("data"), pd.DataFrame):
        return obj["data"].copy()

    raise TypeError(
        f"Storm {storm_id} must be a DataFrame or a dictionary containing "
        "a DataFrame under the key 'data'."
    )


def get_aligned_reference_and_stage(
    all_storms,
    storm_id,
    parameter,
    classified_storms=None,
):
    """
    Return the reference WQ vector and aligned Stage_smoothed vector.

    The same finite WQ observations are used for both vectors, preventing
    length/timestamp mismatch when a parameter contains missing values.
    """

    df = _storm_dataframe(all_storms, storm_id)

    if parameter not in df.columns:
        raise KeyError(
            f"Parameter '{parameter}' was not found for Storm {storm_id}. "
            f"Available columns: {list(df.columns)}"
        )

    wq = pd.to_numeric(df[parameter], errors="coerce")
    valid_wq = wq.notna()
    ref_index = df.index[valid_wq]
    x_ref = wq.loc[valid_wq].to_numpy(dtype=float)

    if len(x_ref) == 0:
        raise ValueError(
            f"No finite {parameter} observations are available for Storm {storm_id}."
        )

    # Prefer an already-computed hydrologic classification.
    phase_df = None
    if classified_storms is not None and storm_id in classified_storms:
        obj = classified_storms[storm_id]
        if isinstance(obj, pd.DataFrame):
            phase_df = obj
        elif isinstance(obj, dict) and isinstance(obj.get("data"), pd.DataFrame):
            phase_df = obj["data"]

    # Otherwise reproduce the same manuscript phase classification here.
    if phase_df is None:
        phase_df, _ = classify_storm_phases(df, radar_column="Radar")

    if "Stage_smoothed" not in phase_df.columns:
        raise KeyError(
            f"Stage_smoothed was not available for Storm {storm_id}."
        )

    stage = (
        pd.to_numeric(phase_df["Stage_smoothed"], errors="coerce")
        .reindex(ref_index)
        .to_numpy(dtype=float)
    )

    if len(stage) != len(x_ref):
        raise ValueError(
            f"Stage/reference length mismatch for {parameter}, Storm {storm_id}: "
            f"{len(stage)} vs {len(x_ref)}."
        )

    return x_ref, stage


def choose_algorithm_pool(scenario, stage=None):
    """
    Build the manuscript algorithm pool.

    Missingness:
        8 methods
    Additive noise / field-realistic fault:
        9 methods
    """

    if scenario in ("random_missing", "block_missing"):
        if stage is None:
            raise ValueError(
                "An aligned Stage_smoothed vector is required for missingness "
                "because Delta-H weighted interpolation and HPWI are included."
            )

        pool = dict(BASE_GAP_ALGORITHMS)
        linear_fn = pool["Linear_Interpolation"]

        pool["Local_Moving_Mean"] = (
            lambda x: local_moving_mean_gapfill(x=x, n_each_side=2)
        )
        pool["Delta_H_Weighted_Interpolation"] = (
            lambda x, stage=stage, linear_fn=linear_fn:
            delta_h_weighted_gapfill(
                x=x,
                stage=stage,
                linear_function=linear_fn,
            )
        )
        pool["HPWI"] = (
            lambda x, stage=stage, linear_fn=linear_fn:
            hpwi_gapfill(
                x=x,
                stage=stage,
                linear_function=linear_fn,
            )
        )

        return pool

    if scenario in ("additive_noise", "field_realistic_fault"):
        return dict(DENOISE_ALGORITHMS)

    raise ValueError(f"Unknown scenario: {scenario}")


def _prediction_only(result):
    """Extract prediction when an algorithm also returns diagnostics."""
    if isinstance(result, tuple):
        if len(result) == 0:
            raise ValueError("Algorithm returned an empty tuple.")
        return result[0]
    return result


def run_fixed_rule_experiment(
    all_storms,
    config,
    classified_storms=None,
    checkpoint_dir="progress_checkpoints",
    checkpoint_every=500,
    verbose=True,
):
    """
    Run the fixed-rule manuscript experiment.

    Parameters
    ----------
    all_storms : dict
        Four manuscript storm-event dataframes.
    config : ExperimentConfig
        Experiment configuration.
    classified_storms : dict, optional
        Output from classify_all_storms(all_storms). If omitted, the same
        phase-classification function is applied internally to create
        Stage_smoothed for the hydrology-informed gap methods.
    checkpoint_dir : str or Path
        Folder for progress checkpoints.
    checkpoint_every : int
        Save a checkpoint after this many algorithm evaluations.
    verbose : bool
        Print checkpoint/runtime information.

    Returns
    -------
    pandas.DataFrame
        Raw algorithm-performance results.
    """

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results = []
    base_rng = np.random.default_rng(config.random_seed)

    # Cache reference and aligned hydrologic vectors once per parameter × storm.
    reference_cache = {}
    stage_cache = {}

    for parameter in config.wq_parameters:
        for storm_id in config.storm_ids:
            try:
                x_ref, stage = get_aligned_reference_and_stage(
                    all_storms=all_storms,
                    storm_id=storm_id,
                    parameter=parameter,
                    classified_storms=classified_storms,
                )
                reference_cache[(parameter, storm_id)] = x_ref
                stage_cache[(parameter, storm_id)] = stage
            except Exception as exc:
                reference_cache[(parameter, storm_id)] = exc
                stage_cache[(parameter, storm_id)] = None

    # Exact task count: 8 methods for each missingness scenario,
    # 9 methods for each correction scenario.
    total_tasks = 0
    for parameter in config.wq_parameters:
        for storm_id in config.storm_ids:
            stage = stage_cache[(parameter, storm_id)]
            for scenario in config.degradation_scenarios:
                try:
                    algorithm_pool = choose_algorithm_pool(
                        scenario,
                        stage=stage if scenario in ("random_missing", "block_missing") else None,
                    )
                    n_algorithms = len(algorithm_pool)
                except Exception:
                    # Reference/stage errors are handled in the main loop.
                    n_algorithms = (
                        8 if scenario in ("random_missing", "block_missing") else 9
                    )

                total_tasks += (
                    len(config.severity_levels)
                    * config.repetitions
                    * n_algorithms
                )

    start_time = time.time()
    completed_tasks = 0

    pbar = tqdm(
        total=total_tasks,
        desc="Fixed-rule experiment",
        unit="alg",
    )

    for parameter in config.wq_parameters:
        for storm_id in config.storm_ids:

            cached = reference_cache[(parameter, storm_id)]

            if isinstance(cached, Exception):
                results.append({
                    "experiment_mode": "fixed_rule",
                    "parameter": parameter,
                    "storm_id": storm_id,
                    "scenario": "",
                    "severity": "",
                    "repetition": np.nan,
                    "seed": np.nan,
                    "algorithm": "",
                    "evaluation_domain": "error",
                    "status": "reference_signal_error",
                    "error_message": str(cached),
                    "best_loss": np.nan,
                    "best_params": "",
                })
                continue

            x_ref = np.asarray(cached, dtype=float)
            stage = np.asarray(stage_cache[(parameter, storm_id)], dtype=float)

            if len(x_ref) < 5:
                results.append({
                    "experiment_mode": "fixed_rule",
                    "parameter": parameter,
                    "storm_id": storm_id,
                    "scenario": "",
                    "severity": "",
                    "repetition": np.nan,
                    "seed": np.nan,
                    "algorithm": "",
                    "evaluation_domain": "error",
                    "status": "too_short",
                    "error_message": f"Reference signal has only {len(x_ref)} points.",
                    "best_loss": np.nan,
                    "best_params": "",
                })
                continue

            for scenario in config.degradation_scenarios:

                if scenario not in DEGRADATION_FUNCTIONS:
                    raise ValueError(
                        f"Scenario '{scenario}' not found in DEGRADATION_FUNCTIONS."
                    )

                algorithm_pool = choose_algorithm_pool(
                    scenario,
                    stage=stage if scenario in ("random_missing", "block_missing") else None,
                )

                for severity in config.severity_levels:
                    for rep in range(config.repetitions):

                        seed = int(base_rng.integers(0, 1_000_000_000))
                        rng = np.random.default_rng(seed)

                        pbar.set_postfix_str(
                            (
                                f"{parameter} | Storm {storm_id} | {scenario} | "
                                f"{severity} | rep {rep + 1}/{config.repetitions}"
                            )[:80]
                        )

                        try:
                            x_deg, corruption_mask = DEGRADATION_FUNCTIONS[scenario](
                                x_ref,
                                severity,
                                rng,
                            )
                        except Exception as exc:
                            results.append({
                                "experiment_mode": "fixed_rule",
                                "parameter": parameter,
                                "storm_id": storm_id,
                                "scenario": scenario,
                                "severity": severity,
                                "repetition": rep,
                                "seed": seed,
                                "algorithm": "",
                                "evaluation_domain": "error",
                                "status": "degradation_error",
                                "error_message": str(exc),
                                "best_loss": np.nan,
                                "best_params": "",
                            })
                            # The degradation failed before any algorithm ran.
                            for _ in algorithm_pool:
                                completed_tasks += 1
                                pbar.update(1)
                            continue

                        for algorithm_name, algorithm_fn in algorithm_pool.items():
                            try:
                                result = algorithm_fn(x_deg.copy())
                                x_hat = _prediction_only(result)
                                x_hat = match_length(x_hat, len(x_ref))

                                for domain, y_true, y_pred in evaluation_slices(
                                    scenario=scenario,
                                    x_ref=x_ref,
                                    x_hat=x_hat,
                                    corruption_mask=corruption_mask,
                                ):
                                    metrics = compute_metrics(
                                        y_true=y_true,
                                        y_pred=y_pred,
                                        parameter=parameter,
                                    )

                                    results.append({
                                        "experiment_mode": "fixed_rule",
                                        "parameter": parameter,
                                        "storm_id": storm_id,
                                        "scenario": scenario,
                                        "severity": severity,
                                        "repetition": rep,
                                        "seed": seed,
                                        "algorithm": algorithm_name,
                                        "evaluation_domain": domain,
                                        "status": "ok",
                                        "error_message": "",
                                        "best_loss": np.nan,
                                        "best_params": "",
                                        **metrics,
                                    })

                            except Exception as exc:
                                results.append({
                                    "experiment_mode": "fixed_rule",
                                    "parameter": parameter,
                                    "storm_id": storm_id,
                                    "scenario": scenario,
                                    "severity": severity,
                                    "repetition": rep,
                                    "seed": seed,
                                    "algorithm": algorithm_name,
                                    "evaluation_domain": "error",
                                    "status": "error",
                                    "error_message": str(exc),
                                    "best_loss": np.nan,
                                    "best_params": "",
                                })

                            completed_tasks += 1
                            pbar.update(1)

                            if checkpoint_every and completed_tasks % checkpoint_every == 0:
                                checkpoint_file = (
                                    checkpoint_dir /
                                    f"checkpoint_{completed_tasks}_tasks.csv"
                                )
                                pd.DataFrame(results).to_csv(
                                    checkpoint_file,
                                    index=False,
                                )

                                if verbose:
                                    elapsed = time.time() - start_time
                                    rate = completed_tasks / elapsed if elapsed > 0 else np.nan
                                    remaining = total_tasks - completed_tasks
                                    eta_seconds = remaining / rate if rate > 0 else np.nan

                                    print(
                                        f"\nCheckpoint saved: {checkpoint_file}\n"
                                        f"Completed: {completed_tasks}/{total_tasks} "
                                        f"({100 * completed_tasks / total_tasks:.2f}%)\n"
                                        f"Elapsed: {elapsed / 3600:.2f} hours\n"
                                        f"Estimated remaining: {eta_seconds / 3600:.2f} hours\n"
                                    )

    pbar.close()

    final_df = pd.DataFrame(results)
    final_checkpoint = checkpoint_dir / "checkpoint_final.csv"
    final_df.to_csv(final_checkpoint, index=False)

    elapsed_total = time.time() - start_time

    if verbose:
        print("\nFixed-rule experiment complete.")
        print(f"Completed tasks: {completed_tasks}/{total_tasks}")
        print(f"Total elapsed time: {elapsed_total / 3600:.2f} hours")
        print(f"Final checkpoint saved: {final_checkpoint}")

    return final_df
