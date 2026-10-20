"""Primary fixed-rule experiment runner."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .data_io import get_reference_signal
from .degradation import DEGRADATION_FUNCTIONS
from .gap_filling import BASE_GAP_ALGORITHMS, match_length
from .denoising import DENOISE_ALGORITHMS
from .metrics import evaluation_slices, compute_metrics

def choose_algorithm_pool(scenario):
    if scenario in ("random_missing", "block_missing"):
        return BASE_GAP_ALGORITHMS
    if scenario in ("additive_noise", "field_realistic_fault"):
        return DENOISE_ALGORITHMS
    raise ValueError(f"Unknown scenario: {scenario}")

def run_fixed_rule_experiment(
    all_storms,
    config,
    checkpoint_dir="progress_checkpoints",
    checkpoint_every=500,
    verbose=True
):
    """
    Fixed-rule experiment with progress monitoring and checkpoint saving.

    Parameters
    ----------
    all_storms : dict
        Dictionary containing all storm data.
    config : ExperimentConfig
        Experiment configuration.
    checkpoint_dir : str
        Folder where partial results are saved.
    checkpoint_every : int
        Save partial results after this many algorithm evaluations.
    verbose : bool
        Print progress messages if True.

    Returns
    -------
    pd.DataFrame
        Raw results dataframe.
    """

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results = []
    base_rng = np.random.default_rng(config.random_seed)

    # Estimate total algorithm evaluations
    total_tasks = 0

    for parameter in config.wq_parameters:
        for storm_id in config.storm_ids:
            for scenario in config.degradation_scenarios:
                algorithm_pool = choose_algorithm_pool(scenario)
                total_tasks += (
                    len(config.severity_levels)
                    * config.repetitions
                    * len(algorithm_pool)
                )

    start_time = time.time()
    completed_tasks = 0

    pbar = tqdm(total=total_tasks, desc="Fixed-rule experiment", unit="alg")

    for parameter in config.wq_parameters:

        for storm_id in config.storm_ids:

            try:
                x_ref = get_reference_signal(
                    all_storms=all_storms,
                    storm_id=storm_id,
                    parameter=parameter,
                    dropna=True
                )

            except Exception as exc:
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
                    "error_message": str(exc),
                    "best_loss": np.nan,
                    "best_params": "",
                })
                continue

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

                algorithm_pool = choose_algorithm_pool(scenario)

                for severity in config.severity_levels:

                    for rep in range(config.repetitions):

                        seed = int(base_rng.integers(0, 1_000_000_000))
                        rng = np.random.default_rng(seed)

                        current_label = (
                            f"{parameter} | Storm {storm_id} | "
                            f"{scenario} | {severity} | rep {rep + 1}/{config.repetitions}"
                        )

                        pbar.set_postfix_str(current_label[:80])

                        try:
                            x_deg, corruption_mask = DEGRADATION_FUNCTIONS[scenario](
                                x_ref,
                                severity,
                                rng
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
                            continue

                        for algorithm_name, algorithm_fn in algorithm_pool.items():

                            try:
                                x_hat = algorithm_fn(x_deg)
                                x_hat = match_length(x_hat, len(x_ref))

                                for domain, y_true, y_pred in evaluation_slices(
                                    scenario=scenario,
                                    x_ref=x_ref,
                                    x_hat=x_hat,
                                    corruption_mask=corruption_mask
                                ):
                                    metrics = compute_metrics(y_true=y_true, y_pred=y_pred, parameter=parameter)

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

                            # Periodic checkpoint save
                            if completed_tasks % checkpoint_every == 0:
                                checkpoint_file = (
                                    checkpoint_dir /
                                    f"checkpoint_{completed_tasks}_tasks.csv"
                                )

                                pd.DataFrame(results).to_csv(
                                    checkpoint_file,
                                    index=False
                                )

                                elapsed = time.time() - start_time
                                rate = completed_tasks / elapsed if elapsed > 0 else np.nan
                                remaining = total_tasks - completed_tasks
                                eta_seconds = remaining / rate if rate > 0 else np.nan

                                if verbose:
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

    print("\nFixed-rule experiment complete.")
    print(f"Completed tasks: {completed_tasks}/{total_tasks}")
    print(f"Total elapsed time: {elapsed_total / 3600:.2f} hours")
    print(f"Final checkpoint saved: {final_checkpoint}")

    return final_df
