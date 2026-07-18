import time
from pathlib import Path

import numpy as np
import pandas as pd
from ax.api.client import Client

from src.grating_opt.acquisition_fcns import ucb
from src.grating_opt.plotting import plot_predicted_vs_actual
from src.grating_opt.simulation_calls import Result, call_on_same_node
from src.grating_opt.utils import (
    calc_crit_ne,
    get_data_root,
    get_plot_root,
    get_results_root,
    load_params,
    query_surrogate,
)


def build_surrogate_client(
        param_configs: list,
        preexisting_trials: list,
        seed_trials: int = 10,
        kappa: float = 2.0,
) -> Client:
    """
    Build an Ax client whose surrogate model is fit entirely from preexisting
    (already-completed) trial data, with no new trials generated.

    :param param_configs: List of RangeParameterConfigs for optimizable parameters
    :type param_configs: list
    :param preexisting_trials: List of (parameters, data) tuples to attach and complete
    :type preexisting_trials: list
    :param seed_trials: Sobol seed count before the acquisition function's node takes over (default: 10)
    :type seed_trials: int
    :param kappa: Kappa parameter for the UCB acquisition function (default: 2.0)
    :type kappa: float

    :return: An Ax client with a fitted, predict()-ready surrogate model
    :rtype: ax.api.client.Client
    """
    client = Client()
    client.configure_experiment(parameters=param_configs)
    client.configure_optimization(objective="reward")
    client.set_generation_strategy(generation_strategy=ucb(seed_trials, kappa))

    for parameters, data in preexisting_trials:
        trial_index = client.attach_trial(parameters=parameters)
        client.complete_trial(trial_index=trial_index, raw_data=data)

    # Attaching trials directly doesn't drive the generation strategy through
    # its normal gen() flow, so the BoTorch node never gets fit and predict()
    # fails with "GenerationNode is not predictive". fit() advances to the
    # right node and fits it without generating (and cluttering the
    # experiment with) an unwanted new trial.
    client._generation_strategy.fit(experiment=client._experiment)

    return client


def sample_param_sets(config: dict, n_sets: int) -> list:
    """
    Draw random parameter sets uniformly from a range config's [lb, ub] bounds.

    :param config: Parameter dictionary with range info (as loaded by utils.load_config)
    :type config: dict
    :param n_sets: Number of random parameter sets to draw
    :type n_sets: int

    :return: List of parameter dictionaries
    :rtype: list
    """
    all_params = []

    for _ in range(n_sets):
        param_dict = {k: float(np.random.uniform(v["lb"], v["ub"])) for k, v in config.items()}
        all_params.append(param_dict)

    return all_params


def compute_reward(
        res: dict,
        crit_ne: float,
        a: float = 5,
        b: float = 10,
        DE_col: str = None,
        wavelength_col: str = None,
        comp_wavelength: float = 1950,
        DE_peak_threshold: float = 0.98,
        DE_avg_threshold: float = 0.92,
) -> float:
    """
    Compute the reward for a completed trial from its result files.

    :param res: Trial metadata dict as returned by simulation_calls.call_on_same_node
        (must contain 'result_fdtd', 'result_rcwa', 'DE_filename')
    :type res: dict
    :param crit_ne: Critical electron density used to normalize ne_peak
    :type crit_ne: float

    :return: Reward value
    :rtype: float
    """
    rcwa_df = pd.read_csv(res["result_rcwa"])
    fdtd_df = pd.read_csv(res["result_fdtd"])

    result = Result(
        norm_ne=fdtd_df["ne_peak"].iloc[0] / crit_ne,
        peak_diff_eff=rcwa_df["DE_m1_peak"].iloc[0],
        diff_eff_avg=rcwa_df["DE_m1_avg"].iloc[0],
        DE_peak_threshold=DE_peak_threshold,
        DE_avg_threshold=DE_avg_threshold,
    )

    return result.calc_reward(
        a=a,
        b=b,
        filepath=res["DE_filename"],
        DE_col=DE_col,
        wavelength_col=wavelength_col,
        comp_wavelength=comp_wavelength,
    )


def evaluate_actual_values(
        experiment_root: Path,
        results_root: Path,
        all_params: list,
        parallelism: int,
        reward_kwargs: dict,
        poll_interval: int = 30,
        timeout: float = 4.5 * 3600,
) -> list:
    """
    Launch a simulation for every parameter set, keeping up to `parallelism`
    trials running at once (one per available node). As each trial's result
    files land, its reward is computed and a new trial is launched into the
    freed slot, until every parameter set has been evaluated.

    :param experiment_root: Root directory of the experiment (anchors matlab_code/ etc.)
    :type experiment_root: pathlib.Path
    :param results_root: Directory under which per-trial result folders are written
    :type results_root: pathlib.Path
    :param all_params: List of parameter dictionaries to evaluate
    :type all_params: list
    :param parallelism: Number of trials to keep running concurrently (typically SLURM_JOB_NUM_NODES)
    :type parallelism: int
    :param reward_kwargs: Keyword arguments forwarded to compute_reward (crit_ne, a, b, DE_col, ...)
    :type reward_kwargs: dict
    :param poll_interval: Seconds between polls of running trials (default: 30)
    :type poll_interval: int
    :param timeout: Seconds before an individual trial is abandoned as timed out (default: 4.5 hours)
    :type timeout: float

    :return: List of reward values (None for trials that timed out), ordered like all_params
    :rtype: list
    """
    n = len(all_params)
    actual_mf = [None] * n
    pending = list(range(n))
    running = {}  # trial_index -> (trial_metadata dict, launch_time)

    while pending or running:
        while pending and len(running) < parallelism:
            trial_index = pending.pop(0)
            res = call_on_same_node(
                experiment_root=experiment_root,
                results_root=results_root,
                trial=trial_index,
                params=all_params[trial_index],
            )
            running[trial_index] = (res, time.time())
            print(f"Launched trial {trial_index} ({len(running)}/{parallelism} slots busy, "
                  f"{len(pending)} pending).", flush=True)

        finished = []
        for trial_index, (res, launch_time) in running.items():
            fdtd_path = Path(res["result_fdtd"])
            rcwa_path = Path(res["result_rcwa"])
            files_ready = (
                fdtd_path.exists() and rcwa_path.exists()
                and fdtd_path.stat().st_size > 0 and rcwa_path.stat().st_size > 0
            )

            if files_ready:
                actual_mf[trial_index] = compute_reward(res, **reward_kwargs)
                print(f"Trial {trial_index} complete: actual reward = {actual_mf[trial_index]}", flush=True)
                finished.append(trial_index)
            elif time.time() - launch_time > timeout:
                print(f"Trial {trial_index} timed out after {timeout} seconds; leaving as missing.", flush=True)
                finished.append(trial_index)

        for trial_index in finished:
            del running[trial_index]

        if pending or running:
            time.sleep(poll_interval)

    return actual_mf


def run_surrogate_test(
        experiment_root: Path,
        surrogate_root: Path,
        config: dict,
        n_sets: int,
        preexisting_trials: list,
        seed_trials: int = 10,
        kappa: float = 2.0,
        parallelism: int = 1,
        csv_filename: str = "surrogate_comparison.csv",
        DE_col: str = None,
        wavelength_col: str = None,
        a: float = 5,
        b: float = 10,
        DE_peak_threshold: float = 0.98,
        DE_avg_threshold: float = 0.92,
        central_wavelength: float = 1950,
        results_dir: str = "Results",
        poll_interval: int = 30,
        timeout: float = 4.5 * 3600,
) -> pd.DataFrame:
    """
    Sample random parameter sets from `config`'s ranges, predict their reward
    with a surrogate fit on `preexisting_trials`, evaluate their actual reward
    via parallel simulation, and save a predicted-vs-actual comparison CSV and
    plot under `surrogate_root`.

    :param experiment_root: Root directory of the experiment (anchors matlab_code/ etc.)
    :type experiment_root: pathlib.Path
    :param surrogate_root: Root directory for this surrogate test's own Results/data
        (e.g. experiment_root/Surrogate)
    :type surrogate_root: pathlib.Path
    :param config: Parameter range config, as loaded by utils.load_config (lb/ub per parameter)
    :type config: dict
    :param n_sets: Number of random parameter sets to sample and evaluate
    :type n_sets: int
    :param preexisting_trials: List of (parameters, data) tuples used to fit the surrogate
    :type preexisting_trials: list
    :param seed_trials: Sobol seed count before the acquisition function's node takes over (default: 10)
    :type seed_trials: int
    :param kappa: Kappa parameter for the UCB acquisition function (default: 2.0)
    :type kappa: float
    :param parallelism: Number of simulations to run concurrently (typically SLURM_JOB_NUM_NODES)
    :type parallelism: int
    :param csv_filename: Name of the CSV file to save predicted-vs-actual data (in surrogate_root/data/csvs)
    :type csv_filename: str
    :param DE_col: Column name for diffraction efficiency in the DE-vs-wavelength CSV
    :type DE_col: str
    :param wavelength_col: Column name for wavelength in the DE-vs-wavelength CSV
    :type wavelength_col: str
    :param a: Weight parameter for peak effective index in reward calculation (default: 5)
    :type a: float
    :param b: Weight parameter for average diffraction efficiency in reward calculation (default: 10)
    :type b: float
    :param DE_peak_threshold: Threshold for peak diffraction efficiency (default: 0.98)
    :type DE_peak_threshold: float
    :param DE_avg_threshold: Threshold for average diffraction efficiency (default: 0.92)
    :type DE_avg_threshold: float
    :param central_wavelength: Wavelength in nm against which to compare (default: 1950 nm)
    :type central_wavelength: float
    :param results_dir: Directory (under surrogate_root) to save simulation results (default: 'Results')
    :type results_dir: str
    :param poll_interval: Seconds between polls of running trials (default: 30)
    :type poll_interval: int
    :param timeout: Seconds before an individual trial is abandoned as timed out (default: 4.5 hours)
    :type timeout: float

    :return: DataFrame with one row per sampled parameter set (params, mf_predicted, mf_predicted_sem, mf_actual)
    :rtype: pandas.DataFrame
    """
    # STEP 1: Fit a surrogate model from preexisting trial data
    param_configs = load_params(config)
    client = build_surrogate_client(param_configs, preexisting_trials, seed_trials=seed_trials, kappa=kappa)

    # STEP 2: Obtain sets of input parameters within config range
    all_params = sample_param_sets(config, n_sets)

    # STEP 3: Obtain merit function predictions from the client
    predictions = [query_surrogate(client, params)["reward"] for params in all_params]
    mf_pred = [p[0] for p in predictions]
    mf_pred_sem = [p[1] for p in predictions]

    # STEP 4: Obtain actual merit function values via simulation, run in
    # parallel across as many nodes as are available in this allocation.
    results_root = get_results_root(surrogate_root, results_dir)
    reward_kwargs = dict(
        crit_ne=calc_crit_ne(central_wavelength),
        a=a,
        b=b,
        DE_col=DE_col,
        wavelength_col=wavelength_col,
        comp_wavelength=central_wavelength,
        DE_peak_threshold=DE_peak_threshold,
        DE_avg_threshold=DE_avg_threshold,
    )
    print(f"Evaluating {n_sets} parameter sets with parallelism={parallelism}.", flush=True)
    mf_actual = evaluate_actual_values(
        experiment_root=experiment_root,
        results_root=results_root,
        all_params=all_params,
        parallelism=parallelism,
        reward_kwargs=reward_kwargs,
        poll_interval=poll_interval,
        timeout=timeout,
    )

    # Save predicted vs. actual comparison data
    df = pd.DataFrame(all_params)
    df.insert(0, "trial_index", range(n_sets))
    df["mf_predicted"] = mf_pred
    df["mf_predicted_sem"] = mf_pred_sem
    df["mf_actual"] = mf_actual

    csv_path = get_data_root(surrogate_root) / "csvs" / csv_filename
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved comparison data to {csv_path}")

    # STEP 5: Plot predictions vs. actual values
    plot_path = get_plot_root(surrogate_root) / "surrogate_vs_actual.png"
    plot_predicted_vs_actual(
        predicted=mf_pred,
        actual=mf_actual,
        output_filepath=str(plot_path),
        predicted_sem=mf_pred_sem,
        metric_name="Reward",
    )
    print(f"Saved plot to {plot_path}")

    return df
