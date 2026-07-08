from ax.api.client import Client
from src.grating_opt.acquisition_fcns import ucb
from src.grating_opt.automatic_orchestration import Runner, Metric
from src.grating_opt.utils import calc_crit_ne


def opt_async(
        params: list,
        num_trials: int,
        seed_trials: int=10,
        kappa: float=2.0,
        parallelism: int=1,
        preexisting_trials: list=None,
        csv_filename: str=None,
        DE_col: str=None,
        wavelength_col: str=None,
        results_dir: str='Results',
        a: float=5,
        b: float=10,
        DE_peak_threshold: float=0.98,
        DE_avg_threshold: float=0.92,
        central_wavelength: float=1950
):
    """
    Asynchronous version of optimize_params. Trials are submitted to the cluster and then polled for completion.

    :param params: List of RangeParameterConfigs for optimizable parameters
    :type params: list
    :param num_trials: Number of optimization trials
    :type num_trials: int
    :param seed_trials: Number of seed trials for acquisition function (default: 10)
    :type seed_trials: int
    :param kappa: Kappa parameter for UCB acquisition function (default: 2.0)
    :type kappa: float
    :param parallelism: Number of trials to run in parallel (default: 1)
    :type parallelism: int
    :param preexisting_trials: List of tuples containing (parameters, data) for pre-existing trials
    :type preexisting_trials: list
    :param csv_filename: Name of CSV file to save trial data (will be placed in data/csvs directory)
    :type csv_filename: str
    :param DE_col: Column name for diffraction efficiency in CSV file (default: None)
    :type DE_col: str
    :param wavelength_col: Column name for wavelength in CSV file (default: None)
    :type wavelength_col: str
    :param results_dir: Directory to save results (default: 'Results')
    :type results_dir: str
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

    :return: The Ax client used for the optimization
    :rtype: ax.api.client.Client
    """
    # Initialize asynchronous runner and metric
    runner = Runner(results_dir=results_dir)
    metric = Metric(
        name="reward",
        a=a,
        b=b,
        crit_ne=calc_crit_ne(central_wavelength),
        csv_filename=csv_filename,
        DE_col=DE_col,
        wavelength_col=wavelength_col,
        comp_wavelength=central_wavelength,
        DE_peak_threshold=DE_peak_threshold,
        DE_avg_threshold=DE_avg_threshold
    )

    # Configure client
    client = Client()
    client.configure_experiment(parameters=params)
    client.configure_optimization(objective="reward")
    client.configure_runner(runner=runner)
    client.configure_metrics(metrics=[metric])
    client.set_generation_strategy(generation_strategy=ucb(seed_trials, kappa))

    # Load preexisting data
    if preexisting_trials is not None:
        for parameters, data in preexisting_trials:
            # Attach the parameterization to the Client as a trial and immediately complete it with the preexisting data
            trial_index = client.attach_trial(parameters=parameters)
            client.complete_trial(trial_index=trial_index, raw_data=data)

    # Run trials in parallel
    client.run_trials(
        max_trials=num_trials,
        parallelism=parallelism,
        tolerated_trial_failure_rate=0.1,
        initial_seconds_between_polls=30,
    )

    return client