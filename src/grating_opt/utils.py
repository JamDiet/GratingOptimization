from ruamel.yaml import YAML
from pathlib import Path
from ax.api.configs import RangeParameterConfig
import pandas as pd
from ax.api.client import Client
from ax.service.utils.report_utils import exp_to_df

_yaml = YAML(typ='safe')


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_results_root() -> Path:
    return get_project_root() / "Results"


def get_prototype_results_root() -> Path:
    return get_project_root() / "prototype_results"


def get_config_root() -> Path:
    return get_project_root() / "configs"


def get_data_root() -> Path:
    return get_project_root() / "data"


def get_plot_root() -> Path:
    return get_data_root() / "plots"


def load_config(config: str) -> dict:
    file = get_config_root() / config

    with file.open() as f:
        params = _yaml.load(f)
    
    return params


def load_params(params: dict):
    """
    Create list of RangeParameterConfigs for Ax optimization.

    :param params: Parameter dictionary with range and type info
    :type params: dict
    """
    param_list = []

    for k, v in params.items():
        param_config = RangeParameterConfig(
            name=k,
            bounds=(v["lb"], v["ub"]),
            parameter_type=v["parameter_type"]
        )

        param_list.append(param_config)
    
    return param_list


def load_preexisting_trials(csv_filename: str):
    full_path = get_data_root() / "csvs" / csv_filename
    # 1. Read the CSV back into a DataFrame
    df = pd.read_csv(full_path)
    # 2. Define which columns belong to which group
    metric_cols = ["reward"]
    exclude_cols = set(metric_cols + ["trial_index"])
    param_cols = [col for col in df.columns if col not in exclude_cols]
    preexisting_trials = []
    # 3. Iterate through rows and build the nested dictionary structure
    for _, row in df.iterrows():
        # Convert row subsets to dictionaries, ignoring the pandas index
        params_dict = row[param_cols].to_dict()
        metrics_dict = row[metric_cols].to_dict()

        # Append as a tuple of the two dicts
        preexisting_trials.append((params_dict, metrics_dict))
    return preexisting_trials


def calc_crit_ne(wavelength: float):
    """
    :param wavelength: Wavelength in nm
    :type wavelength: float
    """
    return 1.1149 / ((wavelength / 1000) ** 2)  # in nm


def query_surrogate(client: Client, params: dict):
    """
    Predict metric function value from surrogate mean.

    :param client: Ax Bayesian optimization client
    :type client: Client
    :param params: Input parameter names and values
    :type params: dict
    :return: List of metrics mapped to mean values and uncertainty
    :rtype: list
    """
    return client.predict(params)


def save_client_data(client: Client, filename: str):
    full_path = get_data_root() / "csvs" / filename

    # client.get_trials_data_frame() aggregates parameters and metrics
    df_step = exp_to_df(client._experiment)
    
    # Ensure the target directory exists before writing
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Overwrite the CSV with all trials (mode='w' is default, header is always True)
    df_step.to_csv(full_path, mode='w', index=False, header=True)


def save_trial_data(trial_index: int, reward: float, params: dict, filename: str):
    full_path = get_data_root() / "csvs" / filename

    # Create a DataFrame for the new trial data
    trial_data = {"trial_index": trial_index, "reward": reward, **params}
    df_trial = pd.DataFrame([trial_data])

    # Ensure the target directory exists before writing
    full_path.parent.mkdir(parents=True, exist_ok=True)

    # Append the new trial data to the CSV file
    if full_path.exists():
        df_trial.to_csv(full_path, mode='a', index=False, header=False)
    else:
        df_trial.to_csv(full_path, mode='w', index=False, header=True)