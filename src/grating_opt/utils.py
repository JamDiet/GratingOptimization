from ruamel.yaml import YAML
from pathlib import Path
from ax.api.configs import RangeParameterConfig
import pandas as pd
from ax.api.client import Client

_yaml = YAML(typ='safe')


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


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
    param_cols = ["duty_cycle", "pillar_thickness", "aoi"]
    metric_cols = ["reward"]

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
    :param wavelength: Wavelength in um
    :type wavelength: float
    """
    return 1.1149 / (wavelength ** 2)  # in nm


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