import os
import argparse
from src.grating_opt.optimization import opt_async
from src.grating_opt import utils

trial_data_filename = "trial_data.csv"
DE_col = "DE_reflected_m1"
wavelength_col = "Wavelength_um"
results_dir = "Results"

def main(args):
    # Load optimizable parameters
    config = utils.load_config(args.config)
    params = utils.load_params(config)

    # Load preexisting trials if available
    preexisting_trials = utils.load_preexisting_trials(trial_data_filename)

    # Set number of parallel trials based on nodes available
    parallelism = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))

    # ======= Specify optimizer arguments ======= #
    # ============== MANUAL INPUT ============== #
    num_trials = 30
    seed_trials = 10
    kappa = 2
    a = 5
    b = 10
    DE_peak_threshold = 0.98
    DE_avg_threshold = 0.92
    central_wavelength = 1950  # [nm]

    # Run optimization
    client = opt_async(
        params=params,
        num_trials=num_trials,
        seed_trials=seed_trials,
        kappa=kappa,
        parallelism=parallelism,
        preexisting_trials=preexisting_trials,
        csv_filename=trial_data_filename,
        DE_col=DE_col,
        wavelength_col=wavelength_col,
        results_dir=results_dir,
        a=a,
        b=b,
        DE_peak_threshold=DE_peak_threshold,
        DE_avg_threshold=DE_avg_threshold,
        central_wavelength=central_wavelength
    )

    return client


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="default.yml")

    args = parser.parse_args()

    _ = main(args)
