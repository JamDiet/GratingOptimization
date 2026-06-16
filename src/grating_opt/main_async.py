import os
import argparse
import src.grating_opt.optimizer as opt
import src.grating_opt.simulation_calls as sim
from src.grating_opt.plotting import plot_reward_from_csv
from src.grating_opt import utils
from ax.analysis.plotly.arm_effects import ArmEffectsPlot
from ax.analysis.summary import Summary

csv_filename = "trial_data.csv"
result_filename = utils.get_plot_root() / "trial_rewards.png"
DE_col = "DE_reflected_m1"
wavelength_col = "Wavelength_um"
WAVELENGTH = 1.95   # [um]

def main(args):
    # Load optimizable parameters
    config = utils.load_config(args.config)
    params = utils.load_params(config)

    # ======= Specify optimizer arguments ======= #
    # ============== MANUAL INPUT ============== #
    num_trials = 10
    crit_ne = utils.calc_crit_ne(WAVELENGTH)
    seed_trials = 10
    kappa = 2.0
    local = False
    parallelism = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    preexisting_trials = utils.load_preexisting_trials(csv_filename)

    # Initialize optimizer
    optimizer = opt.Optimizer(
        crit_ne=crit_ne,
        local=local,
        DE_col=DE_col,
        wavelength_col=wavelength_col,
        comp_wavelength=WAVELENGTH*1000
    )

    # Optimize parameters
    client = optimizer.opt_async(
        params=params,
        num_trials=num_trials,
        seed_trials=seed_trials,
        kappa=kappa,
        parallelism=parallelism,
        preexisting_trials=preexisting_trials
    )

    utils.save_client_data(client, csv_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="default.yml")

    args = parser.parse_args()

    main(args)
