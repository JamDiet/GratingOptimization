import argparse
import src.grating_opt.optimizer as opt
import src.grating_opt.simulation_calls as sim
from src.grating_opt.plotting import plot_reward_from_csv
from src.grating_opt import utils
from ax.analysis.plotly.arm_effects import ArmEffectsPlot
from ax.analysis.summary import Summary

csv_filename = "trial_data.csv"
result_filename = utils.get_plot_root() / "trial_rewards.png"
DE_filename = "DE_vs_wavelength.csv"
DE_col = "DE_reflected_m1"
wavelength_col = "Wavelength_um"
WAVELENGTH = 1.95   # [um]

def main(args):
    # Load optimizable parameters
    config = utils.load_config(args.config)
    params = utils.load_params(config)

    # ======= Specify optimizer arguments ======= #
    # ============== MANUAL INPUT ============== #
    func = sim.call_simulations
    num_trials = 3
    crit_ne = utils.calc_crit_ne(WAVELENGTH)
    a = 1.0
    b = 1.0
    c = 1.0
    DE_peak_threshold = 0.98
    DE_avg_threshold = 0.92
    norm_ne_default = 1.5
    seed_trials = 10
    kappa = 2.0
    local = False
    max_trials = 1
    preexisting_trials = utils.load_preexisting_trials(csv_filename)

    # Initialize optimizer
    optimizer = opt.Optimizer(
        func=func,
        a=a,
        b=b,
        c=c,
        crit_ne=crit_ne,
        DE_peak_threshold=DE_peak_threshold,
        DE_avg_threshold=DE_avg_threshold,
        norm_ne_default=norm_ne_default,
        local=local,
        filename=DE_filename,
        DE_col=DE_col,
        wavelength_col=wavelength_col,
        comp_wavelength=WAVELENGTH*1000
    )

    # Optimize parameters
    client, best_values, all_values = optimizer.optimize_params(
        params=params,
        num_trials=num_trials,
        seed_trials=seed_trials,
        kappa=kappa,
        max_trials=max_trials,
        preexisting_trials=preexisting_trials,
        csv_filename=csv_filename
    )

    client.compute_analyses(
        analyses=[ArmEffectsPlot(metric_name="reward", use_model_predictions=False), Summary()]
    )

    full_path = utils.get_data_root() / "csvs" / csv_filename
    plot_reward_from_csv(full_path, result_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="default.yml")

    args = parser.parse_args()

    main(args)
