import os
from src.grating_opt.plotting import plot_reward_from_csv, plot_DE_vs_wavelength, plot_norm_ne, plot_pareto_frontier, update_ne_peak_csv
from src.grating_opt import utils

# For trials
csv_filepath = os.path.join(utils.get_data_root(), "csvs", "trial_data.csv")
output_filepath = os.path.join(utils.get_data_root(), "plots", "reward_plot.png")
results_path = utils.get_results_root()
DE_output_filepath = os.path.join(utils.get_data_root(), "plots", "worst_ne_DE_plot.png")
norm_ne_output_filepath = os.path.join(utils.get_data_root(), "plots", "norm_ne_plot.png")
pareto_output_filepath = os.path.join(utils.get_data_root(), "plots", "pareto_frontier.png")

# For prototype
p_csv_path = os.path.join(utils.get_data_root(), "csvs", "prototype_data.csv")
p_DE_output = os.path.join(utils.get_data_root(), "plots", "prototype_DE.png")
p_results = utils.get_prototype_results_root()

# indcs = [72, 75, 103, 104, 107, 110, 113, 115, 126, 138, 152]
# indcs = [75]
# indcs = [4, 8, 70, 100, 148, 154, 163]
indcs = [162, 171, 46]  # top 10 trials by norm_ne (data/csvs/ne_peak_data.csv)
# indcs = [140, 2, 151]  # trials with norm_ne < 1 (data/csvs/ne_peak_data.csv)
# indcs = [139, 20, 73]  # top 10 trials by DE_m1_avg (Results/<trial_idx>/RCWA_result.csv)

# plot_reward_from_csv(csv_filepath, output_filepath)
# plot_DE_vs_wavelength(csv_filepath, indcs, DE_output_filepath, results_path, include_prototype=True)
plot_norm_ne(str(results_path), norm_ne_output_filepath, wavelength=1950)
# plot_pareto_frontier(str(results_path), pareto_output_filepath, wavelength=1950, de_modes=range(1, 4))
# update_ne_peak_csv(wavelength=1950)