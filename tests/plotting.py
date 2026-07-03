import os
from src.grating_opt.plotting import plot_reward_from_csv, plot_DE_vs_wavelength, plot_norm_ne
from src.grating_opt import utils

csv_filepath = os.path.join(utils.get_data_root(), "csvs", "trial_data_edited.csv")
output_filepath = os.path.join(utils.get_data_root(), "plots", "reward_plot.png")
results_path = utils.get_results_root()
DE_output_filepath = os.path.join(utils.get_data_root(), "plots", "DE_vs_wavelength_plot.png")
norm_ne_output_filepath = os.path.join(utils.get_data_root(), "plots", "norm_ne_plot.png")

indcs = [35, 57, 69, 72, 75, 78]

# plot_reward_from_csv(csv_filepath, output_filepath)
# plot_DE_vs_wavelength(csv_filepath, indcs, DE_output_filepath, results_path)
plot_norm_ne(str(results_path), norm_ne_output_filepath, wavelength=1.95)