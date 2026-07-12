import pandas as pd

from src.grating_opt.simulation_calls import Result
from src.grating_opt.utils import calc_crit_ne, save_trial_data

results_dir = "prototype_results"
trial_idx = 0
csv_filename = "prototype_data.csv"

DE_col = "DE_reflected_m1"
wavelength_col = "Wavelength_um"
a = 5
b = 10
DE_peak_threshold = 0.98
DE_avg_threshold = 0.92
central_wavelength = 1950  # [nm]

params = {
    "dc": 0.5,
    "tp": 1375.,
    "tr": 100.,
    "lmm": 630,
    "aoi": 38.1847,
    "sa": 90.
}

trial_dir = f"{results_dir}/{trial_idx}"
rcwa_res = f"{trial_dir}/RCWA_result.csv"
fdtd_res = f"{trial_dir}/FDTD_result.csv"
DE_filepath = f"{trial_dir}/DE_vs_wavelength.csv"

rcwa_df = pd.read_csv(rcwa_res)
fdtd_df = pd.read_csv(fdtd_res)

res = Result(
    norm_ne=fdtd_df['ne_peak'].iloc[0] / calc_crit_ne(central_wavelength),
    peak_diff_eff=rcwa_df['DE_m1_peak'].iloc[0],
    diff_eff_avg=rcwa_df['DE_m1_avg'].iloc[0],
    DE_peak_threshold=DE_peak_threshold,
    DE_avg_threshold=DE_avg_threshold
)

reward = res.calc_reward(
    a=a,
    b=b,
    filepath=DE_filepath,
    DE_col=DE_col,
    wavelength_col=wavelength_col,
    comp_wavelength=central_wavelength
)

print(f"Reward for prototype trial {trial_idx}: {reward}")

save_trial_data(
    trial_index=trial_idx,
    reward=reward,
    params=params,
    filename=csv_filename
)
