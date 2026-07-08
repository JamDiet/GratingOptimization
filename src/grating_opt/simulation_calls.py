import pandas as pd
from src.grating_opt.SA_both import run_both


class Result:
    def __init__(
            self,
            norm_ne: float,
            peak_diff_eff: float,
            diff_eff_avg: float,
            DE_peak_threshold: float=0.98,
            DE_avg_threshold: float=0.92
    ):
        self.norm_ne = norm_ne
        self.peak_diff_eff = peak_diff_eff
        self.diff_eff_avg = diff_eff_avg
        self.DE_peak_threshold = DE_peak_threshold
        self.DE_avg_threshold = DE_avg_threshold
    

    def calc_reward(
            self,
            a: float=5,
            b: float=10,
            filepath: str=None,
            DE_col: str=None,
            wavelength_col: str=None,
            comp_wavelength: float=1950
    ):
        reward = (
            a * (self.peak_diff_eff - self.DE_peak_threshold) +
            b * (self.diff_eff_avg - self.DE_avg_threshold) +
            (1 - self.norm_ne)
        )

        # Apply penalty for peak wavelength deviation if all required parameters are provided
        if all(v is not None for v in [filepath, DE_col, wavelength_col]):
            peak_wavelength = self.get_peak_wavelength(filepath, DE_col, wavelength_col)
            reward -= 0.3 * abs(peak_wavelength - comp_wavelength) / 100

        return reward
    

    def get_peak_wavelength(
            self,
            filepath: str,
            DE_col: str,
            wavelength_col: str
    ):
        # Retrieve data
        df = pd.read_csv(filepath)
        
        return df.loc[df[DE_col].idxmax(), wavelength_col] * 1000


def call_on_same_node(trial: int, params: dict, results_dir: str = 'Results'):
    result_fdtd, result_rcwa, DE_filename = run_both(trial, params, results_dir)

    res_dict = {
        "result_fdtd": result_fdtd,
        "result_rcwa": result_rcwa,
        "DE_filename": DE_filename
    }

    return res_dict