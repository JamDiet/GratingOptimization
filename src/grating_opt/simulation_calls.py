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
            a: float,
            b: float,
            c: float,
            include_penalties: bool=False,
            filename: str=None,
            DE_col: str=None,
            wavelength_col: str=None,
            comp_wavelength: float=1950
    ):
        main_term = (
            5 * (self.peak_diff_eff - 0.98) +
            10 * (self.diff_eff_avg - 0.92) +
            (1 - self.norm_ne)
        )

        if include_penalties:
            DE_peak_penalty = a * max(0, self.DE_peak_threshold - self.peak_diff_eff) ** 2
            DE_avg_penalty = b * max(0, self.DE_avg_threshold - self.diff_eff_avg) ** 2
            ne_penalty = c * max(0, self.norm_ne - 1) ** 2
            penalties = DE_peak_penalty + DE_avg_penalty + ne_penalty
        else:
            penalties = 0

        if all(v is not None for v in [filename, DE_col, wavelength_col]):
            peak_wavelength = self.get_peak_wavelength(filename, DE_col, wavelength_col)
            penalties += 0.3 * abs(peak_wavelength - comp_wavelength) / 100

        return main_term - penalties
    

    def get_peak_wavelength(
            self,
            filename: str,
            DE_col: str,
            wavelength_col: str
    ):
        # Retrieve data
        df = pd.read_csv(filename)
        
        return df.loc[df[DE_col].idxmax(), wavelength_col] * 1000


def call_on_same_node(trial: int, params: dict, local: bool=False):
    result_fdtd, result_rcwa, DE_filename = run_both(trial, params)

    res_dict = {
        "result_fdtd": result_fdtd,
        "result_rcwa": result_rcwa,
        "DE_filename": DE_filename
    }

    return res_dict