import os
import numpy as np
import pandas as pd
from src.grating_opt.SA_FDTD import run_fdtd
from src.grating_opt.SA_RCWA import run_rcwa
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
            trial_idx: int=None,
            comp_wavelength: float=1950
    ):
        main_term = (
            5 * (self.peak_diff_eff - 0.98) +
            10 * (self.diff_eff_avg - 0.92) -
            self.norm_ne
        )

        if include_penalties:
            DE_peak_penalty = a * max(0, self.DE_peak_threshold - self.peak_diff_eff) ** 2
            DE_avg_penalty = b * max(0, self.DE_avg_threshold - self.diff_eff_avg) ** 2
            ne_penalty = c * max(0, self.norm_ne - 1) ** 2
            penalties = DE_peak_penalty + DE_avg_penalty + ne_penalty
        else:
            penalties = 0

        if all(v is not None for v in [filename, DE_col, wavelength_col, trial_idx]):
            peak_wavelength = self.get_peak_wavelength(filename, DE_col, wavelength_col, trial_idx)
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


# def feasible_random(
#         trial: int,
#         params: dict,
#         crit_ne: float,
#         DE_peak_threshold: float=0.98,
#         DE_avg_threshold: float=0.92,
#         norm_ne_default: float=2.0,
#         local: bool=True
# ):
#     rng = np.random.default_rng()

#     # Generate feasible random results
#     peak_ne = rng.uniform(6e20, 1e21)
#     diff_eff = rng.uniform(0.93, 0.97)

#     return Result(peak_ne, diff_eff)


def call_simulations(
        trial: int,
        params: dict,
        crit_ne: float,
        DE_peak_threshold: float=0.98,
        DE_avg_threshold: float=0.92,
        norm_ne_default: float=1.5,
        local: bool=True
):
    res_rcwa = run_rcwa(trial, params, local)

    # Don't run FDTD if diffraction efficiency is unacceptable
    if res_rcwa['DE_m1_peak'] > DE_peak_threshold and res_rcwa['DE_m1_avg'] > DE_avg_threshold:
        res_fdtd = run_fdtd(trial, params, local)
        norm_ne = res_fdtd['ne_peak'] / crit_ne
    else:
        norm_ne = norm_ne_default

    return Result(
        norm_ne=norm_ne,
        peak_diff_eff=res_rcwa['DE_m1_peak'],
        diff_eff_avg=res_rcwa['DE_m1_avg'],
        DE_peak_threshold=DE_peak_threshold,
        DE_avg_threshold=DE_avg_threshold
    )


def call_sims_async(
        trial: int,
        params: dict,
        local: bool=False
):
    rcwa_id, rcwa_res, DE_filename = run_rcwa(trial, params, local)
    fdtd_id, fdtd_res = run_fdtd(trial, params, local)

    res_dict = {
        "rcwa_id": rcwa_id,
        "fdtd_id": fdtd_id,
        "rcwa_res": rcwa_res,
        "fdtd_res": fdtd_res,
        "DE_filename": DE_filename
    }

    return res_dict


def call_on_same_node(trial: int, params: dict, local: bool=False):
    result_fdtd, result_rcwa, DE_filename = run_both(trial, params)

    res_dict = {
        "result_fdtd": result_fdtd,
        "result_rcwa": result_rcwa,
        "DE_filename": DE_filename
    }

    return res_dict