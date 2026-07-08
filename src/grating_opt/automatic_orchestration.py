import os
from typing import Any, Mapping

import pandas as pd
from ax.api.protocols.metric import IMetric
from ax.api.protocols.runner import IRunner, TrialStatus
from ax.api.types import TParameterization

from src.grating_opt.simulation_calls import call_on_same_node, Result
from src.grating_opt.utils import save_trial_data


class Runner(IRunner):
    def __init__(self, results_dir: str = 'Results', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_dir = results_dir

    def run_trial(
        self, trial_index: int, parameterization: TParameterization
    ) -> dict[str, Any]:
        
        res = call_on_same_node(
            trial=trial_index,
            params=parameterization,
            results_dir=self.results_dir
        )

        res["params"] = parameterization
        res["reward"] = None
        
        return res

    def poll_trial(
        self, trial_index: int, trial_metadata: Mapping[str, Any]
    ) -> TrialStatus:
        if trial_metadata["reward"] is not None:
            return TrialStatus.COMPLETED

        rcwa_res = trial_metadata["result_rcwa"]
        fdtd_res = trial_metadata["result_fdtd"]

        # 1. Check if the files exist yet
        if not (os.path.exists(rcwa_res) and os.path.exists(fdtd_res)):
            # The external job hasn't finished or written the files yet.
            # We tell Ax to keep waiting.
            return TrialStatus.RUNNING

        # 2. Optional: Check if the files are still being written to (are empty)
        if os.path.getsize(rcwa_res) == 0 or os.path.getsize(fdtd_res) == 0:
            return TrialStatus.RUNNING

        # 3. If the files exist and have data, the trial is done!
        return TrialStatus.COMPLETED


class Metric(IMetric):
    def __init__(
            self,
            crit_ne: float,
            csv_filename: str,
            a: float=5,
            b: float=10,
            DE_col: str=None,
            wavelength_col: str=None,
            comp_wavelength: float=1950,
            DE_peak_threshold: float=0.98,
            DE_avg_threshold: float=0.92,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.a = a
        self.b = b
        self.crit_ne = crit_ne
        self.csv_filename = csv_filename
        self.DE_col = DE_col
        self.wavelength_col = wavelength_col
        self.comp_wavelength = comp_wavelength
        self.DE_peak_threshold = DE_peak_threshold
        self.DE_avg_threshold = DE_avg_threshold

    def fetch(
        self,
        trial_index: int,
        trial_metadata: Mapping[str, Any],
    ) -> tuple[int, float | tuple[float, float]]:
        if trial_metadata["reward"] is not None:
            reward = trial_metadata["reward"]
        else:
            rcwa_res = trial_metadata["result_rcwa"]
            fdtd_res = trial_metadata["result_fdtd"]
            DE_filepath = trial_metadata["DE_filename"]

            rcwa_df = pd.read_csv(rcwa_res)
            fdtd_df = pd.read_csv(fdtd_res)

            res = Result(
                norm_ne=fdtd_df['ne_peak'].iloc[0] / self.crit_ne,
                peak_diff_eff=rcwa_df['DE_m1_peak'].iloc[0],
                diff_eff_avg=rcwa_df['DE_m1_avg'].iloc[0],
                DE_peak_threshold=self.DE_peak_threshold,
                DE_avg_threshold=self.DE_avg_threshold
            )

            reward = res.calc_reward(
                a=self.a,
                b=self.b,
                filepath=DE_filepath,
                DE_col=self.DE_col,
                wavelength_col=self.wavelength_col,
                comp_wavelength=self.comp_wavelength
            )

        save_trial_data(
            trial_index=trial_index,
            reward=reward,
            params=trial_metadata["params"],
            filename=self.csv_filename
        )

        return (0, reward)