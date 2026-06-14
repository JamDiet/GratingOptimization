from xmlrpc import client

from ax.api.client import Client
from tqdm import tqdm
from src.grating_opt.acquisition_fcns import ucb
import os
from src.grating_opt.utils import get_data_root
from ax.service.utils.report_utils import exp_to_df
import numpy as np
from src.grating_opt.automatic_orchestration import Runner, Metric

class Optimizer:
    def __init__(
            self,
            crit_ne: float,
            local: bool,
            a: float=0.0,
            b: float=0.0,
            c: float=0.0,
            DE_peak_threshold: float=0.98,
            DE_avg_threshold: float=0.92,
            norm_ne_default: float=1.5,
            comp_wavelength: float=1950,
            func=None,
            filename: str=None,
            DE_col: str=None,
            wavelength_col: str=None
    ):
        """
        :param func: Function with calls to FDTD and RCWA simulations.
            Returns a Result object.
        :param a: Weight parameter for peak effective index in reward calculation
        :type a: float
        :param b: Weight parameter for diffraction efficiency in reward calculation
        :type b: float
        :param c: Weight parameter for average effective index in reward calculation
        :type c: float
        :param crit_ne: Critical refractive index for optimization
        :type crit_ne: float
        :param DE_peak_threshold: Threshold for peak diffraction efficiency (default: 0.98)
        :type DE_peak_threshold: float
        :param DE_avg_threshold: Threshold for average diffraction efficiency (default: 0.92)
        :type DE_avg_threshold: float
        :param norm_ne_default: Default value for refractive index normalization (default: 2.0)
        :type norm_ne_default: float
        :param local: Flag to run locally
        :type local: bool
        :param filename: Name of DE vs wavelength CSV file
        :type filename: str
        :param DE_col: DE column name
        :type DE_col: str
        :param wavelength_col: Wavelength column name
        :type wavelength_col: str
        :param comp_wavelength: Wavelength against which to compare
        :type comp_wavelength: float
        """
        self.func = func
        self.crit_ne = crit_ne
        self.DE_peak_threshold = DE_peak_threshold
        self.DE_avg_threshold = DE_avg_threshold
        self.norm_ne_default = norm_ne_default
        self.local = local
        self.a = a
        self.b = b
        self.c = c
        self.filename = filename
        self.DE_col = DE_col
        self.wavelength_col = wavelength_col
        self.comp_wavelength = comp_wavelength


    def run_trial_sim(
            self,
            trial_index: int,
            parameters: dict
    ):
        # Call simulations
        res = self.func(
            trial=trial_index,
            params=parameters,
            crit_ne=self.crit_ne,
            DE_peak_threshold=self.DE_peak_threshold,
            DE_avg_threshold=self.DE_avg_threshold,
            norm_ne_default=self.norm_ne_default,
            local=self.local
        )

        reward = res.calc_reward(
            a=self.a,
            b=self.b,
            c=self.c,
            filename=self.filename,
            DE_col=self.DE_col,
            wavelength_col=self.wavelength_col,
            trial_idx=trial_index,
            comp_wavelength=self.comp_wavelength
        )

        return trial_index, reward


    def optimize_params(
            self,
            params: list,
            num_trials: int,
            seed_trials: int=10,
            kappa: float=2.0,
            max_trials: int=1,
            preexisting_trials: list=None,
            csv_filename: str=None
    ):
        """
        :param params: List of RangeParameterConfigs for optimizable parameters
        :type params: list
        :param num_trials: Number of optimization trials
        :type num_trials: int
        :param seed_trials: Number of seed trials for acquisition function (default: 10)
        :type seed_trials: int
        :param kappa: Kappa parameter for UCB acquisition function (default: 2.0)
        :type kappa: float
        :param max_trials: Maximum trials to run in parallel (default: 1)
        :type max_trials: int
        :param preexisting_trials: List of tuples containing (parameters, data) for pre-existing trials
        :type preexisting_trials: list
        :param csv_filename: Name of CSV file to save trial data (will be placed in data/csvs directory)
        :type csv_filename: str
        :return: The Ax client used for the optimization
        :rtype: ax.api.client.Client
        """
        # Ensure max_trials is 1 for local run
        if self.local:
            max_trials = 1
        else:
           from mpi4py.futures import MPIPoolExecutor
           executor = MPIPoolExecutor(max_workers=max_trials)

        # Initialize client
        client = Client()

        # Configure search space
        client.configure_experiment(params)

        # Configure objective
        client.configure_optimization("reward")

        # Configure acquisition function
        client.set_generation_strategy(generation_strategy=ucb(seed_trials, kappa))

        # Track best values for visualization
        best_values = []
        all_values = []
        current_best = float("-inf")

        if csv_filename is not None:
            # Construct full path in data/csvs directory using utility function
            full_path = get_data_root() / "csvs" / csv_filename
            # Ensure directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)

        # Load preexisting data
        if preexisting_trials is not None:
            for parameters, data in preexisting_trials:
                # Attach the parameterization to the Client as a trial and immediately complete it with the preexisting data
                trial_index = client.attach_trial(parameters=parameters)
                client.complete_trial(trial_index=trial_index, raw_data=data)

                # Log data
                all_values.append(data["reward"])

                if data["reward"] > current_best:
                    current_best = data["reward"]
                
                best_values.append(current_best)

        # Conduct experiment with num_trials
        for _ in tqdm(range(num_trials)):
            trial_params = np.array(list(client.get_next_trials(max_trials=max_trials).items()))

            if self.local:
                trials_result = [self.run_trial_sim(*trial_params[0])]
            else:
                trials_result = executor.map(self.run_trial_sim, trial_params)

                if max_trials == 1:
                    trials_result = [trials_result]

            
            for trial_index, reward in trials_result:
                client.complete_trial(trial_index=trial_index, raw_data={"reward": reward})

                # Log data
                all_values.append(reward)

                if reward > current_best:
                    current_best = reward
                
                best_values.append(current_best)

                if csv_filename is not None:
                    # client.get_trials_data_frame() aggregates parameters and metrics
                    df_step = exp_to_df(client._experiment)
                    
                    # To save file overhead, filter for just the current trial index
                    df_trial = df_step[df_step["trial_index"] == trial_index]
                    
                    # Append to CSV (write header only if the file doesn't exist yet)
                    file_exists = os.path.exists(full_path)
                    df_trial.to_csv(full_path, mode='a', index=False, header=not file_exists)

        return client, all_values, best_values


    def opt_async(
            self,
            params: list,
            num_trials: int,
            seed_trials: int=10,
            kappa: float=2.0,
            parallelism: int=1,
            preexisting_trials: list=None,
    ):
        """
        Asynchronous version of optimize_params. Trials are submitted to the cluster and then polled for completion.
        :param params: List of RangeParameterConfigs for optimizable parameters
        :type params: list
        :param num_trials: Number of optimization trials
        :type num_trials: int
        :param seed_trials: Number of seed trials for acquisition function (default: 10)
        :type seed_trials: int
        :param kappa: Kappa parameter for UCB acquisition function (default: 2.0)
        :type kappa: float
        :param parallelism: Number of trials to run in parallel (default: 1)
        :type parallelism: int
        :param preexisting_trials: List of tuples containing (parameters, data) for pre-existing trials
        :type preexisting_trials: list
        :return: The Ax client used for the optimization
        :rtype: ax.api.client.Client
        """
        # Initialize asynchronous runner and metric
        runner = Runner(local=self.local)
        metric = Metric(
            name="reward",
            a=self.a,
            b=self.b,
            c=self.c,
            crit_ne=self.crit_ne,
            include_penalties=False,
            DE_col=self.DE_col,
            wavelength_col=self.wavelength_col,
            comp_wavelength=self.comp_wavelength
        )

        # Congigure client
        client = Client()
        client.configure_experiment(parameters=params)
        client.configure_optimization(objective="reward")
        client.configure_runner(runner=runner)
        client.configure_metrics(metrics=[metric])
        client.set_generation_strategy(generation_strategy=ucb(seed_trials, kappa))

        # Load preexisting data
        if preexisting_trials is not None:
            for parameters, data in preexisting_trials:
                # Attach the parameterization to the Client as a trial and immediately complete it with the preexisting data
                trial_index = client.attach_trial(parameters=parameters)
                client.complete_trial(trial_index=trial_index, raw_data=data)

        # Run trials in parallel
        client.run_trials(
            max_trials=num_trials,
            parallelism=parallelism,
            tolerated_trial_failure_rate=0.1,
            initial_seconds_between_polls=30,
        )

        return client