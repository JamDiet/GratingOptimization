import os
import time

from src.grating_opt.SA_both import run_both

results_dir = "prototype_results"
trial_idx = 0
params = {
    "dc": 0.5,
    "tp": 1375.,
    "tr": 100.,
    "lmm": 630,
    "aoi": 38.1847,
    "sa": 90.
}

result_fdtd, result_rcwa, DE_filename = run_both(trial_idx, params, results_dir)

# run_both launches the MATLAB jobs in the background and returns immediately,
# so poll for the result files (same convention as Runner.poll_trial in
# automatic_orchestration.py) to keep this process alive until they finish.
poll_interval = 30  # seconds
timeout = 4.5 * 3600  # stay under the job's 5:00:00 walltime
elapsed = 0

while elapsed < timeout:
    files_exist = os.path.exists(result_fdtd) and os.path.exists(result_rcwa)
    files_nonempty = files_exist and os.path.getsize(result_fdtd) > 0 and os.path.getsize(result_rcwa) > 0

    if files_nonempty:
        print(f"Simulations complete after {elapsed} seconds.")
        break

    time.sleep(poll_interval)
    elapsed += poll_interval
else:
    print(f"Timed out after {elapsed} seconds waiting for simulations to finish.")