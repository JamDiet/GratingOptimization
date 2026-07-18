import subprocess
from pathlib import Path


def run_both(experiment_root: Path, results_root: Path, trial, params: dict):
    matlab_dir = Path(experiment_root) / "matlab_code"
    prototype_fdtd = matlab_dir / "Prototype_FDTD.m"
    prototype_rcwa = matlab_dir / "Prototype_RCWA.m"

    trial_dir = Path(results_root) / str(trial)
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Result files
    result_fdtd = trial_dir / 'FDTD_result.csv'
    result_rcwa = trial_dir / 'RCWA_result.csv'
    DE_filename = trial_dir / 'DE_vs_wavelength.csv'

    # ======= Replace matlab code and copy here ======= #

    # FDTD
    newcode = prototype_fdtd.read_text(encoding='utf-8')

    # Replace placeholders in the prototype code with actual parameter values
    for k, v in params.items():
        newcode = newcode.replace(f"<{k}>", str(v))

    newcode = newcode.replace("<result_fdtd>", str(result_fdtd))

    FDTD_file = trial_dir / f'code_fdtd_{trial}.m'
    FDTD_file.write_text(newcode, encoding='utf-8')

    # RCWA
    newcode = prototype_rcwa.read_text(encoding='utf-8')

    # Replace placeholders in the prototype code with actual parameter values
    for k, v in params.items():
        newcode = newcode.replace(f"<{k}>", str(v))

    newcode = newcode.replace("<result_rcwa>", str(result_rcwa))
    newcode = newcode.replace("<DE_filename>", str(DE_filename))

    RCWA_file = trial_dir / f'code_rcwa_{trial}.m'
    RCWA_file.write_text(newcode, encoding='utf-8')

    # 1. We allocate a "Job Step" container of 1 node and 2 cores dynamically from your pool.
    # 2. We then run both MATLAB scripts inside that specific container sequentially or simultaneously.

    # We wrap the execution in a single background shell command string
    # This guarantees that the 'salloc' step picks ONE node from your pool,
    # and both srun commands execute inside that exact same picked node.
    cluster_cmd = (
        f"srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 --mpi=none matlab -batch \"run('{RCWA_file}')\" && "
        f"srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=40 --mpi=none matlab -batch \"run('{FDTD_file}')\""
    )
    subprocess.Popen(cluster_cmd, shell=True)

    return str(result_fdtd), str(result_rcwa), str(DE_filename)
