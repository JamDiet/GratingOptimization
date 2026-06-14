import subprocess
import os

from src.grating_opt.utils import get_project_root


submission_script= 'sub_both.pbs'
prototype_fdtd = 'Prototype_FDTD.m'
prototype_rcwa = 'Prototype_RCWA.m'


def run_both(trial: int, params: dict):
    # Input parameters
    aoi = params["aoi"]
    dc = params["duty_cycle"]
    tp = params["pillar_thickness"]

    # Make result directories
    if not os.path.exists('Results'):
        os.mkdir('Results')

    #make new folder
    trial_dir = os.path.join(get_project_root(), 'Results', str(trial))
    os.makedirs(trial_dir, exist_ok=True)

    # Result files
    result_fdtd = os.path.join(trial_dir, 'FDTD_result.csv')
    result_rcwa = os.path.join(trial_dir, 'RCWA_result.csv')
    DE_filename = os.path.join(trial_dir, 'DE_vs_wavelength.csv')

    # ======= Replace matlab code and copy here ======= #

    # FDTD
    f = open(prototype_fdtd,'r',encoding='utf-8')
    inputFile=f.read()
    f.close()

    newcode = inputFile.replace("<aoi>",str(aoi))		
    newcode = newcode.replace("<dc>",str(dc))
    newcode = newcode.replace("<tp>",str(tp))
    newcode = newcode.replace("<trial>",str(trial))
    newcode = newcode.replace("<result_fdtd>", result_fdtd)

    FDTD_file = f'{trial_dir}/code_fdtd_{trial}.m'

    f = open(FDTD_file,'w',encoding='utf-8')
    f.write(newcode)
    f.close()

    # RCWA
    f = open(prototype_rcwa,'r',encoding='utf-8')
    inputFile=f.read()
    f.close()

    newcode = inputFile.replace("<aoi>",str(aoi))        
    newcode = newcode.replace("<dc>",str(dc))
    newcode = newcode.replace("<tp>",str(tp))
    newcode = newcode.replace("<trial>",str(trial))
    newcode = newcode.replace("<result_rcwa>", result_rcwa)
    newcode = newcode.replace("<DE_filename>", DE_filename)

    RCWA_file = f'code_rcwa_{trial}.m'

    f = open(RCWA_file,'w',encoding='utf-8')
    f.write(newcode)
    f.close()

    # 1. We allocate a "Job Step" container of 1 node and 2 cores dynamically from your pool.
    # 2. We then run both MATLAB scripts inside that specific container sequentially or simultaneously.
    
    # We wrap the execution in a single background shell command string
    # This guarantees that the 'salloc' step picks ONE node from your pool, 
    # and both srun commands execute inside that exact same picked node.
    cluster_cmd = (
        f"srun --ntasks=1 --cpus-per-task=40 --mpi=none matlab -batch \"run('{FDTD_file}')\" && "
        f"srun --ntasks=1 --cpus-per-task=1 --mpi=none matlab -batch \"run('{RCWA_file}')\""
    )
    subprocess.Popen(cluster_cmd, shell=True)
    
    return result_fdtd, result_rcwa, DE_filename