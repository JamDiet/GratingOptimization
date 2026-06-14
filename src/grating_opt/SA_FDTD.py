import subprocess
import os
import pandas as pd


submission_script= 'sub_FDTD.pbs'
prototype_code = 'Prototype_FDTD.m'
result_filename = 'FDTD_result.csv'


def run_fdtd(trial: int, params: dict, local: bool=True):
    aoi = params["aoi"]
    dc = params["duty_cycle"]
    tp = params["pillar_thickness"]

    #Replace matlab code and copy here
    f = open('../' +prototype_code,'r',encoding='utf-8')
    inputFile=f.read()
    f.close()

    newcode = inputFile.replace("<aoi>",str(aoi))		
    newcode = newcode.replace("<dc>",str(dc))
    newcode = newcode.replace("<tp>",str(tp))
    newcode = newcode.replace("<trial>",str(trial))

    sim_file = f'code_fdtd_{trial}.m'

    f = open(sim_file,'w',encoding='utf-8')
    f.write(newcode)
    f.close()

    result_file = f'FDTD_{trial}_result.csv'

    # Determine if local or cluster job
    if local:
        path_cmd = "addpath(genpath('../../lib/FDTD'));"
        run_cmd = f"run('{sim_file}');"
        cmd = f"matlab -batch \"{path_cmd} {run_cmd} exit;\""

        # Call simulation
        subprocess.run(cmd, shell=True, check=True) #Switch to subprocess

        res = pd.read_csv(result_file).iloc[0] if os.path.exists(result_file) else None

        print('SUCCESS')

        return {
            'trial': trial,
            'aoi': res['aoi'] if res is not None else None,
            'tp': res['tp'] if res is not None else None,
            'dc': res['dc'] if res is not None else None,
            'ne_peak': res['ne_peak'] if res is not None else None
        }
    else:
        # Create a unique submission script for this trial
        f = open('../src/grating_opt/' +submission_script,'r',encoding='utf-8')
        inputFile=f.read()
        f.close()

        newcode = inputFile.replace("<sim_file>", str(sim_file))

        sub_file = f'sub_FDTD_{trial}.pbs'

        f = open(sub_file,'w',encoding='utf-8')
        f.write(newcode)
        f.close()

        cmd = 'sbatch ' + sub_file

        # Run sbatch, grab the Slurm Job ID from stdout, and return immediately
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        slurm_job_id = result.stdout.strip().split()[-1] 
        
        print(f"Trial {trial} submitted successfully to Slurm (Job ID: {slurm_job_id})")
        
        # Return the Job ID so poll_trial can monitor it
        return slurm_job_id, result_file