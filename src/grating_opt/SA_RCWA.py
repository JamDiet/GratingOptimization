import subprocess
import os, sys
import numpy as np
import pandas as pd


submission_script= 'sub_RCWA.pbs'
prototype_code = 'Prototype_RCWA.m'
result_filename = 'RCWA_result.csv'


def run_rcwa(trial: int, params: dict, local: bool=True):
    aoi = params["aoi"]
    dc = params["duty_cycle"]
    tp = params["pillar_thickness"]
    lmm = params["lines_per_mm"]
    tr = params["residual_thickness"]
    sa = params["slope_angle"]

    #Replace matlab code and copy here
    f = open('../' +prototype_code,'r')
    inputFile=f.read()
    f.close()

    newcode = inputFile.replace("<aoi>",str(aoi))		
    newcode = newcode.replace("<dc>",str(dc))
    newcode = newcode.replace("<tp>",str(tp))
    newcode = newcode.replace("<trial>",str(trial))
    newcode = newcode.replace("<tr>",str(tr))
    newcode = newcode.replace("<lmm>",str(lmm))
    newcode = newcode.replace("<sa>",str(sa))

    sim_file = f'code_rcwa_{trial}.m'

    f = open(sim_file,'w')
    f.write(newcode)
    f.close()

    result_file = f'RCWA_{trial}_result.csv'
    DE_filename = f'DE_vs_wavelength_{trial}.csv'

    # Determine if local or cluster job
    if local:
        path_cmd = "addpath(genpath('../../lib/RCWA'));"
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
            'tr': res['tr'] if res is not None else None,
            'lmm': res['lmm'] if res is not None else None,
            'sa': res['sa'] if res is not None else None,
            'DE_m1_peak': res['DE_m1_peak'] if res is not None else None,
            'DE_m1_avg': res['DE_m1_avg'] if res is not None else None
        }
    else:
        # Create a unique submission script for this trial
        f = open('../src/grating_opt/' +submission_script,'r',encoding='utf-8')
        inputFile=f.read()
        f.close()

        newcode = inputFile.replace("<sim_file>", str(sim_file))

        sub_file = f'sub_RCWA_{trial}.pbs'

        f = open(sub_file,'w',encoding='utf-8')
        f.write(newcode)
        f.close()

        cmd = 'sbatch ' + sub_file

        # Run sbatch, grab the Slurm Job ID from stdout, and return immediately
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        slurm_job_id = result.stdout.strip().split()[-1] 
        
        print(f"Trial {trial} submitted successfully to Slurm (Job ID: {slurm_job_id})")
        
        # Return the Job ID so poll_trial can monitor it
        return slurm_job_id, result_file, DE_filename