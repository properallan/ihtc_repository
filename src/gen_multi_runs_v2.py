import subprocess
import configparser
import os
import shutil
import sys
import numpy as np
import pathlib

class CaseConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr):
        return optionstr

def read_doe(doefile):
    doe = np.loadtxt(doefile, delimiter=',', skiprows=1)
    return doe

def doe_run(config_file):
    config = CaseConfigParser()
    config.read(config_file)

    DOEFILE = config['DOE']['DOEFILE']
    
    N = int(config['MULTIRUN']['N'])
    DOEFILE = config['DOE']['DOEFILE']
    MULTIPLE_RUNS_PATH = f"./{pathlib.Path(config['MULTIRUN']['MULTIPLE_RUNS_PATH']).absolute().relative_to(os.getcwd())}"
    MASTER_FILE = pathlib.Path(config['MULTIRUN']['MASTER_FILE']).absolute()
    DATASET_ROOT = pathlib.Path(config['DATASET']['DATASET_ROOT']).absolute()

    config['HF_PARAMS']['GMSH_SOLVER'] = str( pathlib.Path(config['HF_PARAMS']['GMSH_SOLVER']).absolute())
    config['HF_PARAMS']['SU2_SOLVER'] = str( pathlib.Path(config['HF_PARAMS']['SU2_SOLVER']).absolute()) 
    config['LF_PARAMS']['EULER_Q1D_SOLVER'] = str( pathlib.Path(config['LF_PARAMS']['EULER_Q1D_SOLVER']).absolute() )

    HF_PARAMS = dict(config['HF_PARAMS'])
    LF_PARAMS = dict(config['LF_PARAMS'])

    doe = read_doe(DOEFILE)
    if len(doe.shape) == 1:
        doe = np.array([doe])
    list_of_index = doe[:,0]
    index_ranges = np.array_split(list_of_index, N)
  
    try:
        os.makedirs(MULTIPLE_RUNS_PATH, exist_ok=False)
    except:
        shutil.rmtree(MULTIPLE_RUNS_PATH)
        os.makedirs(MULTIPLE_RUNS_PATH, exist_ok=False)
  
    for i, index_range in enumerate(index_ranges):
        index_range = [int(val) for val in list(index_range)]

        with open(f'{MULTIPLE_RUNS_PATH}/run_{i+1}.py','w') as wf, open(f'{MASTER_FILE}', 'r') as rf:
            lines = rf.readlines()

            for line in lines:
                wf.write(line)
                if "__main__" in line:
                    wf.writelines([
                                f"    import sys\n",
                                f"    import os\n",
                                f"    sys.path.append('{os.getcwd()}')\n",])
            #wf.write("\n")
            #wf.writelines([            
            #    f"    setGmsh('{GMSH_SOLVER}')\n",
            #    f"    SU2_SOLVER = '{SU2_SOLVER}'\n"])

            wf.write("\n")
            wf.writelines([ f"    doe_file = '{DOEFILE}'\n",
                            f"    hf_params = {HF_PARAMS}\n",
                            f"    lf_params = {LF_PARAMS}\n"
                            f"    index_range = {index_range}\n",
                            f"    dataset_root = '{DATASET_ROOT}'\n"
                            f"    gen_dataset(hf_model, dataset_root, doe_file,index_range,  hf_params )\n",
                            f"    gen_dataset(lf_model, dataset_root, doe_file,index_range,  lf_params )\n"
                            ])
    
    subprocess.call(["./multirun.sh", f"{MULTIPLE_RUNS_PATH}"])

def single_run(config_file):
    config = CaseConfigParser()
    config.read(config_file)

    if len(config['HF_PARAMS']) > 0:
        root_file = config['HF_PARAMS']['rootfile']
        other_params = dict(config['HF_PARAMS'])
        hf_model(**other_params)

    try:
        if len(config['HF_PARAMS']) > 0:
            root_file = config['HF_PARAMS']['rootfile']
            other_params = dict(config['HF_PARAMS'])
            hf_model(**other_params)
    except:
        print("no high fidelity model")

    try:
        if len(config['LF_PARAMS']) > 0 :
            root_file = config['LF_PARAMS']['rootfile']
            other_params = dict(config['LF_PARAMS'])
            lf_model(**other_params)
    except:
        print("no low fidelity model")

if __name__ == "__main__":
    import argparse
    from models import *

    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--doeFile', type=str, default=False, help='configfile for running a DoE')
    parser.add_argument('-S', '--singleFile', type=str, default=False, help='configfile for single run')
    parser.add_argument('-N', '--nprocess', type=int, default=1, help='number of concurrent process')

    args = parser.parse_args()

    if args.doeFile:
        doe_run(args.doeFile)

    if args.singleFile:
        single_run(args.singleFile)

