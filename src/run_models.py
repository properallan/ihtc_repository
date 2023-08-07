import subprocess
from subprocess import PIPE
import configparser
import os
import shutil
import sys
import numpy as np
import pathlib
from ast import literal_eval
from doe_sampling import write_lhs_file, write_doe_file

class CaseConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr):
        return optionstr

def read_doe(doefile):
    doe = np.loadtxt(doefile, delimiter=',', skiprows=1)
    return doe

def read_config_file(config_file):
    config = CaseConfigParser()
    config.read(config_file)

    # convert all path to the absolute
    for each_section in config.sections():
        for (each_key, each_val) in config.items(each_section):
            value = pathlib.Path(config[each_section][each_key])
            if value.is_file() or value.is_dir():
                config[each_section][each_key] = f"{value.resolve()}"
    return config

def doe_run(config_file):
    config = read_config_file(config_file)

    
    DATASET_ROOT = pathlib.Path(config['DATASET']['DATASET_ROOT']).resolve()
    DOEFILE = DATASET_ROOT /  pathlib.Path(config['DATASET']['DOEFILE'])

    DOE_VARIABLES = dict(config['DOE_VARIABLES'])

    for key, val in DOE_VARIABLES.items():
        DOE_VARIABLES[key] = literal_eval(val)
        print(key, DOE_VARIABLES[key])

    if config['DOE_SAMPLING']['FUNCTION'] == 'lhs':
        SAMPLES = int(config['DATASET']['SAMPLES'])
        write_lhs_file(DOE_VARIABLES, SAMPLES, DOEFILE)
    elif config['DOE_SAMPLING']['FUNCTION'] in ['fullfact', 'ff2n', 'ccdesign']:
        write_doe_file(config_file)
    else:
        print('unknown sampling method')
    
    
    N = int(config['MULTIRUN']['N'])
    #DOEFILE = pathlib.Path(config['DOE']['DOEFILE']).resolve()
    MULTIPLE_RUNS_PATH = f"./{pathlib.Path(config['MULTIRUN']['MULTIPLE_RUNS_PATH']).resolve().relative_to(os.getcwd())}"
    MASTER_FILE = pathlib.Path(config['MULTIRUN']['MASTER_FILE']).resolve()
    DATASET_ROOT = pathlib.Path(config['DATASET']['DATASET_ROOT']).resolve()

    config['HF_PARAMS']['GMSH_SOLVER'] = str( pathlib.Path(config['HF_PARAMS']['GMSH_SOLVER']).resolve())
    config['HF_PARAMS']['SU2_SOLVER'] = str( pathlib.Path(config['HF_PARAMS']['SU2_SOLVER']).resolve()) 
    config['LF_PARAMS']['EULER_Q1D_SOLVER'] = str( pathlib.Path(config['LF_PARAMS']['EULER_Q1D_SOLVER']).resolve() )


    HF_PARAMS = dict(config['HF_PARAMS'])
    LF_PARAMS = dict(config['LF_PARAMS'])

    HF_PARAMS['only_generate_mesh'] = False

    MODELS_PATH = pathlib.Path(config['MODELS']['models_file']).resolve()
    models_to_include = f"{MODELS_PATH.stem}"
    models_path_to_include = f"{MODELS_PATH.parents[0]}"

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
                                f"    sys.path.append('{os.getcwd()}')\n",
                                f"    sys.path.append('{models_path_to_include}')\n",
                                f'    from {models_to_include} import *\n'
        ])
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
                            f"    gen_dataset(lf_model, dataset_root, doe_file,index_range,  lf_params )\n"
                            f"    gen_dataset(hf_model, dataset_root, doe_file,index_range,  hf_params )\n",
                            ])
    
    subprocess.call(["./multirun.sh", f"{MULTIPLE_RUNS_PATH}"], shell=True)#, stdin=PIPE, stderr=PIPE, stdout=PIPE)

def doe_run_from_file(config_file):
    config = read_config_file(config_file)

    
    DATASET_ROOT = pathlib.Path(config['DATASET']['DATASET_ROOT']).resolve()
    DOEFILE = DATASET_ROOT /  pathlib.Path(config['DATASET']['DOEFILE'])
    
    #DOEFILE = pathlib.Path(config['DOE']['DOEFILE']).resolve()
    MULTIPLE_RUNS_PATH = f"./{pathlib.Path(config['MULTIRUN']['MULTIPLE_RUNS_PATH']).resolve().relative_to(os.getcwd())}"
    MASTER_FILE = pathlib.Path(config['MULTIRUN']['MASTER_FILE']).resolve()
    DATASET_ROOT = pathlib.Path(config['DATASET']['DATASET_ROOT']).resolve()

    config['HF_PARAMS']['GMSH_SOLVER'] = str( pathlib.Path(config['HF_PARAMS']['GMSH_SOLVER']).resolve())
    config['HF_PARAMS']['SU2_SOLVER'] = str( pathlib.Path(config['HF_PARAMS']['SU2_SOLVER']).resolve()) 
    config['LF_PARAMS']['EULER_Q1D_SOLVER'] = str( pathlib.Path(config['LF_PARAMS']['EULER_Q1D_SOLVER']).resolve() )


    HF_PARAMS = dict(config['HF_PARAMS'])
    LF_PARAMS = dict(config['LF_PARAMS'])

    HF_PARAMS['only_generate_mesh'] = False

    MODELS_PATH = pathlib.Path(config['MODELS']['models_file']).resolve()
    models_to_include = f"{MODELS_PATH.stem}"
    models_path_to_include = f"{MODELS_PATH.parents[0]}"

    doe = read_doe(DOEFILE)
    if len(doe.shape) == 1:
        doe = np.array([doe])
    list_of_index = doe[:,0]
    
    try:
        N = int(config['MULTIRUN']['N'])
    except:
        N = int(1)
    
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
                                f"    sys.path.append('{os.getcwd()}')\n",
                                f"    sys.path.append('{models_path_to_include}')\n",
                                f'    from {models_to_include} import *\n'
        ])
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
                            f"    gen_dataset(lf_model, dataset_root, doe_file,index_range,  lf_params )\n"
                            f"    gen_dataset(hf_model, dataset_root, doe_file,index_range,  hf_params )\n",
                            ])
    
    subprocess.call(["./multirun.sh", f"{MULTIPLE_RUNS_PATH}"], shell=True)#, stdin=PIPE, stderr=PIPE, stdout=PIPE)


def single_run(config_file):
    config = CaseConfigParser()
    config.read(config_file)

    try:
        if len(config['HF_PARAMS']) > 0:
            root_file = pathlib.Path(config['HF_PARAMS']['rootfile']).resolve()
            other_params = dict(config['HF_PARAMS'])
            hf_model(**other_params)
    except:
        print("no high fidelity model")

    try:
        if len(config['LF_PARAMS']) > 0 :
            root_file = pathlib.Path(config['LF_PARAMS']['rootfile']).resolve()
            other_params = dict(config['LF_PARAMS'])
            lf_model(**other_params)
    except:
        print("no low fidelity model")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--doeFile', type=str, default=False, help='configfile for running a DoE')
    parser.add_argument('-S', '--singleFile', type=str, default=False, help='configfile for single run')
    parser.add_argument('-N', '--nprocess', type=int, default=1, help='number of concurrent process')
    parser.add_argument('-M', '--models', type=str, default='models.py', help='file to find models implementation')


    args = parser.parse_args()

    if args.doeFile:
        doe_run(args.doeFile)

    if args.singleFile:
        exec(f'from {args.models.split(".")[0]} import *')
        single_run(args.singleFile)

