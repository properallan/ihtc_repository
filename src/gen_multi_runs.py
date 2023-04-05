from run_master import *
import subprocess
import numpy as np

def read_doe(doefile):
    doe = np.loadtxt(doefile, delimiter=',', skiprows=1)
    return doe

if __name__ == "__main__":
    # generate doe file
    #gen_lhs()
    # number of concurrent process

    N = args.nprocess
    
    doe = read_doe(DOEFILE)

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
            wf.writelines(lines[:-1])
            wf.writelines([ f"\tdoe_file = {DOEFILE}\n",
                            f"\tother_params = {OTHER_PARAMS}"
                            f"\tindex_range = {index_range}\n",
                            f"\tgen_dataset(nozzleRun, index_range, doe_file, other_params )"],)
    
    subprocess.call("./multirun.sh")