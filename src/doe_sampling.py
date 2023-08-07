import numpy as np
np.random.seed(42)
import os
from pathlib import Path
import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Lhs
from scipy.spatial.distance import pdist
from explann_doe import fullfact, ff2n, ccdesign
import pandas as pd
import configparser

class CaseConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr):
        return optionstr

def config_to_dict(config):
    new_dict = {}
    for k, v in config._sections.items():
        new_dict[k] = {}
        for k_, v_ in v.items():
            try:
                new_dict[k][k_] = eval(v_)
            except:
                new_dict[k][k_] = v_
    return new_dict

def write_lhs_file(doe_variables, samples, path):
    variable_constants = []
    variable_variables = []
    variable_ranges = doe_variables.values()
    labels=','.join(['ID'] + [key for key in doe_variables.keys()])
    
    for var in variable_ranges:
        if not isinstance(var, tuple):
            variable_constants.append(var)
        elif var[0] == var[1]:
            variable_constants.append(var[0])
        else:
            variable_variables.append(var)

    space = Space(variable_variables)
    lhs = Lhs(criterion="maximin", iterations=10000)
    x = lhs.generate(space.dimensions, samples)

    for xi in x:
        [xi.append(var) for var in variable_constants]

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(f'{labels}\n')
        for i in range(len(x)):
            f.write(f'{i+1}')
            for j in range(len(np.array(x)[i,:])):
                f.write(f', {np.array(x)[i,j]}')
            f.write('\n')
    print(path)

def write_doe_file(config_file):

    config = CaseConfigParser()
    config.read(config_file)
    config = config_to_dict(config)

    
    DOE_VARIABLES = config['DOE_VARIABLES']
    #SAMPLES = config['DATASET']['SAMPLES']
    DOEFILE = config['DATASET']['DOEFILE']
    DATASET_ROOT = config['DATASET']['DATASET_ROOT']
    DOEFILE_PATH = f"{Path(DATASET_ROOT) / Path(DOEFILE).resolve()}"
    os.makedirs(Path(DOEFILE_PATH).resolve().parent, exist_ok=True)

    NVARS = config['DOE_VARIABLES'].__len__()
    ### ccc stands for central composite circunscribed
    DOE_FUNCTION = config['DOE_SAMPLING']['FUNCTION']
    KWARGS = config['DOE_SAMPLING_KWARGS']

    experimental_planning = DOE_FUNCTION(NVARS, **KWARGS )

    levels = np.unique(experimental_planning)

    DOE_VARIABLES_LEVELS = {}
    for var, var_range in DOE_VARIABLES.items():
        DOE_VARIABLES_LEVELS[var] = np.interp(levels, (levels.min(), levels.max()), var_range)

    np.set_printoptions(precision=5, suppress=True)

    converted_levels = np.array([array for array in DOE_VARIABLES_LEVELS.values()])
    
    experimental_planning_converted = np.copy(experimental_planning)
    for i in range(experimental_planning.shape[1]):
        experimental_planning_converted[:,i] = experimental_planning[:,i]*(converted_levels[i][-2]-np.mean(converted_levels[i])) + np.mean(converted_levels[i])

    experimental_planning_converted_ids = np.concatenate([np.arange(1,experimental_planning_converted.shape[0]+1)[:,None],experimental_planning_converted], axis=1)
    experimental_planning_converted_ids_dataframe = pd.DataFrame(data=experimental_planning_converted_ids, columns=['ID','Thickness','CP3_y','T0in','p0in','outerTemperature'])
    experimental_planning_converted_ids_dataframe.to_csv(DOEFILE_PATH, sep=',', index=False)
    
