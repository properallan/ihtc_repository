# %%
from doe_sampling import Space, Lhs, Path
import numpy as np
import pandas as pd
import logging

# %%

#def write_lhs_file(variable_ranges, samples, path, labels):
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

# %%
SAMPLES = 30
DATASET_ROOT = f'../data/doe_30/'
DOEFILE = f'{DATASET_ROOT}doe_lhs.txt'

DOE_VARIABLES = {
                #'T0in' : (285.0, 1115.0),
                 'Thickness' : (0.001, 0.010),
                #'p0in': (0.5e6, 1e6),
                 'CP3_y': (-0.01255805, 0.0),
                 }

OTHER_PARAMS = {
                'Nx': 210,
                'Ny': 330,
                'tol': 1e-8,
                'cores': None,
                'inflationRate': 1.0015,
                'baselineCP' : '/home/ppiper/Dropbox/local/ihtc_repository/src/baselineCP.txt',
                'metal': 'AISI406',
                'itmaxSU2': 4_000,
                'rootfile': DATASET_ROOT, 
}

# %%
write_lhs_file(DOE_VARIABLES, SAMPLES, DOEFILE)
