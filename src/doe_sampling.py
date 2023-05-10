import numpy as np
np.random.seed(42)
import os
from pathlib import Path
import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Lhs
from scipy.spatial.distance import pdist


#def write_lhs_file(variable_ranges, samples, path):
#    space = Space(variable_ranges)
##    lhs = Lhs(criterion="maximin", iterations=10000)
#    x = lhs.generate(space.dimensions, samples)
#    p = Path(path)
#    p.parent.mkdir(parents=True, exist_ok=True)
#    with open(path, 'w') as f:
#        f.write('ID, p0in, T0in, thickness\n')
#        for i in range(len(x)):
#            f.write(f'{i+1}, {np.array(x)[i,0]}, {np.array(x)[i,1]}, {np.array(x)[i,2]}\n')
#    return 0

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