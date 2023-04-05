import numpy as np
np.random.seed(42)
import os
from pathlib import Path
import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Lhs
from scipy.spatial.distance import pdist


def write_lhs_file(variable_ranges, samples, path):
    space = Space(variable_ranges)
    lhs = Lhs(criterion="maximin", iterations=10000)
    x = lhs.generate(space.dimensions, samples)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write('ID, p0in, T0in, thickness\n')
        for i in range(len(x)):
            f.write(f'{i+1}, {np.array(x)[i,0]}, {np.array(x)[i,1]}, {np.array(x)[i,2]}\n')
    return 0
