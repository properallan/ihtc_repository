from models_parametric_shape import hf_model
import pathlib

hf_params = dict(
    Nx =  210,
    Ny = 330,
    tol = 1e-8,
    cores = None,
    inflationRate =  1.0015,
    metal = 'AISI406',
    itmaxSU2 = 2,
    GMSH_SOLVER = './gmsh',
    SU2_SOLVER = './SU2_CFD',
    baselineCP = './baselineCP.txt',
    T0in = 600.0,
    p0in = 8.0e5,
    itprintSU2 = 1,
    CP3_y =  -0.0024504689702769863,
    Thickness = 0.004209215007771409,
    rootfile = '../data/single_run/'
)

hf_model( **hf_params)
#single_run(**run_config)