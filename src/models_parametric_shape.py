import numpy as np
from solvers_2full import *
from fluid_constants import *
from bezier import find_control_points, bezier_curve
import pathlib


def gen_snapshot(solution_path, variable_getter, variables, design_variables=None):
    data = {}
    for dvariable in design_variables:
        data[dvariable] = design_variables[dvariable]
    for variable_name in variables:
        variable_data = variable_getter(solution_path, variable_name)
        data[variable_name][..., id-1] = variable_data


def get_q1d_variable(solution_path, variable_name):
    solution_path = pathlib.Path(solution_path)
    solution_subpath = solution_path / 'Q1D' / 'outputs'
    name_to_variable_conversion = {
        'Pressure': np.loadtxt(solution_subpath / 'p.txt'),
        'Temperature': np.loadtxt(solution_subpath / 'T.txt'),
        'Mach': np.loadtxt(solution_subpath / 'M.txt'),
    }
    return name_to_variable_conversion[variable_name]


def get_block_recursive(block, name_index):
    for n in name_index:
        block = block[n]
    return block


def get_su2_variable(solution_path, variable_name):
    solution_path = pathlib.Path(solution_path)
    solution_subpath = solution_path / 'SU2' / 'outputs' / "cht_setupSU2.vtm"
    block = pv.read(solution_subpath)
    name_to_variable_conversion = {
        'Pressure': get_block_recursive(block,
                                        ['Zone 0 (Comp. Fluid)', 'Internal', 'Internal', 'Pressure']),
        'Temperature': get_block_recursive(block,
                                           ['Zone 0 (Comp. Fluid)', 'Internal', 'Internal', 'Temperature']),
        'Mach': get_block_recursive(block,
                                    ['Zone 0 (Comp. Fluid)', 'Internal', 'Internal', 'Mach']),
        'Temperature_Solid': get_block_recursive(block,
                                                 ['Zone 1 (Solid Heat)', 'Internal', 'Internal', 'Temperature']),
        'Temperature_Solid_INNERWALL': get_block_recursive(block,
                                                           ['Zone 1 (Solid Heat)', 'Boundary', 'INNERWALL', 'Temperature']),
        'Heat_Flux_UPPER_WALL': get_block_recursive(block,
                                                    ['Zone 0 (Comp. Fluid)', 'Boundary', 'UPPER_WALL', 'Heat_Flux']),
    }

    return name_to_variable_conversion[variable_name]


def hf_model(rootfile, **arg ):
    rootfile = pathlib.Path(rootfile).absolute()
    rootfile = f"{rootfile}"
    # xr = np.loadtxt(arg['nozzle_shape'], delimiter=',')

    xr_cp = np.loadtxt(arg['baselineCP'], delimiter=',')
    xr_cp[2, 1] = float(arg['CP3_y'])

    xr = bezier_curve(xr_cp, num_points=1000)

    x = xr[:, 0]
    r = xr[:, 1]
    A = np.pi*r**2

    n = nozzle()
    n.setX(x)
    n.setS(A)

    T0in = float(arg['T0in'])
    p0in = float(arg['p0in'])

    pr = 101e3/p0in
    Min = 0.01
    n.setBC(p0in, T0in, Min, pr*p0in)

    # Fluid properties
    fluid = 'Air'
    gamma, R, Tcrit, pcrit, acentricFactor = getFLuidProperties(
        p0in, T0in, fluid)
    n.R = R
    n.gamma = gamma
    # STANDARD_AIR, IDEAL_GAS, VW_GAS, PR_GAS, CONSTANT_DENSITY, INC_IDEAL_GAS, INC_IDEAL_GAS_POLY
    n.fluidModel = 'STANDARD_AIR'
    n.criticalTemperature = Tcrit
    n.criticalPressure = pcrit
    n.acentricFactor = acentricFactor

    n.turbulenceModel = 'SST'
    n.viscosityModel = 'SUTHERLAND'
    n.sutherlandViscosity = sutherland_viscosity[fluid]
    n.sutherlandTemperature = sutherland_temperature[fluid]
    n.sutherlandConstant = sutherland_constant[fluid]

    # CONSTANT_CONDUCTIVITY, CONSTANT_PRANDTL
    n.conductivityModel = 'CONSTANT_PRANDTL'
    n.laminarPrandtl = laminar_prandtl[fluid]
    n.turbulentPrandtl = turbulent_prandtl[fluid]

    n.outerTemperature = 300.0
    n.thickness = float(arg['Thickness'])

    # n.outerTemperature = 300.0
    # n.thickness = 0.0074
    # AISI406
    if arg['metal'] == 'AISI406':
        n.solidDensity = 7800.0  # kg/m^3
        n.solidHeatCP = 460.0  # 504.0 #J/kg*K
        n.solidConductivity = 27.0  # 15.2 # W/m*K

    # AISI302

    if arg['metal'] == 'AISI302':
        n.solidDensity = 8055.0  # kg/m^3
        n.solidHeatCP = 512.0  # J/kg*K
        n.solidConductivity = 17.3  # W/m*K

    itmaxSU2 = int(arg['itmaxSU2'])
    itprintSU2 = int(arg['itprintSU2'])
    CFLSU2 = 1000

    tolSU2 = np.log10(float(arg['tol']))
    tschemeSU2 = 'EULER_IMPLICIT'  # 'EULER_IMPLICIT'
    fschemeSU2 = 'ROE'
    dimSU2 = 'DIMENSIONAL'  # 'FREESTREAM_PRESS_EQ_ONE' #'DIMENSIONAL'

    n.NxWall = int(arg['Nx'])
    n.NyWall = int(arg['Nx'])/2

    n.NxSU2 = int(arg['Nx'])
    n.NySU2 = int(arg['Ny'])
    n.inflationRate = float(arg['inflationRate'])
    setGmsh(arg['GMSH_SOLVER'])

    if arg['only_generate_mesh'] == True:
        # to generate the mesh run for just 1 iteration
        itmaxSU2 = 1
    n.setupSU2Solver(itmaxSU2, itprintSU2, CFLSU2, tolSU2,
                     tschemeSU2, fschemeSU2, dimSU2)
    n.setupSU2CHT(rootfile + '/SU2/', 'setupSU2.cfg')
    n.solveSU2CHT(arg['SU2_SOLVER'], cores=eval(f"{arg['cores']}"))

    with open(n.su2outfilepath+'solver.log', 'w') as f:
        f.write(f"runtime={n.runtimeSU2}\n")
        f.write(f"p0in={n.p0in}\n")
        f.write(f"T0in={n.T0in}\n")
        f.write(f"wallThicknes={n.thickness}\n")
        f.write(f"converged={n.su2_converged}\n")

    return n

def lf_model(rootfile, **arg):
    # rootfile = pathlib.Path(rootfile)
    # xr = np.loadtxt(arg['nozzle_shape'], delimiter=',')
    xr_cp = np.loadtxt(arg['baselineCP'], delimiter=',')
    xr_cp[2, 1] = float(arg['CP3_y'])

    xr = bezier_curve(xr_cp, num_points=1000)

    x = xr[:, 0]
    r = xr[:, 1]
    A = np.pi*r**2

    n = nozzle()
    n.setX(x)
    n.setS(A)

    T0in = float(arg['T0in'])
    p0in = float(arg['p0in'])

    pr = 101e3/p0in
    Min = 0.01
    n.setBC(p0in, T0in, Min, pr*p0in)

    fluid = 'Air'
    gamma, R, Tcrit, pcrit, acentricFactor = getFLuidProperties(
        p0in, T0in, fluid)
    n.R = R
    n.gamma = gamma

    itmaxQ1D = 50000
    NxQ1D = 100
    itprintQ1D = 1000
    CFLQ1D = 0.15
    tolQ1D = 1e-8
    tschemeQ1D = 'RK4'
    fschemeQ1D = 'AUSM'
    dttypeQ1D = 'Global'
    dimQ1D = 'Dimensionless'

    n.NxQ1D = int(arg['Nx'])
    n.setupQ1DSolver(itmaxQ1D, itprintQ1D, CFLQ1D, tolQ1D,
                     tschemeQ1D, fschemeQ1D, dttypeQ1D, dimQ1D)
    n.setupQ1D(rootfile + 'Q1D/', 'setupQ1D.txt')
    n.solveQ1D(arg['EULER_Q1D_SOLVER'])

    with open(n.q1doutfilepath+'solver.log', 'w') as f:
        f.write(f"runtime={n.runtimeQ1D}\n")
        f.write(f"p0in={n.p0in}\n")
        f.write(f"T0in={n.T0in}\n")

    return n