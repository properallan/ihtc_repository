def gen_dataset(model, dataset_root,  doe_file, index_range, other_params):
    doe = pd.read_csv(fr'{doe_file}')
    doe.index += 1

    if index_range is None:
        index_range = doe.index
        
    doe_lhs = np.loadtxt(doe_file, delimiter=',', skiprows=1)

    for id, row in doe.iterrows():
        if id in index_range:
            model( rootfile = f"{dataset_root}/{int(id)}/", 
                        **dict(row), **other_params)
                                     
if __name__ == "__main__":
    import sys
    import os
    sys.path.append('/home/ppiper/ihtc_repository/src')
    sys.path.append('/home/ppiper/ihtc_repository/src')
    from models_more_design_variables import *
    pass
    doe_file = '/home/ppiper/ihtc_repository/data/doe_more_design_variables/doe_more_design_variables.txt'
    hf_params = {'Nx': '210', 'Ny': '330', 'tol': '1e-8', 'cores': 'None', 'inflationRate': '1.0015', 'metal': 'AISI406', 'itmaxSU2': '3_000', 'GMSH_SOLVER': '/home/ppiper/ihtc_repository/src/gmsh', 'SU2_SOLVER': '/home/ppiper/ihtc_repository/src/SU2_CFD', 'baselineCP': '/home/ppiper/ihtc_repository/src/baselineCP.txt', 'itprintSU2': '1', 'only_generate_mesh': False}
    lf_params = {'Nx': '400', 'EULER_Q1D_SOLVER': '/home/ppiper/ihtc_repository/src/eulerQ1D', 'baselineCP': '/home/ppiper/ihtc_repository/src/baselineCP.txt'}
    index_range = [79, 80, 81, 82, 83, 84, 85, 86, 87]
    dataset_root = '/home/ppiper/ihtc_repository/data/doe_more_design_variables'
    gen_dataset(lf_model, dataset_root, doe_file,index_range,  lf_params )
    gen_dataset(hf_model, dataset_root, doe_file,index_range,  hf_params )
