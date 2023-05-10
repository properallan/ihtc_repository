# %%
import h5py
import glob
import os

# %%
import pyvista as pv
def plotMesh(meshfile, arr, property):
    p = pv.Plotter()
    p.enable()
    p.enable_anti_aliasing()

    mesh = pv.read(meshfile)
    mesh[property] = arr[:]
    mesh.set_active_scalars(property)
    p.add_mesh(mesh, opacity=0.85, render=True, cmap='plasma')
    p.add_mesh(mesh.contour(), color="white", line_width=2, render=True, cmap='plasma')

    p.set_viewup([0, 1, 0])
    #p.fly_to([5,0,0])

    #p.set_position([5.0, -0.01, 7.5])
    p.window_size = [1280,480]
    #p.save_graphic('./annSU2_'+str(pod.modes.shape[1])+'_modes.pdf')
    p.show(interactive_update=False, auto_close=True)
    #p.show()

# %%
import h5py
import numpy as np
import glob
import pyvista as pv
import pandas as pd

def _genHDF5(doe_file, outputfile, solutions_id):
    # solutions indexes to be read
    solutions_id 

    # read doe_file
    pd.read_csv(doe_file, delimiter=',')

    with h5py.File(outputfile, 'w') as snaps:
        # create entries for every design variable in doe_file
        pass

def _genQ1DHDF5(path, variables, outputfile, solutions_id):
    #path = path_ + 'Q1D/*'
    print(f'Reading data from {path}, generating HDF5 file...')
    #npaths = len(list(filter( os.path.isdir, glob.glob(f'{path}*') ))) 

    arrsize = len(np.loadtxt(f"{path}/{1}/Q1D/outputs/{variables[0]}"))
    arrsize = arrsize*len(variables)

    snaps = h5py.File(outputfile,'w')
    snaps.create_dataset('snapshots', shape=(arrsize, len(solutions_id)), dtype=np.float64)
    snaps.create_dataset('id', shape=(len(solutions_id)), dtype=np.int32)
    snaps.create_dataset('p0in', shape=(len(solutions_id)), dtype=np.float64)
    snaps.create_dataset('T0in', shape=(len(solutions_id)), dtype=np.float64)
    snaps.create_dataset('thickness', shape=(len(solutions_id)), dtype=np.float64)
    

    #for i in range(1, npaths+1):
    for i,j in enumerate(solutions_id):
        idx0 = 0
        idxf = idx0
        for var in variables:
            #filename = path.split('*')[0]+str(i)+'/outputs/'+var
            filename = f"{path}/{j}/Q1D/outputs/{var}"
            print(f"reading file {filename}")
            arr = np.loadtxt(filename)
            idx0 = idxf
            idxf = idx0 + len(arr)
            snaps['snapshots'][idx0:idxf,i-1] = arr[:]
    
        #snaps['DoE'] = df

        snaps['id'][i] = j

        p0in, T0in, thickness = get_boundary_condition(f"{path}/doe_lhs.txt", j)
        snaps['p0in'][i] = p0in
        snaps['T0in'][i] = T0in
        snaps['thickness'][i] = thickness

    snaps.close()

    doe = pd.read_csv(f"{path}/doe_lhs.txt")
    doe.to_hdf(outputfile, key='DoE', mode='a', append=True)    

    print(f'File {outputfile} written')

def get_wall_idx(vtkfile):
    output = pv.read(vtkfile)
    #edges = output.extract_feature_edges()
    edges = output
    wall_idx = np.where(edges['Heat_Flux'] != 0)[0]
    return wall_idx

def genSU2HDF5_Fluid(path, vtkfile, variables, outputfile, solutions_id):
    #path = path_ + 'SU2/*'
    print(f'Reading data from {path}, generating HDF5 file...')
    #npaths = len(list(filter( os.path.isdir, glob.glob(f'{path}*') )))

    #mesh = pv.read(path.split('*')[0]+'1/outputs/solution.vtk')
    mesh = pv.read(f"{path}/1/SU2/outputs/{vtkfile}")
    if 'Heat_Flux' in variables:
        wall_idx = get_wall_idx(f"{path}/1/SU2/outputs/{vtkfile}")
        arrsize = len(mesh[variables[0]])
        arrsize = arrsize*(len(variables)-1) + len(wall_idx)
    else:
        arrsize = len(mesh[variables[0]])
        arrsize = arrsize*len(variables)

    snaps = h5py.File(outputfile,'w')
    snaps.create_dataset('snapshots', shape=(arrsize, len(solutions_id)), dtype=np.float64)
    snaps.create_dataset('id', shape=(len(solutions_id)), dtype=np.int32)
    snaps.create_dataset('p0in', shape=(len(solutions_id)), dtype=np.float64)
    snaps.create_dataset('T0in', shape=(len(solutions_id)), dtype=np.float64)
    snaps.create_dataset('thickness', shape=(len(solutions_id)), dtype=np.float64)

    meshfiles=[]
    for i,j in enumerate(solutions_id):
        idx0 = 0
        idxf = idx0
        #filename = path.split('*')[0]+str(i)+'/outputs/solution.vtk'
        filename = f"{path}/{j}/SU2/outputs/{vtkfile}"
        meshfile = f"{path}/{j}/SU2/outputs/{vtkfile.split('.vtk')[0]}_mesh.vtk"
        print(f'reading file {filename}')
        mesh = pv.read(filename)
        for var in variables:
            if var == 'Heat_Flux':
                arr = mesh[var][wall_idx]
                
                #import matplotlib.pyplot as plt
                #print(var)
                #print(mesh[var].shape)
                #plotMesh(f"{path}/{j}/SU2/outputs/{vtkfile}", mesh[var], var)
                #plt.plot(mesh[var][wall_idx])
                #plt.show()
                #input()
                
                idx0 = idxf
                idxf = idx0 + len(arr)
                snaps['snapshots'][idx0:idxf,i-1] = arr[:]
            else:
                arr = mesh[var]
                idx0 = idxf
                idxf = idx0 + len(arr)
                snaps['snapshots'][idx0:idxf,i-1] = arr[:]
        meshfiles.append(meshfile)

        snaps['id'][i] = j

        p0in, T0in, thickness = get_boundary_condition(f"{path}/doe_lhs.txt", j)
        snaps['p0in'][i] = p0in
        snaps['T0in'][i] = T0in
        snaps['thickness'][i] = thickness

    snaps.create_dataset("meshfile", data=np.array(meshfiles, dtype='S'))
    snaps.close()

    doe = pd.read_csv(f"{path}/doe_lhs.txt")
    doe.to_hdf(outputfile, key='DoE', mode='a', append=True)

    print(f'File {outputfile} written')

def genSU2HDF5(path, vtkfile, variables, outputfile, solutions_id):
    #path = path_ + 'SU2/*'
    print(f'Reading data from {path}, generating HDF5 file...')
    #npaths = len(list(filter( os.path.isdir, glob.glob(f'{path}*') )))

    #mesh = pv.read(path.split('*')[0]+'1/outputs/solution.vtk')
    mesh = pv.read(f"{path}/1/SU2/outputs/{vtkfile}")
    arrsize = len(mesh[variables[0]])
    arrsize = arrsize*len(variables)

    snaps = h5py.File(outputfile,'w')
    snaps.create_dataset('snapshots', shape=(arrsize, len(solutions_id)), dtype=np.float64)
    snaps.create_dataset('id', shape=(len(solutions_id)), dtype=np.int32)
    snaps.create_dataset('p0in', shape=(len(solutions_id)), dtype=np.float64)
    snaps.create_dataset('T0in', shape=(len(solutions_id)), dtype=np.float64)
    snaps.create_dataset('thickness', shape=(len(solutions_id)), dtype=np.float64)

    meshfiles=[]
    for i,j in enumerate(solutions_id):
        idx0 = 0
        idxf = idx0
        #filename = path.split('*')[0]+str(i)+'/outputs/solution.vtk'
        filename = f"{path}/{j}/SU2/outputs/{vtkfile}"
        meshfile = f"{path}/{j}/SU2/outputs/{vtkfile.split('.vtk')[0]}_mesh.vtk"
        print(f'reading file {filename}')
        mesh = pv.read(filename)
        for var in variables:
            arr = mesh[var]
            idx0 = idxf
            idxf = idx0 + len(arr)
            snaps['snapshots'][idx0:idxf,i-1] = arr[:]
        meshfiles.append(meshfile)

        snaps['id'][i] = j

        p0in, T0in, thickness = get_boundary_condition(f"{path}/doe_lhs.txt", j)
        snaps['p0in'][i] = p0in
        snaps['T0in'][i] = T0in
        snaps['thickness'][i] = thickness

    snaps.create_dataset("meshfile", data=np.array(meshfiles, dtype='S'))
    snaps.close()

    doe = pd.read_csv(f"{path}/doe_lhs.txt")
    doe.to_hdf(outputfile, key='DoE', mode='a', append=True)

    print(f'File {outputfile} written')


def get_converged_solutions(root_path, solution_path):
    solutions_id = []
    n_directories = len(list(filter( os.path.isdir, glob.glob(f'{root_path}/*') ))) 
    for i in range(n_directories):
        solution_file = f'{root_path}/{i+1}/{solution_path}'
        if len(list( glob.glob(f'{solution_file}*') )) > 0:
            solutions_id.append(i+1)

    return solutions_id

def get_boundary_condition(doe_file, id):
    #dataset_root = '/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200'
    #id = 6 # case to run
    #doe_file = '/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200/doe_lhs.txt'
    doe_table = np.loadtxt(doe_file, skiprows=1, delimiter=',')
    idx = np.where(doe_table[:,0] == id) # idx related to case number

    p0in = doe_table[idx,1][0][0]
    T0in = doe_table[idx,2][0][0]
    thickness = doe_table[idx,3][0][0]

    return p0in, T0in, thickness


def main():
    
    solutions_id = get_converged_solutions('/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200')

    genQ1DHDF5('/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200', 
               ['p.txt','T.txt','M.txt'],
               '/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200/Q1D.hdf5', 
               solutions_id)       
    
    genSU2HDF5_Fluid('/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200', 
               'fluid.vtk',
               ['Pressure','Temperature','Mach','Heat_Flux'],
               '/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200/SU2_fluid.hdf5',
               solutions_id)
    
    #genSU2HDF5('/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200', 
    #           'solid.vtk',
    #           ['Temperature'],
    #           '/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200/SU2_solid.hdf5',
    #           solutions_id)
    

# %%


# %%
import pathlib

def get_q1d_variable(solution_path, variable_name):
    solution_subpath = solution_path / 'Q1D' / 'outputs'
    name_to_variable_conversion = {
        'Pressure': np.loadtxt(solution_subpath / 'p.txt'),
        'Temperature': np.loadtxt(solution_subpath / 'T.txt') ,
        'Mach' : np.loadtxt(solution_subpath / 'M.txt'),
    }
    return name_to_variable_conversion[variable_name]

def get_block_recursive(block, name_index):
    for n in name_index:
        block = block[n]
    return block

def get_su2_variable(solution_path, variable_name):
    solution_subpath = solution_path / 'SU2' / 'outputs' / "cht_setupSU2.vtm"
    block = pv.read(solution_subpath)
    name_to_variable_conversion ={
        'Pressure' : get_block_recursive( block,
            ['Zone 0 (Comp. Fluid)','Internal','Internal','Pressure'] ),
        'Temperature' : get_block_recursive( block,
            ['Zone 0 (Comp. Fluid)','Internal','Internal','Temperature'] ),
        'Mach' : get_block_recursive( block,
            ['Zone 0 (Comp. Fluid)','Internal','Internal','Mach'] ),
        'Temperature_Solid' : get_block_recursive( block,
            ['Zone 1 (Solid Heat)', 'Internal', 'Internal', 'Temperature'] ),
        'Temperature_Solid_INNERWALL' : get_block_recursive( block,
            ['Zone 1 (Solid Heat)', 'Boundary', 'INNERWALL', 'Temperature'] ),
        'Heat_Flux_UPPER_WALL' : get_block_recursive( block,
            ['Zone 0 (Comp. Fluid)', 'Boundary','UPPER_WALL', 'Heat_Flux'] ),
    }

    return  name_to_variable_conversion[variable_name]

# %%
def genHDF5(dataset_path, variables, outputfile, solutions_id, variable_getter, doe_file):
    doe_ds = pd.read_csv(doe_file, delimiter=',').set_index('ID')

    with h5py.File(outputfile, 'w') as h5file:
        for key in doe_ds.keys():
            h5file[key] = doe_ds[key]

        for id in solutions_id:
            for variable_name in variables:
                solution_path = pathlib.Path(dataset_path / str(id))
                variable_data = variable_getter(solution_path, variable_name)

                if id == 1:
                    h5file.create_dataset(variable_name,shape=(len(variable_data), len(solutions_id)))
                    
                h5file[variable_name][...,id-1] = variable_data
