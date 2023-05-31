import pathlib
import configparser
import os
import glob
import numpy as np
import pyvista as pv
import pandas as pd
import h5py
from svd_dataset import SVD
import tensorflow as tf
import pickle
import inspect

def rename_attribute(obj, old_name, new_name):
    obj.__dict__[new_name] = obj.__dict__.pop(old_name)


def uniquify(path : str):
    """
    Generate a new filename with a number, if that filename already exists.

    Args:
        path (str): The original filename.

    Returns:    
        str: A unique filename, that didn't exist before.
    """

    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path
def clear_vtm(solution_file, output_file):
    multiblock = pv.read(solution_file)
    for i in range(len(multiblock)):
        for j in range(len(multiblock[i])):
            for k in range(len(multiblock[i][j])):
                multiblock[i][j][k].clear_point_data()
    multiblock.save(output_file)

    return pv.read(output_file)

    
def read_config_file(config_file):
    """
    Deprecated, use static method read_config_file of FlowReconstruction class instead.
    """
    
    class CaseConfigParser(configparser.ConfigParser):
        def optionxform(self, optionstr):
            return optionstr
    
    config = CaseConfigParser()
    config.read(config_file)

    # convert all path to the absolute
    for each_section in config.sections():
        for (each_key, each_val) in config.items(each_section):
            value = pathlib.Path(config[each_section][each_key])
            if value.is_file() or value.is_dir():
                config[each_section][each_key] = f"{value.resolve()}"
    return config


def get_converged_solutions(root_path, solution_path):
    solutions_id = []
    n_directories = len(
        list(filter(os.path.isdir, glob.glob(f'{root_path}/*'))))
    for i in range(n_directories):
        solution_file = f'{root_path}/{i+1}/{solution_path}'
        if len(list(glob.glob(f'{solution_file}*'))) > 0:
            solutions_id.append(i+1)

    return solutions_id


def sliceDataAlongAxis(data, fractions, axis):
    data_size = data.shape[axis]
    fractions_ = np.zeros_like(fractions, dtype=int)

    total_size = 0
    for i, fraction in enumerate(fractions):
        total_size += int(data_size*fraction)
    remain = data_size-total_size

    slices = ()
    for i, fraction in enumerate(fractions):
        fractions_[i] = int(data_size*fraction)
        if i > 0:
            fractions_[i] += fractions_[i-1]
            slice = data.take(range(fractions_[i-1], fractions_[i]), axis)

        else:
            slice = data.take(range(0, fractions_[i]+remain), axis)

        slices += (slice,)

    return slices


class NeuralNetwork:
    def __init__(self):

        self.name = 'neural_network'

    def set_model(self,
                  layers,
                  activation_function='tanh',
                  optimizer=tf.keras.optimizers.Adam(),
                  hidden_layers=None):

        self.nn = self.get_model(layers, activation_function, optimizer)

    def get_model(self, layers, activation_function, optimizer):
        # Input layer
        ph_input = tf.keras.Input(shape=(layers[0],), name='input_placeholder')
        # Hidden layers
        hidden_layer = ph_input
        for num_neurons in layers[1:-1]:
            hidden_layer = tf.keras.layers.Dense(
                num_neurons, activation=activation_function)(hidden_layer)

        # Output layer
        output = tf.keras.layers.Dense(
            layers[-1], activation='linear', name='output_value')(hidden_layer)

        model = tf.keras.Model(inputs=[ph_input], outputs=[output])
        # Compilation
        model.compile(optimizer=optimizer, loss={
                      'output_value': 'mean_squared_error'})

        return model

    def fit(self,
            train_input,
            train_target,
            validation_input=None,
            validation_target=None,
            epochs=1000,
            batch_size=16):

        if validation_input is not None and validation_target is not None:
            validation_tuple = (validation_input, validation_target)

        self.history = self.nn.fit(train_input, train_target, epochs=epochs,
                                   batch_size=batch_size, validation_data=validation_tuple)

        return self.nn, self.history

    def predict(self, input):
        if len(input.shape) == 1:
            input = input[None, :]
        return self.nn.predict(input)

    def save(self, file):
        self.nn.save(file)

    def load(self, file):
        self.nn = tf.keras.models.load_model(file)


class dataHandler:
    def __init__(self, file, variables):
        self.file = file
        self.variables = variables
        self.stack()

    def stack(self):
        with h5py.File(self.file, 'r') as f:
            self.data = np.vstack([f[var][()] for var in self.variables])

            self.indexes = {}
            start_idx = 0
            for var in self.variables:
                try:
                    end_idx = start_idx + f[var][()].shape[0]
                except:
                    # f[var] is a scalar
                    end_idx = start_idx + 1
                self.indexes[var] = np.arange(start_idx, end_idx)
                start_idx = end_idx

            if 'meshfile' in f.keys():
                self.meshfile = f['meshfile'][()]

    def get_variable(self, variable=None, data=None,):
        if data is None:
            data = self.data

        return data[self.indexes[variable], :]

    def split_train_validation_test(self, fractions):
        self.train, self.validation, self.test = sliceDataAlongAxis(
            self.data, fractions, 1)


class ROM:
    def __init__(self, dataset, rank=None, energy=None):

        self.svd = SVD(dataset)

        self.bounds = [0, 1]

        self.svd.normalize(bounds=self.bounds)
        self.svd.subtractMean()
        self.svd.SVD()
        if rank is not None and energy is None:
            self.setRank(rank)
        if energy is not None:
            self.findRank(energy)

    def setRank(self, rank):
        self.rank = rank
        self.svd.setRank(self.rank)

    def findRank(self, energy):
        rank = self.svd.findRank(energy)
        self.rank = rank

    def setEnergy(self, e):
        self.energyPreserved = e
        self.rank = self.svd.findRank(self.energyPreserved)
        self.svd.setRank(self.rank)

    def reduce(self, snapshot):
        if len(snapshot.shape) == 1:
            snapshot = snapshot[:, None]
        snapshot, min, max = self.svd.normalize(
            snapshot, self.bounds, self.svd.min, self.svd.max)
        snapshot, mean = self.svd.subtractMean(snapshot, self.svd.mean)
        L = (self.svd.u.T @ snapshot).T
        return L

    @property
    def data(self):
        return self.svd.L

    @property
    def L(self):
        return self.svd.L

    def reconstruct(self, input):
        return self.svd.reconstruct(input)

class CaseConfigParser(configparser.ConfigParser):
            def optionxform(self, optionstr):
                return optionstr
class FlowReconstruction:
    def __init__(self):
        pass

    @staticmethod
    def read_config_file(config_file):   
        class CaseConfigParser(configparser.ConfigParser):
            def optionxform(self, optionstr):
                return optionstr
             
        config = CaseConfigParser()
        config.read(config_file)

        # convert all path to the absolute
        for each_section in config.sections():
            for (each_key, each_val) in config.items(each_section):
                value = pathlib.Path(config[each_section][each_key])
                if value.is_file() or value.is_dir():
                    config[each_section][each_key] = f"{value.resolve()}"
        return config

    def set_config_file(self, config_file):
        self.config_file = pathlib.Path(config_file).absolute()
        self.config = FlowReconstruction.read_config_file(self.config_file)
        self.dataset_root = pathlib.Path(
            self.config['DATASET']['DATASET_ROOT']).absolute()
        self.doe_file = self.dataset_root / \
            pathlib.Path(self.config['DATASET']['DOEFILE'])

        self.LF_PARAMS = dict(self.config['LF_PARAMS'])
        self.HF_PARAMS = dict(self.config['HF_PARAMS'])

    def get_converged_solutions(self):
        self.converged_solutions_id = get_converged_solutions(
            self.dataset_root, self.hf_solution_file)

    def set_lf_solution_path(self, path):
        self.lf_solution_path = pathlib.Path(path)

    def set_hf_solution_path(self, path):
        self.hf_solution_path = pathlib.Path(path)

    def set_hf_solution_file(self, file):
        self.hf_solution_file = pathlib.Path(self.hf_solution_path / file)

    def set_lf_variable_getter(self, variable_getter, variables_dict):
        self.lf_variables_dict = variables_dict
        self.lf_variable_getter = variable_getter

    def get_lf_variable(self, variable, idx=None, solution_path=None):
        if solution_path is None and idx is not None:
            solution_path = self.dataset_root / \
                str(idx) / self.lf_solution_path
        else:
            # solution_path = pathlib.Path(solution_path) / self.lf_solution_path
            pass
        return self.lf_variable_getter(self.lf_variables_dict, variable, solution_path)

    def get_lf_variables(self, solution_path=None, variables=None, include_design_variable=True):
        if solution_path is None:
            solution_path = self.lf_fom_rootfile

        snapshot = []

        if include_design_variable:
            for var, value in self.lf_design_variables.items():
                snapshot.append(value)
        if variables is None:
            variables = self.lf_variables_dict.keys()
        for var in variables:
            value = self.get_lf_variable(
                variable=var, solution_path=solution_path)
            snapshot.append(value)

        return np.hstack(snapshot)

    def get_hf_variables(self, solution_file=None):
        if solution_file is None:
            solution_file = self.hf_fom_

    def set_hf_variable_getter(self, variable_getter, variables_dict):
        self.hf_variables_dict = variables_dict
        self.hf_variable_getter = variable_getter


    def get_hf_variable(self, variable, idx=None, solution_file=None):
        if solution_file is None and idx is not None:
            solution_file = self.dataset_root / \
                str(idx) / self.hf_solution_file
        else:
            # inconsistent !
            solution_file = pathlib.Path(solution_file) / self.hf_solution_path

        return self.hf_variable_getter(self.hf_variables_dict, variable, solution_file)

    def get_hf_snapshot(self, idx=None, solution_file=None, include_design_variables=True):
        if solution_file is None and idx is not None:
            solution_file = self.dataset_root / \
                str(idx) / self.hf_solution_file

        snapshot = {}
        if include_design_variables == True:
            df = pd.read_csv(self.doe_file).set_index('ID')
            design_variables = df.keys()
        elif include_design_variables == False:
            design_variables = []
        else:
            design_variables = include_design_variables

        for var in design_variables:
            snapshot[var] = df[var][idx]

        for var in self.hf_variables_dict.keys():
            if idx is not None:
                snapshot[var] = self.get_hf_variable(var, idx)
            elif solution_file is not None:
                snapshot[var] = self.get_hf_variable(
                    variable=var, solution_file=solution_file)

        return snapshot

    def get_lf_snapshot(self, idx=None, solution_path=None, include_design_variables=True):
        if solution_path is None and idx is not None:
            solution_path = self.dataset_root / \
                str(idx) / self.lf_solution_path
        elif solution_path is None:
            solution_path = pathlib.Path(solution_path) / self.lf_solution_path

        snapshot = {}
        if include_design_variables is True:
            df = pd.read_csv(self.doe_file).set_index('ID')
            design_variables = df.keys()
            for var in design_variables:
                snapshot[var] = df[var][idx]
        elif include_design_variables is False:
            design_variables = []
        else:
            design_variables = include_design_variables

            for key, var in design_variables.items():
                snapshot[key] = var

        for var in self.lf_variables_dict.keys():
            if idx is not None:
                snapshot[var] = self.get_lf_variable(var, idx)
            elif solution_path is not None:
                snapshot[var] = self.get_lf_variable(
                    variable=var, solution_path=solution_path)

        return snapshot

    def get_lf_snapshots(self, include_design_variables=True):
        snapshots = []
        for id in self.converged_solutions_id:
            snapshot = self.get_lf_snapshot(
                idx=id, include_design_variables=include_design_variables)
            snapshots.append(snapshot)
        # self.lf_variables = snapshot.keys()
        return snapshots

    def get_hf_snapshots(self, include_design_variables=True):
        snapshots = []
        for id in self.converged_solutions_id:
            snapshot = self.get_hf_snapshot(
                idx=id, include_design_variables=include_design_variables)
            snapshots.append(snapshot)

        # self.hf_variables = snapshot.keys()
        return snapshots

    def gen_HDF5(self, snapshots, h5_filename):
        with h5py.File(h5_filename, 'w') as h5file:
            len(snapshots)
            for key in snapshots[0].keys():
                n_rows = snapshots[0][key].size
                h5file[key] = np.zeros((n_rows, len(snapshots)))

                for i, snap_i in enumerate(snapshots):
                    h5file[key][:, i] = snap_i[key]

    def gen_lf_HDF5(self, h5_filename, snapshots=None, include_design_variables=True):
        self.lf_h5file = self.dataset_root / h5_filename
        if snapshots is None:
            snapshots = self.get_lf_snapshots(
                include_design_variables=include_design_variables)
        self.gen_HDF5(snapshots, self.lf_h5file)
        self.lf_variables = list(snapshots[0].keys())

    def gen_hf_HDF5(self, h5_filename, snapshots=None, include_design_variables=True):
        self.hf_h5file = self.dataset_root / h5_filename
        if snapshots is None:
            snapshots = self.get_hf_snapshots(
                include_design_variables=include_design_variables)
        self.gen_HDF5(snapshots, self.hf_h5file)
        self.hf_variables = list(snapshots[0].keys())

    def set_lf_data_handler(self, train_validation_test_fractions):
        self.lf_data_handler = dataHandler(
            self.lf_h5file, self.lf_variables)
        self.lf_data_handler.split_train_validation_test(
            train_validation_test_fractions)

    def set_hf_data_handler(self, train_validation_test_fractions):
        self.hf_data_handler = dataHandler(
            self.hf_h5file, self.hf_variables)
        self.hf_data_handler.split_train_validation_test(
            train_validation_test_fractions)

    def set_lf_rom(self, rom, **rom_config):
        self.lf_rom_config = rom_config
        self.lf_rom = rom(self.lf_data_handler.train, **rom_config)
        self.lf_rom_rank = self.lf_rom.rank

    def set_hf_rom(self, rom, **rom_config):
        self.hf_rom_config = rom_config
        self.hf_rom = rom(self.hf_data_handler.train, **rom_config)
        self.hf_rom_rank = self.hf_rom.rank

    def set_lf_fom(self, lf_fom):
        self.lf_fom = lf_fom

    def set_hf_fom(self, hf_fom):
        self.hf_fom = hf_fom

    def run_lf_fom(self, rootfile, **other_params):
        self.lf_fom_rootfile = f"{pathlib.Path(self.dataset_root / rootfile)}/"
        print(self.LF_PARAMS)
        self.lf_design_variables = dict(other_params)
        self.lf_fom(rootfile=self.lf_fom_rootfile, **
                    self.LF_PARAMS, **other_params)

    def run_hf_fom(self, rootfile, **other_params):
        self.hf_fom_rootfile = f"{pathlib.Path(self.dataset_root / rootfile)}/"
        print(self.HF_PARAMS)
        self.hf_design_variables = dict(other_params)
        self.hf_fom(rootfile=self.hf_fom_rootfile, **
                    self.HF_PARAMS, **other_params)

    def reconstruct_snapshot(self):
        pass

    def reconstruct(self, rootfile, **other_params):
        def sub_dict(d, keys): return dict((key, d[key]) for key in keys)

        self.run_lf_fom(rootfile, **other_params)
        self.run_hf_fom(rootfile, only_generate_mesh=True, **other_params)

        self.hf_reconstructed_file = self.hf_fom_rootfile / \
            self.hf_solution_path / 'reconstructed.vtm'
        clear_vtm(self.hf_fom_rootfile / self.hf_solution_file,
                  self.hf_reconstructed_file)

        # lf_fom_solution = self.get_lf_variables(solution_path=self.lf_fom_rootfile)
        lf_solution_path = self.lf_fom_rootfile
        lf_fom_solution = self.get_lf_snapshot(
            lf_solution_path, include_design_variables=sub_dict(other_params, ['Thickness']))

        lf_fom_solution = set_thickness_distribution(
            snapshots=[lf_fom_solution])

        lf_fom_solution = np.hstack(
            [val for key, val in lf_fom_solution[0].items()])

        lf_projected = self.lf_rom.reduce(lf_fom_solution)
        hf_projected = self.surrogate.predict(lf_projected)
        hf_reconstructed = self.hf_rom.reconstruct(hf_projected)

        reconstructed_mesh = pv.read(self.hf_reconstructed_file)

        for variable, index in fr.hf_variables_dict.items():
            value = self.hf_data_handler.get_variable(
                variable=variable, data=hf_reconstructed)
            get_block_recursive(reconstructed_mesh,
                                index[:-1])[variable] = value

        reconstructed_mesh.save(self.hf_reconstructed_file)

    def set_surrogate(self, surrogate, surrogate_config):
        self.surrogate = surrogate()
        if self.surrogate.name == 'neural_network':
            surrogate_config['layers'] = [self.lf_rom_rank] + \
                surrogate_config['hidden_layers'] + [self.hf_rom_rank]

            self.surrogate.set_model(**surrogate_config)
    
    def save(self, filename='flow_reconstruction.pickle'):   
        pathlib.Path(filename).parent.absolute().mkdir(parents=True, 
        exist_ok=True)

        with open( uniquify(filename), 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, 'rb') as handle:
            b = pickle.load(handle)
        self = b