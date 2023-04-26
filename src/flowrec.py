import numpy as np
import configparser
import pathlib

class CaseConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr):
        return optionstr

class DesignOfExperiment:
    def __ini__(self, doe_config):
        self.set_config(doe_config)

    def set_config(self, doe_config):
        self.doe_variables = doe_config['doe_variables']
        self.samples = doe_config['samples']
        self.doe_file = doe_config['doe_file']

    def gen_doe(self):
        self.write_lhs_file(
            self.doe_variables, 
            self.samples, 
            self.doe_file)

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

class FlowReconstruction:
    def __init__(self):
        pass
    
    def set_lf_model(self, lf_model):

        self.lf_model = lf_model

    def run_lf_model(self, **other_params):

        config = CaseConfigParser()
        config.read(self.config_file)

        # convert all path to the absolute
        for each_section in config.sections():
            for (each_key, each_val) in config.items(each_section):
                value = pathlib.Path(config[each_section][each_key])
                if value.is_file() or value.is_dir():
                    config[each_section][each_key] = f"{value.resolve()}"
        

        LF_PARAMS = dict(config['LF_PARAMS'])

        self.lf_model(rootfile = self.rootfile, **LF_PARAMS, **other_params)


    def gen_doe(self, doe_config):
        self.DoE = DesignOfExperiment()
        self.DoE.set_config(doe_config)
        self.DoE.gen_doe()

    def set_fr(self, 
                 lf_rom = None, 
                 hf_rom = None, 
                 surrogate = None,
                 lf_data_handler = None,
                 hf_data_handler = None, 
                 rootfile = None, 
                 config_file = None):
        # reciev data handler for low fidelity and high fidelity models
        #self.lf_data = lf_data
        #self.hf_data = hf_data
        self.lf_rom = lf_rom
        self.hf_rom = hf_rom
        self.surrogate = surrogate
        self.lf_data_handler = lf_data_handler
        self.hf_data_handler = hf_data_handler
        self.rootfile = rootfile
        self.config_file = config_file
        pass
    
    def save(self, file):
        self.surrogate_model.save(file)
    
    def load(self, file):
        self.surrogate_model = self.surrogate_model.load(file)

    def reconstruct(self, lf_data, variable=None):
        if len(lf_data.shape) == 1:
            lf_data = lf_data[:,None]
                
        self.lf_data_reduced = self.lf_rom.reduce(lf_data)
        self.hf_data_predicted = self.surrogate.predict(self.lf_data_reduced)
        self.hf_data_reconstructed = self.hf_rom.reconstruct(self.hf_data_predicted)

        if variable is not None:
            self.hf_data_reconstructed = self.hf_data_handler.get_variable(data=self.hf_data_reconstructed, variable=variable)
        
        return self.hf_data_reconstructed
    
    def fit(self, 
            input_data, 
            target_data, 
            input_validation=None, 
            target_validation=None):
        
        self.input_data = input_data
        self.target_data = target_data

        if self.lf_rom is not None:
            self.input_data = self.lf_rom.reduce(input_data)

        if self.hf_rom is not None:
            self.target_data = self.hf_rom.reduce(target_data)

        self.surrogate.fit(
            self.input_data, 
            self.target_data, 
            self.input_validation, 
            self.target_validation)