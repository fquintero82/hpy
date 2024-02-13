
#""" bmi compatible version of Hillslope Link Model"""
import pandas as pd
import numpy as np
# from models.model400 import runoff1
from models.model400polars import runoff1

from models.model400names import CF_LOCATION , CF_UNITS, VAR_TYPES
from models.routing import transfer5,transfer9
#from solver import create_solver
from yaml import Loader
import yaml
from utils.params.params_manager import get_params_from_manager
from utils.forcings.forcing_manager import get_default_forcings
# from utils.states.states_default import get_default_states
from utils.states.states_manager import get_states_from_manager
from utils.network.network import get_network_from_file
from utils.serialization import save_to_netcdf
from utils.serialization import save_to_csv
from utils.outputs.outputs_manager import get_output_links
from utils.outputs.outputs_manager import get_output_states
from utils.outputs.outputs_manager import get_output_timestamps
from utils.network.network_symbolic import NetworkSymbolic
import importlib.util
from utils.check_yaml import check_yaml1
import time as mytime
import os
import shutil


class HLM(object):
    """Creates a new HLM model """

    def __init__(self):
        self.name = 'HLM'
        self.description = None
        self.time = None
        self.time_step_sec = None
        self.init_time = None
        self.end_time = None
        self.states = None
        self.params = None
        self.forcings = None
        self.network = None
        self.output_folder = None
        self.output_file_name = ""
        self.output_links = None
        self.output_states = ["discharge"]
        self.output_file_type = "csv"
        self.output_timestamps = None
        self.configuration = None
        # self.pathsolver=None
        # self.ODESOLVER =None
        self.NetworkSymbolic = None

    def init_from_file(self, config_file: str):
        """
        Initialize the model based on a configuration file.
        config_file: a string representing the path to a YAML configuration file
        """

        with open(config_file) as stream:
            # open the yaml file and is referred to as "stream"
            try:
                # try to load the yaml content
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as e:
                print(e)
                return

        check_yaml1(d)
        self.configuration = d
        self.description = d['description']
        self.time = d['init_time']
        self.init_time = d['init_time']
        self.end_time = d['end_time']
        self.time_step_sec = d['time_step']
        self.network = get_network_from_file(self.configuration)
        # self.states = get_default_states(self.network)
        self.states = get_states_from_manager(d, self.network)
        self.params = get_params_from_manager(self.configuration, self.network)
        self.forcings = get_default_forcings(self.network)
        self.output_folder = d['output_file']['folder']  # set up the output file path
        self.output_file_name = d['output_file']['file_name']  # set up the otuput file name
        self.output_links = get_output_links(d['output_file']['output_links'])  # set up the output links
        self.output_states = get_output_states(d['output_file']['output_states'])  # set up the output states
        self.output_timestamps = get_output_timestamps(
            d['output_file']['output_timestamps'])  # set up the output timestamps
        self.output_file_type = d['output_file']['output_file_type']  # set up the output type
        self.NetworkSymbolic = NetworkSymbolic(self)

    def create_folder(self, folder_name):
        try:
            # Create the folder
            os.makedirs(folder_name)

        except OSError:
            # If the folder already exist, ignore the error
            # Remove the current folder and all the files within it
            shutil.rmtree(folder_name)
            # Create it again
            os.makedirs(folder_name)

            pass

    def get_time(self) -> int:
        return self.time

    def set_time(self, t: int):
        self.time = t

    def get_time_step(self) -> int:
        return self.time_step_sec

    def set_time_step(self, time_step: int):
        self.time_step = time_step

    def set_forcings(self):
        # print('reading forcings')
        # start to record the current time
        t = mytime.time()
        # extract the column names from the forcing dataframe
        modelforcings = list(self.forcings.columns)[1:]
        # retrieve the keys from the forcings dictionary
        config_forcings = list(self.configuration['forcings'].keys())
        # for each force in the columns
        for ii in range(len(modelforcings)):
            # if a forcing in the column is not found in the "config", print a message
            if modelforcings[ii] not in config_forcings:
                print(modelforcings[ii] + ' not found in yaml file')
            # if a forcing in the column is in the "config"
            if modelforcings[ii] in config_forcings:
                # get the option form config
                options = self.configuration['forcings'][modelforcings[ii]]
                if options is not None:
                    try:
                        # try to load and apply the forcing
                        # retrieve the name of the script from the option dictionary
                        modname = options['script']
                        # create a module specification for a module named "mod" from the file
                        spec = importlib.util.spec_from_file_location('mod', modname)
                        # create a new module based on the specification "spec"
                        foo = importlib.util.module_from_spec(spec)
                        # load and excute the module "foo"
                        spec.loader.exec_module(foo)
                        # retrieving values from the module
                        lid, values = foo.get_values(self.time, options)
                        # set values
                        # var_name is "forcings.precipitation", for example
                        # if there is no file at this step, lid = None, values = 0
                        self.set_values(
                            var_name='forcings.' + modelforcings[ii],
                            linkids=lid,
                            values=values
                        )
                    except Exception as e:
                        print(e)
                        quit()
        # print the execution time
        # x = int((mytime.time()-t)*1000)
        # print('forcings loaded in {x} msec'.format(x=x))

    def save_output(self):
        """
        Call the function to save outputs.
        """
        # check the file type to be saved
        if self.output_file_type == "csv":
            save_to_csv(states=self.states, time=self.time,
                        output_folder=self.output_folder, output_file_name=self.output_file_name,
                        output_links=self.output_links, output_states=self.output_states)
        elif self.output_file_type == 'nc':
            # saving state to file
            # TODO: This output netcdf option needs improvement.
            save_to_netcdf(self.states, self.params, self.time, self.output_folder, self.output_file_name)
        else:
            print('Unknown output file type.')
            quit()

    def advance_one_step(self):
        # print(self.time)

        # set up and update forcings
        self.set_forcings()
        # calculate runoff
        runoff1(self.states, self.forcings, self.params, self.network, self.time_step_sec)
        # transfer
        transfer9(self)  # volume, discharge symbolic
        transfer5(self)  # basin vars

        # check the save mode
        if isinstance(self.output_timestamps, str):
            if self.output_timestamps == 'all':
                # save the output every timestamps
                self.save_output()

            # if the output timestamps is an integer
        elif isinstance(self.output_timestamps, int):
            if self.time % self.output_timestamps == 0:
                # save the output at certain intervals
                self.save_output()

            # if the output timestamps is an array
        elif isinstance(self.output_timestamps, np.ndarray):
            # check if the current timestamp is in the required output timestamps
            if np.isin(self.time, self.output_timestamps):
                # save it
                self.save_output()
        else:
            print('Unknown saving settings.')
            quit()

        # add the time step
        self.time += self.time_step_sec

    def advance(self, time_to_advance: float = None):
        """

        time_to_advance: optional parameter, defaults to "none"
        """
        # start counting the time
        computation_start_time = mytime.time()

        # if time_to_advance is none, set it to the end_time.
        if time_to_advance is None:
            time_to_advance = self.end_time

        # check if the output folder exist, if not create one
        self.create_folder(self.output_folder)

        # check the save mode
        if isinstance(self.output_timestamps, str):
            if self.output_timestamps == 'all':
                print("The model will save all timestamps.")
            else:
                print("Unknown saving parameters")
                quit()

            # if the output timestamps is an integer
        elif isinstance(self.output_timestamps, int):
            print("The model will save timestamps every {0} sec.".format(self.output_timestamps))
            # if the output timestamps is an array
        elif isinstance(self.output_timestamps, np.ndarray):
            print('The model will save at specific timestamps.')
        else:
            print('Unknown saving settings.')
            quit()

        # if the current time is less than "time_to_advance"
        while self.time < time_to_advance:
            # call the method "advance_one_step"
            self.advance_one_step()

        # end counting the time
        computation_duration = int((mytime.time() - computation_start_time))
        print('Computation finished in {x} sec'.format(x=computation_duration))

    def finalize(self):
        self = None

    def check_var_exists(self, variable: str) -> bool:
        flag = False
        flag = flag or (variable in self.params.columns)
        flag = flag or (variable in self.forcings.columns)
        flag = flag or (variable in self.states.columns)
        flag = flag or (variable in self.network.columns)
        return flag

    def get_value_ptr(self, var_name: str):
        # this method is still having some issues.
        # it should return a view (pointer) to the variable
        # not a copy
        print(var_name)
        items = var_name.split('.')
        print(items)
        group = items[0]
        variable = items[1]
        if self.check_var_exists(variable) == False:
            print('{} not exists in HLM variables'.format(var_name))
            return
        if group == 'params':
            return self.params[[variable]]
        elif group == 'forcings':
            # return self.forcings[variable] #this return series
            # return self.forcings.loc[:,variable] #this return series

            # doing [[variable]] returns datafarame instead of series
            # doing .loc[:] returns view instead of copy
            return self.forcings[[variable]].loc[:]  # this returns dataframe
        elif group == 'states':
            return self.states[[variable]]
        elif group == 'network':
            return self.network[[variable]]

    def set_values(self, var_name: str, values: np.ndarray, linkids=None):
        """
        var_name: the name of the variable
        values: the values to be set
        linkids:
        """
        # var = self.get_value_ptr(var_name)
        # print(var_name)

        # split the var_name string into a list "items" using "." as the delimiter
        items = var_name.split('.')
        # print(items)

        # extract the group name
        group = items[0]
        # extract the variable name
        variable = items[1]

        # check if the variable exists in the model
        if self.check_var_exists(variable) == False:
            print('{} not in HLM variables'.format(var_name))
            return

        # check if the group is "params"
        if group == 'params':
            pass

        # check if the group is "forcings"
        elif group == 'forcings':

            # if "linkids" is None, this means the file at current step does not exist
            if linkids is None:
                # if the variable is temperature, use the previous forcing values
                if variable == 'temperature':
                    pass
                # if the variable is precipitation, set the new forcing to zero
                else:
                    self.forcings.loc[:, variable] = values

            # if "linkids" is provided, set the column in the dataframe to zero, and then update at specific links
            # this is because for precipitation, only links with rainfall are provided in the binary files
            else:
                self.forcings.loc[:, variable] = 0
                # create a dataframe with values and indexed by link ids
                df = pd.DataFrame({'val': values}, index=linkids)
                # find intersections between forcing and df
                ix = self.forcings.index.intersection(df.index)
                # assign forcing values
                self.forcings.loc[ix, variable] = df.loc[ix, 'val']
                del df, ix

        # check if the group is "states"
        elif group == 'states':
            pass

        # check if the group is "network"
        elif group == 'network':
            pass

    def get_values(self, var_name: str, linkids=None) -> np.ndarray:
        var = self.get_value_ptr(var_name)
        if linkids == None:
            return var.to_numpy().flatten()
        else:
            return var.loc[linkids].to_numpy().flatten()

    def get_var_location(self, var_name: str) -> str:
        return CF_LOCATION[var_name]

    def get_var_units(self, var_name: str) -> str:
        return CF_UNITS[var_name]

    def get_var_type(self, var_name: str) -> str:
        return VAR_TYPES[var_name]

    def main():
        instance = HLM()

    # if __name__ == "__main__":
    #    main()