
#""" bmi compatible version of Hillslope Link Model"""
import pandas as pd
import numpy as np
from model400 import runoff1
from model400names import CF_LOCATION , CF_UNITS, VAR_TYPES
from routing import linear_velocity1,transfer0
from yaml import Loader
import yaml
from utils.network.network import combine_rvr_prm
from utils.params.params_from_prm_file import params_from_prm_file
from utils.params.params_default import get_default_params
from utils.forcings.forcing_manager import get_default_forcings
from utils.states.states_default import get_default_states
from utils.network.network import get_default_network
from utils.serialization import save_to_pickle
#from io3.forcing import check_forcings
import importlib.util

class HLM(object):
    """Creates a new HLM model """

    def __init__(self):
        self.name='HLM'
        self.description=None
        self.time=None
        self.time_step=None
        self.init_time=None
        self.end_time=None
        self.states= None
        self.params= None
        self.forcings= None
        self.network= None
        self.outputfile =None
        self.configuration = None

    
    def init_from_file(self,config_file:str):
        with open(config_file) as stream:
            try:
                d = yaml.load(stream,Loader=Loader)
            except yaml.YAMLError as e:
                print(e)
                return
        
        self.configuration = d
        self.description = d['description']
        self.time = d['init_time']
        self.init_time = d['init_time']
        self.end_time = d['end_time']
        self.time_step= d['time_step']
        #self.network = combine_rvr_prm(d['parameters'],d['network'])
        self.network = get_default_network()
        self.states = get_default_states(self.network)
        self.params = get_default_params(self.network)
        self.forcings = get_default_forcings(self.network)
        self.outputfile = d['output_file']


        ...


    def get_time(self)->int:
        return self.time

    def set_time(self,t:int):
        self.time=t
    
    def get_time_step(self)->int:
        return self.time_step

    def set_time_step(self,time_step:int):
        self.time_step=time_step

    def set_forcings(self):
        modelforcings = list(self.forcings.columns)[1:]
        config_forcings = list(self.configuration['forcings'].keys())
        for ii in range(len(modelforcings)):
            if modelforcings[ii] in config_forcings:
                modname = self.configuration['forcings'][modelforcings[ii]]['script']
                spec = importlib.util.spec_from_file_location('mod',modname)
                foo = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(foo)

            lid, values = foo.get_values(self.time)
            self.set_values(
                var_name='forcings.'+ modelforcings[ii],
                linkids = lid,
                values= values
            )
    
    def advance_one_step(self):
        runoff1(self.states,self.forcings,self.params,self.network,self.time_step)
        #linear_velocity1(self.states,self.params,self.network,self.time_step)
        transfer0(self.states,self.params,self.network,self.time_step)
        save_to_pickle(self.states,self.time)
        self.time += self.time_step*60

    def advance(self,time_to_advance:float):
        while self.time < time_to_advance:
            self.advance_one_step()
        
    def finalize(self):
        self = None

    def check_var_exists(self,variable:str)->bool:
        flag =False
        flag = flag or (variable in self.params.columns)
        flag = flag or (variable in self.forcings.columns)
        flag = flag or (variable in self.states.columns)
        flag = flag or (variable in self.network.columns)
        return flag

    def get_value_ptr(self,var_name:str):
        print(var_name)
        items = var_name.split('.')
        print(items)
        group = items[0]
        variable = items[1]
        if self.check_var_exists(variable)==False:
            print('{} not exists in HLM variables'.format(var_name))
            return
        if group == 'params':
            return self.params[variable]
        elif group == 'forcings':
            return self.forcings[variable]
        elif group == 'states':
            return self.states[variable]
        elif group == 'network':
            return self.network[variable]

    def set_values(self,var_name:str,values:np.ndarray,linkids=None):
        var = self.get_value_ptr(var_name)
        
        if linkids.any() == None:
            var[:] = values
        else:
            var[:]=0
            var.loc[linkids] = values #do intersection ?

    def get_values(self,var_name: str,linkids=None)->np.ndarray:
        var = self.get_value_ptr(var_name)
        if linkids == None:
            return var.to_numpy().flatten()
        else:
            return var.loc[linkids].to_numpy().flatten()

    def get_var_location(self,var_name:str)->str:
        return CF_LOCATION[var_name]
    
    def get_var_units(self,var_name:str)->str:
        return CF_UNITS[var_name]
    
    def get_var_type(self,var_name:str)->str:
        return VAR_TYPES[var_name]

    

    def main():
        instance = HLM()
        

    #if __name__ == "__main__":
    #    main()