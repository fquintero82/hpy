
#""" bmi compatible version of Hillslope Link Model"""
import pandas as pd
import numpy as np
# from models.model400 import runoff1
from models.model400polars import runoff1

from models.model400names import CF_LOCATION , CF_UNITS, VAR_TYPES
from models.routing import transfer5,transfer9,transfer10
#from solver import create_solver
from yaml import Loader
import yaml
from utils.params.params_manager import get_params_from_manager
from utils.forcings.forcing_manager import get_default_forcings
# from utils.states.states_default import get_default_states
from utils.states.states_manager import get_states_from_manager
from utils.network.network import get_network_from_file
# from utils.serialization import save_to_netcdf
from utils.network.network_symbolic import NetworkSymbolic
import importlib.util
from utils.check_yaml import check_yaml1
import time as mytime
from utils.output_manager import OutputManager

class HLM(object):
    """Creates a new HLM model """

    def __init__(self):
        self.name='HLM'
        self.description=None
        self.time=None
        self.time_step_sec=None
        self.init_time=None
        self.end_time=None
        self.states= None
        self.params= None
        self.forcings= None
        self.network= None
        self.outputfile =None
        self.configuration = None
        #self.pathsolver=None
        #self.ODESOLVER =None
        self.NetworkSymbolic=None
        self.OutputManager = None

    
    def init_from_file(self,config_file:str):
        with open(config_file) as stream:
            try:
                d = yaml.load(stream,Loader=Loader)
            except yaml.YAMLError as e:
                print(e)
                return
        check_yaml1(d)
        self.configuration = d
        self.description = d['description']
        self.time = d['init_time']
        self.init_time = d['init_time']
        self.end_time = d['end_time']
        self.time_step_sec= d['time_step']
        self.network = get_network_from_file(self.configuration)
        # self.states = get_default_states(self.network)
        self.states = get_states_from_manager(d,self.network)
        self.params = get_params_from_manager(self.configuration,self.network)
        self.forcings = get_default_forcings(self.network)
        # self.outputfile = d['output']['path']
        self.NetworkSymbolic = NetworkSymbolic(self)
        self.OutputManager = OutputManager(self)

        



    def get_time(self)->int:
        return self.time

    def set_time(self,t:int):
        self.time=t
    
    def get_time_step(self)->int:
        return self.time_step_sec

    def set_time_step(self,time_step:int):
        self.time_step=time_step

    def set_forcings(self):
        # print('reading forcings')
        t = mytime.time()
        modelforcings = list(self.forcings.columns)[1:]
        config_forcings = list(self.configuration['forcings'].keys())
        for ii in range(len(modelforcings)):
            if modelforcings[ii] not in config_forcings:
                print(modelforcings[ii]+ ' not found in yaml file')
            if modelforcings[ii] in config_forcings:
                options = self.configuration['forcings'][modelforcings[ii]]
                if options is not None:
                    try:
                        modname = options['script']
                        spec = importlib.util.spec_from_file_location('mod',modname)
                        foo = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(foo)
                        lid, values = foo.get_values(self.time,options)
                        self.set_values(
                            var_name='forcings.'+ modelforcings[ii],
                            linkids = lid,
                            values= values
                        )
                    except Exception as e:
                        print(e)
                        quit()
        x = int((mytime.time()-t)*1000)
        print('forcings loaded in {x} msec'.format(x=x))
                
    
    def advance_one_step(self):
        # print(self.time)
        if self.time == self.init_time:
            self.OutputManager.save(self)
            # save_to_netcdf(self.states,self.params,self.time,self.outputfile)
        # self.set_forcings()
        # runoff1(self.states,self.forcings,self.params,self.network,self.time_step_sec)
        # transfer9(self) # volume, discharge symbolic
        transfer10(self) # volume, discharge symbolic
        transfer5(self) #basin vars
        self.time += self.time_step_sec
        self.OutputManager.save(self)
        # save_to_netcdf(self.states,self.params,self.time,self.outputfile)
        

    def advance(self,time_to_advance:float=None):
        if time_to_advance is None:
            time_to_advance = self.end_time
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
        #this method is still having some issues.
        #it should return a view (pointer) to the variable
        #not a copy
        print(var_name)
        items = var_name.split('.')
        print(items)
        group = items[0]
        variable = items[1]
        if self.check_var_exists(variable)==False:
            print('{} not exists in HLM variables'.format(var_name))
            return
        if group == 'params':
            return self.params[[variable]]
        elif group == 'forcings':
            #return self.forcings[variable] #this return series
            #return self.forcings.loc[:,variable] #this return series

            #doing [[variable]] returns datafarame instead of series
            #doing .loc[:] returns view instead of copy
            return self.forcings[[variable]].loc[:] #this returns dataframe
        elif group == 'states':
            return self.states[[variable]]
        elif group == 'network':
            return self.network[[variable]]

    def set_values(self,var_name:str,values:np.ndarray,linkids=None):
        #var = self.get_value_ptr(var_name)
        #print(var_name)
        items = var_name.split('.')
        #print(items)
        group = items[0]
        variable = items[1]
        if self.check_var_exists(variable)==False:
            print('{} not in HLM variables'.format(var_name))
            return
        if group == 'params' or group =='parameters':
            if linkids is None:
                self.params.loc[:,variable]=values
        elif group == 'forcings':
            if linkids is None:
                self.forcings.loc[:,variable] = values
            else:
                self.forcings.loc[:,variable]=0
                df = pd.DataFrame({'val':values},index=linkids)
                ix = self.forcings.index.intersection(df.index)
                self.forcings.loc[ix,variable] = df.loc[ix,'val']
                del df, ix
        elif group == 'states':
            if linkids is None:
                self.states.loc[:,variable]=values
            else:
                self.states.loc[:,variable]=0
                df = pd.DataFrame({'val':values},index=linkids)
                ix = self.states.index.intersection(df.index)
                self.states.loc[ix,variable] = df.loc[ix,'val']
                del df, ix                
        elif group == 'network':
            pass
        

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