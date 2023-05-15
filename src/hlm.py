
#""" bmi compatible version of Hillslope Link Model"""
import pandas as pd
import numpy as np
from model400 import runoff1
from model400names import CF_LOCATION , CF_UNITS, VAR_TYPES
from routing import linear_velocity1,transfer1,transfer2
from yaml import Loader
import yaml
from utils.network.network import combine_rvr_prm
from utils.params.params_from_prm_file import params_from_prm_file
from utils.params.params_default import get_default_params
from utils.forcings.forcing_manager import get_default_forcings
from utils.states.states_default import get_default_states
from utils.network.network import get_default_network, get_adjacency_matrix
from utils.serialization import save_to_pickle,save_to_netcdf
#from io3.forcing import check_forcings
import importlib.util
from utils.check_yaml import check_yaml1

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
        self.adjmatrix=None
        self.outputfile =None
        self.configuration = None

    
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
        self.time_step= d['time_step']
        self.network = get_default_network()
        self.adjmatrix = get_adjacency_matrix(self.network,default=False)
        self.states = get_default_states(self.network)
        self.params = get_default_params(self.network)
        self.forcings = get_default_forcings(self.network)
        self.outputfile = d['output_file']['path']
        

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
        print('reading forcings')
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
        print('forcings loaded')
                
    
    def advance_one_step(self):
        print(self.time)
        self.set_forcings()
        runoff1(self.states,self.forcings,self.params,self.network,self.time_step)
        #linear_velocity1(self.states,self.params,self.network,self.time_step)
        #transfer0(self.states,self.params,self.network,self.time_step)
        transfer2(self)
        transfer1(self)
        save_to_netcdf(self.states,self.time,self.outputfile)
        #save_to_pickle(self.states,self.time)
        self.time += self.time_step*60

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
        if group == 'params':
            pass
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
            pass
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