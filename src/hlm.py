
#""" bmi compatible version of Hillslope Link Model"""
import pandas as pd
import numpy as np
from test_dataframes import getTestDF1
from model400pandas import runoff1
from routing import linear_velocity1
from yaml import Loader
import yaml
from utils.network.network_from_rvr_file import combine_rvr_prm
from utils.params.params_from_prm_file import params_from_prm_file
from utils.params.params_default import get_default_params
from utils.forcings.forcing_manager import get_default_forcings
from utils.states.states_default import get_default_states
from utils.network.network_default import get_default_network

class HLM(object):
    """Creates a new HLM model """

    def __init__(self):
        self.name='HLM'
        self.description=None
        self.time=None
        self.time_step=None
        self.states= None #getTestDF1('states')
        self.params= None #getTestDF1('params')
        self.forcings= None #getTestDF1('forcings')
        self.network= None #getTestDF1('network')
        self.outputfile =None

    
    def from_file(self,config_file:str):
        with open(config_file) as stream:
            try:
                d = yaml.load(stream,Loader=Loader)
            except yaml.YAMLError as e:
                print(e)
                return
            
            self.description = d['description']
            self.time = d['init_time']
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

    
    def advance_in_time(self):
        runoff1(self.states,self.forcings,self.params,self.network,self.time_step)
        linear_velocity1(self.states,self.params,self.network,self.time_step)

        

    def main():
        instance = HLM()
        

    #if __name__ == "__main__":
    #    main()