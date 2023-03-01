
#""" bmi compatible version of Hillslope Link Model"""
import pandas as pd
import numpy as np
from test_dataframes import getTestDF1
from model400pandas import runoff1
from routing import linear_velocity
import yaml
from utils.network.network_from_rvr_file import combine_rvr_prm

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
        d = yaml.load(config_file)
        self.description = d['description']
        self.time = d['init_time']
        self.time_step=d['time_step']
        self.network = combine_rvr_prm(d['parameters'],d['network'])
        ...


    def get_time(self):
        return self.time

    def set_time(self,t:int):
        self.time=t
    
    def get_time_step(self):
        return self.time_step

    def set_time_step(self,time_step:int):
        self.time_step=time_step

    
    def advance_in_time(self):
        runoff1(self.states,self.forcings,self.params,self.network,self.time_step)
        linear_velocity(self.states,self.params,self.network,self.time_step)
        
        

    def main():
        instance = HLM()
        

    #if __name__ == "__main__":
    #    main()