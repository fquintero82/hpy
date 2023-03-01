
#""" bmi compatible version of Hillslope Link Model"""
import pandas as pd
import numpy as np
from test_dataframes import getTestDF1
from model400pandas import runoff1
from routing import linear_velocity

class HLM(object):
    """Creates a new HLM model """

    def __init__(self):
        self.name='HLM'
        self.time=None
        self.time_step=None
        self.next_state = None
        self.pd_states= getTestDF1('states')
        self.pd_params= getTestDF1('params')
        self.pd_forcings=getTestDF1('forcings')
        self.pd_network= getTestDF1('network')
    

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