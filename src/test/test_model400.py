import pandas as pd
import numpy as np
from test_dataframes import getTestDF1
from model400names import *
from model400pandas import runoff1
from routing import linear_velocity

def test_runoff1():
    states= getTestDF1('states')
    params= getTestDF1('params')
    forcings= getTestDF1('forcings')
    network= getTestDF1('network')

    old_states = states.copy()
    runoff1(states=states,
    forcings=forcings,
    params=params,
    network=network,
    DT=1440)
    print('old')
    print(old_states)
    print('new')
    print(states)

def test_runoff2():
    states= getTestDF1('states')
    params= getTestDF1('params')
    forcings= getTestDF1('forcings')
    network= getTestDF1('network')

    NSTEPS = 24
    DT = 10
    init_states = states.copy()
    for tt in range(NSTEPS):
        #runoff1(states,forcings,params,network,DT)
        #print(states.loc[3,'discharge'])
        linear_velocity(states,params,network,DT)
        print(states.loc[3,'discharge'])
    end_states = states.copy()
    print('complete test runoff 2')


test_runoff2()
