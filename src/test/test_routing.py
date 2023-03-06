from routing import linear_velocity
from utils.network.network import combine_rvr_prm
from test_dataframes import getDF_by_size
from model400names import STATES_NAMES
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def test1():
    rvr_file ='../examples/cedarrapids1/367813.rvr'
    prm_file ='../examples/cedarrapids1/367813.prm'
    network = combine_rvr_prm(prm_file,rvr_file)
    nlinks = network.shape[0]
    states = pd.DataFrame(
        data = np.zeros(shape=(nlinks,len(STATES_NAMES))),
        columns=STATES_NAMES)
    states['link_id'] = network['link_id'].to_numpy()
    states.index = states['link_id'].to_numpy()
    states['discharge']=1
    NSTEPS = 480
    DT = 60 #min
    velocity = 0.1 #m/s
    for tt in range(NSTEPS):
        print(tt)
        linear_velocity(states,velocity,network,DT)
        f = '../examples/cedarrapids1/out/{}.pkl'.format(tt)
        states['discharge'].to_pickle(f)

def test2():
    NSTEPS = 480
    NGAGE =1
    _X = 367813
    sites = {
        'id':['05464500','05464315','05458900','05463050','05458500'],
        'link':[367813,367697,406874,367567,522980],
        'area':[6492,6022,852,4714,1675]
    }
    out = np.zeros(shape=(NGAGE,NSTEPS))
    for tt in range(NSTEPS):
        f = '../examples/cedarrapids1/out/{}.pkl'.format(tt)
        states = pd.read_pickle(f)
        out[0,tt] = states[_X]
    
    plt.plot(out[0,:])
    plt.show()

def test3():
    rvr_file ='../examples/cedarrapids1/367813.rvr'
    prm_file ='../examples/cedarrapids1/367813.prm'
    network = combine_rvr_prm(prm_file,rvr_file)
    network = pd.read_pickle('../exa')
    nlinks = network.shape[0]
    states = pd.DataFrame(
        data = np.zeros(shape=(nlinks,len(STATES_NAMES))),
        columns=STATES_NAMES)
    states['link_id'] = network['link_id'].to_numpy()
    states.index = states['link_id'].to_numpy()
    states['discharge']=1
    NSTEPS = 480
    DT = 60 #min
    velocity = 0.1 #m/s
    for tt in range(NSTEPS):
        print(tt)
        linear_velocity(states,velocity,network,DT)
        f = '../examples/cedarrapids1/out/{}.pkl'.format(tt)
        states['discharge'].to_pickle(f)