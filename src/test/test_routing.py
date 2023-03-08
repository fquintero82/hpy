from routing import linear_velocity
from utils.network.network import combine_rvr_prm
from test_dataframes import getDF_by_size
from model400names import STATES_NAMES
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils.params.params_default import get_default_params
from utils.states.states_default import get_default_states
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
    network = pd.read_pickle('../examples/cedarrapids1/367813_network.pkl')
    nlinks = network.shape[0]
    states = get_default_states(network)
    params = get_default_params(network)

    routing_order = network.loc[:,['link_id','idx_downstream_link','drainage_area','channel_length']]
    routing_order['idx_upstream_link']=np.arange(nlinks)
    routing_order['river_velocity']= params['river_velocity']
    routing_order = routing_order.sort_values(by=['drainage_area'])
    DT = 3600
    idxd = routing_order['idx_downstream_link'].to_numpy()
    idxu = routing_order['idx_upstream_link'].to_numpy()
    vel = routing_order['river_velocity'].to_numpy()
    len1 = routing_order['channel_length'].to_numpy()
    routing_order.describe()
    states['discharge']=1
    q=states['discharge'].to_numpy()


    for ii in np.arange(nlinks):
        dq = np.min([q[idxu[ii]] , q[idxu[ii]] * vel[ii] / len1[ii] * DT ])
        #dq = q[idxu[ii]]
        if(idxd[ii])>=0:
            q[idxd[ii]] += dq
            q[idxu[ii]] -= dq 
    
    states['discharge']=q
    states.describe()
    NSTEPS = 480
    DT = 60 #min
    velocity = 0.1 #m/s
    for tt in range(NSTEPS):
        print(tt)
        linear_velocity(states,velocity,network,DT)
        f = '../examples/cedarrapids1/out/{}.pkl'.format(tt)
        states['discharge'].to_pickle(f)

def test3():
    q = 1 #m3/s
    v = 1 #m/s
    l = 100 #m
    dt = 60 #s
    nt = 60
    out = np.zeros(shape=(2,nt))
    for tt in np.arange(nt):
        dq = q* v / l * dt #m3/s
        q -=dq
        out[1,tt] = q
        out[0,tt] = tt
    plt.plot(out[0,:],out[1,:])
    plt.show()