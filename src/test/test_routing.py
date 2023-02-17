from routing import linear_velocity2
from utils.network.network_from_rvr_file import combine_rvr_prm
from test_dataframes import getDF_by_size
from model400names import STATES_NAMES
import pandas as pd
import numpy as np

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
    NSTEPS = 24
    DT = 60
    velocity = 0.1 #m/s
    wh = [network['upstream_link']==None]==True
    for tt in range(NSTEPS):
        linear_velocity2(states,velocity,network,DT)
        print(states.loc[3,'discharge'])
    end_states = states.copy()
    