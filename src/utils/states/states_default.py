import pandas as pd
from model400names import STATES_NAMES
import numpy as np

def get_default_states(network:pd.DataFrame):
    nlinks = network.shape[0]
    df = pd.DataFrame(
        data = np.zeros(shape=(nlinks,len(STATES_NAMES))),
        columns=STATES_NAMES)
    df['link_id'] = network['link_id'].to_numpy()
    df.index = network['link_id'].to_numpy()
    df['snow'] =0
    df['static']=0
    df['surface']=0
    df['subsurface']=0
    df['groundwater']=0.1
    df['discharge'] = 0.1
    return df