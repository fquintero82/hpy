import pandas as pd
from model400names import STATES_NAMES
import numpy as np

def get_default_states(network:pd.DataFrame):
    nlinks = network.shape[0]
    df = pd.DataFrame(
        data = np.zeros(shape=(nlinks,len(STATES_NAMES))),
        columns=list(STATES_NAMES.keys()))
    df = df.astype(STATES_NAMES)
    df.loc[:,'link_id'] = network['link_id'].to_numpy()
    df.index = network['link_id'].to_numpy()
    df.loc[:,'snow'] = np.float16(0)
    df.loc[:,'static']=np.float16(0)
    df.loc[:,'surface']=np.float16(0)
    df.loc[:,'subsurface']=np.float16(0)
    df.loc[:,'groundwater']=np.float16(0.1)
    df.loc[:,'discharge'] = np.float16(1)
    
    return df