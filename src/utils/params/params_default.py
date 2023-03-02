import pandas as pd
from model400names import PARAM_NAMES
import numpy as np

def get_default_params(network:pd.DataFrame):
    nlinks = network.shape[0]
    df = pd.DataFrame(
        data = np.zeros(shape=(nlinks,len(PARAM_NAMES))),
        columns=PARAM_NAMES)
    df['link_id'] = network['link_id'].to_numpy()
    df.index = network['link_id'].to_numpy()
    df['river_velocity']=0.3
    df['lambda1']=0
    df['lambda2']=0
    df['max_storage']=100
    df['infiltration']=2
    df['percolation']=1
    df['surface_velocity']=0.1
    df['alfa3'] = 1
    df['alfa4']=1
    df['temp_threshold']=0
    df['melt_factor']=1
    return df
