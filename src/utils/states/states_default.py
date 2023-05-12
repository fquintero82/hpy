import pandas as pd
from model400names import STATES_NAMES, STATES_DEFAULT_VALUES
import numpy as np

def get_default_states(network:pd.DataFrame):
    nlinks = network.shape[0]
    df = pd.DataFrame(
        data = np.zeros(shape=(nlinks,len(STATES_NAMES))),
        columns=list(STATES_NAMES.keys()))
    df = df.astype(STATES_NAMES)
    aux = list(STATES_NAMES.keys())
    for ii in range(len(STATES_NAMES)):
        df.loc[:,aux[ii]] = np.array(STATES_DEFAULT_VALUES[aux[ii]],dtype=STATES_NAMES[aux[ii]])
    df['link_id'] = network['link_id'].to_numpy()
    df.index = network['link_id'].to_numpy()
    
    return df