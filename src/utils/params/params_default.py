import pandas as pd
from models.model400names import PARAM_NAMES,PARAM_DEFAULT_VALUES
import numpy as np

def get_default_params(network:pd.DataFrame):
    nlinks = network.shape[0]
    df = pd.DataFrame(
        data = np.zeros(shape=(nlinks,len(PARAM_NAMES))),
        columns=list(PARAM_NAMES.keys()))
    df = df.astype(PARAM_NAMES)
    aux = list(PARAM_NAMES.keys())
    for ii in range(len(PARAM_NAMES)):
        df.loc[:,aux[ii]] = np.array(PARAM_DEFAULT_VALUES[aux[ii]],dtype=PARAM_NAMES[aux[ii]])

    df['link_id'] = network['link_id'].to_numpy()
    df.index = network['link_id'].to_numpy()
    return df
