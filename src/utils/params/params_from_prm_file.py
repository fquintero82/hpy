import pandas as pd
import numpy as np
from models.model400names import PARAM_NAMES
import sys

def params_from_prm_file_old(prm_file):
    df = pd.read_table(prm_file,
        header=None,
        names=['link_id','drainage_area','channel_length','area_hillslope'],
        sep=' ',
        skiprows=1,
        dtype={
            'link_id':np.uint64,
            'drainage_area':np.float32,
            'channel_length':np.float32,
            'area_hillslope':np.float32
        },
    )
    df.index = df['link_id']
    df['channel_length'] *= 1e3 # from km to m
    df['area_hillslope'] *= 1e6 #from km2 to m2
    df['drainage_area'] *= 1e6 #from km2 to m2
    df.info()
    return df

def params_from_prm_file(prm_file):
    data = np.loadtxt(prm_file,skiprows=1)
    df = pd.DataFrame(data)
    d = {'link_id':np.uint32,            'drainage_area':np.float32,            'channel_length':np.float32,            'area_hillslope':np.float32        }
    df.columns = list(d.keys())
    df = df.astype(d)
    df.index = df['link_id']
    df['channel_length'] *= 1e3 # from km to m
    df['area_hillslope'] *= 1e6 #from km2 to m2
    df['drainage_area'] *= 1e6 #from km2 to m2
    df.info()
    return df

def test():
    inputfile ='../examples/cedarrapids1/367813.prm'
    df = params_from_prm_file(inputfile)
    df = df.sort_values(by=['drainage_area'])