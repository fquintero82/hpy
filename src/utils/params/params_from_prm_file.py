import pandas as pd
import numpy as np
from model400names import PARAM_NAMES
import sys

def params_from_prm_file(inputfile):
    df = pd.read_table(inputfile,
        header=None,
        names=['link_id','drainage_area','channel_length','area_hillslope'],
        sep=' ',
        skiprows=1,
        dtype={
            'link_id':np.uint32,
            'drainage_area':np.float32,
            'channel_length':np.float16,
            'area_hillslope':np.float32
        },
    )
    df.index = df['link_id']
    df['channel_length'] *= 1e3 # from km to m
    df['area_hillslope'] *= 1e6 #from km2 to m2
    df.info()
    return df

def test():
    inputfile ='../examples/cedarrapids1/367813.prm'
    df = params_from_prm_file(inputfile)
    df = df.sort_values(by=['drainage_area'])