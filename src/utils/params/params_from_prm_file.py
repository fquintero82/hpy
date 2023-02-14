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
            'drainage_area':np.float16,
            'channel_length':np.float16,
            'area_hillslope':np.float16
        },
    )
    df.index = df['link_id']
    df.info()
    return df

inputfile ='../examples/cedarrapids1/367813.prm'
df = params_from_prm_file(inputfile)