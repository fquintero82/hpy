import pandas as pd
import numpy as np
from model400names import NETWORK_NAMES

def _process_line(f):
    line = f.readline()
    items = line.split()
    _lid = int(items[0])
    _n = int(items[1])
    _uplinks = 0
    if(_n>0):
        _uplinks = tuple(np.array(items[2:],dtype=np.int32))
    return _lid,_uplinks

def network_from_rvr_file(inputfile):
    f = open(inputfile,'r')
    nlines = int(f.readline())
    _ = f.readline()
    df = pd.DataFrame(data=np.zeros(shape=(nlines,3)),
        columns=NETWORK_NAMES,
        dtype=object
    )
    for ii in range(nlines):
        _lid,_up = _process_line(f)
        df.iloc[ii] = [_lid,0,_up]
    f.close()
    df.index = df[NETWORK_NAMES[0]]
    return df   

inputfile ='../examples/cedarrapids1/367813.rvr'
df = network_from_rvr_file(inputfile)

"""     #deprecated. growing df row by row is slow
def network_from_rvr_file1(inputfile):
    f = open(inputfile,'r')
    nlines = int(f.readline())
    _ = f.readline()
    df = pd.DataFrame()
    for ii in range(nlines):
        _lid,_up = _process_line(f)
        _df2 = pd.DataFrame({
            NETWORK_NAMES[0]: _lid,
            NETWORK_NAMES[1]:0,
            NETWORK_NAMES[2]: [_up] #this need to go in brackets to make it a tuple
            })
        df = pd.concat([df,_df2])
    f.close()
    df.index = df[NETWORK_NAMES[0]]
    return df """