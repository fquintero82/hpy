import pandas as pd
import numpy as np
from utils.params.params_from_prm_file import params_from_prm_file
import os

NETWORK_NAMES ={
    'link_id':np.uint32,
    'downstream_link':np.int32,
    'idx_downstream_link':np.int32,
    'upstream_link':object,
    'idx_upstream_link':object,
    'channel_length':np.float16,
    'area_hillslope':np.float32,
    'drainage_area':np.float32
    }

NETWORK_UNITS ={
    'link_id':'',
    'downstream_link':'',
    'idx_downstream_link':'',
    'upstream_link':'',
    'idx_upstream_link':'',
    'channel_length':'',
    'area_hillslope':'m2',
    'drainage_area':'m2'
    }

def get_default_network():
    f = 'examples/cedarrapids1/367813_network.pkl'
    if os.path.isfile(f)==False:
        f = '../examples/cedarrapids1/367813_network.pkl'
    if os.path.isfile(f)==False:
        return None
    df = pd.read_pickle(f)
    return df

def _process_line(f):
    line = f.readline()
    items = line.split()
    _lid = int(items[0])
    _n = int(items[1])
    _uplinks = 0
    if(_n>0):
        _uplinks = np.array(items[2:],dtype=np.int32)
    return _lid,_uplinks

def get_idx_up_down(df):
    seq = np.arange(df.shape[0])+1
    d2 = pd.DataFrame({'link_id':df['link_id'],'idx':seq})
    #df.loc[:,'idx_downstream_link']=np.NAN
    #df.loc[:,'downstream_link']= np.NAN
    #df.loc[:,'idx_upstream_link']=None #i need zeros for the ode solver
    for ii in np.arange(df.shape[0]):
        _up = df.iloc[ii]['upstream_link'] #get linkids upstream
        _mylink = df.iloc[ii]['link_id']
        _mylink_idx = ii
        if(np.array([_up !=0]).any()): #if there are no zeros in the linkids upstream
            _idx = d2.loc[_up]['idx'].to_numpy() #find the indices of those linkids
            df.iloc[ii]['idx_upstream_link']=_idx #and write them in the idx_upstream column
            df.loc[_up,'downstream_link'] = _mylink #the downstream link of those uplinks is the ii-th linkid
            df.loc[_up,'idx_downstream_link'] = _mylink_idx#and the idx_downstream is ii



def network_from_rvr_file(rvr_file):
    f = open(rvr_file,'r')
    nlines = int(f.readline())
    _ = f.readline()
    df = pd.DataFrame(data=np.zeros(shape=(nlines,len(NETWORK_NAMES))),
        columns=list(NETWORK_NAMES.keys()),
        dtype=object)
    df[:]=-1
    for ii in np.arange(nlines):
        _lid,_up = _process_line(f)
        #df.iloc[ii] = [_lid,0,0,_up,0,0,0,0] #dangerous line
        df.iloc[ii]['link_id'] = _lid
        df.iloc[ii]['upstream_link'] = _up

    f.close()
    df.index = df[list(NETWORK_NAMES.keys())[0]]
    df.info()
    return df

def combine_rvr_prm(prm_file,rvr_file):
    df1 = network_from_rvr_file(rvr_file)
    df2 = params_from_prm_file(prm_file)
    print('indexing network')
    get_idx_up_down(df1)
    print('done indexing network')
    df1 = df1.iloc[:,0:5]
    df2 = df2.iloc[:,1:4]
    df = df1.merge(df2,left_index=True,right_index=True)
    df = df.astype(NETWORK_NAMES)    
    del df1, df2
    return df

# def routing_aux(network,params):
#     routing_order = network.loc[:,['link_id','idx_downstream_link','drainage_area']]
#     #routing_order['idx_upstream_link']=np.arange(nlinks)
#     routing_order = routing_order.sort_values(by=['drainage_area'])
#     idxd = routing_order['idx_downstream_link'].to_numpy()
#     idxu = routing_order['idx_upstream_link'].to_numpy()
#     ...


def test1():
    rvr_file ='../examples/cedarrapids1/367813.rvr'
    #df = network_from_rvr_file(rvr_file)
    prm_file ='../examples/cedarrapids1/367813.prm'
    
    df = combine_rvr_prm(prm_file,rvr_file)
    df.to_pickle('../examples/cedarrapids1/367813_network.pkl')