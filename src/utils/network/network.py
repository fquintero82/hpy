import pandas as pd
import numpy as np
from utils.params.params_from_prm_file import params_from_prm_file,params_from_prm_file_split
from utils.network.network_symbolic import set_routing_expression
import os
from os.path import splitext
import pickle
import multiprocessing


NETWORK_NAMES ={
    'link_id':np.uint32,
    'idx':np.uint32,
    'downstream_link':np.int32,
    'idx_downstream_link':np.int32,
    'upstream_link':object,
    'idx_upstream_link':object,
    'channel_length':np.float32,
    'area_hillslope':np.float32,
    'drainage_area':np.float32
    }

NETWORK_UNITS ={
    'link_id':'',
    'idx':'',
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

def get_network_from_file(options=None):
    if 'network' not in list(options.keys()):
        print('Error. No network option in yaml')
        quit()
    f = options['network']        
    if os.path.isfile(f)==False:
        print('Error. Network file not found')
        quit()
    
    _, extension = os.path.splitext(f)
    if extension =='.pkl':
        try:
            df = pd.read_pickle(f)
            return df
        except AttributeError as e:
            print('network pickle file created with different version of pandas')
            print(e)
        quit()

    if extension =='.rvr':
        if 'parameters' not in list(options.keys()):
            print('Error. No parameter option in yaml')
            quit()
        prm = options['parameters']
        # flag, f = check_if_pkl_exists(f)
        df = combine_rvr_prm(prm,f)
        return df
    
def update_network_pickle(network:pd.DataFrame,fileout:str):
    network.to_pickle(fileout)

def check_if_pkl_exists(rvr_file:str):
    r, ext = os.path.splitext(rvr_file)
    f = r+'.pkl'
    flag = False
    if os.path.isfile(f)==True:
        flag=True
    return flag, f

def get_idx_up_down(df)->pd.DataFrame:
    print('indexing')
    upstream_link = np.array(df['upstream_link']) #get  upstream linkids
    #_up1 = np.array([np.min(x) for x in _up])
    link_id = df['link_id'].to_numpy()
    idx = df['idx'].to_numpy()
    #by default, no upstream links
    idx_upstream_link = np.zeros(shape=(len(idx)),dtype=object)
    downstream_link = -1 * np.ones(shape=(len(idx)))
    idx_downstream_link = df['idx_downstream_link'].to_numpy()

    def process_row(ii,upstream_link,link_id,idx,idx_upstream_link,downstream_link,idx_downstream_link):
        
        if ii % 10000 ==0:
            print('row %s'%ii)
        _up = upstream_link[ii]
        _mylink = link_id[ii]
        _myidx = idx[ii]
        if(np.array([_up ==-1]).any()):
            #por definicion, idx_upstream_link es cero en las cabeceras
            idx_upstream_link[ii]=np.array([0],dtype=np.int32)
        if(np.array([_up !=-1]).any()):
            wh = np.where(np.isin(link_id, _up))[0]
            _upidx = idx[wh]
            idx_upstream_link[ii] = np.array(_upidx ,dtype=np.int32)
            downstream_link[wh] = _mylink
            idx_downstream_link[wh]= _myidx
        return True
    
    ii = np.arange(df.shape[0])
    #ii = [0,1,2]
    for i in ii:
        process_row(i,upstream_link,link_id,idx,idx_upstream_link,downstream_link,idx_downstream_link)
    df['idx_upstream_link'] = idx_upstream_link
    df['downstream_link'] = downstream_link
    df['idx_downstream_link'] = idx_downstream_link
    return df

# def network_from_rvr_file(rvr_file)->pd.DataFrame:
#     def get_lid(line:str):
#         try:
#             items = line.split()
#             _lid = np.int32(items[0])
#             _n = int(items[1])
#             _uplinks = -1
#             if(_n>0):
#                 _uplinks = np.array(items[2:],dtype=np.int32)
#             return _lid
#         except ValueError as e:
#             print('Error reading rvr file at line %s'%line)
#             print(e)
#             quit()
    
#     def get_uplink(line:str):
#         items = line.split()
#         _lid = int(items[0])
#         _n = int(items[1])
#         _uplinks = -1
#         if(_n>0):
#             _uplinks = np.array(items[2:],dtype=np.int32)
#         return _uplinks

#     def reshape_rvr(data):
#         n = len(data)
#         x = data[2:n:3]
#         y = data[3:n:3]
#         z = [x + y for i in range(n)]

#     f = open(rvr_file,'r')
#     data = f.readlines()
#     nlines = int(data[0])
#     df = pd.DataFrame(data=np.zeros(shape=(nlines,len(NETWORK_NAMES))),
#         columns=list(NETWORK_NAMES.keys()),
#         dtype=object)
#     df[:]=-1
#     data = data[2:]
#     _lid = list(map(get_lid,data))
#     df['link_id'] = np.array(_lid)
#     _up = list(map(get_uplink,data))
#     df['upstream_link'] = _up
#     df['idx']= np.arange(nlines) + 1 #index starts at 1. idx 0 is needed for operations
#     df.index = df[list(NETWORK_NAMES.keys())[0]]
#     df.info()
#     return df

def network_from_rvr_file(rvr_file)->pd.DataFrame:
    def get_lid(line:str):
        try:
            items = line.split()
            _lid = np.int32(items[0])
            _n = int(items[1])
            _uplinks = -1
            if(_n>0):
                _uplinks = np.array(items[2:],dtype=np.int32)
            return _lid
        except ValueError as e:
            print('Error reading rvr file at line %s'%line)
            print(e)
            quit()
    
    def get_uplink(line:str):
        items = line.split()
        _lid = int(items[0])
        _n = int(items[1])
        _uplinks = -1
        if(_n>0):
            _uplinks = np.array(items[2:],dtype=np.int32)
        return _uplinks

    def lids_and_ups1(data):
        n = len(data)
        x = data[0:n:3]
        x = [int(a) for a in x]
        y = data[1:n:3]
        lids = np.array(x,dtype=np.int32)
        uplinks = np.empty(shape=len(x),dtype=object)
        for i in range(len(x)):
            a = [eval(j) for j in y[i].split()]
            if len(a)>1:
                uplinks[i] = a[2:]
            else:
                uplinks[i] = [-1]
        return lids,uplinks

    def lids_and_ups2(data):
        data.remove('\n') #here expects 1st line to be the number of links, second the first links ant topo, and so on
        # n = len(data)
        n = int(data[0])
        x = data[1:]
        lids = np.empty(shape=len(x),dtype=np.int32)
        uplinks = np.empty(shape=len(x),dtype=object) 
        for i in range(len(x)):
            a = [eval(j) for j in x[i].split()]
            lids[i] = int(a[0])
            if a[1]>1:
                uplinks[i] = a[2:]
            else:
                uplinks[i] = [-1]
        return lids,uplinks
    
    f = open(rvr_file,'r')
    data = f.readlines()
    # data.remove('\n')
    nlines = int(data[0])
    df = pd.DataFrame(data=np.zeros(shape=(nlines,len(NETWORK_NAMES))),
        columns=list(NETWORK_NAMES.keys()),
        dtype=object)
    df[:]=-1
    try:
        #data = data[2:]
        if len(data) > 1.5*nlines:
            lids,uplinks = lids_and_ups1(data)
        else:
            lids,uplinks = lids_and_ups2(data)
        # _lid = list(map(get_lid,data))
        df['link_id'] = np.array(lids)
        # _up = list(map(get_uplink,data))
        df['upstream_link'] = uplinks
        df['idx']= np.arange(nlines) + 1 #index starts at 1. idx 0 is needed for operations
        df.index = df[list(NETWORK_NAMES.keys())[0]]
        df.info()
        return df
    except ValueError as e:
        print('Check that param files do not have blank lines')
        print(e)
        quit()


def combine_rvr_prm(prm_file,rvr_file,paramsplit=False)->pd.DataFrame:
    df1 = network_from_rvr_file(rvr_file)
    if paramsplit:
        df2 = params_from_prm_file_split(prm_file)
    else:
        df2 = params_from_prm_file(prm_file)
    print('indexing network')
    get_idx_up_down(df1)
    print('done indexing network')
    df1 = df1.iloc[:,0:6]
    df2 = df2.iloc[:,1:4]
    df = df1.merge(df2,left_index=True,right_index=True)
    df = df.astype(NETWORK_NAMES)    
    del df1, df2
    set_routing_expression(df)
    r, ext = os.path.splitext(rvr_file)
    f2 = r+'.pkl'
    print('saved network to %s'%f2)
    df.to_pickle(f2)
    return df



# test1()

