import pandas as pd
import numpy as np
from utils.params.params_from_prm_file import params_from_prm_file
import os
import pickle


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
    if 'network' in list(options.keys()):
        f = options['network']    
    if os.path.isfile(f)==False:
        print('Error. Network file not found')
        quit()
    df = pd.read_pickle(f)
    return df

def get_idx_up_down(df):
    for ii in np.arange(df.shape[0]):
        print(ii)
        #get  upstream linkids
        _up = df.iloc[ii]['upstream_link'] 
        #get  my linkid
        _mylink = df.iloc[ii]['link_id']
        #get my index
        _myidx = df.iloc[ii]['idx']
        if(np.array([_up ==-1]).any()):
            # for mylink, set the idx_upstream column
            #df.iloc[ii]['idx_upstream_link']=0 #this is necesary for ode evaluation
            df.iloc[ii]['idx_upstream_link']=np.array([0],dtype=np.int32) #this is necesary for ode evaluation
        if(np.array([_up !=-1]).any()): #if there are no zeros in the linkids upstream
             #get the index of upstream links
            _upidx = df.loc[_up]['idx'].to_numpy()
            # for mylink, set the idx_upstream column
            #df.iloc[ii]['idx_upstream_link']=_upidx 
            df.iloc[ii]['idx_upstream_link']=np.array(_upidx ,dtype=np.int32)
            #for the upstream links,set their downstream link (mylink)
            df.loc[_up,'downstream_link'] = _mylink 
            #for the upstream links, set  their idx_downstream
            df.loc[_up,'idx_downstream_link'] = _myidx 

def get_idx_up_down(df):
    for ii in np.arange(df.shape[0]):
        print(ii)
        #get  upstream linkids
        _up = df.iloc[ii]['upstream_link'] 
        #get  my linkid
        _mylink = df.iloc[ii]['link_id']
        #get my index
        _myidx = df.iloc[ii]['idx']
        if(np.array([_up ==-1]).any()):
            # for mylink, set the idx_upstream column
            #df.iloc[ii]['idx_upstream_link']=0 #this is necesary for ode evaluation
            df.iloc[ii]['idx_upstream_link']=np.array([0],dtype=np.int32) #this is necesary for ode evaluation
        if(np.array([_up !=-1]).any()): #if there are no zeros in the linkids upstream
             #get the index of upstream links
            _upidx = df.loc[_up]['idx'].to_numpy()
            # for mylink, set the idx_upstream column
            #df.iloc[ii]['idx_upstream_link']=_upidx 
            df.iloc[ii]['idx_upstream_link']=np.array(_upidx ,dtype=np.int32)
            #for the upstream links,set their downstream link (mylink)
            df.loc[_up,'downstream_link'] = _mylink 
            #for the upstream links, set  their idx_downstream
            df.loc[_up,'idx_downstream_link'] = _myidx 

def get_idx_up_down2(df):
    # Get the index of upstream links.
    #df['idx_upstream_link'] = df['upstream_link'].apply(lambda x: df.loc[x]['idx'].to_numpy() if x != -1 else np.array([0]))
    df['idx_upstream_link'] = df['upstream_link'].apply(lambda x: df.loc[x]['idx'].to_numpy() if min(x) != -1 else np.array([0]))

    # Set the downstream link and idx_downstream for upstream links.
    up_idxs = df['idx_upstream_link'].values
    down_links = df['link_id'].values
    for i in range(len(df)):
        if up_idxs[i] != 0:
            df.loc[up_idxs[i], 'downstream_link'] = down_links[i]
            df.loc[up_idxs[i], 'idx_downstream_link'] = i


def get_idx_up_down3(df):
    _up = np.array(df['upstream_link']) #get  upstream linkids
    _up1 = np.array([np.min(x) for x in _up])
    _mylink = df['link_id']
    _myidx = df['idx']

    _upidx = df.loc[_up]['idx'].to_numpy()

    wh = np.where(_up1==-1,True,False)
    df.loc[wh,'idx_upstream_link']=np.float32(0)
    wh = np.where(_up1!=-1,True,False)
    df.iloc[wh]['idx_upstream_link'] =df.iloc[wh]['idx'].to_numpy()
    _upidx = [df.loc[_up[x]]['idx'].to_numpy() for x in wh]
    for ii in np.arange(df.shape[0]):
        print(ii)
        #get  upstream linkids
        _up = df.iloc[ii]['upstream_link'] 
        #get  my linkid
        _mylink = df.iloc[ii]['link_id']
        #get my index
        _myidx = df.iloc[ii]['idx']
        if(np.array([_up ==-1]).any()):
            # for mylink, set the idx_upstream column
            #df.iloc[ii]['idx_upstream_link']=0 #this is necesary for ode evaluation
            df.iloc[ii]['idx_upstream_link']=np.array([0],dtype=np.int32) #this is necesary for ode evaluation
        if(np.array([_up !=-1]).any()): #if there are no zeros in the linkids upstream
             #get the index of upstream links
            _upidx = df.loc[_up]['idx'].to_numpy()
            # for mylink, set the idx_upstream column
            #df.iloc[ii]['idx_upstream_link']=_upidx 
            df.iloc[ii]['idx_upstream_link']=np.array(_upidx ,dtype=np.int32)
            #for the upstream links,set their downstream link (mylink)
            df.loc[_up,'downstream_link'] = _mylink 
            #for the upstream links, set  their idx_downstream
            df.loc[_up,'idx_downstream_link'] = _myidx 

def get_adjacency_matrix(network:pd.DataFrame,default=False):

    if default==False:
        nlinks = len(network)
        #A = np.eye(nlinks,dtype=np.byte)*-1
        A = np.eye(nlinks,dtype=np.float32)*-1
        for ii in np.arange(nlinks):
            idx_up = network.iloc[ii]['idx_upstream_link']
            if np.array([idx_up !=-1]).any():
                A[ii,(idx_up-1).tolist()]=1
        return A
    else:
        file1 = open('examples/cedarrapids1/367813_adj.pkl','rb')
        return pickle.load(file1)
        file1.close()


# def network_from_rvr_file(rvr_file):
#     f = open(rvr_file,'r')
#     nlines = int(f.readline())
#     _ = f.readline()
#     df = pd.DataFrame(data=np.zeros(shape=(nlines,len(NETWORK_NAMES))),
#         columns=list(NETWORK_NAMES.keys()),
#         dtype=object)
#     df[:]=-1
#     for ii in np.arange(nlines):
#         _lid,_up = _process_line(f)
#         df.iloc[ii]['link_id'] = _lid
#         df.iloc[ii]['upstream_link'] = _up
#         df.iloc[ii]['idx']= ii + 1 #index starts at 1. idx 0 is needed for operations

#     f.close()
#     df.index = df[list(NETWORK_NAMES.keys())[0]]
#     df.info()
#     return df

def network_from_rvr_file(rvr_file):
    def get_lid(line:str):
        items = line.split()
        _lid = int(items[0])
        _n = int(items[1])
        _uplinks = -1
        if(_n>0):
            _uplinks = np.array(items[2:],dtype=np.int32)
        return _lid
    
    def get_uplink(line:str):
        items = line.split()
        _lid = int(items[0])
        _n = int(items[1])
        _uplinks = -1
        if(_n>0):
            _uplinks = np.array(items[2:],dtype=np.int32)
        return _uplinks

    
    f = open(rvr_file,'r')
    data = f.readlines()
    nlines = int(data[0])
    df = pd.DataFrame(data=np.zeros(shape=(nlines,len(NETWORK_NAMES))),
        columns=list(NETWORK_NAMES.keys()),
        dtype=object)
    df[:]=-1
    data = data[2:]
    _lid = list(map(get_lid,data))
    df['link_id'] = np.array(_lid)
    _up = list(map(get_uplink,data))
    df['upstream_link'] = _up
    df['idx']= np.arange(nlines) + 1 #index starts at 1. idx 0 is needed for operations
    df.index = df[list(NETWORK_NAMES.keys())[0]]
    df.info()
    return df

def combine_rvr_prm(prm_file,rvr_file):
    df1 = network_from_rvr_file(rvr_file)
    df2 = params_from_prm_file(prm_file)
    print('indexing network')
    get_idx_up_down(df1)
    print('done indexing network')
    df1 = df1.iloc[:,0:6]
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
    rvr_file ='examples/cedarrapids1/367813.rvr'
    #df = network_from_rvr_file(rvr_file)
    prm_file ='examples/cedarrapids1/367813.prm'
    df = combine_rvr_prm(prm_file,rvr_file)
    df.to_pickle('examples/cedarrapids1/367813_network.pkl')



def test2():
    rvr_file ='examples/small/small.rvr'
    prm_file ='examples/small/small.prm'
    df = combine_rvr_prm(prm_file,rvr_file)
    df.to_pickle('examples/small/small.pkl')

def test3():
    rvr_file ='examples/hydrosheds/conus.rvr'
    prm_file ='examples/hydrosheds/conus.prm'
    df = combine_rvr_prm(prm_file,rvr_file)
    df.to_pickle('examples/hydrosheds/conus.pkl')

def testadjmat():
    network = get_default_network()
    A = get_adjacency_matrix(network,False)
    file1 = open('examples/cedarrapids1/367813_adj.pkl','wb')
    #file2 = open('examples/cedarrapids1/367813_adj.np','wb')
    pickle.dump(A,file1)
 #   np.save(file2,A) #same size as pkl
    file1.close()
#    file2.close()


