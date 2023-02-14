import numpy as np
import pandas as pd
from model400names import PARAM_NAMES , NETWORK_NAMES,STATES_NAMES,FORCINGS_NAMES

def getDF_by_size(mycase,nlinks):
    out = None
    if mycase =='states':
        out = pd.DataFrame(
        data = np.zeros(shape=(nlinks,len(STATES_NAMES))),
        columns=STATES_NAMES)
    elif mycase =='params':
        out = pd.DataFrame(
        data=np.zeros((nlinks,len(PARAM_NAMES))),
        columns=PARAM_NAMES)
    elif mycase == 'forcings':
        out = pd.DataFrame(
        data=np.zeros((nlinks,len(FORCINGS_NAMES))),
        columns=FORCINGS_NAMES)
    elif mycase=='network':
        out = pd.DataFrame(
        data=np.zeros((nlinks,len(NETWORK_NAMES))),
        dtype=object,
        columns=NETWORK_NAMES)
    return out

def getTestDF1(mycase):
    if mycase =='states':
        out = pd.DataFrame(
        #data=np.zeros((3,7)),
        data = np.zeros(shape=(3,len(STATES_NAMES))),
        columns=STATES_NAMES)
        out['link_id']=np.array([1,2,3])
        out.index = out['link_id']
        out['static'] = 0
        out['surface']=0.1
        out['subsurface']=0.1
        out['groundwater']=0.1
        out['snow'] = 0.1
        out['discharge']=100
        return out
    elif mycase =='params':
        out = pd.DataFrame(
        data=np.zeros((3,len(PARAM_NAMES))),
        columns=PARAM_NAMES)
        out['link_id']=np.array([1,2,3])
        out.index = out['link_id']
        out['alfa3']=1
        out['alfa4']=1
        out['temp_threshold']=0
        out['melt_factor']=10
        out['max_storage']=200
        out['lambda2'] = .1
        out['river_velocity']=.05
        out['lambda1'] = 0
        out['lambda2'] = 0

        return out
    elif mycase == 'forcings':
        out = pd.DataFrame(
        data=np.zeros((3,len(FORCINGS_NAMES))),
        columns=FORCINGS_NAMES)
        out['link_id']=np.array([1,2,3])
        out['temperature']=np.float16([-10,10,0])
        out['precipitation']=np.float16([10,0,0])
        out['frozen_ground']=np.float16([1,0,0])
        out.index = out['link_id']
        return out
    elif mycase=='network':
        out = pd.DataFrame(
        data=np.array([[1,3,0],[2,3,0],[3,0,[1,2]]],
        dtype=object),
        columns=NETWORK_NAMES)
        out.index = out['link_id']
        out['area_hillslope']=100
        out['channel_length']=100
        out['drainage_area']=100
        
        return out

def getTestDF2(mycase):
    N=5
    if mycase =='states':
        out = pd.DataFrame(
        #data=np.zeros((3,7)),
        data = np.zeros(shape=(N,len(STATES_NAMES))),
        columns=STATES_NAMES)
        out['link_id']=np.arange(N)+1
        out.index = out['link_id']
        out['static'] = 0
        out['surface']=0.1
        out['subsurface']=0.1
        out['groundwater']=0.1
        out['snow'] = 0.1
        return out
    elif mycase =='params':
        out = pd.DataFrame(
        data=np.zeros((N,len(PARAM_NAMES))),
        columns=PARAM_NAMES)
        out['link_id']=np.arange(N)+1
        out.index = out['link_id']
        out['alfa3']=1
        out['alfa4']=1
        out['temp_threshold']=0
        out['melt_factor']=10
        out['max_storage']=200
        out['lambda2'] = .1
        out['river_velocity']=.3
        out['lambda1'] = .33
        out['lambda2'] = 0.1

        return out
    elif mycase == 'forcings':
        out = pd.DataFrame(
        data=np.zeros((N,len(FORCINGS_NAMES))),
        columns=FORCINGS_NAMES)
        out['link_id']=np.arange(N)+1
        out['temperature']=10
        out['precipitation']=0
        out['frozen_ground']=1
        out.index = out['link_id']
        return out
    elif mycase=='network':
        out = pd.DataFrame(
        data=np.array([[1,0,[2,3]],[2,1,[4,5]],[3,1,0],[4,2,0],[5,2,0]],
        dtype=object),
        columns=NETWORK_NAMES)
        out.index = out['link_id']
        out['area_hillslope']=100
        out['channel_length']=100
        out['drainage_area']=100
        return out

def test_multiplication():
    df1 = pd.DataFrame({'val':[1,2,3],'link_id':[1,2,3]})
    df1.index = df1['link_id']
    df2 = pd.DataFrame({'val':[1,2,3],'link_id':[4,5,6]})
    df2.index = df2['link_id']
    out = df1 * df2
    print(out)
    
    df2 = pd.DataFrame({'val':[1,2,3],'link_id':[3,4,5]})
    df2.index = df2['link_id']
    out = df1 * df2
    print(out)

    df2 = pd.DataFrame({'val':[1,2,3],'link_id':[3,4,5]})
    df2.index = df2['link_id']
    out = df1 * df2
    print(out)