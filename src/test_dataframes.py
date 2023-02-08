import numpy as np
import pandas as pd
from model400names import PARAM_NAMES , NETWORK_NAMES,STATES_NAMES,FORCINGS_NAMES

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
            return out
        elif mycase =='params':
            out = pd.DataFrame(
            data=np.zeros((3,len(PARAM_NAMES))),
            columns=PARAM_NAMES)
            out['link_id']=np.array([1,2,3])
            out.index = out['link_id']
            out['area_hillslope']=1
            out['channel_length']=1
            out['alfa3']=1
            out['alfa4']=1
            out['temp_threshold']=0
            out['melt_factor']=10
            out['max_storage']=200

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
            return out