import numpy as np
import pandas as pd

def getTestDF1(mycase):
        if mycase =='states':
            out = pd.DataFrame(
            data=np.zeros((3,7)),
            columns=['link_id','discharge','static','surface',
            'subsurface','groundwater','snow'])
            out['link_id']=np.array([1,2,3])
            out.index = out['link_id']
            return out
        elif mycase =='params':
            out = pd.DataFrame(
            data=np.zeros((3,15)),
            columns=['link_id',
            'length', 'area_hillslope','drainage_area',
            'v0','lambda1','lambda2','max_storage','infiltration',
            'percolation','alfa2','alfa3','alfa4',
            'temp_threshold','melt_factor'])
            out['link_id']=np.array([1,2,3])
            out.index = out['link_id']
            return out
        elif mycase == 'forcings':
            out = pd.DataFrame(
            data=np.zeros((3,6)),
            columns=['link_id',
            'precipitation','evapotranspiration','temperature',
            'frozen_ground','discharge'])
            out['link_id']=np.array([1,2,3])
            out['temperature']=np.float16([1,0,0])
            out['precipitation']=np.float16([1,0,0])
            out['frozen_ground']=np.float16([1,0,0])
            out.index = out['link_id']
            return out
        elif mycase=='network':
            out = pd.DataFrame(
            data=np.array([[1,3,None],[2,3,None],[3,None,[1,2]]],
            dtype=object),
            columns=['link_id',
            'downstream_link','upstream_link'])
            out.index = out['link_id']
            return out