import pandas as pd
import numpy as np
from test_dataframes import getTestDF1
from model400names import PARAM_NAMES,STATES_NAMES,FORCINGS_NAMES
from utils.network.network import NETWORK_NAMES
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time as mytime
import swifter

CF_MMHR_M_MIN = np.float32(1./1000.)*(1/60.) #factor .converts [mm/hr] to [m/min]
CF_MELTFACTOR= np.float32((1/(24*60.0)) *(1/1000.0)) # mm/day/degree to m/min/degree
CF_ET = np.float32((1e-3 / (30.0*24.0*60.0)))
CF_METER_TO_MM = 1000
CF_DAYS_TO_MINUTES = 24 * 60

def func(X):
    #snow storage
    x1 =0
    if X['temperature'] ==0:
        x1 = X['precipitation'] * CF_MMHR_M_MIN * X['DT'] #[m]
    snowmelt =0
    if X['temperature']>=X['temp_threshold']:
        snowmelt = min([X['snow'],X['temperature']*X['melt_factor']*CF_MELTFACTOR * X['DT']])
        X['snow']-=snowmelt
        x1 = (CF_MMHR_M_MIN*X['DT']*X['precipitation']) + snowmelt #[m]  
    if (X['temperature'] !=0) & (X['temperature']<X['temp_threshold']):
        X['snow'] += CF_MMHR_M_MIN*X['DT']*X['precipitation'] #[m]
        x1 = 0
    X['basin_precipitation'] = X['precipitation']*X['area_hillslope']#[mm x m2]
    X['basin_swe'] = CF_METER_TO_MM * X['snow'] * X['area_hillslope'] #[mm x m2]
    
    #static
    x2 = max([x1 + X['static'] - X['max_storage']/1000.,0])
    if X['frozen_ground']==1:
        x2 = x1
    d1 = x1 - x2 # the input to static tank [m/min]
    out1= min([X['evapotranspiration']*CF_ET*X['DT'],  X['static']])         #[m]
    X['static'] += d1 - out1
    X['basin_evapotranspiration'] = CF_METER_TO_MM*out1*X['area_hillslope'] #[mm x m2]
    X['basin_static'] = CF_METER_TO_MM*X['static']*X['area_hillslope'] # [mm x m2]

    #surface storage
    infiltration = X['infiltration'] * CF_MMHR_M_MIN * X['DT'] #infiltration rate [m/min] to [m]
    if X['frozen_ground']==1:
        infiltration=0
    x3 = min([x2,infiltration])
    d2 = x2 -x3
    w = X['surface_velocity']* 60 *X['channel_length'] / X['area_hillslope']
    w = min([1,w])
    out2 = X['surface'] * w * X['DT']
    out2 = min([out2,X['surface']])
    X['surface']+=d2 - out2
    X['basin_surface']= CF_METER_TO_MM*out2 *X['area_hillslope']

    #subsurface storage
    percolation = X['percolation'] * CF_MMHR_M_MIN * X['DT'] # percolation rate to aquifer [m/min] to [m]
    x4 = min([x3,percolation])
    d3 = x3 -x4
    out3  = X['DT'] * X['subsurface'] / (X['tr_subsurface']* CF_DAYS_TO_MINUTES)#[m]
    X['subsurface'] += d3 - out3 #[m]
    X['basin_subsurface'] = CF_METER_TO_MM*out3 *X['area_hillslope']

    #aquifer storage
    d4 = x4
    out4= X['DT'] * X['groundwater'] / (X['tr_groundwater']* CF_DAYS_TO_MINUTES) #[m]
    X['groundwater'] += d4 - out4
    X['basin_groundwater'] = CF_METER_TO_MM*out4 *X['area_hillslope']
   
   #channel update
    X['volume'] += (out2 + out3 + out4) * X['area_hillslope'] #[m]*[m2]  = [m3]
    


def runoff1(states:pd.DataFrame,
    forcings:pd.DataFrame,
    params:pd.DataFrame,
    network:pd.DataFrame,
    time_step_sec:int):
    
    DT = time_step_sec / 60. #minutes
    if check_input_names(states=states,forcings=forcings,params=params,network=network)==False:
        return
    if check_input_values(states=states,forcings=forcings,params=params,network=network,DT=DT)==False:
        return
    X = pd.concat([states,forcings,params,network],axis=1)
    X['DT']=DT
    t1 = mytime.time()
    df = X.swifter.apply(func,axis=1)
    print('completed runoff in %f sec'%(mytime.time()-t1))

def check_input_names(states:pd.DataFrame,
    forcings:pd.DataFrame,
    params:pd.DataFrame,
    network:pd.DataFrame):
    flag = True

    for i in STATES_NAMES:
        if(i not in states.columns):
            flag = False
            print('column {} in dataframe state was not found'.format(i))
            return flag
    
    for i in PARAM_NAMES:
        if(i not in params.columns):
            flag = False
            print('column {} in dataframe params was not found'.format(i))
            return flag
    
    for i in FORCINGS_NAMES:
        if(i not in forcings.columns):
            flag = False
            print('column {} in dataframe forcings was not found'.format(i))
            return flag
    
    for i in NETWORK_NAMES:
        if(i not in network.columns):
            flag = False
            print('column {} in dataframe network was not found'.format(i))
            return flag
    
    return flag

def check_input_values(states:pd.DataFrame,
    forcings:pd.DataFrame,
    params:pd.DataFrame,
    network:pd.DataFrame,
    DT:int):
    flag=True
    
    flag = (DT==0)
    if flag==True:
        print("Error DT is zero")
        return False
    
    flag = network['channel_length'].to_numpy().all()
    if flag==False:
        print("Error Parameter channel_length has zeros")
        return False
    
    flag = network['area_hillslope'].to_numpy().all()
    if flag==False:
        print("Error Parameter area_hillslope has zeros")
        return False
    
    flag = params['tr_subsurface'].to_numpy().all()
    if flag==False:
        print("Error Parameter tr_subsurface has zeros")
        return False
    
    flag = params['tr_groundwater'].to_numpy().all()
    if flag==False:
        print("Error Parameter tr_groundwater has zeros")
        return False
    
    flag = states.index.to_numpy().all()
    if flag==False:
        print("Error States cannot have linkid index zero")
        return False
    
    flag = np.array(states['static'].to_numpy()>=0.5).all()
    if flag==True:
        print("static storage cannot be larger than 0.5m")
        quit()
    
    flag = np.array(states['static'].to_numpy()<0).all()
    if flag==True:
        print("static storage cannot be negative")
        quit()

    flag = np.array(states['surface'].to_numpy()<0).all()
    if flag==True:
        print("surface storage cannot be negative")
        quit()

    flag = np.array(states['subsurface'].to_numpy()<0).all()
    if flag==True:
        print("subsurface storage cannot be negative")
        quit()

    flag = np.array(states['groundwater'].to_numpy()<0).all()
    if flag==True:
        print("groundwater storage cannot be negative")
        quit()

    flag = np.array(states['volume'].to_numpy()<0).all()
    if flag==True:
        print("volume storage cannot be negative")
        quit()

    flag = np.array(states['discharge'].to_numpy()<0).all()
    if flag==True:
        print("discharge storage cannot be negative")
        quit()

    flag = params.index.to_numpy().all()
    if flag==False:
        print("Error Params cannot have linkid index zero")
        return False
    
    flag = forcings.index.to_numpy().all()
    if flag==False:
        print("Error Forcings cannot have linkid index zero")
        return False
    
    flag = network.index.to_numpy().all()
    if flag==False:
        print("Error Network cannot have linkid index zero")
        return False
    
    return flag


