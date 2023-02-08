import pandas as pd
import numpy as np
from test_dataframes import getTestDF1
from model400names import *

def runoff1(states:pd.DataFrame,
    forcings:pd.DataFrame,
    params:pd.DataFrame,
    network:pd.DataFrame,
    DT:int):
    
    if check_input_names(states=states,forcings=forcings,params=params,network=network)==False:
        return
    if check_input_values(states=states,forcings=forcings,params=params,network=network,DT=DT)==False:
        return
    
    N = len(network.index)
    CF_MMHR_M_MIN = np.float16(1./1000.)*(1/60.) #factor .converts [mm/hr] to [m/min]
    CF_MELTFACTOR= np.float16((1/(24*60.0)) *(1/1000.0)) # mm/day/degree to m/min/degree
    CF_ET = np.float16((1e-3 / (30.0*24.0*60.0)))
    
    #snow storage
    x1=pd.DataFrame({'val':0},dtype=np.float16,index=network.index)
    #temperature =0 is the flag for no forcing the variable. no snow process
    wh = forcings['temperature']==0 #maybe i should trigger this condition with temperature = none
    x1.loc[wh,'val'] = forcings['precipitation'][wh] * CF_MMHR_M_MIN * DT #[m]
    #if(temperature>=temp_thres):
    snowmelt=pd.DataFrame({'val':0},dtype=np.float16,index=network.index)
    wh = forcings['temperature']>=params['temp_threshold'] #indices where true
    snowmelt.loc[wh,'val'] = pd.DataFrame({
            'val1':states['snow'][wh],
            'val2':forcings['temperature'][wh]*params['melt_factor'][wh]*CF_MELTFACTOR * DT
        },dtype=np.float16).min(axis=1) #[m]
    states.loc[wh,'snow'] -= snowmelt['val'][wh] #[m]
    x1.loc[wh,'val'] = (CF_MMHR_M_MIN*DT*forcings['precipitation'][wh]) + snowmelt['val'][wh] #[m]
    #if(temperature != 0 and temperature <temp_thres):
    wh = (forcings['temperature'] !=0) & (forcings['temperature']<params['temp_threshold']) 
    states.loc[wh,'snow'] += CF_MMHR_M_MIN*DT*forcings['precipitation'][wh] #[m]
    x1.loc[wh,'val'] = 0
    del snowmelt #garbage collection

    #static storage
    x2=pd.DataFrame({
        'val1':0,
        'val2': x1['val'] + states['static'] - params['max_storage']/1000
    },dtype=np.float16).max(axis=1) #[m]
    x2 = pd.DataFrame({'val':x2},dtype=np.float16)
    #if ground is frozen, x1 goes directly to the surface
    #therefore nothing is diverted to static tank
    wh = forcings['frozen_ground']==1
    x2[wh] = x1[wh]
    d1 = x1 - x2 # the input to static tank [m/min]
    out1= pd.DataFrame({
        'val1':forcings['evapotranspiration']*CF_ET*DT, #mm/month to m/min to m
        'val2':states['static']
        },dtype=np.float16).min(axis=1) #[m]
    out1=pd.DataFrame({'val':out1})
    states['static'] += d1['val'] - out1['val']
    del d1,x1

    #surface storage
    infiltration = params['infiltration'] * CF_MMHR_M_MIN * DT #infiltration rate [m/min] to [m]
    #if(frozen_ground == 1):
    wh = forcings['frozen_ground']==1
    infiltration[wh]=0
    x3 = pd.DataFrame({
       'val1' : x2['val'],
        'val2':infiltration
    },dtype=np.float16).min(axis=1) #[m]
    d2 = x2['val'] - x3 # the input to surface storage [m]
    w=pd.Series(params['surface_velocity'] * params['channel_length'] / params['area_hillslope'] * 60,dtype=np.float16) #[1/min]
    # water can take less than 1 min (dt) to leave surface
    w=pd.DataFrame({'val1':1,
        'val2':w},dtype=np.float16).min(axis=1)
    out2 = pd.Series((states['surface'] * w * DT), dtype=np.float16)  #[m]
    states['surface']+= (d2 - out2) #[m]
    del x2,w,d2,infiltration

    #subsurface storage
    percolation = pd.Series(params['percolation'] * CF_MMHR_M_MIN * DT,dtype=np.float16) # percolation rate to aquifer [m/min] to [m]
    x4 = pd.DataFrame({
        'val1':x3,
        'val2':percolation
    },dtype=np.float16).min(axis=1) #[m]
    d3 = x3 - x4 # input to gravitational storage [m]
    #out3=pd.DataFrame({0}, dtype=np.float16,index=np.arange(N))
    out3  = pd.Series(DT * states['subsurface'] / (params['alfa3']* 24*60),dtype=np.float16) #[m]
    states['subsurface'] += (d3 - out3) #[m]
    del x3,percolation,d3

	#aquifer storage
    d4 = x4
    out4= pd.Series(DT * states['groundwater'] / (params['alfa4']* 24*60),dtype=np.float16) #[m]
    states['groundwater'] += (d4 - out4)
    del x4,d4
    print('run completed')

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
    flag = params['channel_length'].to_numpy().all()
    if flag==False:
        print("Error Parameter channel_length has zeros")
        return False
    flag = params['area_hillslope'].to_numpy().all()
    if flag==False:
        print("Error Parameter area_hillslope has zeros")
        return False
    flag = params['alfa3'].to_numpy().all()
    if flag==False:
        print("Error Parameter alfa3 has zeros")
        return False
    flag = params['alfa4'].to_numpy().all()
    if flag==False:
        print("Error Parameter alfa4 has zeros")
        return False
    flag = states['link_id'].to_numpy().all()
    if flag==False:
        print("Error States cannot have linkid zero")
        return False
    flag = params['link_id'].to_numpy().all()
    if flag==False:
        print("Error Params cannot have linkid zero")
        return False
    flag = forcings['link_id'].to_numpy().all()
    if flag==False:
        print("Error Forcings cannot have linkid zero")
        return False
    flag = network['link_id'].to_numpy().all()
    if flag==False:
        print("Error Network cannot have linkid zero")
        return False
    return flag


def test_runoff1():
    states= getTestDF1('states')
    params= getTestDF1('params')
    forcings= getTestDF1('forcings')
    network= getTestDF1('network')

    old_states = states.copy()
    runoff1(states=states,
    forcings=forcings,
    params=params,
    network=network,
    DT=1440)
    print('old')
    print(old_states)
    print('new')
    print(states)


test_runoff1()