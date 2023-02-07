import pandas as pd
import numpy as np
from test_dataframes import getTestDF1

def runoff1(states:pd.DataFrame,
    forcings:pd.DataFrame,
    params:pd.DataFrame,
    network:pd.DataFrame,
    DT:int):
    
    N = len(network.index)
    CF_MMHR_M_MIN = (1./1000.)*(1/60.) #factor .converts [mm/hr] to [m/min]
    CF_MELTFACTOR= (1/(24*60.0)) *(1/1000.0) # mm/day/degree to m/min/degree
    CF_ET = (1e-3 / (30.0*24.0*60.0))
    
    #snow storage
    x1=pd.DataFrame({0},dtype=np.float16,index=np.arange(N))
    #temperature =0 is the flag for no forcing the variable. no snow process
    x1[forcings['temperature']==0] = forcings['precipitation'] * CF_MMHR_M_MIN * DT #[m]
    #if(temperature>=temp_thres):
    snowmelt=pd.DataFrame({0},dtype=np.float16,index=np.arange(N))
    wh = forcings['temperature']>=params['temp_threshold'] #indices where true
    snowmelt[wh] = pd.Dataframe({
            states['snow'][wh],
            (forcings['temperature']*params['melt_factor']*CF_MELTFACTOR * DT)[wh]
        }).min(axis=1) #[m]
    states['snow'][wh] -= snowmelt[wh] #[m]
    x1[wh] = (CF_MMHR_M_MIN*DT*forcings['precipitation'])[wh] + (snowmelt)[wh] #[m]
    #if(temperature != 0 and temperature <temp_thres):
    wh = (forcings['temperature'] !=0) & (forcings['temperature']<params['temp_threshold']) 
    states['snow'][wh] += (CF_MMHR_M_MIN*DT*forcings['precipitation'])[wh] #[m]
    x1[wh] = 0
    
    #static storage
    x2=pd.DataFrame({
        0,
        x1 + states['static'] - params['max_storage']/1000
    }).max(axis=1) #[m]
    #if ground is frozen, x1 goes directly to the surface
    #therefore nothing is diverted to static tank
    wh = forcings['frozen_ground']==1
    x2[wh] = x1[wh]
    d1 = x1 - x2 # the input to static tank [m/min]
    out1= pd.DataFrame({
        forcings['et']*CF_ET*DT, #mm/month to m/min to m
        states['static']
        }).min(axis=1) #[m]
    states['static']+= (d1 - out1)

    #surface storage
    infiltration = params['infiltration'] * CF_MMHR_M_MIN * DT #infiltration rate [m/min] to [m]
    #if(frozen_ground == 1):
    wh = forcings['frozen_ground']==1
    infiltration[wh]=0
    x3 = pd.DataFrame({
        x2,
        infiltration
    }).min(axis=1) #[m]
    d2 = x2 - x3 # the input to surface storage [m]
    w=params['velocity'] * network['channel_length'] / network['area_hillslope'] * 60 #[1/min]
    # water can take less than 1 min (dt) to leave surface
    w=pd.DataFrame({1,w}).min(axis=1)
    out2 = states['surface']* w * DT #[m]
    states['surface']+= (d2 - out2) #[m]

    #subsurface storage
    percolation = params['percolation'] * CF_MMHR_M_MIN * DT # percolation rate to aquifer [m/min] to [m]
    x4 = pd.DataFrame({
        x3,
        percolation
    }).min(axis=1) #[m]
    d3 = x3 - x4 # input to gravitational storage [m]
    out3=pd.DataFrame({0}, dtype=np.float16,index=np.arange(N))
    alfa3 = params['alfa3']* 24*60 #residence time [days] to [min].
    out3[alfa3>=1] = DT * states['subsurface'] / alfa3 #[m]
    states['subsurface'] += (d3 - out3) #[m]

	#aquifer storage
    d4 = x4
    out4=pd.DataFrame({0},dtype=np.float16,index=np.arange(N))
    alfa4 = params['alfa4']* 24*60 #residence time [days] to [min].
    out4[alfa4>=1]= DT * states['groundwater'] / alfa4 #[m]
    states['groundwater'] += (d4 - out4)

def check_variables(states:pd.DataFrame,
    forcings:pd.DataFrame,
    params:pd.DataFrame,
    network:pd.DataFrame):
    flag = True
    
    name_states = ['link_id','snow','static','surface','subsurface','groundwater']
    for i in name_states:
        if(i not in states.columns):
            flag = False
            print('column {} in dataframe state was not found'.format(i))
            return flag
    
    name_params=['link_id',
            'length', 'area_hillslope','drainage_area',
            'v0','lambda1','lambda2','max_storage','infiltration',
            'percolation','alfa2','alfa3','alfa4',
            'temp_thres','melt_factor']
    for i in name_params:
        if(i not in params.columns):
            flag = False
            print('column {} in dataframe params was not found'.format(i))
            return flag
    
    name_forcings=['link_id',
            'precipitation','evapotranspiration','temperature',
            'frozen_ground','discharge']
    for i in name_forcings:
        if(i not in forcings.columns):
            flag = False
            print('column {} in dataframe forcings was not found'.format(i))
            return flag
    
    name_network=['link_id',
            'downstream_link','upstream_link']
    for i in name_network:
        if(i not in network.columns):
            flag = False
            print('column {} in dataframe network was not found'.format(i))
            return flag
    
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
    DT=1)


test_runoff1()