from hlm import HLM
import polars as pl
import numpy as np

instance= HLM()

config_file = 'examples/cedarrapids1/cedar_example.yaml'
instance.init_from_file(config_file,option_solver=False)

a = pl.from_pandas(instance.states)
b = pl.from_pandas(instance.forcings.iloc[:,1:-1])
c = pl.from_pandas(instance.params.iloc[:,1:-1])
d = pl.from_pandas(instance.network[['channel_length','area_hillslope']],include_index=True)
N = len(instance.network)
X = pl.concat([a,b,c,d],how='horizontal')

    
CF_MMHR_M_MIN = np.float32(1./1000.)*(1/60.) #factor .converts [mm/hr] to [m/min]
CF_MELTFACTOR= np.float32((1/(24*60.0)) *(1/1000.0)) # mm/day/degree to m/min/degree
CF_ET = np.float32((1e-3 / (30.0*24.0*60.0)))
CF_METER_TO_MM = 1000
#snow storage
# x1=pd.DataFrame({'val':0},dtype=np.float32,index=network.index)
x1 = pl.zeros(N)
#temperature =0 is the flag for no forcing the variable. no snow process
wh = X['temperature']==0 #maybe i should trigger this condition with temperature = none
if len(wh)>0:
    x1.loc[wh,'val'] = X['precipitation'][wh] * CF_MMHR_M_MIN * DT #[m]      #FF
#if(temperature>=temp_thres):
snowmelt=pd.DataFrame({'val':0},dtype=np.float32,index=network.index)
wh = X['temperature']>=X['temp_threshold'] #indices where true
snowmelt.loc[wh,'val'] = pd.DataFrame({                                         #FF
        'val1':X['snow'][wh],
        'val2':X['temperature'][wh]*X['melt_factor'][wh]*CF_MELTFACTOR * DT
    },dtype=np.float32).min(axis=1) #[m]
X.loc[wh,'snow'] -= snowmelt['val'][wh] #[m]
x1.loc[wh,'val'] = (CF_MMHR_M_MIN*DT*X['precipitation'][wh]) + snowmelt['val'][wh] #[m]      #FF
#if(temperature != 0 and temperature <temp_thres):
wh = (X['temperature'] !=0) & (X['temperature']<X['temp_threshold']) 
X.loc[wh,'snow'] += CF_MMHR_M_MIN*DT*X['precipitation'][wh] #[m]
x1.loc[wh,'val'] = 0
X['basin_precipitation'] = X['precipitation'].copy()*X['area_hillslope']#[mm x m2]
X['basin_swe'] = CF_METER_TO_MM * X['snow'].copy() * X['area_hillslope'] #[mm x m2]
del snowmelt #garbage collection

#static storage
x2=pd.DataFrame({
    'val1':0,
    'val2': x1['val'] + X['static'] - X['max_storage']/1000.
},dtype=np.float32).max(axis=1) #[m]
x2 = pd.DataFrame({'val':x2},dtype=np.float32)
#if ground is frozen, x1 goes directly to the surface
#therefore nothing is diverted to static tank
wh = X['frozen_ground']==1
x2[wh] = x1[wh]
d1 = x1 - x2 # the input to static tank [m/min]
out1= pd.DataFrame({
    'val1':X['evapotranspiration']*CF_ET*DT, #mm/month to m/min to m
    'val2':X['static']
    },dtype=np.float32).min(axis=1) #[m]
out1=pd.DataFrame({'val':out1})
X['static'] += d1['val'] - out1['val']

X['basin_evapotranspiration'] = CF_METER_TO_MM*out1['val'].copy()*X['area_hillslope'] #[mm x m2]
X['basin_static'] = CF_METER_TO_MM*X['static'].copy()*X['area_hillslope'] # [mm x m2]

#del d1,x1

#surface storage
infiltration = X['infiltration'] * CF_MMHR_M_MIN * DT #infiltration rate [m/min] to [m]
#if(frozen_ground == 1):
wh = X['frozen_ground']==1
infiltration[wh]=0
x3 = pd.DataFrame({
    'val1' : x2['val'],
    'val2':infiltration
},dtype=np.float32).min(axis=1) #[m]
d2 = x2['val'] - x3 # the input to surface storage [m]
w=pd.Series(X['surface_velocity'] * 60 *X['channel_length'] / X['area_hillslope'],dtype=np.float32) #[1/min]
# water can take less than 1 min (dt) to leave surface
w=pd.DataFrame({'val1':1,
    'val2':w},dtype=np.float32).min(axis=1)
#out2 = pd.Series((states['surface'] * w * DT), dtype=np.float32)  #[m]
out2 = np.array((X['surface'] * w * DT), dtype=np.float32)  #[m]
out2 = np.minimum(out2,X['surface'])

X['surface']+= d2 - out2 #[m]
#states['basin_surface'] = 1000*states['surface'].copy()
X['basin_surface'] = CF_METER_TO_MM*out2 *X['area_hillslope']
del x2,w,d2,infiltration

#subsurface storage
percolation = pd.Series(X['percolation'] * CF_MMHR_M_MIN * DT,dtype=np.float32) # percolation rate to aquifer [m/min] to [m]
x4 = pd.DataFrame({
    'val1':x3,
    'val2':percolation
},dtype=np.float32).min(axis=1) #[m]
d3 = x3 - x4 # input to gravitational storage [m]
CF_DAYS_TO_MINUTES = 24 * 60
out3  = pd.Series(DT * states['subsurface'] / (params['tr_subsurface']* CF_DAYS_TO_MINUTES),dtype=np.float32) #[m]
states['subsurface'] += d3 - out3 #[m]
#states['basin_subsurface'] = 1000*states['subsurface'].copy()
states['basin_subsurface'] = CF_METER_TO_MM*out3 *network['area_hillslope']
del x3,percolation,d3

#aquifer storage
d4 = x4

out4= pd.Series(DT * states['groundwater'] / (params['tr_groundwater']* CF_DAYS_TO_MINUTES),dtype=np.float32) #[m]
states['groundwater'] += d4 - out4
#states['basin_groundwater'] = 1000*states['groundwater'].copy()
states['basin_groundwater'] = CF_METER_TO_MM*out4 *network['area_hillslope']
del x4,d4

#channel update
segs_in_DT = DT * 60.
states['volume'] += (out2 + out3 + out4) * network['area_hillslope'] #[m]*[m2]  = [m3]
#states['discharge'] += (out2 + out3 + out4) * network['area_hillslope'] / segs_in_DT #[m]*[m2] / [s] = [m3/s]
print('completed runoff in %f sec'%(mytime.time()-t1))
