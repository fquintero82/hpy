
#""" Simplified, bmi compatible version of Hillslope Link Model"""
#https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1
import pandas as pd
import numpy as np

def getTestDF(mycase):
        if mycase =='pd_states':
            out = pd.DataFrame(
            data=np.zeros((3,7)),
            columns=['link_id','discharge','static','surface',
            'subsurface','groundwater','snow'])
            out['link_id']=np.array([1,2,3])
            out.index = out['link_id']
            return out
        elif mycase =='pd_params':
            out = pd.DataFrame(
            data=np.zeros((3,15)),
            columns=['link_id',
            'length', 'area_hillslope','drainage_area',
            'v0','lambda1','lambda2','max_storage','infiltration',
            'percolation','alfa2','alfa3','alfa4',
            'temp_thres','melt_factor'])
            out['link_id']=np.array([1,2,3])
            out.index = out['link_id']
            return out
        elif mycase == 'pd_forcings':
            out = pd.DataFrame(
            data=np.zeros((3,6)),
            columns=['link_id',
            'precipitation','evapotranspiration','temperature',
            'frozen_ground','discharge'])
            out['link_id']=np.array([1,2,3])
            out.index = out['link_id']
            return out
        elif mycase=='pd_network':
            out = pd.DataFrame(
            data=np.array([[1,3,None],[2,3,None],[3,None,[1,2]]],
            dtype=object),
            columns=['link_id',
            'downstream_link','upstream_link'])
            out.index = out['link_id']
            return out

def solve_one_link(pd_states,pd_forcings,pd_params,pd_network,mylink):
        #test only
        # pd_states= instance.pd_states
        # pd_params=instance.pd_params
        # pd_forcings=instance.pd_forcings
        # pd_network=instance.pd_network
        # mylink = 1
        
        L = pd_params.loc[mylink,'length']
        A_h = pd_params.loc[mylink,'area_hillslope']
        A_i = pd_params.loc[mylink,'drainage_area']
        c_1 = (1./1000.)*(1/60.) #factor .converts [mm/hr] to [m/min]
        rainfall = pd_forcings.loc[mylink,'precipitation'] * c_1 #rainfall. from [mm/hr] to [m/min]
        e_pot = pd_forcings.loc[mylink,'evapotranspiration'] * (1e-3 / (30.0*24.0*60.0)) #potential et[mm/month] -> [m/min]
        temperature = pd_forcings.loc[mylink,'temperature'] # daily temperature in Celsius
        temp_thres=pd_params.loc[mylink,'temp_thres'] # celsius degrees
        melt_factor = pd_params.loc[mylink,'melt_factor'] *(1/(24*60.0)) *(1/1000.0) # mm/day/degree to m/min/degree
        frozen_ground = pd_forcings.loc[mylink,'frozen_ground'] # 1 if ground is frozen, 0 if not frozen 
        

        #INITIAL VALUES
        h1 = pd_states.loc[mylink,'static'] # static storage [m]
        h2 = pd_states.loc[mylink,'surface'] #water in the hillslope surface [m]
        h3 = pd_states.loc[mylink,'subsurface'] # water in the gravitational storage in the upper part of soil [m]
        h4 = pd_states.loc[mylink,'groundwater']  # water in the aquifer storage [m]
        h5 = pd_states.loc[mylink,'snow'] #snow storage [m]
        q =  pd_states.loc[mylink,'discharge']  # discharge [m^3/s]

        #snow storage
        x1 =0
        end_SNOW=h5
        #temperature =0 is the flag for no forcing the variable. no snow process
        if(temperature==0):
            x1 = rainfall
            #self.end_SNOW +=0 #no changes to snow
        else:
            if(temperature>=temp_thres):
                snowmelt = min(h5,temperature * melt_factor)# in [m]
                end_SNOW -= snowmelt #melting outs of snow storage
                x1 = rainfall + snowmelt # in [m]
               
            if(temperature != 0 and temperature <temp_thres):
                end_snow += rainfall #all precipitation is stored in the snow storage
                x1=0

        #static storage
        end_STATIC=h1
        Hu = pd_params.loc[mylink,'max_storage']/1000 # max available storage in static tank [mm] to [m]
        x2 = max(0,x1 + h1 - Hu ) # excedance flow to the second storage [m] [m/min] check units
        #if ground is frozen, x1 goes directly to the surface
        #therefore nothing is diverted to static tank
        if(frozen_ground == 1):
            x2 = x1
        d1 = x1 - x2 # the input to static tank [m/min]
        out1 = min(e_pot, h1) # evaporation from the static tank. it cannot evaporate more than h1 [m]
        end_STATIC += (d1 - out1) # equation of static storage

        #surface storage
        end_SURFACE=h2
        infiltration = pd_params.loc[mylink,'infiltration'] * c_1 #infiltration rate [m/min]
        if(frozen_ground == 1):
            infiltration = 0
        x3 = min(x2, infiltration) #water that infiltrates to gravitational storage [m/min]
        d2 = x2 - x3 # the input to surface storage [m] check units
        out2=0
        alfa2 = pd_params.loc[mylink,'alfa2'] # velocity in m/s
        w = alfa2 * L / A_h  * 60 # [1/min]
        w = min(1,w) #water can take less than 1 min (dt) to leave surface
        out2  = h2 * w #direct runoff [m/min]
        end_SURFACE  += (d2 - out2) # 

        #SUBSURFACE storage
        end_SUBSUR = h3
        percolation = pd_params.loc[mylink,'percolation']*c_1 # percolation rate to aquifer [m/min]
        x4 = min(x3,percolation) #water that percolates to aquifer storage [m/min]
        d3 = x3 - x4 # input to gravitational storage [m/min]
        out3=0
        alfa3 = pd_params.loc[mylink,'alfa3']* 24*60 #residence time [days] to [min].
        if(alfa3>=1):
            out3 = h3/alfa3 #interflow [m/min]
        end_SUBSUR += (d3 - out3) #differential equation for gravitational storage

		#aquifer storage
        end_GW = h4
        d4 = x4
        out4=0
        alfa4 = pd_params.loc[mylink,'alfa4']* 24*60 #residence time [days] to [min].
        if(alfa4>=1):
            out4 = h4/alfa4 # base flow [m/min]
        end_GW  += (d4 - out4) #differential equation for aquifer storage

        #channel storage
        lambda_1 = pd_params.loc[mylink,'lambda1']
        lambda_2 = pd_params.loc[mylink,'lambda2']
        v_0 = pd_params.loc[mylink,'v0']
        invtau = 60.0 * v_0 * (A_i ** lambda_2) / ((1.0 - lambda_1)*L) #[1/min]  invtau
        c_2 = A_h / 60.0
        end_DISCHARGE=q
        aux = (out2 + out3 + out4) * c_2 #[m/min] to [m3/s]
        end_DISCHARGE +=  aux
        my_upstream_links = pd_network.loc[mylink,'upstream_link']
        if(my_upstream_links !=None):
            num_parents = len(pd_network.loc[mylink,'upstream_link'])
            for i in range(num_parents):
                q_up = pd_states.loc[my_upstream_links[i],'discharge']
                end_DISCHARGE += q_up
        qout = invtau * (q ** lambda_1) # *ans[discharge]
        end_DISCHARGE -= qout

        data_out=[mylink, end_DISCHARGE,end_STATIC,end_SURFACE,end_SUBSUR,end_GW,end_SNOW]
        return data_out

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
    


class HLM(object):
    """Creates a new HLM model

    PARAMETERS
    ----------
    pd_states: pandas dataframe
        columns are six model states for each link: link_id, discharge, static,
        surface,subsurface,groundwater,snow
        one row per hillslope-link
        pd_states index is link id

    pd_params: pandas dataframe
        columns are hlm parameters: link_id,length, area_hillslope,drainage_area,
        v_0,lambda1,lambda2,max_storage,infiltration,percolation,alfa1,alfa2,alfa3,
        temp_thres,melt_factor
        one row per hillslope-link
        pd_params index is link id

    pd_forcings: pandas dataframe
        columns are five forcings: link_id,precipitation,evapotranspiration,temperature,
        frozen_ground,discharge
        one row per hillslope-link
        pd_forcings index is link id

    pd_network: pandas dataframe
        columns are link_id, downstream_link, upstream_link (array)

    """
    def __init__(self):
        self.name='HLM'
        self.time=0.0
        self.next_state = None
        self.pd_states= getTestDF('pd_states')
        self.pd_params= getTestDF('pd_params')
        self.pd_forcings=getTestDF('pd_forcings')
        self.pd_network= getTestDF('pd_network')
    
    
    

    @property
    def set_time(self):
        """Current model time."""
        return self.time

    def advance_in_time(self):
        """Calculate model states for the next time step"""
        pd_states_new = self.pd_states.copy()
        for ii in range(len(self.pd_network.index)):
            mylink = self.pd_states.index[ii]
            aux = solve_one_link(
            self.pd_states,
            self.pd_forcings,
            self.pd_params,
            self.pd_network,
            mylink)
            pd_states_new.loc[mylink] = aux
        

    def main():
        instance = HLM()
        

    #if __name__ == "__main__":
    #    main()