""" Simplified, bmi compatible version of Hillslope Link Model"""

import pandas as pd

def solve_one_link(pd_states,pd_forcings,pd_params,pd_network):

        L = pd_params['length']
        A_h = pd_params['area_hillslope']
        A_i = pd_params['drainage_area']
        c_1 = (1./1000.)*(1/60.) #factor .converts [mm/hr] to [m/min]
        rainfall = pd_forcings['rainfall'] * c_1 #rainfall. from [mm/hr] to [m/min]
        e_pot = pd_forcings['et'] * (1e-3 / (30.0*24.0*60.0)) #potential et[mm/month] -> [m/min]
        temperature = pd_forcings['temperature'] # daily temperature in Celsius
        temp_thres=pd_params['temp_thres'] # celsius degrees
        melt_factor = pd_params['melt_factor'] *(1/(24*60.0)) *(1/1000.0) # mm/day/degree to m/min/degree
        frozen_ground = pd_forcings['frozen'] # 1 if ground is frozen, 0 if not frozen 
        

        #INITIAL VALUES
        h1 = pd_states['static'] # static storage [m]
        h2 = pd_states['surface'] #water in the hillslope surface [m]
        h3 = pd_states['subsurface'] # water in the gravitational storage in the upper part of soil [m]
        h4 = pd_states['groundwater']  # water in the aquifer storage [m]
        h5 = pd_states['snow'] #snow storage [m]
        q =  pd_states['discharge']  # discharge [m^3/s]

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
        Hu = pd_params['max_storage']/1000 # max available storage in static tank [mm] to [m]
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
        infiltration = pd_params['infiltration'] * c_1 #infiltration rate [m/min]
        if(frozen_ground == 1):
            infiltration = 0
        x3 = min(x2, infiltration) #water that infiltrates to gravitational storage [m/min]
        d2 = x2 - x3 # the input to surface storage [m] check units
        alfa2 = pd_params['alfa2'] # velocity in m/s
        w = alfa2 * L / A_h  * 60 # [1/min]
        w = min(1,w) #water can take less than 1 min (dt) to leave surface
        out2  = h2 * w #direct runoff [m/min]
        end_SURFACE  += (d2 - out2) # 

        #SUBSURFACE storage
        end_SUBSUR = h3
        percolation = pd_params['percolation']*c_1 # percolation rate to aquifer [m/min]
        x4 = min(x3,percolation) #water that percolates to aquifer storage [m/min]
        d3 = x3 - x4 # input to gravitational storage [m/min]
        alfa3 = pd_params['alfa3']* 24*60 #residence time [days] to [min].
        if(alfa3>=1):
            out3 = h3/alfa3 #interflow [m/min]
        end_SUBSUR += (d3 - out3) #differential equation for gravitational storage

		#aquifer storage
        end_GW = h4
        d4 = x4
        alfa4 = pd_params['alfa4']* 24*60 #residence time [days] to [min].
        if(alfa4>=1):
            out4 = h4/alfa4 # base flow [m/min]
        end_GW  += (d4 - out4) #differential equation for aquifer storage

        #channel storage
        lambda_1 = pd_params['lambda1']
        lambda_2 = pd_params['lambda2']
        v_0 = pd_params['v0']
        invtau = 60.0 * v_0 * (A_i ** lambda_2) / ((1.0 - lambda_1)*L) #[1/min]  invtau
        c_2 = A_h / 60.0
        end_DISCHARGE=q
        aux = (out2 + out3 + out4) * c_2 #[m/min] to [m3/s]
        end_DISCHARGE +=  aux
        num_parents = len(pd_network['upstream_link_ids'])
        if(num_parents>0):
            my_upstream_links = pd_network['upstream_link_ids']
        for i in range(num_parents):
            q_up = pd_states['discharge'].loc(my_upstream_links[i])
            end_DISCHARGE += q_up
        qout = invtau * (q ** lambda_1) # *ans[discharge]
        end_DISCHARGE -= qout
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
        self.pd_states= None
        self.pd_params=None
        self.pd_forcings=None
        self.pd_network=None
        self.time=0.0
        self.next_state = None

    def __init__(self,
                pd_states=pd.DataFrame(columns=['link_id',
                'discharge','static','surface',
                'subsurface','groundwater','snow']),
                pd_params=pd.DataFrame(columns=['link_id',
                'length', 'area_hillslope','drainage_area',
                'v_0','lambda1','lambda2','max_storage','infiltration',
                'percolation','alfa1','alfa2','alfa3',
                'temp_thres','melt_factor']),
                pd_forcings=pd.pd.DataFrame(columns=['link_id',
                'precipitation','evapotranspiration','temperature',
                'frozen_ground','discharge']),
                pd_network=pd.pd.DataFrame(columns=['link_id',
                'downstream_link','upstream_link'])
                ):
        pass
    def __init__(self,config_file):
        pass

    @property
    def time(self):
        """Current model time."""
        return self.time

    def advance_in_time(self):
        """Calculate model states for the next time step"""
        solve_one_link(
            self.pd_states,
            self.pd_forcings,
            self.pd_params,
            self.pd_network
        )
        self.next_state = pd.copy