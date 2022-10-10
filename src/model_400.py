from pickle import NONE
#from models.hydrologic_model import Hydrologic_Model
import pandas as pd
from link import Link
from hillslope import Hillslope


class Model_400:
    def __init__(self) :
        self.hillslope=None
        self.link=None
        self.init_DISCHARGE=None
        self.init_SNOW=None
        self.init_STATIC=None
        self.init_SURFACE=None
        self.init_SUBSUR=None
        self.init_GW=None

        self.end_DISCHARGE=None
        self.end_SNOW=None
        self.end_STATIC=None
        self.end_SURFACE=None
        self.end_SUBSUR=None
        self.end_GW=None

    def set_Link(self,myLink):
        if type(myLink) is Link:
            self.link=myLink

    def set_Hillslope(self,myHillslope):
        if type(myHillslope) is Hillslope:
            self.hillslope=myHillslope

    def get_Link(self):
        return(self.link)

    def get_Hillslope(self):
        return self.hillslope

    def kernel1(self,pd_states,pd_forcings,pd_params):
        L = self.get_Link().get_len()
        A_h = self.get_Hillslope().get_area()
        A_i = self.getLink().get_drainage_area()
        c_1 = (1./1000.)*(1/60.) #factor .converts [mm/hr] to [m/min]
        rainfall = pd_forcings['rainfall'] * c_1 #rainfall. from [mm/hr] to [m/min]
        e_pot = pd_forcings['et'] * (1e-3 / (30.0*24.0*60.0)) #potential et[mm/month] -> [m/min]
        temperature = pd_forcings['temperature'] # daily temperature in Celsius
        temp_thres=pd_params['temp_thres'] # celsius degrees
        melt_factor = pd_params['melt_factor'] *(1/(24*60.0)) *(1/1000.0) # mm/day/degree to m/min/degree
        frozen_ground = pd_forcings['frozen'] # 1 if ground is frozen, 0 if not frozen 
        x1 =0

        #INITIAL VALUES
        h1 = self.init_STATIC # static storage [m]
        h2 = self.init_SURFACE #water in the hillslope surface [m]
        h3 = self.init_SUBSUR # water in the gravitational storage in the upper part of soil [m]
        h4 = self.init_GW # water in the aquifer storage [m]
        h5 = self.init_SNOW #snow storage [m]
        q =  self.init_DISCHARGE # discharge [m^3/s]

        #snow storage
        self.end_SNOW=h5
        #temperature =0 is the flag for no forcing the variable. no snow process
        if(temperature==0):
            x1 = rainfall
            #self.end_SNOW +=0 #no changes to snow
        else:
            if(temperature>=temp_thres):
                snowmelt = min(h5,temperature * melt_factor)# in [m]
                self.end_SNOW -= snowmelt #melting outs of snow storage
                x1 = rainfall + snowmelt # in [m]
               
            if(temperature != 0 and temperature <temp_thres):
                self.end_snow += rainfall #all precipitation is stored in the snow storage
                x1=0
        #static storage
        self.end_STATIC=h1
        Hu = pd_params['max_storage']/1000 # max available storage in static tank [mm] to [m]
        x2 = max(0,x1 + h1 - Hu ) # excedance flow to the second storage [m] [m/min] check units
        #if ground is frozen, x1 goes directly to the surface
        #therefore nothing is diverted to static tank
        if(frozen_ground == 1):
            x2 = x1
        d1 = x1 - x2 # the input to static tank [m/min]
        out1 = min(e_pot, h1) # evaporation from the static tank. it cannot evaporate more than h1 [m]
        self.end_STATIC += (d1 - out1) # equation of static storage

        #surface storage
        self.end_SURFACE=h2
        infiltration = pd_params['infiltration'] * c_1 #infiltration rate [m/min]
        if(frozen_ground == 1):
            infiltration = 0
        x3 = min(x2, infiltration) #water that infiltrates to gravitational storage [m/min]
        d2 = x2 - x3 # the input to surface storage [m] check units
        alfa2 = pd_params['alfa2'] # velocity in m/s
        w = alfa2 * L / A_h  * 60 # [1/min]
        w = min(1,w) #water can take less than 1 min (dt) to leave surface
        out2  = h2 * w #direct runoff [m/min]
        self.end_SURFACE  += (d2 - out2) # 

        #SUBSURFACE storage
        self.end_SUBSUR = h3
        percolation = pd_params['percolation']*c_1 # percolation rate to aquifer [m/min]
        x4 = min(x3,percolation) #water that percolates to aquifer storage [m/min]
        d3 = x3 - x4 # input to gravitational storage [m/min]
        alfa3 = pd_params['alfa3']* 24*60 #residence time [days] to [min].
        if(alfa3>=1):
            out3 = h3/alfa3 #interflow [m/min]
        self.end_SUBSURF += (d3 - out3) #differential equation for gravitational storage

		#aquifer storage
        self.end_GW = h4
        d4 = x4
        alfa4 = pd_params['alfa4']* 24*60 #residence time [days] to [min].
        if(alfa4>=1):
            out4 = h4/alfa4 # base flow [m/min]
        self.end_GW  += (d4 - out4) #differential equation for aquifer storage

        #channel storage
        lambda_1 = pd_params['lambda1']
        lambda_2 = pd_params['lambda2']
        v_0 = pd_params['v0']
        invtau = 60.0 * v_0 * (A_i ** lambda_2) / ((1.0 - lambda_1)*L) #[1/min]  invtau
        c_2 = A_h / 60.0
        self.end_DISCHARGE=q
        aux = (out2 + out3 + out4) * c_2 #[m/min] to [m3/s]
        self.end_DISCHARGE +=  aux
        num_parents = self.get_Link().get_num_upstream_links()
        if(num_parents>0):
            my_upstream_links = self.get_Link().get_upstream_links()
        for i in range(num_parents):
            q_up = my_upstream_links[i].get_Model().getDischarge()
            self.end_DISCHARGE += q_up
        qout = invtau * (q ** lambda_1) # *ans[discharge]
        self.end_DISCHARGE -= qout
