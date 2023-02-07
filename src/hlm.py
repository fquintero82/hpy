
#""" Simplified, bmi compatible version of Hillslope Link Model"""
#https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1
import pandas as pd
import numpy as np
from test_dataframes import getTestDF1






    


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
        self.pd_states= getTestDF1('states')
        self.pd_params= getTestDF1('params')
        self.pd_forcings=getTestDF1('forcings')
        self.pd_network= getTestDF1('network')
    
    
    

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