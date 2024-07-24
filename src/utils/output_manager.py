import os
from os.path import splitext
import numpy as np
from utils.to_netcdf import save_to_netcdf


FORMAT_NETCDF =1


class OutputManager(object):
    def __init__(self,hlm_object) -> None:
        self.links_to_save=None
        self.format=None
        self.path = None
        self.path_links=None
        self.links = None
        self.dischargeonly = False
        self.variables=None

        options = hlm_object.configuration
        
        if 'output' not in list(options.keys()):
            print('Error. No output_file option in yaml')
            quit()
        f = options['output']
        if f=='default':
            pass
        if('path' not in options['output'].keys()):
            print('path for netcdf file not provided')
            quit()
        if('discharge_only' in options['output'].keys()):
            self.dischargeonly=options['output']['discharge_only']
        if('variables' in options['output'].keys()):
            items =  options['output']['variables']
            self.variables = items.split(' ')

        _, extension = os.path.splitext(options['output']['path'])
        if extension =='.nc':
            self.format = FORMAT_NETCDF
            self.path = options['output']['path']
            if('links' in options['output'].keys()):
                self.path_links = options['output']['links']

    def save(self,hlm_object):
        if self.format==FORMAT_NETCDF:            
            if self.path_links is None:
                save_to_netcdf(hlm_object.states,
                    hlm_object.params,
                    hlm_object.time,
                    self.path,
                    discharge_only=self.dischargeonly,
                    variables = self.variables)
                
            if self.path_links is not None:
                links = read_links(self.path_links)
                save_to_netcdf(hlm_object.states.loc[links],
                    hlm_object.params.loc[links],
                    hlm_object.time,
                    self.path,
                    discharge_only=self.dischargeonly,
                    variables=self.variables)
        

def read_links(path_links):
    vals = np.loadtxt(path_links,dtype=int)
    return(vals)

