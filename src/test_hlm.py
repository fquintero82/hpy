from hlm import HLM
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import netCDF4 as nc

def plot1():
    NSTEPS = 240
    NGAGE =1
    X = 367813 #oulet
    #X = 502480 #small
    
    out = np.zeros(shape=(NGAGE,NSTEPS))
    ix=0
    for tt in np.arange(start=0,stop=240*3600,step=3600):
        f = 'examples/cedarrapids1/out/{}.pkl'.format(tt)
        states = pd.read_pickle(f)
        out[0,ix] = states.loc[X,'volume']
        ix+=1
    
    plt.plot(out[0,:])
    plt.show()

def plot2():
    f ='/Users/felipe/hpy/hpy/examples/cedarrapids1/out/2008.nc'
    
def test1():
    instance = HLM()
    config_file = 'examples/cedarrapids1/cedar_imac.yaml'
    instance.init_from_file(config_file)
    for ii in range(10):
        print(ii)
        instance.advance_one_step()

def test2():
    instance = HLM()
    config_file = 'examples/cedarrapids1/cedar_example_windows.yaml'
    config_file = 'E:/projects/et/hydrology/modis_climatology/iowa_modis_climatology.yaml'
    config_file = 'examples/cedarrapids1/cedar_imac_routing.yaml'
    config_file = 'examples/small/small.yaml'
    instance.init_from_file(config_file)
    instance.set_values('parameters.river_velocity',0.15,None)
    instance.advance()

def test3():
    instance = HLM()
    config_file = 'examples/hydrosheds/conus_example.yaml'
    config_file = 'examples/miss/miss_imac.yaml'
    instance.init_from_file(config_file)
    instance.advance()   

def test4():
    instance = HLM()
    config_file = 'examples/cedarrapids1/cedar_imac.yaml'
    config_file = 'examples/cedarrapids1/cedar_imac_aorc.yaml'
    instance.init_from_file(config_file)
    instance.advance()

def test5():
    instance = HLM()
    config_file = 'examples/iowa/iowa_macbook.yaml'
    config_file = 'examples/cedarrapids1/cedar_macbook.yaml'

    instance.init_from_file(config_file)
    instance.advance()   

def test6():
    instance = HLM()
    config_file = '/Users/felipe/tmp/iowa_operational/iowa_operational_imac.yaml'
    instance.init_from_file(config_file)
    instance.advance()   

if __name__ == "__main__":
    test2()
    #plot1()

    
