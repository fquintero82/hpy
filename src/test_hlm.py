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
    config_file = 'examples/cedarrapids1/cedar_example2.yaml'
    instance.init_from_file(config_file)
    for ii in range(10):
        print(ii)
        instance.advance_one_step()

def test2():
    instance = HLM()
    config_file = 'examples/cedarrapids1/cedar_example_windows.yaml'
    instance.init_from_file(config_file,option_solver=False)
    instance.advance()

def test3():
    instance = HLM()
    config_file = 'examples/hydrosheds/conus_example.yaml'
    instance.init_from_file(config_file)
    instance.advance()   


if __name__ == "__main__":
    test2()
    #plot1()

    
