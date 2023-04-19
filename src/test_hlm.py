from hlm import HLM
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def plot1():
    NSTEPS = int((1204542000 - 1204344000 ) / 3600)
    NGAGE =1
    _X = 367812
    
    out = np.zeros(shape=(NGAGE,NSTEPS))
    ix=0
    for tt in np.arange(start=1204344000,stop=1204542000,step=3600):
        f = 'examples/cedarrapids1/out/{}.pkl'.format(tt)
        states = pd.read_pickle(f)
        out[0,ix] = states[_X]
        ix+=1
    
    plt.plot(out[0,:])
    plt.show()

if __name__ == "__main__":
    instance = HLM()
    config_file = 'examples/cedarrapids1/cedar_example.yaml'
    instance.init_from_file(config_file)
    #instance.advance_one_step()
    instance.advance(time_to_advance=1212278400)

    
