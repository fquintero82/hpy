from hlm import HLM
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    instance = HLM()
    config_file = 'examples/cedarrapids1/cedar_example.yaml'
    instance.init_from_file(config_file)
    instance.set_forcings()
    #instance.advance_one_step()
    instance.advance(time_to_advance=480*3600) # 3minutes

    NSTEPS = 480
    NGAGE =1
    _X = 367813
    
    out = np.zeros(shape=(NGAGE,NSTEPS))
    for tt in range(NSTEPS):
        f = '../examples/cedarrapids1/out/{}.pkl'.format(tt)
        states = pd.read_pickle(f)
        out[0,tt] = states[_X]
    
    plt.plot(out[0,:])
    plt.show()
