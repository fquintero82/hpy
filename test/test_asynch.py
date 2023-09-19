from utils.network.network import get_network_from_file
import pandas as pd
import numpy as np

f = 'examples/small/small.pkl'
network = pd.read_pickle(f)
n = len(network)
vals = np.zeros(n+1)
resolved = np.zeros(n+1)

def isleaf(_up)->bool:
    out=False
    if(np.array([_up ==-1]).any()):
        out=True
    return out

def test():
    ii=2
    upstream_link = np.array(network['upstream_link']) #get  upstream linkids
    idxup = np.array(network['idx_upstream_link']) #get  upstream linkids
    def giveapass():
        for ii in np.arange(n):
            print(ii)
            _up = upstream_link[ii]
            _iup = np.array(idxup[ii],dtype=np.int32)
            if isleaf(_up)==True and resolved[ii+1]==0:
                print('{ii} ok'.format(ii=ii))
                vals[ii+1]=1
                resolved[ii+1]=1
            
            if isleaf(_up)==False and resolved[ii+1]==0:
                nup = len(_up)
                s = np.sum(resolved[[_iup]])
                if s==nup:
                    print('{ii} ok'.format(ii=ii))
                    sumup = np.sum(vals[[_iup]])
                    vals[ii+1] = sumup + 1
                    resolved[ii+1] =1
    
    giveapass()
    giveapass()
    giveapass()
    print(vals)
    print(resolved)

test()