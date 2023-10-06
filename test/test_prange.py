import pandas as pd
import numpy as np
from numba import njit, prange


def isleaf(_up:np.ndarray)->bool:
    out=False
    if(np.array([_up ==-1]).any()):
        out=True
    return out


# def action_leaf(ii):
#     vals[ii+1]=1
#     resolved[ii+1]=1    

# def action_noleaf(ii):
#     vals[ii+1]=1
#     resolved[ii+1]=1
@njit
def check_unit(ii:int,
               vals:np.ndarray,
               resolved:np.ndarray,
               upstream_link:np.ndarray,
               idxup:np.ndarray)->bool:
    try:
        if(resolved[ii+1]==True):
            return True
        _up = upstream_link[ii]
        _iup = np.array(idxup[ii],dtype=np.int32)
        if isleaf(_up)==True and resolved[ii+1]==0:
            # print('{ii} ok'.format(ii=ii))
            vals[ii+1]=1
            resolved[ii+1]=1
        if isleaf(_up)==False and resolved[ii+1]==0:
            nup = len(_up)
            s = np.sum(resolved[[_iup]])
            if s!=nup:
                return False
            if s==nup:
                sumup = np.sum(vals[[_iup]])
                vals[ii+1] = sumup + 1
                resolved[ii+1] =1
    except IndexError as e:
        print(e)
        return False
    return True


def test_unit():
    # f = 'examples/hydrosheds/conus.pkl'
    f = 'examples/small/small.pkl'
    #f = 'examples/cedarrapids1/367813.pkl'

    network = pd.read_pickle(f)
    n = len(network)
    vals = np.zeros(n+1)
    resolved = np.zeros(n+1)
    upstream_link = np.array(network['upstream_link']) #get  upstream linkids
    idxup = np.array(network['idx_upstream_link']) #get  upstream linkids

    ii=0
    print(check_unit(ii,vals,resolved,upstream_link,idxup))

def check_all1(vals:np.ndarray,
               resolved:np.ndarray,
               upstream_link:np.ndarray,
               idxup:np.ndarray,n:int):
    unresolved = np.where(resolved[1:]==0)[0]
    for x in range(len(unresolved)):
        ii = unresolved[x]
        check_unit(ii,vals,resolved,upstream_link,idxup)
    total_resolved = np.sum(resolved)
    print('solved {s} out of {n}'.format(s=total_resolved,n=n))

@njit(parallel=True)
def check_all(vals:np.ndarray,
               resolved:np.ndarray,
               upstream_link:np.ndarray,
               idxup:np.ndarray,n:int):
    unresolved = np.where(resolved[1:]==0)[0]
    for x in prange(len(unresolved)):
        ii = unresolved[x]
        check_unit(ii,vals,resolved,upstream_link,idxup)
    total_resolved = np.sum(resolved)
    print('solved {s} out of {n}'.format(s=total_resolved,n=n))

    
def test_check_all():
    # f = 'examples/hydrosheds/conus.pkl'
    f = 'examples/small/small.pkl'
    #f = 'examples/cedarrapids1/367813.pkl'

    network = pd.read_pickle(f)
    n = len(network)
    vals = np.zeros(n+1)
    resolved = np.zeros(n+1)
    upstream_link = np.array(network['upstream_link']) #get  upstream linkids
    idxup = np.array(network['idx_upstream_link']) #get  upstream linkids
    check_all(vals,resolved,upstream_link,idxup,n)

def run(): 
    # f = 'examples/hydrosheds/conus.pkl'
    f = 'examples/small/small.pkl'
    # f = 'examples/cedarrapids1/367813.pkl'

    network = pd.read_pickle(f)
    n = len(network)
    vals = np.zeros(n+1)
    resolved = np.zeros(n+1)
    upstream_link = np.array(network['upstream_link']) #get  upstream linkids
    idxup = np.array(network['idx_upstream_link']) #get  upstream linkids
      
    total_resolved1 = np.sum(resolved)
    while total_resolved1 < n:
        check_all(vals,resolved,upstream_link,idxup,n)
        total_resolved1 = np.sum(resolved)

    # print(vals)
    # print(resolved)

run()

