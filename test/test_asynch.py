import pandas as pd
import numpy as np
from time import sleep
from multiprocessing import Pool, Manager, Process

# f = 'examples/hydrosheds/conus.pkl'
f = 'examples/small/small.pkl'
# f = 'examples/cedarrapids1/367813.pkl'

network = pd.read_pickle(f)
n = len(network)
_vals = np.zeros(n+1)
_resolved = np.zeros(n+1)
_idxup = np.array(network['idx_upstream_link']) #get  upstream linkids

def upready(ii,idxup,resolved)->bool:
        iup = np.array(idxup[ii],dtype=np.int32) #object to int
        # print(iup)
        # print(ii)
        if np.array(iup ==-1).any():
            return True
        nup = len(iup)
        s = np.sum(np.array(resolved)[iup])
        if s==nup:
            return True
        else:
            return False

def action(ii,idxup,vals):
    iup = np.array(idxup[ii],dtype=np.int32) #object to int
    sumup =0
    if np.array(iup!=-1).any():
        sumup = np.sum(np.array(vals)[iup])
    vals[ii+1] = sumup + 1
                    
def process_one(ii,resolved,vals,idxup)->bool:
    if(resolved[ii+1]==True):
        return True
    if upready(ii,idxup,resolved)==True:
        action(ii,idxup,vals)
        resolved[ii+1] =1
    
def test_process_one():
    process_one(1)
    process_one(3)
    process_one(4)
    process_one(2)
    process_one(0)
    # print(vals)

def process_all1():
    # https://stackoverflow.com/questions/11055303/multiprocessing-global-variable-updates-not-returned-to-parent
    # no se puede modificar variables globales con multiprocessing
    ncpu = 1
    manager = Manager()
    resolved = manager.list(_resolved)
    vals = manager.list(_vals)
    idxup = manager.list(_idxup)
    # idxs = np.arange(n)
    idxs = np.where(np.array(resolved)[1:]==0)[0]
    print(idxs)
    # idxs = [0,1,2,3,4]
    # idxs = [1,3,4]
    # idxs = [1]
    jobs = []
    for idx in idxs:
        j = Process(target=process_one,args=(idx,resolved,vals,idxup))
        jobs.append(j)
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()

    # with Pool(ncpu) as p:
        # p.map(process_one,[1,3,4])
    print(resolved)    
    print(vals)

def process_all():
    ncpu = 2
    manager = Manager()
    resolved = manager.list(_resolved)
    vals = manager.list(_vals)
    idxup = manager.list(_idxup)

    total_resolved1 = np.sum(np.array(resolved))
    while total_resolved1 < n:
        idxs = np.where(np.array(resolved)[1:]==0)[0]
        print(idxs)
        jobs = []
        for idx in idxs:
            j = Process(target=process_one,args=(idx,resolved,vals,idxup))
            jobs.append(j)
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
        # print(resolved)    
        # print(vals)
        total_resolved1 = np.sum(np.array(resolved))
        print('solved {s} out of {n}'.format(s=total_resolved1,n=n))

if __name__ == '__main__':    
    process_all()

