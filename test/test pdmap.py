import pandas as pd
import numpy as np
import time

# f = 'examples/hydrosheds/conus.pkl'
# f = 'examples/small/small.pkl'
f = 'examples/cedarrapids1/367813.pkl'

network = pd.read_pickle(f)
n = len(network)
vals = np.zeros(n)
resolved = np.zeros(n)
idxup = network['idx_upstream_link'] #get  upstream linkids
network['resolved'] = 0
# iup = [np.array(x,dtype=np.int32) for x in idxup]
def fun1(x):
    out = x + 10
    return out
test = network['idx'].map(fun1)

def check_no_upstream_links(x): #input is object type
    #return true if link is has no upstream links
    x2 = np.array(x,dtype=np.int32)
    out = np.array(x2 ==-1).any()
    return out

out_check_no_upstream_links = idxup.map(check_no_upstream_links)

def is_resolved(x):
    out = np.array(x==1)
    return out

test = network['resolved'].map(is_resolved)

def get_count_upstream_links(x):
    x2 = np.array(x,dtype=np.int32)
    out = np.size(x2)
    return out

test = idxup.map(get_count_upstream_links)
print(test)

def get_sum_resolved_up1(idxup,res):
    x2 = np.array(idxup,dtype=np.int32)
    res1=res.to_numpy()
    out = np.array(x2 ==0).any()
    if out == True:
        return 0
    s = np.sum(np.array(res1[x2-1]))    
    return s

t = time.time()
test = idxup.map(lambda x: get_sum_resolved_up(x,network['resolved']))
print(time.time()-t)

def get_sum_resolved_up2(idxup,res):
    out = np.zeros(len(res))
    x2 = np.array([np.array(x,dtype=np.int32) for x in idxup])
    wh = np.array(x2 ==0).any()
    wh = ~wh
    res1=res.to_numpy()
    s = np.sum(np.array(res1[x2-1]))    
    return s

t = time.time()
res = network['resolved']
test = get_sum_resolved_up2(idxup,network['resolved'])
print(time.time()-t)


