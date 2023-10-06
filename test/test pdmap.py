import pandas as pd
import numpy as np

# f = 'examples/hydrosheds/conus.pkl'
f = 'examples/small/small.pkl'
# f = 'examples/cedarrapids1/367813.pkl'

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

def get_sum_resolved_up(idxup,res):
    x2 = np.array(idxup,dtype=np.int32)
    print(x2)
    out = np.array(x2 ==-1).any()
    if out == True:
        return 0
    # print(np.array(res.iloc[x2]))
    s = np.sum(np.array(res.loc[x2]))
    return s

test = idxup.map(lambda x: get_sum_resolved_up(x,network['resolved']))


idxup1 = network['resolved']*out_check_no_upstream_links
a =  ~out_check_no_upstream_links
idxup[out_check_no_upstream_links]
idxup1 = (idxup==-1).any()
np.sum(network['resolved'].iloc[[2,3]])
network['resolved'].iloc[[2,3]]
network['resolved'].iloc[-1]
