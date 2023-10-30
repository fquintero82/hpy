import numpy as np
import pandas as pd
from  scipy.special import factorial
from multiprocessing import Pool
import time
from os import cpu_count
from numba import jit
import copy
# https://stackoverflow.com/questions/69423036/use-eval-in-numba-numbalsoda#:~:text=eval%20is%20fundamentally%20incompatible%20with,calls)%20to%20be%20strongly%20typed.


def _process_unit(idx:np.int32,
                    idx_upstream_links:np.ndarray,
                    order:np.int32,
                    expr:list,source_idx:np.int32):
        if order < 100:
            # myexpr = (order,idx)
            expr.append(order)
            expr.append(idx)
            expr.append(source_idx)
            myidx_upstream_links = idx_upstream_links[idx - 1]
            if (myidx_upstream_links!=0).any():
                for new_idx in myidx_upstream_links:
                    _process_unit(new_idx,idx_upstream_links,order+1,expr,source_idx)

# def _process_unit2(idx:np.int32,
#                     idx_upstream_links:np.ndarray,
#                     order:np.int32,
#                     expr:list):
#         if order < 10:
#             myexpr = '1/float(factorial({order}-1)) * P[{idx}-1]**({order}-1) * X[{idx}-1] * T**({order}-1) * 2.718 ** (-P[{idx}-1] * T)'
#             myexpr = myexpr.format(order=order,idx=idx)
#             expr.append(myexpr)
#             myidx_upstream_links = idx_upstream_links[idx - 1]
#             if (myidx_upstream_links!=0).any():
#                 for new_idx in myidx_upstream_links:
#                     _process_unit(new_idx,idx_upstream_links,order+1,expr)

def process_unit(idx:np.int32,
                idx_upstream_links:np.ndarray):
    expr = []
    order=1
    source_idx = copy.deepcopy(idx)
    _process_unit(idx,idx_upstream_links,order,expr,source_idx)
    # expr = '+'.join(expr)
    return expr
class ProcessLink(object):
    def __init__(self, idx_upstream_links):
        self.idx_upstream_links = idx_upstream_links
    def __call__(self, src):
        out= process_unit(src, self.idx_upstream_links)
        return out

def _eval_unit(x:list,P:np.ndarray,X:np.ndarray,T:int):
    #uneven values of x are the order
    #even values of x are the index
    n=len(P)
    order = np.array(x[0:len(x):2])
    idx = np.array(x[1:len(x):2])
    val = 1/(factorial(order-1)) * P[idx-1]**(order-1) * X[idx-1] * T**(order-1) * 2.718 ** (-P[idx-1] * T)
    out = np.sum(val)
    return out

def eval_unit(idx:np.int32,expr:np.ndarray,P:np.ndarray,X:np.ndarray,T:int):
    x= expr[idx]
    out = _eval_unit(x,P,X,T)
    return out

class Evaluator(object):
    def __init__(self, expr,P,X,T):
        self.expr = expr
        self.P = P
        self.X = X
        self.T = T
    def __call__(self, src):
        out = eval_unit(src, self.expr,self.P,self.X,self.T)
        return out


def process_all(network:pd.DataFrame):
    N = len(network)
    idx_upstream_links = network['idx_upstream_link'].to_numpy()
    idxs = network['idx'].to_numpy()
    expression=np.empty(shape=(N,),dtype=object)
    t=time.time()
    ncpu = cpu_count() - 1
    with Pool(processes=ncpu) as pool:
        out = pool.map(ProcessLink(idx_upstream_links),idxs)
    print('done in %f sec'%(time.time()-t))
    for i in np.arange(len(out)):
        expression[i] = np.array(out[i])
    network['expression'] = expression
    
def eval_all2(expr,P,X,T):
    N = len(expr)
    t=time.time()
    ncpu = cpu_count() - 1
    ncpu=2
    with Pool(processes=ncpu) as pool:
        out = pool.map(Evaluator(expr,P,X,T),range(N))
    print('done in %f sec'%(time.time()-t))
    return np.asarray(out)

def eval_all(expr,P,X,T):
    N = len(expr)
    t=time.time()
    out = np.zeros(N)
    for i in np.arange(N):
        out [i] =_eval_unit(expr[i],P,X,T)
    print('done in %f sec'%(time.time()-t))
    return out

def expression_to_arrays():
    f = '/Users/felipe/tmp/iowa/iowa_network.pkl'
    network = pd.read_pickle(f)
    expr = network['expression'].to_numpy()
    x = np.concatenate(expr)
    order = np.array(x[0:len(x):3])
    idx = np.array(x[1:len(x):3])
    source_idx = np.array(x[2:len(x):3])
    
    


# def process_all2(network:pd.DataFrame):
#     N = len(network)
#     idx_upstream_links = network['idx_upstream_link'].to_numpy()
#     idxs = network['idx'].to_numpy()
#     expression=np.empty(shape=(N,),dtype=object)
#     for i in np.arange(N):
#         if i % 10000 ==0:
#             print('row %s'%i)
#         out=process_unit(idxs[i],idx_upstream_links)
#         expression[i] = out
#     network['expression'] = expression