# from sympy import *
import numpy as np
# from sympy.stats import Exponential
from utils.network.network import get_default_network
import sys
from math import factorial


def test_eval():
    var = 'x[0] + x[1]'
    x = np.array([1,1])
    # eval(var)
    var='0'
    N = int(1e5) #este es el maximo que da mi portatil 100,000
    print(N)
    sys.setrecursionlimit(N)
    for i in np.arange(N):
        var+= '+ x[%d]'%i
    x=np.ones(N)
    eval(var)

def process_link1(link_id):
    N = len(network)
    X =symbols('x0:%d'%N)
    P =symbols('p0:%d'%N)
    T =symbols('t')
    depth = 10
    link_id = 393217
    idx = network.loc[link_id,'idx']
    expr = X[idx]* 2.718**(-P[idx]*T)
    # expr.evalf(subs={X[idx]: 2,P[idx]:1,t:1})


def process_unit(idx:np.int32,
                 idx_upstream_links:np.ndarray,
                 order:np.int32,
                 expr:list)->str:
    # expr = 'X[{idx}] * 2.718 ** (-P[{idx}] * T)'
    myexpr = '1/factorial({order}-1) * P**({order}-1) * X[{idx}] * T**({order}-1) * 2.718 ** (-P[{idx}] * T)'
    # myexpr = 'X[{idx}] * P[{idx}] * 2.718 ** (-P[{idx}] * T) * ({order} * 1/factorial({order}-1) * (T * P[{idx}])**({order}-1)
    
    myexpr = myexpr.format(order=order,idx=idx)
    # expr = expr + myexpr
    expr.append(myexpr)
    # print(order)
    # print(expr)
    myidx_upstream_links = idx_upstream_links[idx - 1]
    if (myidx_upstream_links!=0).any():
        for new_idx in myidx_upstream_links:
            process_unit(new_idx,idx_upstream_links,order+1,expr)
    

network = get_default_network()
N = len(network)
link_id = 367813
idx = 32715
order = 1
idx_upstream_links = network['idx_upstream_link'].to_numpy()
expr = ['0']
count = 0
process_unit(idx,idx_upstream_links,order,expr)
print(len(expr))
X = np.ones(N)
T = 1
P = np.ones(N)
out = eval(expr[1])