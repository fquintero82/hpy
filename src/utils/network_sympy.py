from sympy import *
from sympy.stats import Exponential
from utils.network.network import get_default_network
network = get_default_network()

def process_link(link_id):
    N = len(network)
    X =symbols('x0:%d'%N)
    P =symbols('p0:%d'%N)
    T =symbols('t')

    link_id = 393217
    idx = network.loc[link_id,'idx']
    expr = X[idx]* 2.718**(-P[idx]*T)
    expr.evalf(subs={X[idx]: 2,P[idx]:1,t:1})
    
