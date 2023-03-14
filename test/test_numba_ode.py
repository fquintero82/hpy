import nbkode
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.network.network_from_rvr_file import combine_rvr_prm
from model400names import STATES_NAMES
import time

def test1():
    def rhs(t, y):
        return -0.1 * y
    y0 = 1.
    t0 = 0
    solver = nbkode.RungeKutta45(rhs, t0, y0)
    ts = np.linspace(0, 10, 100)
    ts, ys = solver.run(ts)
    plt.plot(ts, ys) 
    plt.show()

def test2(t,q,p):
    def fun(t,q,p): #t in minutes, q in m3/h
        velocity = (0.1)
        channel_len_m = (500.0)
        #idx_up = p[2]
        #q_aux = np.zeros(shape=(q.shape[0]+1),dtype=np.float16)#fail
        q_aux = np.zeros(shape=(q.shape[0]+1)) #wont fail. dont pass type. numba infer it
        q_aux[1:] = q #dont fail
        #q_upstream = np.zeros(q.shape[0])
        #q_upstream = np.array([np.sum(q_aux[x]) for x in idx_up]) #m3/h
        #velocity *=60*60 #m/s to m/h
        #dq_dt = np.array((1/channel_len_m )* velocity * (-1*q_aux[1:] + q_upstream))
        dq_dt = ((1.0)/channel_len_m )* velocity #* (-1*q_aux[1:])
        return dq_dt

    rvr_file ='../examples/cedarrapids1/367813.rvr'
    prm_file ='../examples/cedarrapids1/367813.prm'
    network = combine_rvr_prm(prm_file,rvr_file)
    nlinks = network.shape[0]
    states = pd.DataFrame(
        data = np.zeros(shape=(nlinks,len(STATES_NAMES))),
        columns=STATES_NAMES)
    states['link_id'] = network['link_id'].to_numpy()
    states.index = states['link_id'].to_numpy()
    states['discharge']=1
    velocity = 0.5 #m/s
    idx_up = network['idx_upstream_link'].to_numpy()
    q = np.array(states['discharge'],dtype=np.float16) #force everything to float16
    t0=0
    channel_len_m = np.array(network['channel_length'])
    start_time = time.time()
    p =[velocity,channel_len_m,idx_up] #p must be array, not tuple?
    p=[0.1,0.2]
    solver = nbkode.RungeKutta45(fun, t0, q, params=p)
    ts, ys = solver.step(n=1)