from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import numpy as np
from test_dataframes import getTestDF1, getTestDF2
import pandas as pd

def test1():
    '''
    #v[0] = V11'(s) = -12*v12(s)**2 
    #v[1] v22'(s) = 12*v12(s)**2  
    #v[2] v12'(s) = 6*v11(s)*v12(s) - 6*v12(s)*v22(s) - 36*v12(s) 
    '''
    def fun(s,v):
        return [-12*v[2]**2 , 12*v[2]**2, 6*v[0]*v[2]-6*v[2]*v[1] - 36*v[2]]

    res = solve_ivp(fun,
        t_span=(0,0.1),
        y0=[2,3,4]
    )
    res['y'].shape
    plt.plot(res['t'],res['y'][0])
    plt.plot(res['t'],res['y'][1])
    plt.plot(res['t'],res['y'][2])
    plt.show()

def test2():
    #channel 1 and 2 drain to 3
    def fun(t,q):
        v0=0.05 #[m/s]
        lambda1= 0.0#0.33
        lambda2= 0#0.1
        A_i=1     #[km2]
        L_i = 100   #[m]
        invtau = (v0*A_i**lambda2) /((1.0 - lambda1)*L_i)
        return [invtau*(q[0]**lambda1)*(-1*q[0]),
                invtau*(q[1]**lambda1)*(-1*q[1]),
                invtau*(q[2]**lambda1)*(-1*q[2]+q[0]+q[1])
        ]
    q =[100,110,100]
    DT = 60 #min
    t_end = DT * 60 #seg
    res = solve_ivp(fun,t_span=(0,t_end),y0=q)
    plt.plot(res['t'],res['y'][0])
    #plt.plot(res['t'],res['y'][1])
    #plt.plot(res['t'],res['y'][2])
    plt.show()

def test3():
    #channel 1 and 2 drain to 3
    def fun(t,q,v0,lambda1,lambda2,A_i,L_i):
        invtau = (v0*A_i**lambda2) /((1.0 - lambda1)*L_i)
        return [invtau*(q[0]**lambda1)*(-1*q[0]),
                invtau*(q[1]**lambda1)*(-1*q[1]),
                invtau*(q[2]**lambda1)*(-1*q[2]+q[0]+q[1])
        ]
    args=(.3,.33,.1,100,200)
    res = solve_ivp(fun,t_span=(0,0.1),y0=[100,110,100],args=args)


def test3():
    #channel 1 and 2 drain to 3
    def fun(t,q,v0,invtau,index_up1,index_up2):
        #invtau = (v0*A_i**lambda2) /((1.0 - lambda1)*L_i)
        #invtau = np.divide(np.multiply(v0,np.power(A_i,lambda2)),np.multiply(np.subtract(1,lambda1),L_i))
        #print(q.shape)
        return np.multiply(np.power(np.multiply(invtau,q),lambda1),(-1*q+q[index_up1]+q[index_up2]))


    v0=np.array([0, .3, .4 , .5])
    lambda1=np.array([0, 0.33,.33,.33])
    lambda2=np.array([0, .1,.1,.1])
    A_i=np.array([0, 100,100,100])    #km2
    L_i = np.array([0, 100,100,100])   #m
    invtau = np.divide(np.multiply(v0,np.power(A_i,lambda2)),np.multiply(np.subtract(1,lambda1),L_i)) 
    args1 = (v0,lambda1,lambda2,A_i,L_i)
    args2=(.3,.33,.1,100,200)
    index_up1 = [0,0,0,1]
    index_up2 = [0,0,0,2]
    res = solve_ivp(fun,t_span=(0,0.1),y0=[0,100,110,100],args=args1,)
    res = solve_ivp(fun,t_span=(0,0.1),y0=[0,100,110,100],args=args2,)
    plt.plot(res['t'],res['y'][0])
    plt.plot(res['t'],res['y'][1])
    plt.plot(res['t'],res['y'][2])
    plt.show()

def test4():
    #[-3.243743104813389, -3.6821266577313194, 3.568117415294728]
    #0.007096536682661702
    states= getTestDF1('states')
    states['discharge'] = [100,110,100]
    q = states['discharge']
    q_aux = pd.concat([
            pd.Series(0,index=[0]),
            pd.Series(states['discharge'])
        ])
    params= getTestDF1('params')
    network= getTestDF1('network')
    #q_upstream = [np.sum(q_aux[x]) for x in network['upstream_link']]
    invtau = np.divide(
                np.multiply(
                    params['river_velocity'],
                    np.power(params['drainage_area'],params['lambda2'])
                ),
                np.multiply(
                    np.subtract(1,params['lambda1'])
                    ,params['channel_length']
                )
    )
    
    def fun(t,q,invtau,idx_up,lambda1):
        q_aux = pd.concat([
             pd.Series(0,index=[0]),
             pd.Series(q)
        ]).to_numpy() #it is important to convert this pd df into a nparray,otherwise i got broadcast errors
        q_upstream = np.array([np.sum(q_aux[x]) for x in idx_up])
        #q_aux = q_aux[1:]
        #dq_dt = invtau*np.power(q_aux.iloc[1:],lambda1)*(-1*q_aux.iloc[1:] + q_upstream)
        #dq_dt = invtau*q_aux.iloc[1:]**lambda1*(-1*q_aux.iloc[1:] + q_upstream)
        dq_dt = invtau*q_aux[1:]**lambda1*(-1*q_aux[1:] + q_upstream)
        #dq_dt = invtau*np.power(pd.concat([pd.Series(0,index=[0]),pd.Series(q)]),lambda1)*(-1*pd.concat([pd.Series(0,index=[0]),pd.Series(q)]) + [np.sum(pd.concat([pd.Series(0,index=[0]),pd.Series(q)])[x]) for x in idx_up])
        return dq_dt

    idx_up = network['upstream_link']
    lambda1 =params['lambda1']
    res = solve_ivp(fun,t_span=(0,1000),y0=q,args=(invtau,idx_up,lambda1))
    plt.plot(res['t'],res['y'][0])
    plt.plot(res['t'],res['y'][1])
    plt.plot(res['t'],res['y'][2])
    plt.show()

def test5():
    states= getTestDF2('states')
    states['discharge'] = [100,110,120,130,140]
    q = states['discharge']
    q_aux = pd.concat([
            pd.Series(0,index=[0]),
            pd.Series(states['discharge'])
        ])
    params= getTestDF2('params')
    network= getTestDF2('network')
    #q_upstream = [np.sum(q_aux[x]) for x in network['upstream_link']]
    invtau = np.divide(
                np.multiply(
                    params['river_velocity'],
                    np.power(params['drainage_area'],params['lambda2'])
                ),
                np.multiply(
                    np.subtract(1,params['lambda1'])
                    ,params['channel_length']
                )
    )
    
    def fun(t,q,invtau,idx_up,lambda1):
        q_aux = pd.concat([
             pd.Series(0,index=[0]),
             pd.Series(q)
        ]).to_numpy() #it is important to convert this pd df into a nparray,otherwise i got broadcast errors
        q_upstream = np.array([np.sum(q_aux[x]) for x in idx_up])
        dq_dt = invtau*q_aux[1:]**lambda1*(-1*q_aux[1:] + q_upstream)
        return dq_dt

    idx_up = network['upstream_link']
    lambda1 =params['lambda1']
    DT = 60
    t_end = DT*60 #secs
    res = solve_ivp(fun,t_span=(0,t_end),y0=q,args=(invtau,idx_up,lambda1))
    n_eval = res.t.shape[0] 
    y_1 = res.y[:,n_eval-1]
    for x in range(5):
        plt.plot(res['t'],res['y'][x],label=str(x+1))
    plt.legend()
    plt.show()