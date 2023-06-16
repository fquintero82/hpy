from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
import multiprocessing as mp
import time
from numpy.linalg import inv
import time
from scipy.linalg import solve


#calculates routing using Mantilla 2005 equation, using lambda1 and lambda 2 parameters
def nonlinear_velocity(states:pd.DataFrame,
    params:pd.DataFrame,
    network:pd.DataFrame,DT:int):

    def fun(t,q,invtau,idx_up,lambda1):
        q_aux = pd.concat([
             pd.Series(0,index=[0]),
             pd.Series(q)
        ]).to_numpy() #it is important to convert this pd df into a nparray,otherwise i got broadcast errors
        q_upstream = np.array([np.sum(q_aux[x]) for x in idx_up])
        dq_dt = invtau*q_aux[1:]**lambda1*(-1*q_aux[1:] + q_upstream)
        return dq_dt
    
    q = states['discharge']
    q_aux = pd.concat([
            pd.Series(0,index=[0]),
            pd.Series(states['discharge'])
        ])
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
    idx_up = network['upstream_link']
    lambda1 =params['lambda1']
    t_end_sim = DT*60 #since the ODE flow inputs are in m3/s , t_end_sim is the number of seconds of routing process
    res = solve_ivp(fun,
            t_span=(0,t_end_sim),
            y0=q,
            args=(invtau,idx_up,lambda1),
        )
    n_eval = res.t.shape[0] 
    y_1 = res.y[:,n_eval-1]
    states['discharge'] = y_1

#calculates routing using Mantilla 2005 equation, simplification with lamdbda 1 and 2 equal zero, 
#meaning constant velocity over time, varying on channel length
def linear_velocity1(states:pd.DataFrame,
    params:pd.DataFrame,
    network:pd.DataFrame,
    DT:int):

    def fun(t,q,velocity,channel_len_m,idx_up): #t in minutes, q in m3/h
        #print(type(q))
        q_aux = np.concatenate(([0],q))
        q_upstream = np.zeros(q.shape)
        q_upstream = np.array([np.sum(q_aux[x]) for x in idx_up]) #m3/h
        velocity = np.multiply(velocity,3600) #m/s to m/h
        dq_dt = (1/channel_len_m )* velocity * (-1*q_aux[1:] + q_upstream)
        return dq_dt

    idx_up = network['idx_upstream_link'].to_numpy()
    #t_end_sim = DT*60 #since the ODE flow inputs are in m3/s , t_end_sim is the number of seconds of routing process
    t_end_sim = DT / 60 # in hours
    q = np.array(states['discharge'])
    q = np.multiply(q,3600) #m3/s to m3/h
    channel_len_m = np.array(network['channel_length'])
    velocity = np.array(params['river_velocity'])
    start_time = time.time()
    res = solve_ivp(fun,
            t_span=(0,1),
            y0=q, 
            args=(velocity,channel_len_m,idx_up),
            method='RK23',
            atol=1e-2,
            rtol=1e-2
        )
    print("--- %s seconds ---" % (time.time() - start_time))
    n_eval = res.t.shape[0] 
    y_1 = res.y[:,n_eval-1]/3600. #m3/h to m3/s
    states['discharge'] = y_1


def linear_velocity2(states:pd.DataFrame,
    velocity:np.float16,
    network:pd.DataFrame,DT:int):

    def fun(t,q,velocity,channel_len_m,idx_up): #t in minutes, q in m3/h
        #print(type(q))
        q_aux = np.concatenate(([0],q))
        q_upstream = np.zeros(q.shape)
        q_upstream = np.array([np.sum(q_aux[x]) for x in idx_up]) #m3/h
        velocity *=60*60 #m/s to m/h
        dq_dt = (1/channel_len_m )* velocity * (-1*q_aux[1:] + q_upstream)
        return dq_dt

    idx_up = network['idx_upstream_link'].to_numpy()
    #t_end_sim = DT*60 #since the ODE flow inputs are in m3/s , t_end_sim is the number of seconds of routing process
    t_end_sim = DT / 60 # in hours
    q = np.array(states['discharge'])
    channel_len_m = np.array(network['channel_length'])
    start_time = time.time()
    res = solve_ivp(fun,
            t_span=(0,1),
            y0=q*60*60, #m3/s to m3/h
            args=(velocity,channel_len_m,idx_up),
            method='RK23',
            atol=1e-2,
            rtol=1e-2
        )
    print("--- %s seconds ---" % (time.time() - start_time))
    #takes 90 seconds run one hour
    n_eval = res.t.shape[0] 
    y_1 = res.y[:,n_eval-1]/3600. #m3/h to m3/s
    states['discharge'] = y_1

#calculates routing as linear transfer from upstream to downstream link

# #cancelled because this routing of discharge wont work well for large basins
# def transfer0(states:pd.DataFrame,
#     params:pd.DataFrame,
#     network:pd.DataFrame,DT:int):

#     nlinks = network.shape[0]
#     routing_order = network.loc[:,['link_id','idx_downstream_link','drainage_area','channel_length']]
#     routing_order['idx_upstream_link']=np.arange(nlinks)
#     routing_order['river_velocity']= params['river_velocity']
#     routing_order = routing_order.sort_values(by=['drainage_area'])

#     idxd = routing_order['idx_downstream_link'].to_numpy()
#     idxu = routing_order['idx_upstream_link'].to_numpy()
#     vel = routing_order['river_velocity'].to_numpy()
#     vel[:] = 0.01 #m/s
#     len1 = routing_order['channel_length'].to_numpy()
#     q=states['discharge'].to_numpy() #q is a pointer. changing q results in changing states['discharge']


#     for ii in np.arange(nlinks):
#         #dq = np.min([q[idxu[ii]] , q[idxu[ii]] * vel[ii] / len1[ii] * DT*60 ])
#         dq = q[idxu[ii]]
#         if(idxd[ii])>=0:
#             q[idxd[ii]] += dq
#             q[idxu[ii]] -= dq 


def transfer0(hlm_object):
    def fun(t,v,velocity,channel_len_m,idx_up): #t in minutes, q in m3/h
        #print(type(q))
        v_aux = np.concatenate(([0],v),dtype=np.float32)
        v_upstream = np.zeros(v.shape,dtype=np.float32)
        v_upstream = np.array([np.sum([v_aux[np.array(x,dtype=np.integer)]]) for x in idx_up])
        
        velocity = np.multiply(velocity,3600) #m/s to m/h
        dv_dt = (1/channel_len_m )* velocity * (-1*v_aux[1:] + v_upstream)
        return dv_dt

    idx_up = hlm_object.network['idx_upstream_link'].to_numpy()
    #t_end_sim = DT*60 #since the ODE flow inputs are in m3/s , t_end_sim is the number of seconds of routing process
    #t_end_sim = hlm_object.time_step_sec / 3600 # in hours
    v = np.array(hlm_object.states['volume'],dtype=np.float32)
    channel_len_m = np.array(hlm_object.network['channel_length'],dtype=np.float32)
    velocity = np.array(hlm_object.params['river_velocity'],dtype=np.float32)
    start_time = time.time()
    res = solve_ivp(fun,
            t_span=(0,1),
            y0=v, 
            args=(velocity,channel_len_m,idx_up),
            method='RK23',
            atol=1e-2,
            rtol=1e-2
        )
    print("--- %s seconds ---" % (time.time() - start_time))
    n_eval = res.t.shape[0] 
    y_1 = res.y[:,n_eval-1]
    hlm_object.states['volume'] = np.array(y_1,dtype=np.float32)

#this works well. should not be used with river volume or discharge
def transfer1(hlm_object):
    t = time.time()
    nlinks = hlm_object.network.shape[0]
    routing_order = hlm_object.network.loc[:,['idx','idx_downstream_link','drainage_area']].copy()
    routing_order = routing_order.sort_values(by=['drainage_area'])
    idxd = routing_order['idx_downstream_link'].to_numpy()
    idxu = routing_order['idx'].to_numpy()
    da = np.array(hlm_object.network['drainage_area']*1e6,dtype=np.float32)
    bp=np.array(hlm_object.states['basin_precipitation']* hlm_object.network['area_hillslope'],dtype=np.float32)
    bet=np.array(hlm_object.states['basin_evapotranspiration']* hlm_object.network['area_hillslope'],dtype=np.float32)
    bswe=np.array(hlm_object.states['basin_swe']* hlm_object.network['area_hillslope'],dtype=np.float32)
    bst=np.array(hlm_object.states['basin_static']* hlm_object.network['area_hillslope'],dtype=np.float32)
    bsf=np.array(hlm_object.states['basin_surface']* hlm_object.network['area_hillslope'],dtype=np.float32)
    bsub=np.array(hlm_object.states['basin_subsurface']* hlm_object.network['area_hillslope'],dtype=np.float32)
    bgw=np.array(hlm_object.states['basin_groundwater']* hlm_object.network['area_hillslope'],dtype=np.float32)

    for ii in np.arange(nlinks):
        if idxd[ii]!=-1:
            bp[idxd[ii]-1]+= bp[idxu[ii]-1]
            bet[idxd[ii]-1]+= bet[idxu[ii]-1]
            bswe[idxd[ii]-1]+= bswe[idxu[ii]-1]
            bst[idxd[ii]-1]+= bst[idxu[ii]-1]
            bsf[idxd[ii]-1]+= bsf[idxu[ii]-1]
            bsub[idxd[ii]-1]+= bsub[idxu[ii]-1]
            bgw[idxd[ii]-1]+= bgw[idxu[ii]-1]
    bp /= da
    bet /= da
    bswe /= da
    bswe /= da
    bst /= da
    bsf /= da
    bsub /= da
    bgw /= da

    hlm_object.states['basin_precipitation'] = bp
    hlm_object.states['basin_evapotranspiration'] = bet
    hlm_object.states['basin_swe'] = bswe
    hlm_object.states['basin_static'] = bst
    hlm_object.states['basin_surface'] = bsf
    hlm_object.states['basin_subsurface'] = bsub
    hlm_object.states['basin_groundwater'] = bgw
    del bp,bet,bswe,bsf,bst,bsub,bgw,routing_order,idxd,idxu
    print(time.time()-t)



def transfer2(hlm_object):
    N = hlm_object.network.shape[0]
    initial_state = np.zeros(shape=(N+1))
    initial_state[1:] = hlm_object.states['volume'].to_numpy()
    hlm_object.ODESOLVER.set_initial_value(initial_state,0.0)
    time = hlm_object.time_step_sec / 3600 #hours
    out = hlm_object.ODESOLVER.integrate(time)[1:]#value 0 is auxiliary
    hlm_object.states['volume'] = out
    hlm_object.states['discharge'] = out / hlm_object.time_step_sec

def transfer3(hlm_object):
    da = np.array(hlm_object.network['drainage_area']*1e6,dtype=np.float32)
    vars = np.array(['basin_precipitation',
                     'basin_evapotranspiration',
                     'basin_swe',
                     'basin_static',
                     'basin_surface',
                     'basin_subsurface',
                     'basin_groundwater'
                     ])
    time = hlm_object.time_step_sec / 3600 #hours
    for i in vars:
        var = np.array(hlm_object.states[i]* hlm_object.network['area_hillslope'],dtype=np.float32)
        var = np.concatenate(([0],var))
        hlm_object.accum.set_initial_value(var,0.0)
        out = hlm_object.accum.integrate(time)[1:]#value 0 is auxiliary
        out /= da
        hlm_object.states[i] = out


# def transfer2(hlm_object):
#     #
#     #transfer discharge volume
#     #print('transfer volume')
#     #q0 = np.array(hlm_object.states['volume'],dtype=np.float32)
#     #q0 = hlm_object.states['volume'].copy()
#     q0 = hlm_object.states['volume'].to_numpy()
#     A = hlm_object.adjmatrix
#     #A = np.array(hlm_object.adjmatrix,dtype=np.float32)
#     f = np.array(hlm_object.params['river_velocity'] * (1.0 / hlm_object.network['channel_length']) * hlm_object.time_step_sec,dtype=np.float32)
#     t = time.time()
#     # aux = q0 + sgemm(alpha=1,a=A,b=(f*q0))
#     output = np.multiply(f,q0)
#     output = np.minimum(output,q0) #output cant be larger than the current volume

#     hlm_object.states['volume'] = q0 + np.matmul(A,output)
#     hlm_object.states['discharge'] = np.abs(output) /  hlm_object.time_step_sec
#     #print(time.time()-t)
#     #print('complete transfer volume')
#     #assuming channel of 1m width
#     # discharge  = area wet section x river velocity
#     #[m3/s] = [m3] / [m] * [m/s]
#     #hlm_object.states['discharge'] = np.array(
#     #    hlm_object.states['volume'] / hlm_object.network['channel_length'] * hlm_object.params['river_velocity']
#     #    ,dtype=np.float32)

#     del q0,f,output

#     #cancelled the approach below because the inverse of A
#     #takes forever to be estimated, even pre-processed
#     # A = -1*A
#     # #transfer basin accumulation states
#     # print('transfer precip')
#     # t = time.time()
#     # q0= hlm_object.states['basin_precipitation'].copy()
#     # hlm_object.states['basin_precipitation'] = solve(A,q0)
#     # print(time.time()-t)

# def transfer3(hlm_object):
#     #june 2023. found that this method doesnt work well for transfering discharg
#     #it works well for small basins
#     #but at larger basins, doesnt move volume fast enough.
#     #changing to route discharge using odes
#     #this works well for other variables

#     #this function transfers basin variables, but when transfering discharge and volume , all the flow from the upper
#     #rivers is transported up to the outlet. is different to transfer 2
#     t = time.time()
#     nlinks = hlm_object.network.shape[0]
#     routing_order = hlm_object.network.loc[:,['idx','idx_downstream_link','drainage_area']].copy()
#     routing_order = routing_order.sort_values(by=['drainage_area'])
#     idxd = routing_order['idx_downstream_link'].to_numpy()
#     idxu = routing_order['idx'].to_numpy()
#     da = np.array(hlm_object.network['drainage_area']*1e6,dtype=np.float32)
#     bp=np.array(hlm_object.states['basin_precipitation']* hlm_object.network['area_hillslope'],dtype=np.float32)
#     bet=np.array(hlm_object.states['basin_evapotranspiration']* hlm_object.network['area_hillslope'],dtype=np.float32)
#     bswe=np.array(hlm_object.states['basin_swe']* hlm_object.network['area_hillslope'],dtype=np.float32)
#     bst=np.array(hlm_object.states['basin_static']* hlm_object.network['area_hillslope'],dtype=np.float32)
#     bsf=np.array(hlm_object.states['basin_surface']* hlm_object.network['area_hillslope'],dtype=np.float32)
#     bsub=np.array(hlm_object.states['basin_subsurface']* hlm_object.network['area_hillslope'],dtype=np.float32)
#     bgw=np.array(hlm_object.states['basin_groundwater']* hlm_object.network['area_hillslope'],dtype=np.float32)
#     #f = np.array(hlm_object.params['river_velocity'] * (1.0 / hlm_object.network['channel_length']) * hlm_object.time_step_sec,dtype=np.float32)
#     f = 0.95 #test
#     vol = np.array(hlm_object.states['volume'],dtype=np.float32)
#     outvol = f * vol
#     outvol = np.minimum(outvol,vol)
#     discharge = np.array(outvol / hlm_object.time_step_sec,dtype=np.float32 ) #m3/s
    
#     for ii in np.arange(nlinks):
#         if idxd[ii]!=-1:
#             bp[idxd[ii]-1]+= bp[idxu[ii]-1]
#             bet[idxd[ii]-1]+= bet[idxu[ii]-1]
#             bswe[idxd[ii]-1]+= bswe[idxu[ii]-1]
#             bst[idxd[ii]-1]+= bst[idxu[ii]-1]
#             bsf[idxd[ii]-1]+= bsf[idxu[ii]-1]
#             bsub[idxd[ii]-1]+= bsub[idxu[ii]-1]
#             bgw[idxd[ii]-1]+= bgw[idxu[ii]-1]
#             vol[idxd[ii]-1]+= outvol[idxu[ii]-1]
#             #vol[idxu[ii]-1]-= outvol[idxu[ii]-1]
#         vol[idxu[ii]-1]-= outvol[idxu[ii]-1] #substracts also at the outlets

#     bp /= da
#     bet /= da
#     bswe /= da
#     bswe /= da
#     bst /= da
#     bsf /= da
#     bsub /= da
#     bgw /= da

#     hlm_object.states['basin_precipitation'] = bp
#     hlm_object.states['basin_evapotranspiration'] = bet
#     hlm_object.states['basin_swe'] = bswe
#     hlm_object.states['basin_static'] = bst
#     hlm_object.states['basin_surface'] = bsf
#     hlm_object.states['basin_subsurface'] = bsub
#     hlm_object.states['basin_groundwater'] = bgw
#     hlm_object.states['volume'] = vol
#     hlm_object.states['discharge'] = discharge
#     del bp,bet,bswe,bsf,bst,bsub,bgw,routing_order,idxd,idxu
#     del f, vol,outvol, discharge
#     print(time.time()-t)
    