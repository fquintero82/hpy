from models.model400 import runoff1

# import pandas as pd
# import numpy as np
# from test_dataframes import getTestDF1
# from model400names import *
# from model400pandas import runoff1
# from routing import linear_velocity
# from utils.network.network_from_rvr_file import combine_rvr_prm


# def test_runoff1():
#     states= getTestDF1('states')
#     params= getTestDF1('params')
#     forcings= getTestDF1('forcings')
#     network= getTestDF1('network')

#     old_states = states.copy()
#     runoff1(states=states,
#     forcings=forcings,
#     params=params,
#     network=network,
#     DT=1440)
#     print('old')
#     print(old_states)
#     print('new')
#     print(states)

# def test_runoff2():
#     states= getTestDF1('states')
#     params= getTestDF1('params')
#     forcings= getTestDF1('forcings')
#     network= getTestDF1('network')

#     NSTEPS = 24
#     DT = 10
#     init_states = states.copy()
#     for tt in range(NSTEPS):
#         #runoff1(states,forcings,params,network,DT)
#         #print(states.loc[3,'discharge'])
#         linear_velocity(states,params,network,DT)
#         print(states.loc[3,'discharge'])
#     end_states = states.copy()
#     print('complete test runoff 2')

# def test_runoff3():
#     rvr_file ='../examples/cedarrapids1/367813.rvr'
#     prm_file ='../examples/cedarrapids1/367813.prm'
#     network = combine_rvr_prm(prm_file,rvr_file)
#     nlinks = network.shape[0]
#     states = pd.DataFrame(
#         data = np.zeros(shape=(nlinks,len(STATES_NAMES))),
#         columns=STATES_NAMES)
#     states['link_id'] = network['link_id'].to_numpy()
#     states.index = states['link_id'].to_numpy()
#     states['discharge']=1
#     states['static'] = 0
#     states['surface']=0.1
#     states['subsurface']=0.1
#     states['groundwater']=0.1
#     states['snow'] = 0.1

#     forcings = pd.DataFrame(
#         data = np.zeros(shape=(nlinks,len(FORCINGS_NAMES))),
#         columns=FORCINGS_NAMES)
#     forcings.index = states['link_id'].to_numpy()
#     forcings['precipitation']=1
#     forcings['link_id'] = network['link_id'].to_numpy()

#     params = pd.DataFrame(
#         data = np.zeros(shape=(nlinks,len(PARAM_NAMES))),
#         columns=PARAM_NAMES)
#     params.index = states['link_id'].to_numpy()
#     params['link_id'] = network['link_id'].to_numpy()
#     params['alfa3']=1
#     params['alfa4']=1
#     NSTEPS = 10
#     DT = 60 #min
#     velocity = 0.1 #m/s
#     for tt in range(NSTEPS):
#         print(tt)
#         runoff1(states,forcings,params,network,DT)
#         #linear_velocity(states,velocity,network,DT)
#         #f = '../examples/cedarrapids1/out/{}.pkl'.format(tt)
#         #states['discharge'].to_pickle(f)

# test_runoff3()

def test_model400():
    from hlm import HLM
    instance= HLM()
    config_file = 'examples/hydrosheds/conus_macbook.yaml'
    instance.init_from_file(config_file,option_solver=False)
    runoff1(instance.states,
            instance.forcings,
            instance.params,
            instance.network,
            instance.time_step_sec)

test_model400()