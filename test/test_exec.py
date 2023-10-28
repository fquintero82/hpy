from hlm import HLM
import numpy as np
# from math import factorial
import time
import math

instance = HLM()
config_file = 'examples/cedarrapids1/cedar_example.yaml'
config_file = 'examples/hydrosheds/conus2.yaml'

instance.init_from_file(config_file,option_solver=False)

N=len(instance.network)
initial_state = np.ones(shape=(N,))
expr = instance.network['expression'].to_numpy()
expr2 = ['out[{x1}]='.format(x1=x) for x in np.arange(N)]
expr3 = expr2 + expr
expr3 = np.array(expr3,dtype='str')
expr4 = ';'.join(expr3)
t=time.time()
expr_compiled = compile(expr4,'<string>','exec')
print('compiled in %f sec'%(time.time()-t))
P = (instance.params['river_velocity'] / instance.network['channel_length']).to_numpy()
T=instance.time_step_sec
X = np.ones(shape=(N,))
out = np.zeros(shape=(N,))
t=time.time()
out = exec(expr_compiled)
print('exec in %f sec'%(time.time()-t))

