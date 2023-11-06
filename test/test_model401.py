from hlm import HLM
from model401 import runoff1 as run401
from model400 import runoff1 as run400 
from model402 import runoff1 as run402 
import modin.pandas as pd
import time
#conda install -c conda-forge grpc-cpp


instance= HLM()
# config_file = 'examples/cedarrapids1/cedar_example.yaml'
config_file = 'examples/hydrosheds/conus_macbook.yaml'

instance.init_from_file(config_file,option_solver=False)
t = time.time()
# run400(instance.states,instance.forcings,instance.params,instance.network,instance.time_step_sec)
run401(instance.states,instance.forcings,instance.params,instance.network,instance.time_step_sec)

print(time.time()-t)
t = time.time()
run402(pd.DataFrame(instance.states),
       pd.DataFrame(instance.forcings),
       pd.DataFrame(instance.params),
       pd.DataFrame(instance.network),
       instance.time_step_sec)
print(time.time()-t)
