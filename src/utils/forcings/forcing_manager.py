import pandas as pd
from models.model400names import FORCINGS_NAMES
import numpy as np
from abc import ABC, abstractmethod
import time as mytime
import importlib.util

def set_forcings(hlminstance):
        # print('reading forcings')
        t = mytime.time()
        modelforcings = list(hlminstance.forcings.columns)[1:]
        config_forcings = list(hlminstance.configuration['forcings'].keys())
        for ii in range(len(modelforcings)):
            if modelforcings[ii] not in config_forcings:
                print(modelforcings[ii]+ ' not found in yaml file')
            if modelforcings[ii] in config_forcings:
                options = hlminstance.configuration['forcings'][modelforcings[ii]]
                if options is not None:
                    try:
                        modname = options['script']
                        spec = importlib.util.spec_from_file_location('mod',modname)
                        foo = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(foo)
                        lid, values = foo.get_values(hlminstance.time,options)
                        hlminstance.set_values(
                            var_name='forcings.'+ modelforcings[ii],
                            linkids = lid,
                            values= values
                        )
                    except Exception as e:
                        print(e)
                        quit()
        x = int((mytime.time()-t)*1000)
        print('forcings loaded in {x} msec'.format(x=x))


# class forcing(ABC):
#     file = None
#     in_memory = None

#     def __init__(self) -> None:
#         pass

#     @abstractmethod
#     def get_values(self):
#         pass

def get_default_forcings(network:pd.DataFrame):
    nlinks = network.shape[0]
    df = pd.DataFrame(
        data = np.zeros(shape=(nlinks,len(FORCINGS_NAMES))),
        columns=FORCINGS_NAMES)
    df['link_id'] = network['link_id'].to_numpy()
    df.index = network['link_id'].to_numpy()
    return df



