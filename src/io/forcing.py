from abc import ABC, abstractmethod
from hlm import HLM
import importlib.util

class Forcing(ABC):
    @abstractmethod
    def get_value(hlm:HLM):
        pass

def check_forcings(instance:HLM):
    # modelforcings = list(instance.forcings.columns)[1:]
    # config_forcings = list(instance.configuration['forcings'].keys())
    # for ii in len(modelforcings):
    #     if modelforcings[ii] in config_forcings:
    #         modname = instance.configuration['forcings'][modelforcings[ii]]['script']
    #         spec = importlib.util.spec_from_file_location('mod',modname)
    #         foo = importlib.util.module_from_spec(spec)
    #         spec.loader.exec_module(foo)

    #         lid, values = foo.get_values(instance.time)
    #         instance.set_values(var_name='forcings.'+ modelforcings[ii],
    #                             linkids = lid,
    #                             values= values
    #                             )
            
def test():
    instance = HLM()
    config_file = '../examples/cedarrapids1/cedar_example.yaml'
    instance.init_from_file(config_file)

test()