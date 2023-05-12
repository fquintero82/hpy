from yaml import Loader
import yaml
import sys

def check_yaml1(configuration):
        config_file = 'examples/cedarrapids1/cedar_example2.yaml'

        stream = open(config_file)
        configuration = yaml.load(stream,Loader=Loader)
        
        #check that all forcings exists in the yaml file
        if configuration['forcings'] is None:
                print('forcings not included in configuration file')
                quit()

        if len(configuration['forcings'].keys())<4:
                print('missing forcings in configuration file')
                quit()
                
        if configuration['description'] is None:
                sys.exit('description not included in configuration file')

        if configuration['network'] is None:
                sys.exit('network not included in configuration file')

        if configuration['parameters'] is None:
                sys.exit('parameters not included in configuration file')

        if configuration['initial_states'] is None:
                sys.exit('initial_states not included in configuration file')

        if configuration['init_time'] is None:
                sys.exit('init_time not included in configuration file')

        if configuration['end_time'] is None:
                sys.exit('end_time not included in configuration file')

        if configuration['time_step'] is None:
                sys.exit('time_step not included in configuration file')

        if configuration['output_file'] is None:
                sys.exit('output_file not included in configuration file')

