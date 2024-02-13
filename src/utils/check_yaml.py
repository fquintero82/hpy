from yaml import Loader
import yaml
import sys


def check_yaml1(configuration):
        # config_file = 'examples/cedarrapids1/cedar_example2.yaml'
        # stream = open(config_file)
        # configuration = yaml.load(stream,Loader=Loader)

        # check that all forcings exists in the yaml file
        if configuration['forcings'] is None:
                print('forcings not included in configuration file')
                quit()

        if len(configuration['forcings'].keys()) < 4:
                print('missing forcings in configuration file')
                quit()

        if configuration['description'] is None:
                print('description not included in configuration file')
                quit()

        if configuration['network'] is None:
                print('network not included in configuration file')
                quit()

        if configuration['parameters'] is None:
                print('parameters not included in configuration file')
                quit()

        if configuration['initial_states'] is None:
                print('initial_states not included in configuration file')
                quit()

        if configuration['init_time'] is None:
                print('init_time not included in configuration file')
                quit()

        if configuration['end_time'] is None:
                print('end_time not included in configuration file')
                quit()

        if configuration['time_step'] is None:
                print('time_step not included in configuration file')
                quit()

        if configuration['output_file'] is None:
                print('output_file not included in configuration file')
                quit()

        # if configuration['solver'] is None:
        #         print('a path to save the ode solver (e.g. examples/ode.so) must be provided in the configuration file')
        #         quit()
