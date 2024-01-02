from hlm import HLM
import sys

if __name__ == "__main__":
    instance = HLM()
    config_file = sys.argv[0]
    instance.init_from_file(config_file,option_solver=False)
    instance.advance()  
