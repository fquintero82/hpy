from hlm import HLM

if __name__ == "__main__":
    instance = HLM()
    config_file = 'examples/cedarrapids1/cedar_example.yaml'
    instance.from_file(config_file)
    instance.advance_in_time()

