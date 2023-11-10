from models.model400polars import runoff1

def test_model400polars():
    from hlm import HLM
    instance= HLM()
    config_file = 'examples/hydrosheds/conus_macbook.yaml'
    instance.init_from_file(config_file,option_solver=False)
    runoff1(instance.states,
            instance.forcings,
            instance.params,
            instance.network,
            instance.time_step_sec)

test_model400polars()