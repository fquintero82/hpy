import pandas as pd
from models.model400names import FORCINGS_NAMES
import numpy as np
from abc import ABC, abstractmethod


class forcing(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_values(self):
        pass

def get_default_forcings(network:pd.DataFrame):
    nlinks = network.shape[0]
    df = pd.DataFrame(
        data = np.zeros(shape=(nlinks,len(FORCINGS_NAMES))),
        columns=FORCINGS_NAMES)
    df['link_id'] = network['link_id'].to_numpy()
    df.index = network['link_id'].to_numpy()
    return df



