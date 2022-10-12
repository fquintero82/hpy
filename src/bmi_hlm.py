import bmipy as Bmi

from src.hlm import HLM

class BmiHLM(Bmi):
    name = 'HLM'

    def __init__(self):
        """Creates a HLM Model that is ready for initialization"""
        self.model=None
        self.start_time=0.0

    def initialize(self,filename=None):
        """Initialize the HLM Model
        
        Parameters
        ----------
        filename: str,optional
            Path to the name of input file
        """
        if filename is None:
            self.model=HLM()
        elif isinstance(filename,str):
            with open(filename,'r') as file_obj:
                self.model = HLM.from_file(file_obj.read())
        else:
            self.model=HLM.from_file(filename)

        