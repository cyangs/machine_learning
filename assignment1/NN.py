
import pandas as pd


class NeuralNetwork:

    def testRun(self):
        return NotImplementedError

    def keplerData(self):
        df = pd.read_csv("../data/kepler.csv")
        non_floats = []





    def __init__(self, test=True, verbose=True):
        self.verbose = verbose

        if test:
            return NotImplementedError
        else:
            print("Production Run, using Kepler data set")
            self.keplerData()