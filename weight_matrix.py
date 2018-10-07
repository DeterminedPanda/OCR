import numpy

class WeightMatrix():

    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns

    # generates a 2D-matrix that is filled with random values, in the range +- 1/root(scale)
    def generate_matrix(self, scale):
        return numpy.random.normal(0, pow(scale, -0.5), (self.rows, self.columns))