import numpy

class WeightMatrix():

    def __init__(self):
        pass

    # generates a 2D-matrix that is filled with random values, in the range +- 1/root(scale)
    def generate_matrix(self, scale, size):
        return numpy.random.normal(0.0, pow(scale, -0.5), (size[0], size[1]))
