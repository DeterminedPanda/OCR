from weight_matrix import WeightMatrix
import numpy
import scipy.special

class NeuralNetwork():

    def __init__(self, layer_size, learn_rate):
        self.layer_size = layer_size
        self.learn_rate = learn_rate
        matrix_factory = WeightMatrix(layer_size, layer_size)
        # generates a weight matrix of the size (layer_size)², for the connections between the INPUT and HIDDEN layer
        self.input_hidden_layer_weights = matrix_factory.generate_matrix(layer_size)
        # generates a weight matrix of the size (layer_size)², for the connections between the HIDDEN and OUTPUT layer
        self.hidden_output_layer_weights = matrix_factory.generate_matrix(layer_size)

    def trainNetwork(self, input, training_examples):
        hidden_layer_output, output_layer_output = query(input)

        output_layer_errors = training_examples - output_layer_output
        hidden_layer_errors = numpy.dot(self.hidden_output_layer_weights.T, output_layer_errors)

        self.hidden_output_layer_weights += self.learn_rate * numpy.dot((output_layer_errors * output_layer_output * (1.0 - output_layer_output)), numpy.transpose(hidden_layer_output))


    def query(self, input):
        input = self.listToTransposed2DArray(input)

        hidden_layer_input = self.matrixMultiplication(self.input_hidden_layer_weights, input)
        hidden_layer_output = self.sigmoid(hidden_layer_input)

        output_layer_input = self.matrixMultiplication(self.hidden_output_layer_weights, hidden_layer_output)
        output_layer_output = self.sigmoid(output_layer_input)

        return hidden_layer_output, output_layer_output

    # converts a list to a transposed 2D-array
    def listToTransposed2DArray(self, list):
        return numpy.array(list, ndmin=2).T

    def matrixMultiplication(self, matrix_one, matrix_two):
        return numpy.dot(matrix_one, matrix_two)

    def sigmoid(self, data):
        return scipy.special.expit(data)