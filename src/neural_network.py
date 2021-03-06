from weight_matrix import WeightMatrix
import numpy
import scipy.special

class NeuralNetwork():

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learn_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learn_rate = learn_rate
        matrix_factory = WeightMatrix()
        # generates a weight matrix, for the connections between the INPUT and HIDDEN layer
        self.input_hidden_layer_weights = matrix_factory.generate_matrix(hidden_nodes, (hidden_nodes, input_nodes))
        # generates a weight matrix, for the connections between the HIDDEN and OUTPUT layer
        self.hidden_output_layer_weights = matrix_factory.generate_matrix(output_nodes, (output_nodes, hidden_nodes))

    def train(self, input_list, training_examples):
        # convert lists to transposed 2D arrays
        input = self.listToTransposed2DArray(input_list)
        targets = self.listToTransposed2DArray(training_examples)
        # query the input and save the output of the layers for later use in the backpropagation
        hidden_layer_output, output_layer_output = self.query(input_list)
        # calculate the occured errors
        output_layer_errors = targets - output_layer_output
        hidden_layer_errors = numpy.dot(self.hidden_output_layer_weights.T, output_layer_errors)
        # backpropagation
        # update the weights between the input & hidden layer and the weights between the hidden & output layer
        self.hidden_output_layer_weights += self.learn_rate * numpy.dot((output_layer_errors * output_layer_output 
            * (1.0 - output_layer_output)), numpy.transpose(hidden_layer_output))
        self.input_hidden_layer_weights += self.learn_rate * numpy.dot((hidden_layer_errors * hidden_layer_output 
            * (1.0 - hidden_layer_output)), numpy.transpose(input))

    def query(self, input_list):
        input = self.listToTransposed2DArray(input_list)
        # input to output layer
        hidden_layer_input = numpy.dot(self.input_hidden_layer_weights, input)
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        # hidden to output layer
        output_layer_input = numpy.dot(self.hidden_output_layer_weights, hidden_layer_output)
        output_layer_output = self.sigmoid(output_layer_input)
        return hidden_layer_output, output_layer_output

    # converts a list to a transposed 2D-array
    def listToTransposed2DArray(self, list):
        return numpy.array(list, ndmin=2).T

    def sigmoid(self, data):
        return scipy.special.expit(data)
