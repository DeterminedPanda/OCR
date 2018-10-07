from neural_network import NeuralNetwork

layer_size = 3 # number of neurons in each layer (Input, Hidden and Output layer)
learn_rate = 0.3

neural_network = NeuralNetwork(layer_size, learn_rate)
print(neural_network)