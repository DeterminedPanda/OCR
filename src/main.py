from neural_network import NeuralNetwork
from file_reader import FileReader
import numpy

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learn_rate = 0.3

neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learn_rate)
file_reader = FileReader()

def teachNetwork():
    training_data_list = file_reader.read("../resources/mnist_train.csv")
    for record in training_data_list:
        row_entries = record.split(",")
        targets = numpy.zeros(output_nodes) + 0.01
        shifted_input = shiftList(row_entries)
        # the expected index of the result is set to 0.99, to check wether or not the neural network predicted the number correctly
        targets[int(row_entries[0])] = 0.99
        neural_network.train(shifted_input, targets)

def testNetwork():
    test_data_list = file_reader.read("../resources/mnist_test.csv")
    scoreboard = []
    for record in test_data_list:
        row_entries = record.split(",")
        correct_label = int(row_entries[0])
        # transforms the training data set values that range from 0 - 255 to a training set with the range of 0.01 - 0.99.
        # Every value in the list is transformed except for the first value, since that value contains the expected result
        shifted_input = (numpy.asfarray(row_entries[1:]) / 255.0 * 0.99) + 0.01
        outputs = neural_network.query(shifted_input)[1]
        label = numpy.argmax(outputs)
        scoreboard.append(1) if label == correct_label else scoreboard.append(0)
    scoreboard_array = numpy.asfarray(scoreboard)
    return (scoreboard_array.sum() / scoreboard_array.size) * 100

def shiftList(list):
    # transforms the training data set values that range from 0 - 255 to a training set with the range of 0.01 - 0.99.
    # This is needed because the sigmoid activation function only allows values in that range.
    # Every value in the list is transformed except for the first value, since that value contains the expected result.
    return (numpy.asfarray(list[1:]) / 255.0 * 0.99) + 0.01

def main():
    print("Starting the learning process, this may take a while...")
    teachNetwork()        
    print("Completed the learning process.")
    print("Starting the testing process, this may take a while...")
    results = testNetwork()
    print("Completed the testing process.")
    print(f"The accuracy of the Neural Network is: {results:.2f}%")

if __name__ == "__main__":
    main()
