import numpy as np
import h5py
import matplotlib.pyplot as plt
from network_utils import NeuralNetwork


if __name__ == '__main__':

    # Data
    def load_data():
        train_dataset = h5py.File('../Datasets/train_catvnoncat.h5', "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

        test_dataset = h5py.File('../Datasets/test_catvnoncat.h5', "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

        classes = np.array(test_dataset["list_classes"][:]) # the list of classes
        
        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
        
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    # Flatten the training and test examples 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    # Make the neural neural network
    neural_network = NeuralNetwork()
    layers_dims = [12288, 20, 7, 5, 1]
    iterations = 100

    params = neural_network.L_layer_model(train_x, train_y, layers_dims, learning_rate = 0.0075, num_iterations = iterations, print_cost = True)

    train_acc = neural_network.predict(train_x, train_y, params)
    test_acc = neural_network.predict(test_x, test_y)

    print(f'Training accuracy: after {iterations} is {train_acc} and testing accuracy is {test_acc}')