import numpy as np
from typing import List

from .functions import null, sigmoid, gaussian, identity, mean_squared_error as mse
from .types import Layer, Vector, Data

# Possible options for activation functions
funcs = [null, sigmoid, np.tanh, np.cos, gaussian]


class ANN:
    """
    An implementation of an Artificial Neural Network using weight matrices in a simple
    feed forward architecture.
    :param shape: The shape of the network given by neurons per layer
    :type shape: List(int)
    :param seed: An optional random seed to fix initial conditions
    :type seed: int
    :param output_activation: Allows a non-identity activation function for the output layer
    :type output_activation: bool
    :ivar shape: Instance variable to store the network shape
    :type shape: List(int)
    :ivar layers: A list of Layers storing information about the network
    :type layers: List(Layer)
    """
    def __init__(self, shape: List[int], seed: int = None, output_activation: bool = False) -> None:
        self.shape = shape
        self.layers = []

        # Set to known seed if required
        if seed:
            np.random.seed(seed)

        # Initialise weights, activation function and bias for each node
        for layer in range(len(shape) - 1):
            weights = np.random.randn(shape[layer+1], shape[layer])
            activation = np.array([funcs[i] for i in np.random.randint(len(funcs), size=shape[layer+1])])
            bias = np.random.randn(shape[layer+1])
            self.layers.append(Layer(weights, activation, bias))

        # Set output activation to the identity to allow for all values
        if not output_activation:
            self.layers[-1].activation = np.array([identity for _ in range(shape[-1])])

    def predict_evaluate(self, input_vectors: Data, ground_truth: Data, error_function=mse) -> float:
        """
        Performs the prediction on the sample of input vectors and compares the results against the ground truth using
        the error function (MSE by default). The list of input vectors and output vectors should have the same length.
        The function can also compare a single input vector against the expected output vector.
        :param input_vectors: Single or list of input vectors to base predictions on
        :type input_vectors: Data
        :param ground_truth: Single or list of ground truths to compare against predictions. Should have same length as
        input_vectors
        :type ground_truth: Data
        :param error_function: A function to calculate the error on two equal length lists of input vectors and ground
        truths and returns a float.
        :type error_function: Data, Data -> float
        :return: Returns a floating point error score based on the error function.
        """
        if len(input_vectors) != len(ground_truth):
            raise ValueError(f"Number of test samples ({len(input_vectors)}) does not match ground "
                             f"truth ({len(ground_truth)})")

        # Make inputs conform
        if type(ground_truth[0]) is int:
            ground_truth = [ground_truth]

        # Perform predictions and calculate error
        predictions = self.predict(input_vectors)
        return error_function(predictions, ground_truth)

    def predict(self, x: Data) -> np.ndarray:
        """
        The prediction function of the network which takes a list of input vectors and
        :param x: The input vector
        :type x: Data
        :return: List of output vectors
        :rtype: List[Vector]
        """
        # Check if it's a single input vector or many
        if type(x[0]) is int:
            x = [x]

        # Check if vector has correct input shape and perform predictions
        predictions = []
        for input_vector in x:
            if len(input_vector) == self.shape[0]:
                predictions.append(self._predict_one(input_vector))
            else:
                raise ValueError(f"Size of input vector ({len(input_vector)}) does not match "
                                 f"size of input layer ({self.shape[0]})")
        return np.array(predictions)

    def _predict_one(self, x: Vector) -> np.ndarray:
        """
        Performs a single model prediction by propagating the input values forward through the network using matrix
        multiplication, followed by applying the activation function to the sum of the result and the bias.
        :param x: Single input vector
        :type x: Vector
        :return: Single output vector
        :rtype: Vector
        """
        layer_values = x
        # For each layer, apply the activation function to the sum of the weighted inputs and the bias.
        # Weight summation uses matrix multiplication for speed and brevity.
        for layer in self.layers:
            layer_values = np.matmul(layer.weights, layer_values)
            pre_op = zip(layer_values, layer.activation, layer.bias)
            layer_values = [activation(node_value + bias) for node_value, activation, bias in pre_op]
        return np.array(layer_values)

    @property
    def weights(self):
        return np.array([layer.weights for layer in self.layers])

    @weights.setter
    def weights(self, new_weights):
        new_weights = np.array(new_weights)
        if new_weights.shape == self.weights.shape:
            for layer_weights, layer in zip(new_weights, self.layers):
                layer.weights = layer_weights

    @property
    def activation_funcs(self):
        return np.array([layer.activation for layer in self.layers])

    @activation_funcs.setter
    def activation_funcs(self, new_funcs):
        new_funcs = np.array(new_funcs)
        if new_funcs.shape == self.activation_funcs.shape:
            for layer_funcs, layer in zip(new_funcs, self.layers):
                layer.activation = layer_funcs

    @property
    def biases(self):
        return np.array([layer.bias for layer in self.layers])

    @biases.setter
    def biases(self, new_biases):
        new_biases = np.array(new_biases)
        if new_biases.shape == self.biases.shape:
            for layer_bias, layer in zip(new_biases, self.layers):
                layer.bias = layer_bias
