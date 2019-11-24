import numpy as np

from .my_types import Data

# Activation Functions


def null(_: float) -> float:
    """
    Function that always returns 0
    :param _: Input value to function
    :type _: int
    :return: Always 0
    :rtype: int
    """
    return 0


def sigmoid(x: float) -> float:
    """
    Implementation of the Sigmoid activation function
    :param x: Input value to function
    :type x: float
    :return: sigmoid(x)
    :rtype: float
    """
    return 1 / (1 + np.exp(-x))


def gaussian(x: float) -> float:
    """
    Implementation of the Gaussian activation function
    :param x: Input value to function
    :type x: float
    :return: gaussian(x)
    :rtype: float
    """
    return np.exp(-(x**2/2))


def identity(x: float) -> float:
    """
    Identity function to be used in place of no activation function
    :param x: Input value to function
    :type x: float
    :return: x
    :rtype: float
    """
    return x


# Training Functions
def cubic(x: float) -> float:
    """
    Cubic training function
    :param x: Input value to function
    :type x: float
    :return: x^3
    :rtype: float
    """
    return x ** 3


def xor(x1: int, x2: int) -> int:
    """
    xor training function
    :param x1: Input value to function
    :param x2: Input value to function
    :type x1: int
    :type x2: int
    :return: x1 XOR x2
    :rtype: int
    """
    return x1 ^ x2


def complex_train(x1: float, x2: float) -> float:
    """
    complex training function
    :param x1: Input value to function
    :param x2: Input value to function
    :type x1: float
    :type x2: float
    :return: 1.9{ 1.35 + e^x1-x2 sin[13(x1 -0.6)^2]sin[7x2] }
    :rtype: int
    """
    return 1.9 * (1.35 + np.exp(x1 - x2) * np.sin(13*((x1-0.6)**2)) * np.sin(7*x2))


# Error Functions
def mean_squared_error(predictions: Data, ground_truth: Data) -> float:
    """
    Function to take the MSE (Mean Squared Error) of two equal length lists of output vectors.
    Commonly used to take get the MSE of predictions from a network and the ground truth. All vectors in the list must
    have equal length. While the order of the supplied lists doesn't matter, they are named for clarity and convention.
    :param predictions: The predictions from a network
    :type predictions: Data
    :param ground_truth: The ground truths to compare against
    :type ground_truth: Data
    :return: The MSE (Mean Squared Error) of all supplied samples
    :rtype: float
    """
    # Ensure arguments are of the same length
    if len(ground_truth) != len(predictions):
        raise ValueError(f"Number of predictions ({len(predictions)}) does not match ground "
                         f"truth ({len(ground_truth)})")

    # As the function accepts single vectors or lists of vectors, make the arguments conform to expectations
    if type(predictions[0]) == int:
        predictions = [predictions]
    if type(ground_truth[0]) == int:
        ground_truth = [ground_truth]

    # Create a list of the absolute summed error between corresponding vectors
    per_sample_error = [sum(np.absolute(np.array(x) - np.array(y))) for x, y in zip(ground_truth, predictions)]

    # Square error between samples and take the mean
    return sum(np.array(per_sample_error) ** 2) / len(ground_truth)
