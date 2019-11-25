import os
from os import path
import click
import pandas
import numpy as np
from network.ANN import ANN, funcs
from network.PSO import PSO


def train_ann(ann: ANN, data):
    shape = ann.shape
    bias_and_activation = sum(shape[1:])
    weights = sum([shape[layer] * shape[layer+1] for layer in range(len(shape) - 1)])
    dimensions = 2 * bias_and_activation + weights
    pso = PSO(dimensions, wrap_eval(ann, data))
    best_per_epoch = pso.optimise()
    encode_ann(ann, best_per_epoch[-1][0])
    return best_per_epoch


def wrap_eval(ann: ANN, data: (np.ndarray, np.ndarray)):
    def wrapped(dimensions):
        x, y = data
        encode_ann(ann, dimensions)
        return ann.predict_evaluate(x, y)
    return wrapped


def encode_ann(ann: ANN, dimensions: np.ndarray):
    # Get sizes
    ab_size, weight_size = ann.size

    # Get cumulative sums for indexing
    cumulative_nodes = np.cumsum(ann.shape[1:])
    cumulative_weights = np.cumsum(weight_size)

    # Reshape weights
    weights = [dimensions[start:end] for start, end in zip(np.insert(cumulative_weights[:-1], 0, 0), cumulative_weights)]
    weights = [weight.reshape(height, width) for weight, height, width in zip(weights, ann.shape[:-1], ann.shape[1:])]

    # Reshape biases
    bias = dimensions[-ab_size:]
    bias = [bias[start:end] for start, end in zip(np.insert(cumulative_nodes[:-1], 0, 0), cumulative_nodes)]

    # Reshape activations
    activations = dimensions[-2 * ab_size:-ab_size]
    activations = [activations[start:end] for start, end in zip(np.insert(cumulative_nodes[:-1], 0, 0), cumulative_nodes)]
    activations = [np.clip(np.floor((activation+1)/2 * len(funcs)), 0, len(funcs)-1) for activation in activations]
    activations = [[funcs[int(func)] for func in func_list] for func_list in activations]

    # Update Values
    ann.weights = weights
    ann.biases = bias
    ann.activation_funcs = activations


@click.command()
@click.argument('data_folder')
def network(data_folder):
    data = load_data(data_folder)
    ann = ANN([1, 5, 1])
    result = train_ann(ann, data["1in_linear"])
    print(result)
    print(ann.predict([0.5]))




def load_data(folder: str):
    data = {}
    with click.progressbar(os.listdir(folder)) as bar:
        for file in bar:
            no_inputs = int(file[0])
            frame = pandas.read_table(path.join(folder, file), delim_whitespace=True, header=None).to_numpy()
            file = file[:-4]
            data[file] = frame[:, :no_inputs], frame[:, no_inputs:]
    return data


if __name__ == "__main__":
    network()
