import os
from os import path
import click
import pandas


@click.command()
@click.argument('data_folder')
def network(data_folder):
    test = load_data(data_folder)
    print(test)


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
