import click

from network.ANN import ANN


@click.command()
def network():
    test = ANN([3, 2, 1])
    print(test.weights)


if __name__ == "__main__":
    network()
