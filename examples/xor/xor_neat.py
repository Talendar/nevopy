""" In this example, we're going to use NEvoPy's implementation of the NEAT
algorithm to learn the XOR logic function.

Authors:
    Gabriel Nogueira (Talendar)
"""

import nevopy as ne
import numpy as np


# Number of input variables for the XOR logic function:
NUM_XOR_VARIABLES = 2

# Building XOR data:
xor_inputs, xor_outputs = ne.utils.make_xor_data(NUM_XOR_VARIABLES)


def fitness_function(genome, log=False):
    """ Implementation of the fitness function we're going to use.

    It simply feeds the XOR inputs to the given genome and calculates how well
    it did (based on the squared error).
    """
    # Shuffling the input, in order to prevent our networks from memorizing the
    # sequence of the answers.
    idx = np.random.permutation(len(xor_inputs))

    error = 0
    for x, y in zip(xor_inputs[idx], xor_outputs[idx]):
        # Resetting the cached activations of the genome (optional).
        genome.reset()

        # Feeding the input to the genome. A numpy array with the value
        # predicted by the neural network is returned.
        h = genome.process(x)[0]

        # Calculating the squared error.
        error += (y - h) ** 2

        if log:
            print(f"IN: {x}  |  OUT: {h:.4f}  |  TARGET: {y}")

    if log:
        print(f"\nError: {error}")

    return (1 / error) if error > 0 else 0


if __name__ == "__main__":
    # Creating a new random population of NEAT genomes:
    # (we're going to use the default settings)
    population = ne.neat.NeatPopulation(
        size=200,
        num_inputs=len(xor_inputs[0]),
        num_outputs=1,
    )

    # Evolving the population:
    history = population.evolve(generations=64,
                                fitness_function=fitness_function)

    # Retrieving the fittest genome in the population:
    best_genome = population.fittest()

    # Evaluating the best genome:
    print()
    fitness_function(best_genome, log=True)

    # Visualizing the best genome's topology
    best_genome.visualize()

    # Plotting the evolution of the population's fitness
    history.visualize()
