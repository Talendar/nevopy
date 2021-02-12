""" In this example, we're going to use NEvoPy's implementation of a classic
fixed-topology neuroevolution algorithm to learn the XOR logic function.

Authors:
    Gabriel Nogueira (Talendar)
"""

import os

import nevopy as ne
import numpy as np
from tensorflow import reshape


# Setting this environment variable to "true" will avoid a single process from
# allocating all the GPU memory by itself. Since NEvoPy heavily relies on
# multi-processing, this step is important if you're going to use a GPU.
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Number of input variables for the XOR logic function:
NUM_XOR_VARIABLES = 2

# Building XOR data:
xor_inputs, xor_outputs = ne.utils.make_xor_data(NUM_XOR_VARIABLES)

# Defining a base genome for our population. It will serve as a model for the
# other genomes of the population, which will have the same topology as it.
BASE_GENOME = ne.fixed_topology.FixedTopologyGenome(

    # List with the layers of the base genome. Other genomes in the population
    # will have similar layers (with the same topology).
    layers=[ne.fixed_topology.layers.TFDenseLayer(8, activation="relu"),
            ne.fixed_topology.layers.TFDenseLayer(1, activation="relu")],

    # Shape of the input samples expected by the genome.
    input_shape=xor_inputs.shape,
)


def fitness_function(genome, log=False):
    """ Implementation of the fitness function we're going to use.

    It simply feeds the XOR inputs to the given genome and calculates how well
    it did (based on the squared error).
    """
    # Shuffling the input, in order to prevent our networks from memorizing the
    # sequence of the answers.
    idx = np.random.permutation(len(xor_inputs))

    # Feeding the input to the genome.
    out = genome.process(xor_inputs[idx])

    # Since we're using TensorFlow layers, the value output by our genome is a
    # tf.tensor. Let's reshape it and turn it into a numpy array:
    out = reshape(out, [-1]).numpy()

    # Squared error:
    error = ((out - xor_outputs[idx]) ** 2).sum()

    if log:
        for x, y, h in zip(xor_inputs[idx], xor_outputs[idx], out):
            print(f"IN: {x}  |  OUT: {h:.4f}  |  TARGET: {y}")
        print(f"\nError: {error}")

    return (1 / error) if error > 0 else 0


if __name__ == "__main__":
    # Creating a new random population of fixed-topology genomes:
    # (we're going to use the default settings)
    population = ne.genetic_algorithm.GeneticPopulation(size=150,
                                                        base_genome=BASE_GENOME)

    # Evolving the population:
    # (don't worry if you see a bunch of messages from TensorFlow, it's normal)
    history = population.evolve(generations=15,
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
