"""
NEvoPY
todo
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import nevopy
from nevopy import neat
from nevopy import fixed_topology as fito

from timeit import default_timer as timer
import numpy as np
from tensorflow import reshape

import logging
logging.basicConfig(level=logging.INFO)


ALGORITHM = "fixed_topology"  # "neat" or "fixed_topology"
MAX_GENERATIONS = 100
FITNESS_THRESHOLD = 1e3
NUM_VARIABLES = 2


# =============== MAKING XOR DATA ==================
assert NUM_VARIABLES > 1
xor_inputs = []
xor_outputs = []

for num in range(2 ** NUM_VARIABLES):
    binary = bin(num)[2:].zfill(NUM_VARIABLES)
    xin = [int(binary[0])]
    xout = int(binary[0])
    for bit in binary[1:]:
        xin.append(int(bit))
        xout ^= int(bit)
    xor_inputs.append(np.array(xin))
    xor_outputs.append(np.array(xout))

xor_inputs = np.array(xor_inputs)
xor_outputs = np.array(xor_outputs)
# ===================================================


def eval_neat_genome(genome: neat.genomes.NeatGenome,
                     log=False) -> float:
    idx = np.random.permutation(len(xor_inputs))
    error = 0
    for x, y in zip(xor_inputs[idx], xor_outputs[idx]):
        genome.reset()

        h = genome.process(x)[0]
        error += (y - h) ** 2

        if log:
            print(f"IN: {x}  |  OUT: {h}  |  TARGET: {y}")

    if log:
        print(f"\nError: {error}")

    return 1 / error


def eval_fito_genome(genome: neat.genomes.FixedTopologyGenome,
                     log=False) -> float:
    idx = np.random.permutation(len(xor_inputs))
    H = genome.process(xor_inputs[idx])
    H = reshape(H, [-1]).numpy()
    error = ((H - xor_outputs[idx]) ** 2).sum()

    if log:
        for x, y, h in zip(xor_inputs[idx], xor_outputs[idx], H):
            print(f"IN: {x}  |  OUT: {h}  |  TARGET: {y}")
        print(f"\nError: {error}")

    return 1 / error


def load_and_test(path, cls):
    genome = cls.load(path)
    if cls == neat.NeatGenome:
        print(f"Fitness: {eval_neat_genome(genome, log=True)}")
    else:
        print(f"Fitness: {eval_fito_genome(genome, log=True)}")


if __name__ == "__main__":
    # load_and_test("./.temp/2021-01-31-12-20-06_genome_checkpoint5.pkl",
    #               fito.FixedTopologyGenome)
    runs = 1
    total_time = 0
    pop = history = None

    if ALGORITHM == "fixed_topology":
        base_genome = fito.FixedTopologyGenome(
            layers=[fito.layers.TFDenseLayer(8, activation="relu"),
                    fito.layers.TFDenseLayer(1, activation="relu")],
            input_shape=xor_inputs.shape,
        )
        func = eval_fito_genome
    else:
        func = eval_neat_genome

    for r in range(runs):
        start_time = timer()
        if ALGORITHM == "neat":
            pop = neat.population.NeatPopulation(
                size=200,
                num_inputs=len(xor_inputs[0]),
                num_outputs=1,
            )
        else:
            pop = fito.FixedTopologyPopulation(size=200,
                                               base_genome=base_genome)

        history = pop.evolve(generations=MAX_GENERATIONS,
                             fitness_function=func,
                             verbose=2,
                             callbacks=[
                                 nevopy.callbacks.FitnessEarlyStopping(
                                     fitness_threshold=FITNESS_THRESHOLD,
                                     min_consecutive_generations=3,
                                 ),
                                 # nevopy.callbacks.BestGenomeCheckpoint(
                                 #    output_path="./.temp",
                                 #    min_improvement_pc=0.5,
                                 # ),
                             ])

        deltaT = timer() - start_time
        total_time += deltaT

    best = pop.fittest()
    print()
    func(best, log=True)
    best.visualize()

    print("\n" + 20 * "=")
    print(f"\nTotal time: {total_time}s"
          f"\nAvg. time: {total_time / runs}s\n")

    history.visualize()
    history.visualize(attrs=["processing_time"])
