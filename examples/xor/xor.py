"""
NEvoPY
todo
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import nevopy
from nevopy import neat
from nevopy import fixed_topology as fito
from timeit import default_timer as timer
import numpy as np
from tensorflow import expand_dims
import random

# =============== MAKING XOR DATA ==================
num_variables = 2
assert num_variables > 1

xor_inputs = []
xor_outputs = []

for num in range(2 ** num_variables):
    binary = bin(num)[2:].zfill(num_variables)
    xin = [int(binary[0])]
    xout = int(binary[0])
    for bit in binary[1:]:
        xin.append(int(bit))
        xout ^= int(bit)
    xor_inputs.append(np.array(xin))
    xor_outputs.append(np.array(xout))
# ===================================================


def eval_genome(genome: neat.genomes.NeatGenome,
                shuffle=True,
                log=False) -> float:
    idx = list(range(len(xor_inputs)))
    if shuffle:
        random.shuffle(idx)

    error = 0
    for i in idx:
        x, y = xor_inputs[i], xor_outputs[i]
        genome.reset()
        h = genome.process(expand_dims(x, axis=0))[0].numpy()[0]
        print(h)
        error += (y - h) ** 2
        if log:
            print(f"IN: {x}  |  OUT: {h}  |  TARGET: {y}")
    if log:
        print(f"\nError: {error}")

    return 1/error


if __name__ == "__main__":
    runs = 1
    total_time = 0
    pop = history = None
    base_layers = [
        fito.layers.TFDenseLayer(32, activation="relu"),
        fito.layers.TFDenseLayer(1, activation="relu"),
    ]

    for r in range(runs):
        start_time = timer()
        # pop = neat.population.NeatPopulation(
        #     size=200,
        #     num_inputs=len(xor_inputs[0]),
        #     num_outputs=1,
        # )

        pop = fito.FixedTopologyPopulation(size=100,
                                           base_layers=base_layers)

        history = pop.evolve(generations=100,
                             fitness_function=eval_genome,
                             verbose=2,
                             callbacks=[
                                 nevopy.callbacks.FitnessEarlyStopping(
                                     fitness_threshold=1e3,
                                     min_consecutive_generations=3,
                                 )
                             ])

        deltaT = timer() - start_time
        total_time += deltaT

    best = pop.fittest()
    print()
    eval_genome(best, log=True)
    # print(best.info())
    best.visualize()

    print("\n" + 20*"=" +
          # f"\nReplaced invalid genomes: {sum(history.invalid_genomes_replaced)}"
          f"\nTotal time: {total_time}s"
          f"\nAvg. time: {total_time / runs}s\n")

    history.visualize()
    history.visualize(attrs=["processing_time"])

    # print("\n\n\nSaving population...")
    # pop.save("./test/test_pop.pkl")
    #
    # del pop
    # del best
    #
    # pop = nevopy.neat.population.NeatPopulation.load("./test/test_pop.pkl")
    # print(pop.info())
    # best = pop.fittest()
    # eval_genome(best, log=True)
    # print(best.info())
    # best.visualize()