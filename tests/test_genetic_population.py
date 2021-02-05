# MIT License
#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" Tests the implementation of
:class:`nevopy.genetic_algorithm.GeneticPopulation`.
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from timeit import default_timer as timer
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from tensorflow import reshape

import nevopy as ne


FITNESS_THRESHOLD = 1e3
SPECIATION = True
CONFIG = ne.genetic_algorithm.GeneticAlgorithmConfig(
    # weight mutation
    mutation_chance=(0.6, 0.9),
    weight_mutation_chance=(0.5, 1),
    weight_perturbation_pc=(0.07, 0.5),
    weight_reset_chance=(0.07, 0.5),
    new_weight_interval=(-2, 2),
    # reproduction
    weak_genomes_removal_pc=0.5,
    mating_chance=0.75,
    interspecies_mating_chance=0.05,
    mating_mode="weights_mating",
    rank_prob_dist_coefficient=1.75,
    predatism_chance=0.1,
    # speciation
    species_distance_threshold=1.7,
    species_elitism_threshold=5,
    elitism_pc=0.03,
    species_no_improvement_limit=15,
    # mass extinction
    mass_extinction_threshold=25,
    maex_improvement_threshold_pc=0.03,
)

xor_in, xor_out = ne.utils.make_xor_data(num_variables=2)


def eval_genome(genome, log=False) -> float:
    idx = np.random.permutation(len(xor_in))
    H = genome.process(xor_in[idx])
    H = reshape(H, [-1]).numpy()
    error = ((H - xor_out[idx]) ** 2).sum()

    if log:
        for x, y, h in zip(xor_in[idx], xor_out[idx], H):
            print(f"IN: {x}  |  OUT: {h}  |  TARGET: {y}")
        print(f"\nError: {error}")

    return 1 / error


if __name__ == "__main__":
    print(xor_in, xor_out, end="\n\n")
    base_genome = ne.fixed_topology.FixedTopologyGenome(
        layers=[
            ne.fixed_topology.layers.TFDenseLayer(units=8, activation="relu"),
            ne.fixed_topology.layers.TFDenseLayer(units=1, activation="relu")
        ],
        input_shape=xor_in.shape,
    )

    population = ne.genetic_algorithm.GeneticPopulation(
        size=150,
        base_genome=base_genome,
        config=CONFIG,
        speciation=SPECIATION,
    )

    start_time = timer()
    history = population.evolve(generations=100,
                                fitness_function=eval_genome,
                                verbose=2,
                                callbacks=[
                                    ne.callbacks.FitnessEarlyStopping(
                                        fitness_threshold=FITNESS_THRESHOLD,
                                        min_consecutive_generations=3,
                                    )
                                ])
    total_time = timer() - start_time

    best = population.fittest()
    print()
    eval_genome(best, log=True)
    best.visualize()

    print("\n" + 20 * "=")
    print(f"\nTotal time: {total_time}s")

    history.visualize()
    history.visualize(attrs=["processing_time"])
