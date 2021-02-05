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

""" Tests the implementation in :mod:`.fixed_topology.genome`.
"""

# pylint: disable=wrong-import-position
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# pylint: enable=wrong-import-position
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

from nevopy.fixed_topology.config import FixedTopologyConfig
from nevopy.fixed_topology.genomes import FixedTopologyGenome
from nevopy.fixed_topology.layers import TFConv2DLayer
from nevopy.fixed_topology.layers import TFFlattenLayer
from nevopy.fixed_topology.layers import TFDenseLayer
import test_utils


config = FixedTopologyConfig(  # weight mutation
                             weight_mutation_chance=(0.5, 0.9),
                             weight_perturbation_pc=(0.2, 0.4),
                             weight_reset_chance=(0.2, 0.3),
                             new_weight_interval=(-2, 2),
                               # others
                             mating_mode="exchange_weights_mating")


def test_processing_and_mutation(genome, num_tests=5, verbose=False):
    past_layers_weights = None
    mutation_time = 0
    for t in range(num_tests):
        if verbose:
            print(f"\n>> Test {t} of {num_tests}.")

        layers_weights = [nl.weights for nl in genome.layers]

        if past_layers_weights is not None:
            for l_num, (w_list0, w_list1) in enumerate(zip(past_layers_weights,
                                                       layers_weights)):
                for w0, w1 in zip(w_list0, w_list1):
                    if genome.layers[l_num].mutable:
                        if verbose:
                            print(f"[L{l_num}] Mutable layer.")
                            # print(f"w0 = \n{w0}\n\n")
                            # print(f"w1 = \n{w1}\n\n")

                        if not (w0 == 0).all():
                            assert not (w0 == w1).all()
                    else:
                        if verbose:
                            print(f"[L{l_num}] Immutable layer.")
                        assert (w0 == w1).all()

        past_layers_weights = layers_weights

        if verbose:
            print("Mutating...")

        start_time = timer()
        genome.mutate_weights()
        mutation_time += timer() - start_time

        if verbose:
            print("Processing input...")

        for _ in range(3):
            data = tf.random.uniform(genome.input_shape, dtype=float)
            r1, r2 = genome.process(data).numpy(), genome(data).numpy()
            assert (r1 == r2).all()

            if verbose:
                print(f"Output shape: {r1.shape}")

    if verbose:
        print("\n" + "#"*30 + "\n" + "Results:")
    print(f"\t. Mutation time: {1000 * mutation_time / num_tests:.2f}ms")


def test_random_copy(genome, num_tests=10):
    random_copy_time = 0
    for _ in range(num_tests):
        start_time = timer()
        new_genome = genome.random_copy()
        random_copy_time += timer() - start_time

        for l_new, l_old in zip(new_genome.layers, genome.layers):
            assert l_new.mutable == l_old.mutable

            for w_new, w_old in zip(l_new.weights, l_old.weights):
                assert w_new.shape == w_old.shape

                if l_old.mutable:
                    assert not (w_new == w_old).all()
                else:
                    assert (w_new == w_old).all()

    print(f"\t. Random copy time: {1000 * random_copy_time / num_tests:.2f}ms")


def test_deep_copy(genome, num_tests=10):
    mutation_time = dp_copy_time = 0
    for _ in range(num_tests):
        start_time = timer()
        new_genome = genome.deep_copy()
        dp_copy_time += timer() - start_time

        for l_new, l_old in zip(new_genome.layers, genome.layers):
            assert l_new.mutable == l_old.mutable

            for w_new, w_old in zip(l_new.weights, l_old.weights):
                assert (w_new == w_old).all()

        for _ in range(3):
            data = tf.random.uniform(genome.input_shape, dtype=float)
            r_new, r_old = new_genome(data).numpy(), genome(data).numpy()
            assert (r_new == r_old).all()

        start_time = timer()
        genome.mutate_weights()
        new_genome.mutate_weights()
        mutation_time += (timer() - start_time) / 2

        for l_new, l_old in zip(new_genome.layers, genome.layers):
            for w_new, w_old in zip(l_new.weights, l_old.weights):
                if l_old.mutable:
                    assert not (w_new == w_old).all()
                else:
                    assert (w_new == w_old).all()

    print(f"\t. Mutation time: {1000 * mutation_time / num_tests:.2f}ms")
    print(f"\t. Deep copy time: {1000 * dp_copy_time / num_tests:.2f}ms")


def test_mating(g1, g2, mode, num_tests=10):
    assert mode in ("weights_mating", "exchange_layers")
    config.mating_mode = mode
    parents = (g1, g2)

    mating_time = 0
    for _ in range(num_tests):
        g1.mutate_weights()
        g2.mutate_weights()

        p1, p2 = np.random.choice(parents), np.random.choice(parents)

        start_time = timer()
        g_new = p1.mate(p2)
        mating_time += timer() - start_time

        for l_p1, l_p2, l_new in zip(p1.layers, p2.layers, g_new.layers):
            for w_p1, w_p2, w_new in zip(l_p1.weights,
                                         l_p2.weights,
                                         l_new.weights):
                if p1 == p2:
                    assert (w_new == w_p1).all() and (w_new == w_p2).all()
                else:
                    if mode == "weights_mating":
                        if l_p1.mutable:
                            if not (w_p1 == w_p2).all():
                                assert not (w_new == w_p1).all()
                                assert not (w_new == w_p2).all()
                        else:
                            assert (w_new == w_p1).all()
                    else:
                        if not (w_p1 == w_p2).all():
                            assert (w_new == w_p1).all() ^ (w_new == w_p2).all()

    print(f"\t. Mating time: {1000 * mating_time / num_tests:.2f}ms")


def test_distance(g1, num_tests=10):
    total_time = 0
    dist_history = []
    g2 = g1.deep_copy()
    for _ in range(num_tests):
        start_time = timer()
        d = g1.distance(g2)
        total_time += timer() - start_time

        dist_history.append(d)
        g1.mutate_weights()
        g2.mutate_weights()

    print(f"\t. Distance time: {1000 * total_time / num_tests:.4f}ms")
    print(f"\t. Distance progression: "
          f"{[f'{d:.2f}' for d in dist_history]}")
    print(f"\t. Avg distance: {np.mean(dist_history):.4f}")


def test_save_and_load(genome, num_tests=10):
    saving_time = loading_time = file_size = weights_size = 0
    for _ in range(num_tests):
        genome.mutate_weights()
        file_name = (f"{test_utils.TEMP_SAVE_DIR}/"
                     f"saved_genome_{int(1e6 * timer())}")

        start_time = timer()
        genome.save(file_name)
        saving_time += timer() - start_time

        file_size += os.path.getsize(file_name + ".pkl") / 1024

        start_time = timer()
        loaded_genome = FixedTopologyGenome.load(file_name)
        loading_time += timer() - start_time

        for layer0, layer1 in zip(genome.layers, loaded_genome.layers):
            for w0, w1 in zip(layer0.weights, layer1.weights):
                assert w0.shape == w1.shape
                assert (w0 == w1).all()
                weights_size += w0.nbytes / 1024

        for _ in range(3):
            test_input = tf.random.uniform(shape=genome.input_shape,
                                           dtype=float)
            h0, h1 = genome(test_input), loaded_genome(test_input)
            assert (h0.numpy() == h1.numpy()).all()

    print(f"\t. Saving time: {1000 * saving_time / num_tests:.2f}ms")
    print(f"\t. Loading time: {1000 * loading_time / num_tests:.2f}ms")

    file_size /= num_tests
    file_size = (f"{file_size / 1024:.2f}MB" if file_size >= 1024
                 else f"{file_size:.2f}KB")
    weights_size /= num_tests
    weights_size = (f"{weights_size / 1024:.2f}MB" if weights_size >= 1024
                    else f"{weights_size:.2f}KB")

    print(f"\t. File size: {file_size}")
    print(f"\t. Weights size: {weights_size}")


if __name__ == "__main__":
    test_utils.clear_temp()
    input_shape = (4, 128, 128, 3)

    # Creating genomes:
    genome1 = FixedTopologyGenome(
        config=config,
        layers=[
            TFConv2DLayer(filters=64, kernel_size=(3, 3), mutable=False),
            TFConv2DLayer(filters=64, kernel_size=(5, 5), strides=(3, 3)),
            TFConv2DLayer(filters=32, kernel_size=(4, 4), strides=(2, 2)),
            TFFlattenLayer(),
            TFDenseLayer(units=64, activation="relu"),
            TFDenseLayer(units=64, activation="relu"),
            TFDenseLayer(units=20, activation="softmax"),
        ],
        input_shape=input_shape,
    )

    genome2 = FixedTopologyGenome(
        config=config,
        layers=[
            TFConv2DLayer(filters=64, kernel_size=(3, 3), mutable=False),
            TFConv2DLayer(filters=64, kernel_size=(5, 5), strides=(3, 3)),
            TFConv2DLayer(filters=32, kernel_size=(4, 4), strides=(2, 2)),
            TFFlattenLayer(),
            TFDenseLayer(units=64, activation="relu"),
            TFDenseLayer(units=64, activation="relu"),
            TFDenseLayer(units=20, activation="softmax"),
        ],
        input_shape=input_shape,
    )

    # Weight matrices shapes
    for i, layer in enumerate(genome1.layers):
        for j, w in enumerate(layer.weights):
            print(f"[L{i}][W{j}][{type(layer)}] Shape: {w.shape}")

    # Testing:
    print("\n> Processing and mutation:")
    test_processing_and_mutation(genome1, num_tests=10)

    print("> Random copy:")
    test_random_copy(genome1, num_tests=10)

    print("> Deep copy:")
    test_deep_copy(genome1, num_tests=10)

    print("> Mating (weights):")
    test_mating(genome1, genome2, mode="weights_mating", num_tests=10)

    print("> Mating (layers):")
    test_mating(genome1, genome2, mode="exchange_layers", num_tests=10)

    print("> Distance:")
    test_distance(genome1, num_tests=20)

    print("> Saving and loading:")
    test_save_and_load(genome1, num_tests=10)

    print("\n" + "_"*30 + "\nPassed in all assertions!")
    test_utils.clear_temp()
