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

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from nevopy.fixed_topology.genomes import FixedTopologyGenome
from nevopy.fixed_topology.layers import TFConv2DLayer
from nevopy.fixed_topology.config import FixedTopologyConfig
import tensorflow as tf
import numpy as np


config = FixedTopologyConfig(  # weight mutation
                             weight_mutation_chance=(0.7, 0.9),
                             weight_perturbation_pc=(0.1, 0.4),
                             weight_reset_chance=(0.1, 0.3),
                             new_weight_interval=(-2, 2),
                               # bias mutation
                             bias_mutation_chance=(0.5, 0.7),
                             bias_perturbation_pc=(0.1, 0.3),
                             bias_reset_chance=(0.1, 0.2),
                             new_bias_interval=(-2, 2),
                               # others
                             mating_mode="exchange_weights")


def get_weight_lists(genome):
    w_list, b_list = [], []
    for layer in genome.layers:
        w, b = layer.weights
        w_list.append(w.numpy())
        b_list.append(b.numpy())
    return w_list, b_list


def test_processing_and_mutation(genome, num_tests=100):
    past_w_list = past_b_list = None
    for _ in range(num_tests):
        w_list, b_list = get_weight_lists(genome)
        if past_w_list is not None:
            for (w0, w1), (b0, b1) in zip(zip(past_w_list, w_list),
                                          zip(past_b_list, b_list)):
                assert not (w0 == w1).all() and not (b0 == b1).all()

        past_w_list, past_b_list = w_list, b_list
        for __ in range(3):
            genome.mutate_weights()
            data = tf.random.uniform(genome.input_shape)
            r1, r2 = genome.process(data).numpy(), genome(data).numpy()
            assert (r1 == r2).all()


def test_random_copy(genome):
    new_genome = genome.random_copy()
    for l_new, l_old in zip(new_genome.layers, genome.layers):
        w_new, b_new = [w.numpy() for w in l_new.weights]
        w_old, b_old = [w.numpy() for w in l_old.weights]
        assert w_new.shape == w_old.shape and b_new.shape == b_old.shape
        assert not (w_new == w_old).all()


def test_deep_copy(genome, num_tests=100):
    for _ in range(num_tests):
        new_genome = genome.deep_copy()
        w_new, b_new = get_weight_lists(new_genome)
        w_old, b_old = get_weight_lists(genome)
        for (w0, w1), (b0, b1) in zip(zip(w_new, w_old), zip(b_new, b_old)):
            assert (w0 == w1).all() and (b0 == b1).all()

        for __ in range(5):
            data = tf.random.uniform(genome.input_shape)
            r_new, r_old = new_genome(data).numpy(), genome(data).numpy()
            assert (r_new == r_old).all()

        for __ in range(5):
            genome.mutate_weights()
            new_genome.mutate_weights()

            w_new, b_new = get_weight_lists(new_genome)
            w_old, b_old = get_weight_lists(genome)
            for (w0, w1), (b0, b1) in zip(zip(w_new, w_old), zip(b_new, b_old)):
                assert not (w0 == w1).all() and not (b0 == b1).all()


def test_mating(g1, g2, mode, num_tests=100):
    assert mode in ("exchange_weights", "exchange_layers")
    config.mating_mode = mode
    parents = (g1, g2)

    for _ in range(num_tests):
        g1.mutate_weights()
        g2.mutate_weights()

        p1, p2 = np.random.choice(parents), np.random.choice(parents)
        g_new = p1.mate(p2)

        w_new_list, b_new_list = get_weight_lists(g_new)
        w1_list, b1_list = get_weight_lists(p1)
        w2_list, b2_list = get_weight_lists(p2)

        for (w0, w1, w2), (b0, b1, b2) in zip(zip(w_new_list, w1_list, w2_list),
                                              zip(b_new_list, b1_list, b2_list)):
            if p1 != p2:
                if mode == "exchange_weights":
                    assert not (w0 == w1).all() and not (w0 == w2).all()
                    assert not (b0 == b1).all() and not (b0 == b2).all()
                else:
                    assert (w0 == w1).all() ^ (w0 == w2).all()
                    assert (b0 == b1).all() ^ (b0 == b2).all()
            else:
                assert (w0 == w1).all() and (w0 == w2).all()
                assert (b0 == b1).all() and (b0 == b2).all()


if __name__ == "__main__":
    # fixing TF
    # tf_config = tf.compat.v1.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # tf_session = tf.compat.v1.InteractiveSession(config=tf_config)

    # genomes
    genome1 = FixedTopologyGenome(config=config, layers=[
        TFConv2DLayer(filters=64, kernel_size=(3, 3)),
        TFConv2DLayer(filters=32, kernel_size=(3, 3)),
        TFConv2DLayer(filters=128, kernel_size=(3, 3)),
    ])
    genome2 = FixedTopologyGenome(config=config, layers=[
        TFConv2DLayer(filters=64, kernel_size=(3, 3)),
        TFConv2DLayer(filters=32, kernel_size=(3, 3)),
        TFConv2DLayer(filters=128, kernel_size=(3, 3)),
    ])

    # building layers
    input_shape = (1, 128, 128, 3)
    genome1(tf.random.uniform(input_shape))
    genome2(tf.random.uniform(input_shape))

    # tests
    print("\nProcessing and mutation... ", end="")
    test_processing_and_mutation(genome1, num_tests=100)
    print("passed!\nRandom copy... ", end="")

    test_random_copy(genome1)
    print("passed!\nDeep copy... ", end="")

    test_deep_copy(genome1, num_tests=100)
    print("passed!\nMating (exchange weights)... ", end="")

    test_mating(genome1, genome2, mode="exchange_weights", num_tests=100)
    print("passed!\nMating (exchange layers)... ", end="")

    test_mating(genome1, genome2, mode="exchange_layers", num_tests=100)
    print("passed!")

    print("\n" + "_"*30 + "\nPassed in all assertions!")
