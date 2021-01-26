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

""" Tests the implementation in :mod:`.fixed_topology.layers`.
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from nevopy.fixed_topology.config import FixedTopologyConfig
from nevopy.fixed_topology.layers.tf_layers import TFConv2DLayer
from nevopy.fixed_topology.layers.tf_layers import TensorFlowLayer
from nevopy.fixed_topology.layers.base_layer import IncompatibleLayersError
from nevopy.fixed_topology.layers import mating

from timeit import default_timer as timer
import numpy as np
import tensorflow as tf

config = FixedTopologyConfig(  # weight mutation
                             weight_mutation_chance=(0.7, 0.9),
                             weight_perturbation_pc=(0.1, 0.4),
                             weight_reset_chance=(0.1, 0.3),
                             new_weight_interval=(-2, 2),
                               # bias mutation
                             bias_mutation_chance=(0.5, 0.7),
                             bias_perturbation_pc=(0.1, 0.3),
                             bias_reset_chance=(0.1, 0.2),
                             new_bias_interval=(-2, 2))


def test_mutate_weights(layer, num_tests=100, verbose=False):
    deltaT = 0
    for _ in range(num_tests):
        test_info = {}
        weights_old = layer.weights

        start_time = timer()
        layer.mutate_weights(_test_info=test_info)
        deltaT += timer() - start_time

        weights_new = layer.weights

        for w_num, (w_old, w_new) in enumerate(zip(weights_old, weights_new)):
            if verbose:
                print(f"\n[WEIGHT MATRIX AT INDEX {w_num}]")

            diff_pc = np.divide(w_new - w_old, w_old,
                                out=np.full_like(w_old, fill_value=-123456),
                                where=w_old != 0)
            count_reset = 0
            for i in range(w_old.size):
                d_real = diff_pc.flat[i]
                if d_real == -123456:
                    continue

                d_calc = (test_info[f"w{w_num}_perturbation"].numpy() - 1).flat[i]

                if verbose:
                    print(f"[ABS CHANGE] {w_old.flat[i]:.5f} "
                          f"-> {w_new.flat[i]:.5f}  <>  "
                          f"[% CHANGE] real = {d_real:.2%} | "
                          f"calc = {d_calc:.2%}  <>  "
                          f"Reset: {i in test_info[f'w{w_num}_reset_idx']}")

                if abs(d_real - d_calc) > 1e-5:
                    if i in test_info[f"w{w_num}_reset_idx"]:
                        count_reset += 1
                        continue
                    raise RuntimeError("Incorrect weight value!")
            assert count_reset <= len(test_info[f"w{w_num}_reset_idx"])
            if verbose:
                print(f"Reset weights idx: "
                      f"{test_info[f'w{w_num}_reset_idx']}\n\n")

    if verbose:
        print("\n" + "="*50)
    print(f"> Mutation time: {1000 * deltaT / num_tests}ms")


def test_immutable_layer_mutation(layer, num_tests=100, verbose=False):
    _mutable = layer.mutable
    layer.mutable = False

    deltaT = 0
    for _ in range(num_tests):
        old_weights = layer.weights

        start_time = timer()
        layer.mutate_weights()
        deltaT += timer() - start_time

        new_weights = layer.weights
        for w_new, w_old in zip(new_weights, old_weights):
            assert (w_new == w_old).all()

    layer.mutable = _mutable
    if verbose:
        print("\n" + "=" * 50)
    print("> Immutable layer mutation time: "
          f"{1000 * deltaT / num_tests}ms")


def test_deep_copy(layer, num_tests=100, verbose=False):
    deltaT = 0
    for _ in range(num_tests):
        layer.mutate_weights()

        start_time = timer()
        new_layer = layer.deep_copy()
        deltaT += timer() - start_time

        # Checking if weights are equal
        for w_new, w_old in zip(new_layer.weights, layer.weights):
            if verbose:
                for _ in range(50):
                    i = np.random.randint(low=0, high=w_old.size)
                    print(f"[W] old = {w_old.flat[i]}, new = {w_new.flat[i]}")

            assert w_new.shape == w_old.shape
            assert (w_new == w_old).all()

        # Checking if outputs are equal
        for _ in range(20):
            random_input = tf.random.uniform(shape=layer.input_shape)
            y_new = new_layer(random_input).numpy()
            y_old = layer(random_input).numpy()

            if verbose:
                print(f"[OUTPUT] diff = {np.abs(y_old - y_new).sum()}")

            assert (y_new == y_old).all()

    if verbose:
        print("\n" + "="*50)
    print(f"> Deep copy time: {1000 * deltaT / num_tests}ms")


def test_random_copy(layer, num_tests=100, verbose=False):
    deltaT = 0
    for _ in range(num_tests):
        layer.mutate_weights()

        start_time = timer()
        new_layer = layer.random_copy()
        deltaT += timer() - start_time

        # Checking if weights are different
        for w_old, w_new in zip(layer.weights, new_layer.weights):
            if verbose:
                for _ in range(50):
                    i = np.random.randint(low=0, high=w_old.size)
                    print(f"[W] old = {w_old.flat[i]}, new = {w_new.flat[i]}")

            assert w_old.shape == w_new.shape
            assert not (w_old == w_new).all()

        # Checking if outputs are different
        for _ in range(20):
            random_input = tf.random.uniform(shape=layer.input_shape)
            y_new = new_layer(random_input).numpy()
            y_old = layer(random_input).numpy()

            if verbose:
                print(f"[OUTPUT] diff = {np.abs(y_old - y_new).sum()}")

            assert not (y_new == y_old).all()

    if verbose:
        print("\n" + "=" * 50)
    print(f"> Random copy time: {1000 * deltaT / num_tests}ms")


def test_exchange_units_mating(layer1, layer2, num_tests=100, verbose=False):
    layer1.mating_func = layer2.mating_func = mating.exchange_units_mating
    mate_deltaT = mutation_deltaT = 0
    units_from_p1 = units_from_p2 = 0
    for t in range(num_tests):
        if verbose:
            print("\n" + "#"*50)
            print(f"----- Test {t} of {num_tests} -----\n")

        # Mutating layers
        start_time = timer()
        layer1.mutate_weights()
        layer2.mutate_weights()
        mutation_deltaT += (timer() - start_time) / 2

        # Mating layers
        start_time = timer()
        new_layer = layer1.mate(layer2)
        mate_deltaT += timer() - start_time

        # Checking units inheritance
        for w_new, w_p1, w_p2 in zip(new_layer.weights,
                                     layer1.weights,
                                     layer2.weights):
            assert w_new.shape == w_p1.shape == w_p2.shape
            for i in range(w_p1.shape[-1]):
                unit_new = w_new[..., i]
                unit_p1 = w_p1[..., i]
                unit_p2 = w_p2[..., i]

                if (unit_new == unit_p1).all():
                    units_from_p1 += 1
                    if verbose:
                        print(f"[{i}] Unit from parent 1 <> "
                              f"diff_sum = {np.abs(unit_new - unit_p1).sum()}")
                elif (unit_new == unit_p2).all():
                    units_from_p2 += 1
                    if verbose:
                        print(f"[{i}] Unit from parent 2 <> "
                              f"diff_sum = {np.abs(unit_new - unit_p2).sum()}")
                else:
                    raise AssertionError(
                        f"New unit at index {i} doesn't match any "
                        f"of the parents units at the same index!"
                        f"\n> New unit = \n{unit_new}\n\n"
                        f"> Parent 1 unit = \n{unit_p1}\n\n"
                        f"> Parent 2 unit = \n{unit_p2}")

    if verbose:
        print("\n" + "=" * 50)
        total_units = units_from_p1 + units_from_p2
        print(f"Units from parent 1: {units_from_p1 / total_units:.2%}")
        print(f"Units from parent 2: {units_from_p2 / total_units:.2%}\n")

    print(f"> Exchange units mating:\n"
          f"\t. Mating time: {1000 * mate_deltaT / num_tests}ms\n"
          f"\t. Mutation time: {1000 * mutation_deltaT / num_tests}ms")


def test_exchange_weights_mating(layer1, layer2, num_tests=100, verbose=False):
    layer1.mating_func = layer2.mating_func = mating.exchange_weights_mating
    mate_deltaT = mutation_deltaT = 0
    w_from_p1 = w_from_p2 = 0
    for t in range(num_tests):
        if verbose:
            print("\n" + "#"*50)
            print(f"----- Test {t} of {num_tests} -----\n")

        # Mutating layers:
        start_time = timer()
        layer1.mutate_weights()
        layer2.mutate_weights()
        mutation_deltaT += (timer() - start_time) / 2

        # Mating layers:
        start_time = timer()
        new_layer = layer1.mate(layer2)
        mate_deltaT += timer() - start_time

        # Checking individual weights inheritance:
        for w_new, w_p1, w_p2 in zip(new_layer.weights,
                                     layer1.weights,
                                     layer2.weights):
            assert w_new.shape == w_p1.shape == w_p2.shape
            for i in range(w_p1.size):
                if w_new.flat[i] == w_p1.flat[i]:
                    w_from_p1 += 1
                    if verbose:
                        print(f"[{i}] Weight from parent 1 <> "
                              f"parent = {w_p1.flat[i]}, "
                              f"child = {w_new.flat[i]}")
                elif w_new.flat[i] == w_p2.flat[i]:
                    w_from_p2 += 1
                    if verbose:
                        print(f"[{i}] Weight from parent 2 <> "
                              f"parent = {w_p2.flat[i]}, "
                              f"child = {w_new.flat[i]}")
                else:
                    raise AssertionError(
                        f"New weight at index {i} doesn't match any of the "
                        "parents weights at the same index!\n"
                        f"> New weight = {w_new.flat[i]}\n"
                        f"> Parent 1 weight = {w_p1.flat[i]}\n"
                        f"> Parent 2 weight = {w_p2.flat[i]}"
                    )

    if verbose:
        print("\n" + "=" * 50)
        total_w = w_from_p1 + w_from_p2
        print(f"Weights from parent 1: {w_from_p1 / total_w:.2%}")
        print(f"Weights from parent 2: {w_from_p2 / total_w:.2%}\n")

    print(f"> Exchange weights mating:\n"
          f"\t. Mating time: {1000 * mate_deltaT / num_tests}ms\n"
          f"\t. Mutation time: {1000 * mutation_deltaT / num_tests}ms")


def test_weights_avg_mating(layer1, layer2, num_tests=100, verbose=False):
    layer1.mating_func = layer2.mating_func = mating.weights_avg_mating
    mate_deltaT = mutation_deltaT = 0
    for t in range(num_tests):
        if verbose:
            print(f"> Test {t} of {num_tests}.")

        # Mutating layers:
        start_time = timer()
        layer1.mutate_weights()
        layer2.mutate_weights()
        mutation_deltaT += (timer() - start_time) / 2

        # Mating layers:
        start_time = timer()
        new_layer = layer1.mate(layer2)
        mate_deltaT += timer() - start_time

        # Checking individual weights inheritance:
        for w_new, w_p1, w_p2 in zip(new_layer.weights,
                                     layer1.weights,
                                     layer2.weights):
            assert w_new.shape == w_p1.shape == w_p2.shape
            for i in range(w_p1.size):
                assert w_new.flat[i] == (w_p1.flat[i] + w_p2.flat[i]) / 2

    if verbose:
        print("\n" + "=" * 50)

    print(f"> Weights averaging mating:\n"
          f"\t. Mating time: {1000 * mate_deltaT / num_tests}ms\n"
          f"\t. Mutation time: {1000 * mutation_deltaT / num_tests}ms")


def test_immutable_layer_mating(layer1, layer2, num_tests=100, verbose=False):
    _mutable1 = layer1.mutable
    _mutable2 = layer2.mutable

    deltaT = 0
    for _ in range(num_tests):
        # Valid mating
        layer1.mutable = layer2.mutable = False
        start_time = timer()
        new_layer1 = layer1.mate(layer2)
        new_layer2 = layer2.mate(layer1)
        deltaT += (timer() - start_time) / 2

        for old, new in [(layer1, new_layer1), (layer2, new_layer2)]:
            for w_old, w_new in zip(old.weights, new.weights):
                assert (w_old == w_new).all()

        # Invalid mating
        layer2.mutable = True
        try:
            layer1.mate(layer2)
            layer2.mate(layer1)
            raise AssertionError("Invalid mating occurred!")
        except IncompatibleLayersError:
            pass

    layer1.mutable = _mutable1
    layer2.mutable = _mutable2
    if verbose:
        print("\n" + "=" * 50)

    print("> Immutable layer valid mating time: "
          f"{1000 * deltaT / num_tests}ms")


def run_all_tests(test_layer1, test_layer2):
    if test_layer1.mutable:
        test_mutate_weights(test_layer1, num_tests=100, verbose=False)
        test_exchange_units_mating(test_layer1, test_layer2,
                                   num_tests=100, verbose=False)
        test_exchange_weights_mating(test_layer1, test_layer2,
                                     num_tests=100, verbose=False)
        test_weights_avg_mating(test_layer1, test_layer2,
                                num_tests=100, verbose=False)

    test_deep_copy(test_layer1, num_tests=100, verbose=False)
    test_random_copy(test_layer1, num_tests=100, verbose=False)
    test_immutable_layer_mating(test_layer1, test_layer2,
                                num_tests=100, verbose=False)
    test_immutable_layer_mutation(test_layer1, num_tests=100, verbose=False)


def test_conv2d():
    input_shape = (1, 128, 128, 3)
    test_layer1 = TensorFlowLayer(filters=64,
                                  kernel_size=(3, 3),
                                  layer_type=tf.keras.layers.Conv2D,
                                  mating_func=mating.exchange_units_mating,
                                  config=config,
                                  input_shape=input_shape)
    test_layer2 = TFConv2DLayer(filters=64,
                                kernel_size=(3, 3),
                                config=config,
                                input_shape=input_shape)
    run_all_tests(test_layer1, test_layer2)


def test_dense():
    input_shape = (512, 128)
    test_layer1 = TensorFlowLayer(units=64,
                                  activation="relu",
                                  layer_type=tf.keras.layers.Dense,
                                  mating_func=mating.exchange_units_mating,
                                  config=config,
                                  input_shape=input_shape)
    test_layer2 = TensorFlowLayer(units=64,
                                  activation="relu",
                                  layer_type=tf.keras.layers.Dense,
                                  mating_func=mating.exchange_units_mating,
                                  config=config,
                                  input_shape=input_shape)
    run_all_tests(test_layer1, test_layer2)


def test_flatten():
    flatten_layer = TensorFlowLayer(layer_type=tf.keras.layers.Flatten,
                                    config=config,
                                    mutable=False)

    for _ in range(100):
        dc = flatten_layer.deep_copy()
        rc = flatten_layer.random_copy()
        child = flatten_layer.mate(rc)

        test_input = tf.random.uniform(shape=[16, 32, 64, 100])
        orig_out = flatten_layer(test_input).numpy()
        deep_out = dc(test_input).numpy()
        rand_out = rc(test_input).numpy()
        child_out = child(test_input).numpy()

        assert ((deep_out == orig_out).all()
                and (deep_out == orig_out).all()
                and (rand_out == orig_out).all()
                and (child_out == orig_out).all())
        assert len(orig_out.shape) == 2 and orig_out.shape[1] == (32*64*100)


def test_sequential(verbose=False):
    layers = [
        # Conv2D
        TensorFlowLayer(filters=64,
                        kernel_size=(3, 3),
                        layer_type=tf.keras.layers.Conv2D,
                        mating_func=mating.exchange_units_mating,
                        config=config),
        # Flatten
        TensorFlowLayer(layer_type=tf.keras.layers.Flatten,
                        config=config,
                        mutable=False),
        # Dense hidden
        TensorFlowLayer(units=64,
                        activation="relu",
                        layer_type=tf.keras.layers.Dense,
                        mating_func=mating.exchange_units_mating,
                        config=config),
        # Dense output
        TensorFlowLayer(units=10,
                        activation="softmax",
                        layer_type=tf.keras.layers.Dense,
                        mating_func=mating.exchange_units_mating,
                        config=config),
    ]

    for n in range(10):
        out = tf.random.uniform(shape=[3, 256, 256, 3])
        for layer in layers:
            out = layer(out)

        if verbose:
            print(f"\n\n[OUT {n}]")
            print(out.numpy(), [-1])


if __name__ == "__main__":
    # Conv2D:
    print("\n[CONV2D]")
    test_conv2d()
    print("[CONV2D] Passed all assertions!")

    # Dense:
    print("\n[DENSE]")
    test_dense()
    print("[DENSE] Passed all assertions!")

    # Flatten:
    print("\n[FLATTEN]")
    test_flatten()
    print("[FLATTEN] Passed all assertions!")

    # Sequential layers processing:
    test_sequential(verbose=False)
