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

# pylint: disable=wrong-import-position
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# pylint: enable=wrong-import-position
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

from nevopy.fixed_topology.config import FixedTopologyConfig
from nevopy.fixed_topology.layers import mating
from nevopy.fixed_topology.layers.base_layer import IncompatibleLayersError
from nevopy.fixed_topology.layers.tf_layers import TensorFlowLayer
from nevopy.fixed_topology.layers.tf_layers import TFConv2DLayer
import test_utils


config = FixedTopologyConfig(  # weight mutation
                             weight_mutation_chance=(0.7, 0.9),
                             weight_perturbation_pc=(0.1, 0.4),
                             weight_reset_chance=(0.1, 0.3),
                             new_weight_interval=(-2, 2))


def test_mutate_weights(layer, num_tests=100, verbose=False):
    mutation_time = 0
    mutated_weights_pc = 0
    reset_weights_pc = 0

    for _ in range(num_tests):
        test_info = {}
        weights_old = layer.weights

        start_time = timer()
        layer.mutate_weights(_test_info=test_info)
        mutation_time += timer() - start_time

        weights_new = layer.weights

        for w_num, (w_old, w_new) in enumerate(zip(weights_old, weights_new)):
            if verbose:
                print(f"\n[WEIGHT MATRIX AT INDEX {w_num}]")

            mutate_idx = test_info[f"w{w_num}_mutate_idx"]
            perturbation = test_info[f"w{w_num}_perturbation"] - 1
            assert len(mutate_idx) == len(perturbation)

            if len(mutate_idx) > 0:
                mutate_idx, perturbation = (
                    np.array(list(t))
                    for t in zip(
                        *sorted(zip(test_info[f"w{w_num}_mutate_idx"],
                                    test_info[f"w{w_num}_perturbation"] - 1))
                    )
                )

            reset_idx = test_info[f"w{w_num}_reset_idx"]
            diff_pc = np.divide(w_new - w_old, w_old,
                                out=np.full_like(w_old, fill_value=None),
                                where=w_old != 0)

            mutated_pc = len(mutate_idx) / w_old.size
            reset_pc = len(reset_idx) / w_old.size

            mutated_weights_pc += mutated_pc / len(weights_old)
            reset_weights_pc += reset_pc / len(weights_old)

            count_reset = 0
            count_pert = 0
            for i in range(w_old.size):
                d_real = diff_pc.flat[i] \
                    # pylint: disable=unsubscriptable-object
                if verbose:
                    print(f"[ABS CHANGE] {w_old.flat[i]:.5f} "
                          f"-> {w_new.flat[i]:.5f}  <>  "
                          f"Reset: {i in reset_idx} | "
                          f"Mutate: {i in mutate_idx}"
                          f"  <>  real = {d_real:.2%}", end="")

                if i not in mutate_idx:
                    if verbose:
                        print()

                    if i not in reset_idx:
                        assert w_new.flat[i] == w_old.flat[i]
                    else:
                        assert w_new.flat[i] != w_old.flat[i]

                    if not np.isnan(d_real):
                        assert d_real == 0 or i in reset_idx

                    continue

                d_calc = perturbation.flat[count_pert]
                count_pert += 1

                if verbose:
                    print(f" | calc = {d_calc:.2%}")

                if np.isnan(d_real):
                    continue

                if abs(d_real - d_calc) > 1e-5:
                    if i in reset_idx:
                        count_reset += 1
                        continue
                    print(perturbation)
                    raise RuntimeError("Incorrect weight value!")

            assert count_reset <= len(reset_idx)
            assert count_pert == len(mutate_idx)

            if verbose:
                print(f"\nMutated weights idx: {mutate_idx}")
                print(f"Reset weights idx: {reset_idx}")
                print(f"Mutation pc: {mutated_pc:.2%}")
                print(f"Reset pc: {reset_pc:.2%}\n\n")

    if verbose:
        print("\n" + "="*50)
    print(f"> Mutation time: {1000 * mutation_time / num_tests:.4f}ms")
    print(f"> Mutated weights pc: {mutated_weights_pc / num_tests:.2%}")
    print(f"> Reset weights pc: {reset_weights_pc / num_tests:.2%}")


def test_immutable_layer_mutation(layer, num_tests=100, verbose=False):
    mutable = layer.mutable
    layer.mutable = False

    mutation_time = 0
    for _ in range(num_tests):
        old_weights = layer.weights

        start_time = timer()
        layer.mutate_weights()
        mutation_time += timer() - start_time

        new_weights = layer.weights
        for w_new, w_old in zip(new_weights, old_weights):
            assert (w_new == w_old).all()

    layer.mutable = mutable
    if verbose:
        print("\n" + "=" * 50)
    print("> Immutable layer mutation time: "
          f"{1000 * mutation_time / num_tests:.4f}ms")


def test_deep_copy(layer, num_tests=100, verbose=False):
    dc_time = 0
    for _ in range(num_tests):
        layer.mutate_weights()

        start_time = timer()
        new_layer = layer.deep_copy()
        dc_time += timer() - start_time

        # Checking if weights are equal
        for w_new, w_old in zip(new_layer.weights, layer.weights):
            if verbose:
                for _ in range(50):
                    i = np.random.randint(low=0, high=w_old.size)
                    print(f"[W] old = {w_old.flat[i]}, new = {w_new.flat[i]}")

            assert w_new.shape == w_old.shape
            assert (w_new == w_old).all()

        # Checking if outputs are equal
        for _ in range(3):
            random_input = tf.random.uniform(shape=layer.input_shape,
                                             dtype=float)
            y_new = new_layer(random_input).numpy()
            y_old = layer(random_input).numpy()

            if verbose:
                print(f"[OUTPUT] diff = {np.abs(y_old - y_new).sum()}")

            assert (y_new == y_old).all()

    if verbose:
        print("\n" + "="*50)
    print(f"> Deep copy time: {1000 * dc_time / num_tests:.4f}ms")


def test_random_copy(layer, num_tests=100, verbose=False):
    rc_time = 0
    for _ in range(num_tests):
        layer.mutate_weights()

        start_time = timer()
        new_layer = layer.random_copy()
        rc_time += timer() - start_time

        # Checking if weights are different
        for w_old, w_new in zip(layer.weights, new_layer.weights):
            if verbose:
                for _ in range(50):
                    i = np.random.randint(low=0, high=w_old.size)
                    print(f"[W] old = {w_old.flat[i]}, new = {w_new.flat[i]}")

            assert w_old.shape == w_new.shape
            assert not (w_old == w_new).all()

        # Checking if outputs are different
        for _ in range(3):
            random_input = tf.random.uniform(shape=layer.input_shape,
                                             dtype=float)
            y_new = new_layer(random_input).numpy()
            y_old = layer(random_input).numpy()

            if verbose:
                print(f"[OUTPUT] diff = {np.abs(y_old - y_new).sum()}")

            assert not (y_new == y_old).all()

    if verbose:
        print("\n" + "=" * 50)
    print(f"> Random copy time: {1000 * rc_time / num_tests:.4f}ms")


def test_exchange_units_mating(layer1, layer2, num_tests=100, verbose=False):
    layer1.mating_func = layer2.mating_func = mating.exchange_units_mating
    mating_time = mutation_time = 0
    units_from_p1 = units_from_p2 = 0
    for t in range(num_tests):
        if verbose:
            print("\n" + "#"*50)
            print(f"----- Test {t} of {num_tests} -----\n")

        # Mutating layers
        start_time = timer()
        layer1.mutate_weights()
        layer2.mutate_weights()
        mutation_time += (timer() - start_time) / 2

        # Mating layers
        start_time = timer()
        new_layer = layer1.mate(layer2)
        mating_time += timer() - start_time

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
          f"\t. Mating time: {1000 * mating_time / num_tests:.4f}ms\n"
          f"\t. Mutation time: {1000 * mutation_time / num_tests:.4f}ms")


def test_exchange_weights_mating(layer1, layer2, num_tests=100, verbose=False):
    layer1.mating_func = layer2.mating_func = mating.exchange_weights_mating
    mating_time = mutation_time = 0
    w_from_p1 = w_from_p2 = 0
    for t in range(num_tests):
        if verbose:
            print("\n" + "#"*50)
            print(f"----- Test {t} of {num_tests} -----\n")

        # Mutating layers:
        start_time = timer()
        layer1.mutate_weights()
        layer2.mutate_weights()
        mutation_time += (timer() - start_time) / 2

        # Mating layers:
        start_time = timer()
        new_layer = layer1.mate(layer2)
        mating_time += timer() - start_time

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
          f"\t. Mating time: {1000 * mating_time / num_tests:.4f}ms\n"
          f"\t. Mutation time: {1000 * mutation_time / num_tests:.4f}ms")


def test_weights_avg_mating(layer1, layer2, num_tests=100, verbose=False):
    layer1.mating_func = layer2.mating_func = mating.weights_avg_mating
    mating_time = mutation_time = 0
    for t in range(num_tests):
        if verbose:
            print(f"> Test {t} of {num_tests}.")

        # Mutating layers:
        start_time = timer()
        layer1.mutate_weights()
        layer2.mutate_weights()
        mutation_time += (timer() - start_time) / 2

        # Mating layers:
        start_time = timer()
        new_layer = layer1.mate(layer2)
        mating_time += timer() - start_time

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
          f"\t. Mating time: {1000 * mating_time / num_tests:.4f}ms\n"
          f"\t. Mutation time: {1000 * mutation_time / num_tests:.4f}ms")


def test_immutable_layer_mating(layer1, layer2, num_tests=100, verbose=False):
    mutable1 = layer1.mutable
    mutable2 = layer2.mutable

    mating_time = 0
    for _ in range(num_tests):
        # Valid mating
        layer1.mutable = layer2.mutable = False
        start_time = timer()
        new_layer1 = layer1.mate(layer2)
        new_layer2 = layer2.mate(layer1)
        mating_time += (timer() - start_time) / 2

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

    layer1.mutable = mutable1
    layer2.mutable = mutable2
    if verbose:
        print("\n" + "=" * 50)

    print("> Immutable layer valid mating time: "
          f"{1000 * mating_time / num_tests:.4f}ms")


def test_save_and_load(layer, num_tests=10, pkl=True):
    saving_time = loading_time = file_size = 0
    for _ in range(num_tests):
        file_name = (f"{test_utils.TEMP_SAVE_DIR}/"
                     f"saved_layer_{int(1e6 * timer())}")

        start_time = timer()
        layer.save(file_name)
        saving_time += timer() - start_time

        file_size += os.path.getsize(file_name + ".pkl" if pkl else "")

        start_time = timer()
        loaded_layer = TensorFlowLayer.load(file_name)
        loading_time += timer() - start_time

        for w0, w1 in zip(layer.weights, loaded_layer.weights):
            assert w0.shape == w1.shape
            assert (w0 == w1).all()

        for _ in range(3):
            test_input = tf.random.uniform(shape=layer.input_shape, dtype=float)
            h0, h1 = layer(test_input), loaded_layer(test_input)
            assert (h0.numpy() == h1.numpy()).all()

    print(f"> Saving time: {1000 * saving_time / num_tests:.4f}ms")
    print(f"> Loading time: {1000 * loading_time / num_tests:.4f}ms")
    print(f"> File size: {file_size / (num_tests * 1024):.2f}KB")

    w_kb = np.sum([w.nbytes for w in layer.weights]) / 1024
    print(f"> Weights size: {w_kb:.2f}KB")


def run_all_tests(test_layer1, test_layer2):
    if test_layer1.mutable:
        test_mutate_weights(test_layer1, num_tests=10, verbose=False)
        test_exchange_units_mating(test_layer1, test_layer2,
                                   num_tests=10, verbose=False)
        test_exchange_weights_mating(test_layer1, test_layer2,
                                     num_tests=10, verbose=False)
        test_weights_avg_mating(test_layer1, test_layer2,
                                num_tests=10, verbose=False)

    test_deep_copy(test_layer1, num_tests=10, verbose=False)
    test_random_copy(test_layer1, num_tests=10, verbose=False)
    test_immutable_layer_mating(test_layer1, test_layer2,
                                num_tests=10, verbose=False)
    test_immutable_layer_mutation(test_layer1, num_tests=10, verbose=False)
    test_save_and_load(test_layer1, num_tests=10)


def test_conv2d():
    input_shape = (4, 256, 256, 3)
    test_layer1 = TensorFlowLayer(filters=128,
                                  kernel_size=(3, 3),
                                  layer_type=tf.keras.layers.Conv2D,
                                  mating_func=mating.exchange_units_mating,
                                  config=config,
                                  input_shape=input_shape)
    test_layer2 = TFConv2DLayer(filters=128,
                                kernel_size=(3, 3),
                                config=config,
                                input_shape=input_shape)
    run_all_tests(test_layer1, test_layer2)


def test_dense():
    input_shape = (32, 1024)
    test_layer1 = TensorFlowLayer(units=128,
                                  activation="relu",
                                  layer_type=tf.keras.layers.Dense,
                                  mating_func=mating.exchange_units_mating,
                                  config=config,
                                  input_shape=input_shape)
    test_layer2 = TensorFlowLayer(units=128,
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

        test_input = tf.random.uniform(shape=[16, 32, 64, 100], dtype=float)
        orig_out = flatten_layer(test_input).numpy()
        deep_out = dc(test_input).numpy()
        rand_out = rc(test_input).numpy()
        child_out = child(test_input).numpy()

        assert ((deep_out == orig_out).all()
                and (deep_out == orig_out).all()
                and (rand_out == orig_out).all()
                and (child_out == orig_out).all())
        assert len(orig_out.shape) == 2 and orig_out.shape[1] == (32*64*100)

    test_save_and_load(flatten_layer)


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
        out = tf.random.uniform(shape=[3, 256, 256, 3], dtype=float)
        for layer in layers:
            out = layer(out)

        if verbose:
            print(f"\n\n[OUT {n}]")
            print(out.numpy(), [-1])


if __name__ == "__main__":
    test_utils.clear_temp()

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

    test_utils.clear_temp()
