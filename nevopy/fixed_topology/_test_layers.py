""" Tests the implementation of the method
:meth:`nevopy.fixed_topology.layers.BaseLayer.mutate_weights()`.
"""

from nevopy.fixed_topology.config import FixedTopologyConfig
from nevopy.fixed_topology.layers import TFConv2DLayer

from timeit import default_timer as timer
import numpy as np

import tensorflow as tf
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=tf_config)

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
        w0, b0 = layer.weights[0].numpy(), layer.weights[1].numpy()

        start_time = timer()
        layer.mutate_weights(_test_info=test_info)
        deltaT += timer() - start_time

        w1, b1 = layer.weights[0].numpy(), layer.weights[1].numpy()

        # weights
        diff_pc = (w1 - w0) / w0
        count_reset = 0
        for i in range(w0.size):
            d_real = diff_pc.flat[i]
            d_calc = (test_info["w_perturbation"].numpy() - 1).flat[i]

            if verbose:
                print(f"[ABS CHANGE] {w0.flat[i]:.5f} -> {w1.flat[i]:.5f}"
                      f"  <>  "
                      f"[% CHANGE] real = {d_real:.2%} | calc = {d_calc:.2%}"
                      f"  <>  "
                      f"Reset: {i in test_info['w_reset_idx']}")

            if abs(d_real - d_calc) > 1e-5:
                if i in test_info["w_reset_idx"]:
                    count_reset += 1
                    continue
                raise RuntimeError("Incorrect weight value!")
        assert count_reset <= len(test_info["w_reset_idx"])
        if verbose:
            print(f"Reset weights idx: {test_info['w_reset_idx']}\n\n")

        # bias
        diff_pc = np.divide(b1 - b0, b0, out=np.zeros_like(b0), where=b0 != 0)
        count_reset = 0
        for i in range(b0.size):
            d_real = diff_pc.flat[i]
            if d_real == 0:
                continue
            d_calc = (test_info["b_perturbation"].numpy() - 1).flat[i]
            if verbose:
                print(f"[ABS CHANGE] {b0.flat[i]:.5f} -> {b1.flat[i]:.5f}"
                      f"  <>  "
                      f"[% CHANGE] real = {d_real:.2%} | calc = {d_calc:.2%}"
                      f"  <>  "
                      f"Reset: {i in test_info['b_reset_idx']}")

            if abs(d_real - d_calc) > 1e-5:
                reset_idx = test_info["b_reset_idx"]
                if reset_idx is not None and i in reset_idx:
                    count_reset += 1
                    continue
                raise RuntimeError("Incorrect bias value!")
        assert count_reset <= len(test_info["b_reset_idx"])
        if verbose:
            print(f"Reset bias idx: {test_info['b_reset_idx']}\n")

    if verbose:
        print("\n" + "#" * 30 + "\n")
        print(f"Mutation avg time: {1000 * deltaT / num_tests}ms")


def test_filter_stacking(conv2d, verbose=False):
    array = conv2d.weights[0].numpy()

    filters = [array[:, :, :, i] for i in range(array.shape[-1])]
    stacked_filters = np.stack(filters, axis=-1)

    if verbose:
        print(f"\n>> Filter stacking <<")
        print(f". Original shape: {array.shape}")
        print(f". Filter shape: {filters[0].shape}")
        print(f". Stacked shape: {stacked_filters.shape}\n\n")

    assert array.shape == stacked_filters.shape
    assert (array == stacked_filters).all()

    for _ in range(100):
        i = np.random.randint(low=0, high=array.size)
        if verbose:
            print(array.flat[i], stacked_filters.flat[i])
        assert array.flat[i] == stacked_filters.flat[i]


def test_shallow_copy(layer, verbose=False):
    new_layer = layer.shallow_copy()

    w_new, b_new = (w.numpy() for w in new_layer.weights)
    w_p, b_p = (w.numpy() for w in layer.weights)

    if verbose:
        for _ in range(100):
            i = np.random.randint(low=0, high=w_p.size)
            print(f"[W] old = {w_p.flat[i]}, new = {w_new.flat[i]}")
            i = np.random.randint(low=0, high=len(b_p))
            print(f"[B] old = {b_p[i]}, new = {b_new[i]}")

    assert w_new.shape == w_p.shape and b_new.shape == b_p.shape
    assert not (w_new == w_p).all()


def test_mating_conv2d(conv1, conv2, num_tests=100, verbose=False):
    mate_deltaT = mutation_deltaT = 0
    total_filters_from_p1 = total_filters_from_p2 = 0
    total_bias_from_p1 = total_bias_from_p2 = 0
    for t in range(num_tests):
        if verbose:
            print("\n" + "#"*50)
            print(f"----- Test {t} of {num_tests} -----\n")

        start_time = timer()
        conv1.mutate_weights()
        conv2.mutate_weights()
        mutation_deltaT += (timer() - start_time)/2

        start_time = timer()
        new_layer = conv1.mate(conv2)
        mate_deltaT += timer() - start_time

        w_new, b_new = (w.numpy() for w in new_layer.weights)
        w_p1, b_p1 = (w.numpy() for w in conv1.weights)
        w_p2, b_p2 = (w.numpy() for w in conv2.weights)

        assert w_new.shape == w_p1.shape == w_p2.shape
        assert b_new.shape == b_p1.shape == b_p2.shape

        # checking filters ancestry
        from_p1 = from_p2 = 0
        for i in range(w_p1.shape[-1]):
            f_new = w_new[:, :, :, i]
            f_p1 = w_p1[:, :, :, i]
            f_p2 = w_p2[:, :, :, i]

            if (f_new == f_p1).all():
                from_p1 += 1
                if verbose:
                    print(f"[{i}] Filter from parent 1 <> "
                          f"diff_sum = {(f_new - f_p1).sum()}")
            elif (f_new == f_p2).all():
                from_p2 += 1
                if verbose:
                    print(f"[{i}] Filter from parent 2 <> "
                          f"diff_sum = {(f_new - f_p2).sum()}")
            else:
                raise RuntimeError(f"New filter at index {i} doesn't match any "
                                   f"of the parents filters at the same index!"
                                   f"\n> New filter = \n{f_new}\n\n"
                                   f"> Parent 1 filter = \n{f_p1}\n\n"
                                   f"> Parent 2 filter = \n{f_p2}")
        if verbose:
            print("\n" + "="*20)
            print(
                f"[Filters] "
                f"from parent 1 = {from_p1} ({from_p1 / (from_p1 + from_p2):.2%})"
                f"  |  "
                f"from parent 2 = {from_p2} ({from_p2 / (from_p1 + from_p2):.2%})"
            )
            print("\n" + "#"*40 + "\n\n")

        total_filters_from_p1 += from_p1
        total_filters_from_p2 += from_p2

        # checking bias ancestry
        from_p1 = from_p2 = 0
        for i in range(len(b_p1)):
            b0, b1, b2 = b_new[i], b_p1[i], b_p2[i]

            if b0 == b1:
                from_p1 += 1
                if verbose:
                    print(f"[{i}] Bias from parent 1 <> diff = {b0 - b1}")
            elif b0 == b2:
                from_p2 += 1
                if verbose:
                    print(f"[{i}] Bias from parent 2 <> diff = {b0 - b2}")
            else:
                raise RuntimeError(f"New bias at index {i} doesn't match any of"
                                   f" the parents biases at the same index!\n"
                                   f"> New bias = {b0}\n"
                                   f"> Parent 1 bias = {b1}\n"
                                   f"> Parent 2 bias = {b2}")
        if verbose:
            print("\n" + "=" * 20)
            print(
                f"[Bias] "
                f"from parent 1 = {from_p1} ({from_p1 / (from_p1 + from_p2):.2%})"
                f"  |  "
                f"from parent 2 = {from_p2} ({from_p2 / (from_p1 + from_p2):.2%})"
            )

        total_bias_from_p1 += from_p1
        total_bias_from_p2 += from_p2

    if verbose:
        print("\n" + "_"*50 + "\n")
        print(f"Avg mating time: {1000 * mate_deltaT / num_tests}ms")
        print(f"Avg mutation time: {1000 * mutation_deltaT / num_tests}ms\n")

        fT = total_filters_from_p1 + total_filters_from_p2
        print(
            f"[Filters] "
            f"from parent 1: {total_filters_from_p1 / fT:.3%}"
            f"  |  "
            f"from parent 2: {total_filters_from_p2 / fT:.3%}"
        )

        bT = total_bias_from_p1 + total_bias_from_p2
        print(
            f"[Bias] "
            f"from parent 1: {total_bias_from_p1 / bT:.3%}"
            f"  |  "
            f"from parent 2: {total_bias_from_p2 / bT:.3%}"
        )


if __name__ == "__main__":
    input_shape = (1, 128, 128, 3)

    # layers
    layer1 = TFConv2DLayer(filters=64, kernel_size=(3, 3),
                           config=config, input_shape=input_shape)
    layer2 = TFConv2DLayer(filters=64, kernel_size=(3, 3),
                           config=config, input_shape=input_shape)

    # tests
    # test_filter_stacking(layer1, verbose=False)
    test_mutate_weights(layer1, num_tests=100, verbose=False)
    test_shallow_copy(layer1, verbose=False)
    test_mating_conv2d(layer1, layer2, num_tests=100, verbose=False)
