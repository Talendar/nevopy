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
                print(f"{w0.flat[i]} -> {w1.flat[i]}  ###  {d_real} | {d_calc}")

            if abs(d_real - d_calc) > 1e-5:
                reset_idx = test_info["w_reset_idx"]
                if reset_idx is not None and i in reset_idx:
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
                print(f"{b0.flat[i]} -> {b1.flat[i]}  ###  {d_real} | {d_calc}")

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


def test_shallow_copy(layer, data):
    new_layer = layer.shallow_copy()
    new_layer(data)

    w_new, b_new = (w.numpy() for w in new_layer.weights)
    w_p, b_p = (w.numpy() for w in layer.weights)

    assert w_new.shape == w_p.shape and b_new.shape == b_p.shape
    assert not (w_new == w_p).all()


def test_mating(layer1, layer2, data, verbose=False):
    new_layer = layer1.mate(layer2)
    new_layer(data)

    w_new, b_new = (w.numpy() for w in new_layer.weights)
    w_p1, b_p1 = (w.numpy() for w in layer1.weights)
    w_p2, b_p2 = (w.numpy() for w in layer2.weights)

    assert w_new.shape == w_p1.shape == w_p2.shape
    assert b_new.shape == b_p1.shape == b_p2.shape

    from_p1 = from_p2 = 0
    print(w_new[0])


if __name__ == "__main__":
    # layers
    layer1 = TFConv2DLayer(filters=64, kernel_size=(3, 3), config=config)
    layer2 = TFConv2DLayer(filters=64, kernel_size=(3, 3), config=config)

    # forcing the building of the weight matrices
    test_data = tf.random.uniform(shape=[1, 128, 128, 3])
    layer1(test_data)
    layer2(test_data)

    # tests
    # test_shallow_copy(layer1, test_data)
    test_mating(layer1, layer2, test_data)
    # test_mutate_weights(layer1, num_tests=100, verbose=False)
    # test_filter_stacking(layer1, verbose=True)
