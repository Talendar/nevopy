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


def test_mutate_weights(layer, input_data, num_tests=100, verbose=False):
    output = layer(input_data).numpy()
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

    return 1000 * deltaT / num_tests


if __name__ == "__main__":
    dT = test_mutate_weights(
        layer=TFConv2DLayer(filters=512, kernel_size=(3, 3), config=config),
        input_data=tf.random.uniform(shape=[1, 128, 128, 10]),
        num_tests=100,
        verbose=False,
    )
    print("\n" + "#"*30 + "\n")
    print(f"Mutation avg time: {dT}ms")
