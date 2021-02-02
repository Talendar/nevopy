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

""" Tests the implementation of :class:`FixTopNeatGenome`.
"""

# pylint: disable=wrong-import-position
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# pylint: enable=wrong-import-position

import tensorflow as tf

from nevopy.neat.genomes import FixTopNeatGenome
from nevopy.fixed_topology import FixedTopologyConfig
from nevopy.fixed_topology.genomes import FixedTopologyGenome
from nevopy.fixed_topology.layers import TFConv2DLayer
from nevopy.neat.config import NeatConfig


if __name__ == "__main__":
    # genomes
    genome = FixTopNeatGenome(
        fito_genome=FixedTopologyGenome(
            layers=[
                TFConv2DLayer(filters=32, kernel_size=(16, 16), strides=(4, 4)),
                TFConv2DLayer(filters=32, kernel_size=(16, 16), strides=(4, 4)),
                TFConv2DLayer(filters=8, kernel_size=(4, 4), strides=(1, 1)),
            ],
            config=FixedTopologyConfig(),
        ),
        num_neat_inputs=8,
        num_neat_outputs=5,
        config=NeatConfig(),
    )

    # tests
    print(genome(tf.random.uniform(shape=(1, 128, 128, 3), dtype=float)))
