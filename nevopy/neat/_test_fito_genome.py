from genomes import FixTopNeatGenome
from nevopy.neat.config import NeatConfig
from nevopy.fixed_topology.genomes import FixedTopologyGenome
from nevopy.fixed_topology.layers import TFConv2DLayer
from nevopy.fixed_topology import FixedTopologyConfig
import tensorflow as tf


if __name__ == "__main__":
    # fixing TF
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_session = tf.compat.v1.InteractiveSession(config=tf_config)

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
    print(genome(tf.random.uniform(shape=(1, 128, 128, 3))))
