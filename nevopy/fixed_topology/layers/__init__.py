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

""" Neural network layers to be used with `NEvoPy's` fixed-topology
neuroevolutionary algorithms.
"""

# Mating util functions
from nevopy.fixed_topology.layers import mating

# Base abstract layer
from nevopy.fixed_topology.layers.base_layer import BaseLayer
from nevopy.fixed_topology.layers.base_layer import IncompatibleLayersError

# TensorFlow's layers
from nevopy.fixed_topology.layers.tf_layers import TensorFlowLayer
from nevopy.fixed_topology.layers.tf_layers import TFConv2DLayer
from nevopy.fixed_topology.layers.tf_layers import TFDenseLayer
from nevopy.fixed_topology.layers.tf_layers import TFFlattenLayer
from nevopy.fixed_topology.layers.tf_layers import TFMaxPool2DLayer
