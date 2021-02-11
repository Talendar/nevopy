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

""" Imports the core names of NEvoPy.
"""

# Util submodules
from nevopy import activations
from nevopy import callbacks

# "Fixed topology" subpackage
from nevopy import fixed_topology

# "Genetic algorithm" subpackage
from nevopy import genetic_algorithm

# "NEAT" subpackage
from nevopy import neat

# "Processing" subpackage
from nevopy import processing

# "Utils" subpackage
from nevopy import utils

# Base genome
from nevopy.base_genome import BaseGenome
from nevopy.base_genome import IncompatibleGenomesError
from nevopy.base_genome import InvalidInputError

# Base population
from nevopy.base_population import BasePopulation
