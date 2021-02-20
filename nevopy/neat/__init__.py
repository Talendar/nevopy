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

""" Imports core names of :mod:`nevopy.neat`.
"""

# Config
from nevopy.neat.config import NeatConfig

# Genes
from nevopy.neat.genes import align_connections
from nevopy.neat.genes import ConnectionGene
from nevopy.neat.genes import NodeGene

# Genomes
from nevopy.neat.genomes import FixTopNeatGenome
from nevopy.neat.genomes import NeatGenome

# ID handler
from nevopy.neat.id_handler import IdHandler

# Population
from nevopy.neat.population import NeatPopulation

# Species
from nevopy.neat.species import NeatSpecies

# Visualization
from nevopy.neat.visualization import NodeVisualizationInfo
from nevopy.neat.visualization import visualize_activations
from nevopy.neat.visualization import visualize_genome
