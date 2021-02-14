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

"""
TODO
"""

from timeit import default_timer as timer

import nevopy as ne
import numpy as np
import pygame


if __name__ == "__main__":
    id_handler = ne.neat.id_handler.IdHandler(has_bias=True,
                                              num_inputs=8,
                                              num_outputs=4)
    genome = ne.neat.NeatGenome(num_inputs=8,
                                num_outputs=4,
                                config=ne.neat.NeatConfig())
    genome.visualize()

    for _ in range(5):
        genome.add_random_hidden_node(id_handler)
        genome.add_random_connection(id_handler)

    genome.visualize()

    screen_size = 500, 360
    display = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Activations Visualization")
    clock = pygame.time.Clock()

    for _ in range(1000):
        start = timer()
        surface = genome.visualize_activations(screen_size)

        print(f"Time: {1000 * (timer() - start)}ms")
        # pylint: disable=protected-access
        print("Activations:\n"
              f"\t. Input: {[n.activation for n in genome._input_nodes]}\n"
              f"\t. Output: {[n.activation for n in genome._output_nodes]}\n"
              f"\t. Hidden: {[n.activation for n in genome.hidden_nodes]}")
        print("\n" + "#" * 40 + "\n")

        display.blit(surface, [0, 0])
        pygame.display.update()
        clock.tick(60)

        genome.process(np.random.uniform(size=8))
