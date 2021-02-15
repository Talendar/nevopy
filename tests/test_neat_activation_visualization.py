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

""" Quick test of the :meth:`.NeatGenome.visualize_activations()` method.
"""

from timeit import default_timer as timer

import nevopy as ne
import numpy as np
import pygame

_SCREEN_SIZE = 700, 450
_NUM_INPUTS = 16
_NUM_OUTPUTS = 2


if __name__ == "__main__":
    id_handler = ne.neat.id_handler.IdHandler(has_bias=True,
                                              num_inputs=_NUM_INPUTS,
                                              num_outputs=_NUM_OUTPUTS)
    genome = ne.neat.NeatGenome(num_inputs=_NUM_INPUTS,
                                num_outputs=_NUM_OUTPUTS,
                                config=ne.neat.NeatConfig())
    for _ in range(10):
        genome.add_random_hidden_node(id_handler)
        # genome.add_random_connection(id_handler)

    genome.visualize()

    screen_size = 700, 450
    display = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Activations Visualization")
    clock = pygame.time.Clock()

    for _ in range(500):
        genome.mutate_weights()
        genome.process(np.random.uniform(size=_NUM_INPUTS))

        start = timer()
        surface = genome.visualize_activations(
            surface_size=screen_size,
            input_labels=[f"l{d}_" + "i"*d for d in range(_NUM_INPUTS)],
            output_labels=[f"out{d}" for d in range(_NUM_OUTPUTS)],
        )

        print(f"Time: {1000 * (timer() - start)}ms")

        # pylint: disable=protected-access
        # noinspection PyProtectedMember
        print("Activations:\n"
              f"\t. Input: {[n.activation for n in genome._input_nodes]}\n"
              f"\t. Output: {[n.activation for n in genome._output_nodes]}\n"
              f"\t. Hidden: {[n.activation for n in genome.hidden_nodes]}")
        print("\n" + "#" * 40 + "\n")

        display.blit(surface, [0, 0])
        pygame.display.update()
        clock.tick(60)
