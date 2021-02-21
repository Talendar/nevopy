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

""" Manual testing of the :meth:`.NeatGenome.visualize_activations()` method.
"""

from typing import Optional
from timeit import default_timer as timer

import nevopy as ne
import numpy as np
import pygame

_SCREEN_SIZE = 700, 450
_NUM_INPUTS = 4
_NUM_OUTPUTS = 2

_DISPLAY = pygame.display.set_mode(_SCREEN_SIZE)
_CLOCK = pygame.time.Clock()


def _test(genome: ne.neat.NeatGenome,
          test_title: str,
          num_frames: int = 10,
          fps: Optional[int] = None,
          **kwargs) -> None:
    total_time = 0.0
    for f in range(num_frames):
        pygame.display.set_caption(test_title + f" ({f}/{num_frames})")
        genome.mutate_weights()
        genome.process(np.random.uniform(low=-1, high=1, size=_NUM_INPUTS))

        start_time = timer()
        surface = genome.visualize_activations(surface_size=_SCREEN_SIZE,
                                               **kwargs)
        elapsed_time = timer() - start_time
        total_time += elapsed_time

        print(f"Surface building time: {1000 * elapsed_time}ms")
        print("Activations:\n"
              f"\t. Input: {[n.activation for n in genome.input_nodes]}\n"
              f"\t. Output: {[n.activation for n in genome.output_nodes]}\n"
              f"\t. Hidden: {[n.activation for n in genome.hidden_nodes]}")
        print("_" * 20 + "\n")

        _DISPLAY.blit(surface, [0, 0])
        pygame.display.update()

        if fps is not None:
            _CLOCK.tick(fps)
        else:
            proceed = False
            while not proceed:
                for event in pygame.event.get():
                    if (event.type == pygame.MOUSEBUTTONUP
                        or (event.type == pygame.KEYDOWN
                            and event.key == pygame.K_SPACE)):
                        proceed = True
    print("#" * 40)
    print(f"Avg surface building time: {1000 * total_time / num_frames}ms\n\n")


if __name__ == "__main__":
    # Preparing test genome:
    id_handler = ne.neat.id_handler.IdHandler(has_bias=True,
                                              num_inputs=_NUM_INPUTS,
                                              num_outputs=_NUM_OUTPUTS)
    test_genome = ne.neat.NeatGenome(num_inputs=_NUM_INPUTS,
                                     num_outputs=_NUM_OUTPUTS,
                                     config=ne.neat.NeatConfig())
    for _ in range(10):
        test_genome.add_random_hidden_node(id_handler)
        # genome.add_random_connection(id_handler)

    # Visualizing test genome:
    test_genome.visualize()

    # Test 1: default arguments.
    # _test(genome=test_genome, test_title="[Test 1] Default arguments")

    # Test 2: input labels.
    # _test(genome=test_genome,
    #       test_title="[Test 2] Input labels",
    #       show_input_values=False,
    #       input_visualization_info=["input_" + "x" * (i + 1)
    #                                 for i in range(_NUM_INPUTS)])

    # Test 3: input info.
    # _test(genome=test_genome,
    #       test_title="[Test 3] Input info",
    #       show_input_values=True,
    #       input_visualization_info=[
    #           ne.neat.NodeVisualizationInfo("Input 0", 0.5, "greater"),
    #           ne.neat.NodeVisualizationInfo("Input 1", 0, "less"),
    #           ne.neat.NodeVisualizationInfo("Input 2", -0.5, "equal"),
    #           ne.neat.NodeVisualizationInfo("Input 3", 0, "diff"),
    #       ])

    # Test 4: output labels.
    # _test(genome=test_genome,
    #       test_title="[Test 4] Output labels",
    #       show_output_values=False,
    #       output_visualization_info=["output_" + "x" * (i + 1)
    #                                  for i in range(_NUM_OUTPUTS)])

    # Test 5: output labels v2.
    # _test(genome=test_genome,
    #       test_title="[Test 5] Output labels v2",
    #       show_output_values=True,
    #       output_activate_greatest_only=False,
    #       output_visualization_info=["output_" + "x" * (i + 1)
    #                                  for i in range(_NUM_OUTPUTS)])

    # Test 6: output info.
    # _test(genome=test_genome,
    #       test_title="[Test 6] Output labels activate all",
    #       show_output_values=True,
    #       output_activate_greatest_only=False,
    #       output_visualization_info=[
    #           ne.neat.NodeVisualizationInfo("Output 0", 0.5, "greater"),
    #           ne.neat.NodeVisualizationInfo("Output 1", 0.75, "less"),
    #       ])

    # Test 7: input and output labels.
    # _test(genome=test_genome,
    #       test_title="[Test 7] Input and output labels",
    #       show_input_values=False,
    #       show_output_values=False,
    #       input_visualization_info=["input_" + "x" * (i + 1)
    #                                 for i in range(_NUM_INPUTS)],
    #       output_visualization_info=["output_" + "x" * (i + 1)
    #                                  for i in range(_NUM_OUTPUTS)])

    # Test 8: input and output info.
    # _test(genome=test_genome,
    #       test_title="[Test 8] Input and output info",
    #       show_input_values=True,
    #       show_output_values=True,
    #       output_activate_greatest_only=False,
    #       input_visualization_info=[
    #           ne.neat.NodeVisualizationInfo("Input 0", 0.5, "greater"),
    #           ne.neat.NodeVisualizationInfo("Input 1", 0, "less"),
    #           ne.neat.NodeVisualizationInfo("Input 2", -0.5, "equal"),
    #           ne.neat.NodeVisualizationInfo("Input 3", 0, "diff"),
    #       ],
    #       output_visualization_info=[
    #           ne.neat.NodeVisualizationInfo("Output 0", 0.5, "greater"),
    #           ne.neat.NodeVisualizationInfo("Output 1", 0.75, "less"),
    #       ])

    # Test 9: input and output info continuous.
    _test(genome=test_genome,
          fps=10,
          num_frames=100,
          test_title="[Test 9] Input and output info continuous.",
          show_input_values=True,
          show_output_values=True,
          output_activate_greatest_only=True,
          input_visualization_info=[
              ne.neat.NodeVisualizationInfo("Input 0", 0.5, "greater"),
              ne.neat.NodeVisualizationInfo("Input 1", 0, "less"),
              ne.neat.NodeVisualizationInfo("Input 2", -0.5, "equal"),
              ne.neat.NodeVisualizationInfo("Input 3", 0, "diff"),
          ],
          output_visualization_info=["Output 0", "Output 1"])
