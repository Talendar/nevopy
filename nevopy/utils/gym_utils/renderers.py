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

""" This module implements entities responsible for rendering a
:class:`gym.Env` during the evaluation of a genome's fitness by a
:class:`.GymFitnessFunction`.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import nevopy as ne

if TYPE_CHECKING:
    import gym


class GymRenderer:
    """ Defines the entity responsible for rendering a :class:`gym.Env` during
    the evaluation of a genome's fitness by a :class:`.GymFitnessFunction`.

    Args:
        fps (int): Frames per second.

    Attributes:
        fps (int): Frames per second.
    """

    def __init__(self, fps: int = 45):
        self.fps = fps

    def render(self,
               env: "gym.Env",
               # pylint: disable=unused-argument
               genome: ne.base_genome.BaseGenome,
    ) -> None:
        """ Renders the environment in "human mode".

        Args:
            env (gym.Env): Environment to be rendered.
            genome (BaseGenome): Genome currently being evaluated.
        """
        env.render(mode="human")
        time.sleep(1 / self.fps)

    def flush(self) -> None:
        """ Flushes the internal buffers of the renderer. """


class NeatActivationsGymRenderer(GymRenderer):
    """
    TODO
    """

    def __init__(self,
                 out_path: str = "./gym_videos",
                 fps: int = 45,
                 **kwargs):
        super().__init__(fps=fps)

        self._out_path = out_path
        self.kwargs = kwargs

        self._env_imgs = []
        self._activations_imgs = []

    def render(self,
               env: "gym.Env",
               genome: ne.base_genome.BaseGenome,) -> None:
        """
        TODO
        """
        # Random agents:
        if genome is None:
            return super().render(env=env, genome=genome)

        # Checking if genome is a NeatGenome:
        if not isinstance(genome, ne.neat.NeatGenome):
            raise ValueError("Currently, this renderer is only compatible with "
                             "genomes of the type `NeatGenome`.")

        # Getting images
        env_img = env.render("rgb_array")
        activations_img = genome.visualize_activations(
            return_rgb_array=True,
            **self.kwargs,
        )
        activations_img = np.rot90(activations_img)

        # Saving images:
        self._env_imgs.append(env_img.astype(np.uint8))
        self._activations_imgs.append(activations_img.astype(np.uint8))

    def flush(self) -> None:
        """
        TODO

        Returns:

        """
        if len(self._activations_imgs) > 0:
            # Importing scikit-video:
            try:
                import skvideo.io  # pylint: disable=import-outside-toplevel
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "Couldn't find 'scikit-video'! To use this renderer, make "
                    "sure you have it installed.\nYou can install "
                    "'scikit-video' using pip:\n\t$ pip install sk-video"
                ) from e

            print("Generating video(s)... this might take a while.")
            Path(self._out_path).mkdir(parents=True, exist_ok=True)

            # Current date and time for file names:
            time_now = datetime.today().strftime("%Y%m%d%H%M%S")

            # Activations video:
            act_fn = os.path.join(self._out_path,
                                  f"gym_video_{time_now}_activations.avi")
            skvideo.io.vwrite(fname=act_fn,
                              videodata=self._activations_imgs,
                              inputdict={"-r": str(self.fps)},
                              outputdict={"-r": str(self.fps)})
            print(f"Gym activations video saved to: {act_fn}")

            # Env video:
            env_fn = os.path.join(self._out_path,
                                  f"gym_video_{time_now}_env.avi")
            skvideo.io.vwrite(fname=env_fn,
                              videodata=self._env_imgs,
                              inputdict={"-r": str(self.fps)},
                              outputdict={"-r": str(self.fps)})
            print(f"Gym environment video saved to: {env_fn}")

            # Discarding cached images:
            self._env_imgs = []
            self._activations_imgs = []
