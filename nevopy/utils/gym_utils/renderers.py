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

""" This module implements the entities responsible for rendering a
:class:`gym.Env` during the evaluation of a genome's fitness by a
:class:`.GymFitnessFunction`.
"""

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List

import gym  # pylint: disable=unused-import
import numpy as np

import nevopy as ne


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
               genome: "ne.base_genome.BaseGenome",
    ) -> None:
        """ Renders the environment in "human mode".

        Args:
            env (gym.Env): Environment to be rendered.
            genome (BaseGenome): Genome currently being evaluated.
        """
        env.render(mode="human")
        time.sleep(1 / self.fps)

    def flush(self) -> None:
        """ Flushes the internal buffers of the renderer.

        Doesn't do anything by default. Subclasses should override this method
        in order for any action to occur.
        """


class NeatActivationsGymRenderer(GymRenderer):
    """ Gym env renderer that renders a NEAT genome's neural network while the
    genome is interacting with the environment.

    Three videos will be generated: one containing the recording of the genome's
    interactions with the environment, another containing the genome's neural
    network states during the interactions and another containing the
    concatenation of the two previously videos, side by side.

    Note:
        Compatible with :class:`.NeatGenome` only!

    Note:
        This renderer requires you to have the :mod:`opencv-python` and
        :mod:`scikit-video` packages installed. You can install them using
        `pip`. You'll also need `FFmpeg`.

    Args:
        out_path (str): Path to the output directory.
        fps (int): Frames per second of the generated videos.
        play_video (bool): If ``True``, the concatenated videos will be
            automatically played after the rendering is done.
        **kwargs (Dict[str, Any]): Named arguments to be passed to
            :func:`genome.visualize_activations`.
    """

    def __init__(self,
                 out_path: str = "./gym_videos",
                 fps: int = 30,
                 play_video: bool = True,
                 **kwargs):
        super().__init__(fps=fps)

        self._out_path = out_path
        self._play_video = play_video
        self.kwargs = kwargs

        self._env_imgs = []          # type: List[np.ndarray]
        self._activations_imgs = []  # type: List[np.ndarray]

    def render(self,
               env: "gym.Env",
               genome: "ne.base_genome.BaseGenome") -> None:
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
        activations_img = np.flipud(np.rot90(activations_img))

        # Saving images:
        self._env_imgs.append(env_img.astype(np.uint8))
        self._activations_imgs.append(activations_img.astype(np.uint8))

    @staticmethod
    def play_video(video_file: str, fps: int) -> None:
        """ Plays a video from a file. ????? """
        # Importing OpenCV:
        try:
            import cv2  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Couldn't find 'opencv-python'! To use this renderer, make "
                "sure you have it installed.\nYou can install "
                "'opencv-python' using pip:\n\t$ pip install opencv-python"
            ) from e

        # Loading video:
        video_cap = cv2.VideoCapture(video_file)
        if not video_cap.isOpened():
            raise RuntimeError("Error opening the video!")

        # Playing the video:
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if ret:
                cv2.imshow("Gym video", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(1 / fps)
            else:
                break

        # Closing:
        video_cap.release()
        cv2.destroyAllWindows()

    def flush(self) -> None:
        """ Generates the videos from the images in the cache, closes the
        necessary resources and clears the image cache.
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

            # FFmpeg dicts
            ffmpeg_out_dict = {"-r": str(self.fps),
                               "-vcodec": "mpeg4",
                               "-b": "800k"}

            # Activations video:
            act_fn = os.path.join(self._out_path,
                                  f"gym_video_{time_now}_activations.avi")
            skvideo.io.vwrite(fname=act_fn,
                              videodata=self._activations_imgs,
                              outputdict=ffmpeg_out_dict)
            print(f"Gym activations video saved to: {act_fn}")

            # Env video:
            env_fn = os.path.join(self._out_path,
                                  f"gym_video_{time_now}_env.avi")
            skvideo.io.vwrite(fname=env_fn,
                              videodata=self._env_imgs,
                              outputdict=ffmpeg_out_dict)
            print(f"Gym environment video saved to: {env_fn}")

            # Concatenating videos:
            filter_arg = ("[1:v][0:v]scale2ref=main_w:ih[sec][pri];"
                          "[sec]setsar=1,drawbox=c=black:t=fill[sec];"
                          "[pri][sec]hstack[canvas];"
                          "[canvas][1:v]overlay=main_w-overlay_w")
            combined_fn = os.path.join(self._out_path,
                                       f"gym_video_{time_now}_combined.avi")
            subprocess.run(["ffmpeg",
                            "-i", act_fn,
                            "-i", env_fn,
                            "-r", str(self.fps),
                            "-vcodec", "mpeg4",
                            "-b", "800k",
                            "-filter_complex", filter_arg,
                            combined_fn],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT,
                           check=True)
            print(f"Gym combined video saved to: {combined_fn}")

            # Playing video:
            if self._play_video:
                NeatActivationsGymRenderer.play_video(combined_fn, self.fps)

            # Discarding cached images:
            self._env_imgs = []
            self._activations_imgs = []
