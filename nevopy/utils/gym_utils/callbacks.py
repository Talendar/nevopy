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

""" This module defines an interface for callbacks to be used with
:class:`.GymFitnessFunction`.
"""

from typing import Any, Dict, Optional

import gym  # pylint: disable=unused-import
import numpy as np

import nevopy as ne  # pylint: disable=unused-import


class GymCallback:
    """ Interface for callbacks to be used with :class:`.GymFitnessFunction`.

    Each of the callback's method is called at a different point during the
    evaluation of a genome's fitness by a :class:`.GymFitnessFunction`.

    It's not required for a subclass to implement all the methods of this class
    (you can implement only those that will be useful for your case).
    """

    def __init__(self):
        self._env = None  # type: Optional["gym.Env"]
        self._genome = None  # type: Optional["ne.BaseGenome"]

    def on_env_built(self,
                     env: "gym.Env",
                     genome: "ne.BaseGenome") -> None:
        """ Called right AFTER the gym environment is built.

        This method is called right after the :mod:`gym` environment is built,
        i.e., right after a call to :func:`gym.make()` is made.

        Args:
            env (gym.Env): The :mod:`gym` environment that's going to be used by
                the fitness function.
            genome (nevopy.BaseGenome): The genome currently being evaluated by
                the fitness function.
        """
        self._env = env
        self._genome = genome

    def on_episode_start(self,
                         current_eps: int,
                         total_eps: int) -> None:
        """ Called at the start of a new episode, before the env is reset.

        Subclasses should override this method for any actions to run.

        Args:
            current_eps (int): Number of the current episode.
            total_eps (int): Total number of episodes to run during the current
                session.
        """

    def on_step_start(self,
                      current_step: int,
                      max_steps: int,) -> None:
        """ Called at the start of a new step.

        Subclasses should override this method for any actions to run.

        Args:
            current_step (int): Number of the current step.
            max_steps (int): Maximum number of steps allowed in each episode.
        """

    def on_visualization(self) -> None:
        """ Called right BEFORE the rendering of the environment occurs.

        This method is only called when ``True`` is passed to the ``visualize``
        parameter of :meth:`.GymFitnessFunction.__call__()`.

        Subclasses should override this method for any actions to run.
        """

    def on_obs_processing(self,
                          wrapped_obs: "ne.utils.MutableWrapper[Any]",
    ) -> None:
        """ Called right BEFORE the observation yielded by the environment is
        fed to the genome.

        Changing the observation stored by the wrapper will have effects on the
        fitness function.

        Subclasses should override this method for any actions to run.

        Args:
            wrapped_obs (nevopy.utils.MutableWrapper[Any]): Mutable wrapper
                around he observation yielded by the gym environment.
        """

    def on_action_chosen(self,
                         wrapped_action: "ne.utils.MutableWrapper[Any]",
    ) -> None:
        """ Called right AFTER an action is chosen by the genome.

        Changing the action stored by the wrapper will have effects on the
        fitness function.

        Subclasses should override this method for any actions to run.

        Args:
            wrapped_action (nevopy.utils.MutableWrapper[Any]): Mutable wrapper
                around the action chosen by the genome.
        """

    def on_step_taken(self,
                      obs: Any,
                      reward: float,
                      done: bool,
                      info: Dict[str, Any],
                      total_reward: float,
                      force_stop_eps: "ne.utils.MutableWrapper[bool]",
    ) -> None:
        """ Called right AFTER the environment's :meth:`step()` method is
        called.

        Subclasses should override this method for any actions to run.

        Args:
            obs (Any): The observation yielded by the environment.
            reward (float): The reward yielded by the environment.
            done (bool): Whether or not the episode has finished.
            info (Dict[str, Any]): Extra information yielded by the environment.
            total_reward (float): Total reward obtained by the genome so far.
            force_stop_eps (nevopy.utils.MutableWrapper[bool]): Setting the
                value on this wrapper to `True` will forcefully stop the current
                episode.
        """

    def on_env_close(self) -> None:
        """ Called right BEFORE the environment is closed and the function
        returns the fitness of the genome.

        Subclasses should override this method for any actions to run.
        """


class BatchObsGymCallback(GymCallback):
    """ Simple callback that expands the dimensions of the observations yielded
    by a :class:`gym.Env` before feeding them to a genome.

    Simply turns the observation into a batch of one item (the observation
    itself), so it can be fed to genomes that require batched inputs (like
    genomes that use TensorFlow, for example).
    """

    def on_obs_processing(self,
                          wrapped_obs: "ne.utils.MutableWrapper[Any]",
    ) -> None:
        x = wrapped_obs.value
        wrapped_obs.value = np.reshape(x, [1, len(x)])
