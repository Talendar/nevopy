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

""" This module implements useful utility functions.
"""

from typing import Optional, List, TypeVar, Callable, Any, Iterable, Set, Union
from abc import ABCMeta, abstractmethod

import numpy as np
import click
import os
import time


#: `TypeVar` indicating an undefined type
T = TypeVar("T")


class Comparable(metaclass=ABCMeta):
    """ Indication of a "comparable" type. """
    @abstractmethod
    def __lt__(self, other: Any) -> bool:
        pass


def chance(p: float) -> bool:
    """ Randomly returns `True` or `False`.

    Args:
        p (float): Float between 0 and 1. Specifies the chance of the function
            returning True.

    Returns:
        A randomly chosen boolean value (`True` or `False`).
    """
    return np.random.uniform(low=0, high=1) < p


def align_lists(lists: Iterable[List[T]],
                getkey: Optional[Callable[[T], Comparable]] = None,
                placeholder: Optional[Any] = None) -> List[List[T]]:
    """ Aligns the given lists based on their common values.

    Repeated entries within a single list are discarded.

    Example:
        >>> align_lists(([1, 2, 3, 6], [1, 3, 4, 5]))
        [[1, 2, 3, None, None, 6], [1, None, 3, 4, 5, None]]

    Args:
        lists (Iterable[List[T]]): Iterable that yields lists containing the
            objects to be aligned.
        getkey (Optional[Callable[[T], Comparable]]): Optional function to be
            passed to :py:func:`sorted()` to retrieve comparable keys from the
            objects to be aligned.
        placeholder (Optional[Any]): Value to be used as a placeholder when an
            item doesn't match with any other (see the example above, where
            `None` is the placeholder).

    Returns:
        A list containing the aligned lists. The items in the aligning lists
        will be sorted in ascending order.
    """
    union = set()  # type: Set[T]
    for l in lists:
        union = union | set(l)
    values = sorted(union, key=getkey)

    result = []
    for l in lists:
        result.append([n if n in l else placeholder for n in values])
    return result


def min_max_norm(values: Iterable) -> np.array:
    """ Applies min-max normalization to the given values. """
    a = np.array(values)
    a_min, a_max = np.min(a), np.max(a)
    return (a - a_min) / (a_max - a_min)


def is_jupyter_notebook() -> bool:
    """ Checks whether the program is running on a jupyter notebook.

    Warning:
        This function is not guaranteed to work! It simply checks if
        :py:func:`IPython.get_ipython` returns `None`.

    Returns:
        `True` if the program is running on a jupyter notebook and `False`
        otherwise.
    """
    try:
        # noinspection PyUnresolvedReferences
        from IPython import get_ipython
        if get_ipython() is None:
            return False
    except ModuleNotFoundError:
        return False
    return True


def clear_output() -> None:
    """ Clears the output.

    Should work on Windows and Linux terminals and on Jupyter notebooks. On
    PyCharm, it simply prints a bunch of new lines.
    """
    if "PYCHARM_HOSTED" in os.environ:
        print("\n" * 15)
    elif is_jupyter_notebook():
        # noinspection PyUnresolvedReferences
        from IPython.display import clear_output
        clear_output(wait=True)
    else:
        click.clear()


class GymEnvFitness:
    """ Wrapper for a fitness function to be used with a `gym` environment.

    This utility class implements the basic routine used as a fitness function
    for neuroevolutionary algorithms when a `gym` environment is being used.

    Args:
        make_env (Callable[[], Any]): Callable that creates the environment to
            be used. It should receive not arguments and return an instance of a
            `gym` environment.
        num_episodes (int): Default number of episodes for the simulation to run
            in each call. This value can be overridden during the call.
        pre_process_obs (Optional[Callable[[T], T]]): Function used to
            pre-process the observation yielded by the environment before
            feeding it to the agent. If `None`, no pre-processing is done.
        render_fps (int): Number of frames per second when visualizing the
            simulation.
        max_steps (Optional[int]): Default maximum number of steps a session can
            run. By default, there is no limit. This can be overridden during
            the call.

    Attributes:
        make_env (Callable[[], Any]): Callable that creates the environment to
            be used. It should receive not arguments and return an instance of a
            `gym` environment.
        num_episodes (int): Default number of episodes for the simulation to run
            in each call. This value can be overridden during the call.
        pre_process_obs (Optional[Callable[[T, Any], T]]): Function used to
            pre-process the observation yielded by the environment before
            feeding it to the agent. If `None`, no pre-processing is done. The
            function is expected to receive the observed data from the
            environment (param 1) and the current instance of the environment
            (param 2).
        render_fps (int): Number of frames per second when visualizing the
            simulation.
        max_steps (Optional[int]): Default maximum number of steps a session can
            run. By default, there is no limit. This can be overridden during
            the call.
    """

    def __init__(self,
                 make_env: Callable[[], Any],
                 num_episodes: int,
                 pre_process_obs: Optional[Callable[[T, Any], T]] = None,
                 render_fps: int = 60,
                 max_steps: Optional[int] = None) -> None:
        self.make_env = make_env
        self.num_episodes = num_episodes
        self.pre_process_obs = pre_process_obs
        self.render_fps = render_fps
        self.max_steps = max_steps  # type: Optional[int]

    def __call__(self,
                 agent: Optional[Any] = None,
                 eps: Optional[int] = None,
                 max_steps: Optional[Union[float, int]] = None,
                 visualize: bool = False) -> float:
        """ Runs a simulation of the given agent in the gym environment.

        Todo:
            Currently, this only supports agents that implements the methods
            ``process`` and ``reset_activations`` from the class
            :class:`nevopy.neat.genome.Genome`. Perhaps implement a common agent
            interface for all `NEvoPY's` submodules?

        Args:
            agent (Optional[Any]): Agent that's going to interact with the gym
                environment. If `None`, a random agent is created.
            eps (int): Number of episodes. The default value is the number
                specified when creating the class.
            max_steps (Optional[Union[float, int]]): Default maximum number of
                steps a session can run. The default value is the number
                specified when creating the class.
            visualize (bool): Whether to show the simulation.

        Returns:
            The average reward obtained by the agent during each episode.
        """
        # preparing
        env = self.make_env()

        if eps is None:
            eps = self.num_episodes

        if max_steps is None:
            max_steps = (float("inf") if self.max_steps is None
                         else self.max_steps)  # type: ignore

        total_reward = 0
        for _ in range(eps):
            obs = env.reset()
            if agent is not None:
                agent.reset_activations()

            steps = 0
            while steps < max_steps:
                steps += 1
                if visualize:
                    env.render()
                    time.sleep(1 / self.render_fps)

                if self.pre_process_obs is not None:
                    obs = self.pre_process_obs(obs, env)

                if agent is not None:
                    action = np.argmax(agent.process(obs))
                else:
                    action = env.action_space.sample()

                obs, reward, done, _ = env.step(action)
                total_reward += reward

                if done:
                    break

        env.close()
        return total_reward / eps
