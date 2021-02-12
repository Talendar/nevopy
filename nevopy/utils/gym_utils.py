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

""" This module implements useful utility functions related to the `gym`
package.
"""

import time
from typing import Any, Callable, Optional, TypeVar, Union

import numpy as np

from nevopy.base_genome import BaseGenome

#: `TypeVar` indicating an undefined type
_T = TypeVar("_T")


class GymEnvFitness:
    """ Wrapper for a fitness function to be used with a `gym` environment.

    This utility class implements the basic routine used as a fitness function
    for neuroevolutionary algorithms in the context of a regular `gym`
    environment.

    Args:
        make_env (Callable[[], Any]): Callable that creates the environment to
            be used. It should receive no arguments and return an instance of a
            `gym` environment.
        num_episodes (int): Default number of episodes for the simulation to run
            in each call. This value can be overridden during the call.
        pre_process_obs (Optional[Callable[[_T], _T]]): Function used to
            pre-process the observation yielded by the environment before
            feeding it to the genome. If `None`, no pre-processing is done.
        render_fps (int): Number of frames per second when visualizing the
            simulation.
        max_steps (Optional[int]): Default maximum number of steps a session can
            run. By default, there is no limit. This can be overridden during
            the call.
        num_obs_skip (int):

    Attributes:
        make_env (Callable[[], Any]): Callable that creates the environment to
            be used. It should receive not arguments and return an instance of a
            `gym` environment.
        num_episodes (int): Default number of episodes for the simulation to run
            in each call. This value can be overridden during the call.
        pre_process_obs (Optional[Callable[[_T, Any], _T]]): Function used to
            pre-process the observation yielded by the environment before
            feeding it to the genome. If `None`, no pre-processing is done. The
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
                 pre_process_obs: Optional[Callable[[_T, Any], _T]] = None,
                 render_fps: int = 60,
                 max_steps: Optional[int] = None,
                 num_obs_skip: int = 0) -> None:
        self.make_env = make_env
        self.num_episodes = num_episodes
        self.pre_process_obs = pre_process_obs
        self.render_fps = render_fps
        self.max_steps = max_steps
        self.num_obs_skip = num_obs_skip

    def __call__(self,
                 genome: Optional[BaseGenome] = None,
                 eps: Optional[int] = None,
                 max_steps: Optional[Union[float, int]] = None,
                 visualize: bool = False) -> float:
        """ Runs a simulation of the given agent in the gym environment.

        Args:
            genome (Optional[BaseGenome]): Agent that's going to interact with
                the gym environment. If `None`, a random agent is created.
            eps (int): Number of episodes. The default value is the number
                specified when creating the class.
            max_steps (Optional[Union[float, int]]): Maximum number of steps a
                session can run. The default value is the number specified when
                creating the class.
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

        skip_counter = self.num_obs_skip
        last_action = None
        total_reward = 0

        for _ in range(eps):
            obs = env.reset()
            if genome is not None:
                genome.reset()

            steps = 0
            while steps < max_steps:
                steps += 1
                if visualize:
                    env.render()
                    time.sleep(1 / self.render_fps)

                if self.pre_process_obs is not None:
                    obs = self.pre_process_obs(obs, env)

                if skip_counter == 0 or last_action is None:
                    if genome is not None:
                        result = genome.process(obs)
                        if len(result.shape) > 1:
                            assert result.shape[0] == 1, (
                                "Detected a batched outputs with multiple "
                                "items! Only batches with 1 item are accepted."
                            )
                            result = result[0]

                        action = (round(float(result[0])) if len(result) == 1
                                  else np.argmax(result))
                    else:
                        action = env.action_space.sample()

                    skip_counter = self.num_obs_skip
                    last_action = action
                else:
                    skip_counter -= 1
                    action = last_action

                obs, reward, done, _ = env.step(action)
                total_reward += reward

                if done:
                    break

        env.close()
        return total_reward / eps
