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

""" This module implements a generalizable fitness function that can be used
with most :mod:`gym` environments.
"""

from typing import Callable, List, Optional

import gym
import numpy as np

from nevopy.base_genome import BaseGenome
from nevopy.utils.gym_utils.callbacks import GymCallback
from nevopy.utils.gym_utils.renderers import GymRenderer
from nevopy.utils.utils import MutableWrapper


class GymFitnessFunction:
    """ Wrapper for a fitness function to be used with :mod:`gym`.

    This utility class implements a generalizable fitness function compatible
    with different :mod:`gym` environments.

    Args:
        make_env (Callable[[], Any]): Callable that creates the environment to
            be used. It should receive no arguments and return an instance of
            :class:`gym.Env`.
        env_renderer (Optional[GymRenderer]): Instance of :class:`.GymRenderer`
            (or a subclass) to be used to render the environment. By default, a
            new instance of :class:`.GymRenderer` is created (default rendering
            of the environment).
        callbacks (Optional[List[GymCallback]]): List with callbacks to be
            called at different stages of the evaluation of the genome's
            fitness.
        default_num_episodes (int): Default number of episodes ran in each call
            to the fitness function. This can be overridden during the call to
            the fitness function.
        default_max_steps (Optional[int]): Default maximum number of steps
            allowed in each episode. By default, there is no limit to the number
            of steps. This can be overridden during the call to the fitness
            function.
        num_obs_skip (int): Number of observations to be skipped during an
            episode. As an example, consider this value is set to 3. In this
            case, for each sequence of 4 observations yielded by the
            environment, only the 1st one will be fed to the genome. When,
            during a step, no observation is fed to the genome, the genome's
            last output is used to advance the environment's state.

    Attributes:
        env_renderer (GymRenderer): Instance of :class:`.GymRenderer` (or a
            subclass) to be used to render the environment.
        callbacks (List[GymCallback]): List with callbacks to be called at
            different stages of the evaluation of the genome's fitness.
        num_obs_skip (int): Number of observations to be skipped during an
            episode.
    """

    def __init__(self,
                 make_env: Callable[[], gym.Env],
                 env_renderer: Optional[GymRenderer] = None,
                 callbacks: Optional[List[GymCallback]] = None,
                 default_num_episodes: int = 1,
                 default_max_steps: Optional[int] = None,
                 num_obs_skip: int = 0) -> None:
        self._make_env = make_env
        self.env_renderer = (env_renderer if env_renderer is not None
                             else GymRenderer())
        self.callbacks = (callbacks if callbacks is not None
                          else [])  # type: List[GymCallback]

        self._default_num_episodes = default_num_episodes
        self._default_max_steps = default_max_steps
        self.num_obs_skip = num_obs_skip

        # Checking the environment's action space:
        temp_env = make_env()
        if type(temp_env.action_space) not in (gym.spaces.Discrete,
                                               gym.spaces.Box):
            raise ValueError(
                "This class can only handle discrete or boxed action spaces! "
                "The provided environment has an action space of the type: "
                f"{temp_env.action_space}"
            )

        self._discrete_action_space = isinstance(temp_env.action_space,
                                                 gym.spaces.Discrete)
        temp_env.close()

    def __call__(self,
                 genome: Optional[BaseGenome],
                 num_eps: Optional[int] = None,
                 max_steps: Optional[int] = None,
                 visualize: bool = False,
                 extra_callbacks: Optional[List[GymCallback]] = None,
    ) -> float:
        """ Makes a new environment and uses it to evaluate the genome's
        fitness (average reward obtained during the episodes).

        Args:
            genome (Optional[BaseGenome]): Genome (agent) that's going to
                interact with the environment. If ``None``, a random agent is
                used.
            num_eps (Optional[int]): Number of episodes. If not ``None``,
                overrides the default number of episodes. Otherwise, the number
                of episodes will be equal to the one specified when
                instantiating the class.
            max_steps (Optional[int]): Maximum number of steps a session can
                run.  If not ``None``, overrides the default maximum number of
                steps. Otherwise, the maximum number of steps will be equal to
                the one specified when instantiating the class.
            visualize (bool): Whether or not to render the environment and show
                the simulation running.
            extra_callbacks (Optional[List[GymCallback]]): Optional list with
                extra callbacks to be used only during this call to the fitness
                function.

        Returns:
            The average reward obtained by the agent during the episodes.
        """
        # Preparing variables:
        total_reward = 0.0

        if num_eps is None:
            num_eps = self._default_num_episodes

        if max_steps is None:
            max_steps = (self._default_max_steps  # type: ignore
                         if self._default_max_steps is not None
                         else float("inf"))

        callbacks = self.callbacks
        if extra_callbacks is not None:
            callbacks += extra_callbacks

        # Building the environment:
        env = self._make_env()

        # Callback: `on_env_built`:
        for cb in callbacks:
            cb.on_env_built(env=env, genome=genome)

        # Running the episodes:
        eps = None
        for eps in range(num_eps):
            # Callback: `on_episode_start`:
            for cb in callbacks:
                cb.on_episode_start(current_eps=eps, total_eps=num_eps)

            # Resetting:
            obs = env.reset()
            if genome is not None:
                genome.reset()

            env_done = False
            force_stop_eps = MutableWrapper(False)
            last_action = None

            # Running the steps:
            step = 0
            while (step <= max_steps
                   and not env_done
                   and not force_stop_eps.value):
                # Callback: `on_step_start`:
                for cb in callbacks:
                    cb.on_step_start(current_step=step, max_steps=max_steps)

                # Visualization:
                if visualize:
                    # Callback: `on_visualization`:
                    for cb in callbacks:
                        cb.on_visualization()

                    # Rendering:
                    self.env_renderer.render(env=env, genome=genome)

                # Calculating new action:
                if step % (self.num_obs_skip + 1) == 0:
                    # Callback: `on_obs_processing`:
                    wrapped_obs = MutableWrapper(obs)
                    for cb in callbacks:
                        cb.on_obs_processing(wrapped_obs=wrapped_obs)
                    obs = wrapped_obs.value

                    # Feeding obs to genome:
                    if genome is not None:
                        h = genome.process(obs)

                        # Fixing the output's shape (if needed):
                        if len(h.shape) > 1:
                            assert h.shape[0] == 1, ("Invalid output shape "
                                                     f"{h.shape}!")
                            h = h[0]

                        # Determining the action:
                        if self._discrete_action_space:
                            # Discrete action space
                            action = (round(float(h[0])) if len(h) == 1
                                      else np.argmax(h))
                        else:
                            # Boxed action space
                            action = h
                    # Random agent:
                    else:
                        action = env.action_space.sample()

                    last_action = action
                # Repeating the agent's last action:
                else:
                    action = last_action

                # Callback: `on_action_chosen`
                wrapped_action = MutableWrapper(action)
                for cb in callbacks:
                    cb.on_action_chosen(wrapped_action=wrapped_action)
                action = wrapped_action.value

                # Processing step:
                obs, reward, env_done, info = env.step(action)
                total_reward += reward

                # Callback: `on_step_taken`:
                for cb in callbacks:
                    cb.on_step_taken(obs=obs,
                                     reward=reward,
                                     done=env_done,
                                     info=info,
                                     total_reward=total_reward,
                                     force_stop_eps=force_stop_eps)

        # Callback: `on_env_close`:
        for cb in callbacks:
            cb.on_env_close()

        # Flushing the renderer's buffers:
        if visualize:
            self.env_renderer.flush()

        # Closing the environment:
        env.close()

        # Returning average fitness:
        return (total_reward / (eps + 1)) if eps is not None else 0
