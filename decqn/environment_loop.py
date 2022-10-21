# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications copyright (C) 2022 DecQN team.

"""A simple agent-environment training loop."""

import operator
import time
from typing import Optional

from acme import core
# Internal imports.
from acme.utils import counting
from acme.utils import loggers

import dm_env
from dm_env import specs
import numpy as np
import tree


class EnvironmentLoop(core.Worker):
  """A simple RL environment loop.

  This takes `Environment` and `Actor` instances and coordinates their
  interaction. Agent is updated if `should_update=True`. This can be used as:

    loop = EnvironmentLoop(environment, actor)
    loop.run(num_episodes)

  A `Counter` instance can optionally be given in order to maintain counts
  between different Acme components. If not given a local Counter will be
  created to maintain counts between calls to the `run` method.

  A `Logger` instance can also be passed in order to control the output of the
  loop. If not given a platform-specific default logger will be used as defined
  by utils.loggers.make_default_logger. A string `label` can be passed to easily
  change the label associated with the default logger; this is ignored if a
  `Logger` instance is given.
  """

  def __init__(
      self,
      environment: dm_env.Environment,
      actor: core.Actor,
      domain,
      counter: counting.Counter = None,
      eval_every: int = 10,
      logger: loggers.Logger = None,
      eval_logger: loggers.Logger = None,
      video_logger: loggers.Logger = None,
      should_update: bool = True,
      eval_dir: str = None,
      eval_video: bool = False,
  ):
    # Internalize agent and environment.
    self._environment = environment
    self._actor = actor
    self._counter = counter or counting.Counter()
    self._logger = logger
    self._video_logger = video_logger
    self._should_update = should_update

    self._domain = domain

    self._eval_every = eval_every
    self._eval_actor = actor.eval_actor
    self._eval_logger = eval_logger
    self._eval_counter = counting.Counter()

    self._eval_dir = eval_dir

    self._eval_video = eval_video
    self._frames = None
    self._video_width = 84
    self._video_height = 84
    self._camera_id = dict(quadruped=2).get(domain, 0)

    self.init_time = None


  def run_episode(self, evaluate=False, increment=True) -> loggers.LoggingData:
    """Run one episode.

    Each episode is a loop which interacts first with the environment to get an
    observation and then give that observation to the agent in order to retrieve
    an action.

    Returns:
      An instance of `loggers.LoggingData`.
    """
    # Reset any counts and start the environment.
    start_time = time.time()
    if not self.init_time:
      self.init_time = start_time
    episode_steps = 0

    # For evaluation, this keeps track of the total undiscounted reward
    # accumulated during the episode.
    episode_return = tree.map_structure(_generate_zeros_from_spec,
                                        self._environment.reward_spec())
    timestep = self._environment.reset()

    # Make the first observation.
    if not evaluate:
      self._actor.observe_first(timestep)

    if self._eval_video and evaluate:
      self._frames = [self._render()]

    # Run an episode.
    while not timestep.last():
      # Generate an action from the agent's policy and step the environment.
      if not evaluate:
        action = self._actor.select_action(timestep.observation)
      else:
        action = self._eval_actor.select_action(timestep.observation)

      timestep = self._environment.step(action)

      if self._eval_video and evaluate:
        self._frames.append(self._render())

      # Have the agent observe the timestep and let the actor update itself.
      if not evaluate:
        self._actor.observe(action, next_timestep=timestep)
        if self._should_update:
          self._actor.update()

      # Book-keeping.
      episode_steps += 1

      # Equivalent to: episode_return += timestep.reward
      tree.map_structure(operator.iadd, episode_return, timestep.reward)

    # Record counts.
    if not evaluate:
      counts = self._counter.increment(episodes=1, steps=episode_steps)
    else:
      if increment:
        counts = self._eval_counter.increment(episodes=self._eval_every, steps=episode_steps)
      else:
        counts = self._eval_counter.increment(episodes=0, steps=episode_steps)

    # Collect the results and combine with counts.
    curr_time = time.time()
    steps_per_second = episode_steps / (curr_time - start_time)
    walltime = curr_time - self.init_time
    result = {
        'episode_length': episode_steps,
        'episode_return': float(episode_return),
        'steps_per_second': steps_per_second,
        'walltime': walltime,
    }
    result.update(counts)
    return result


  def run(self,
          num_episodes: Optional[int] = None,
          num_steps: Optional[int] = None):
    """Perform the run loop.

    Run the environment loop either for `num_episodes` episodes or for at
    least `num_steps` steps (the last episode is always run until completion,
    so the total number of steps may be slightly more than `num_steps`).
    At least one of these two arguments has to be None.

    Upon termination of an episode a new episode will be started. If the number
    of episodes and the number of steps are not given then this will interact
    with the environment infinitely.

    Args:
      num_episodes: number of episodes to run the loop for.
      num_steps: minimal number of steps to run the loop for.

    Raises:
      ValueError: If both 'num_episodes' and 'num_steps' are not None.
    """

    if not (num_episodes is None or num_steps is None):
      raise ValueError('Either "num_episodes" or "num_steps" should be None.')

    def should_terminate(episode_count: int, step_count: int) -> bool:
      return ((num_episodes is not None and episode_count >= num_episodes) or
              (num_steps is not None and step_count >= num_steps))

    episode_count, step_count, step_since_eval = 0, 0, 0
    while not should_terminate(episode_count, step_count):

      should_eval = (episode_count % self._eval_every)==0

      if should_eval:
        episode_eval_num = 10
        returns_eval = []
        for id_eval_ep in range(episode_eval_num):
          increment = True if id_eval_ep==0 else False
          increment = increment and (episode_count>0)

          result = self.run_episode(evaluate=True, increment=increment)
          returns_eval.append(result['episode_return'])

        returns_eval_mean = np.mean(np.stack(returns_eval))
        returns_eval_stdv = np.std(np.stack(returns_eval))
        result.update({'episode_return_mean': returns_eval_mean, 'episode_return_stdv': returns_eval_stdv, 'episode_eval_num': episode_eval_num})

        if self._eval_logger:
          self._eval_logger.write(result)

        if self._eval_video and self._video_logger:
          self._video_logger.write({'video': np.stack(self._frames)})

        step_since_eval -= self._eval_every * 1000.

      result = self.run_episode()
      episode_count += 1
      step_count += result['episode_length']

      step_since_eval += result['episode_length']

      # Log the given results.
      if self._logger:
        self._logger.write(result)


  def _render(self):
    return self._environment._physics.render(camera_id=self._camera_id, height=self._video_height, width=self._video_width)


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)
