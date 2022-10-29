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


import numpy as np

import dm_env
from acme import specs, types
from acme.wrappers import base

import itertools
import collections

from typing import Sequence, Optional

from acme.tf import utils
import tree

from typing import Optional, Sequence


class DiscreteWrapper(base.EnvironmentWrapper):
  """Continuous to discrete wrapper."""

  def __init__(self, environment: dm_env.Environment, config):
    super().__init__(environment)

    self._num_bins = config.num_bins

    self._action_min = environment.action_spec().minimum
    self._action_max = environment.action_spec().maximum
    self._action_all = self._get_action_list()

  
  def action_spec(self):
    act_spec = super().action_spec()
    act_shape = act_spec.shape
    return specs.BoundedArray(
      shape=(),
      dtype=np.int32,
      minimum=0,
      maximum=self._num_bins**act_shape[0]-1,
    )


  def _get_action_list(self):
    act_lim = list(np.linspace(self._action_min, self._action_max, num=self._num_bins).transpose())
    act_per = itertools.product(*act_lim)
    act_per = [np.array(e) for e in act_per]
    return act_per


  def reset(self) -> dm_env.TimeStep:
    # Reset the environment
    timestep = super().reset()
    return timestep


  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    # Step the environment
    action = self._action_all[action]
    timestep = self._environment.step(action)

    return timestep


class DecoupledDiscreteWrapper(base.EnvironmentWrapper):
  """Continuous to decoupled discrete wrapper."""

  def __init__(self, environment: dm_env.Environment, config):
      super().__init__(environment)

      self._num_bins = config.num_bins

      self._action_min = environment.action_spec().minimum
      self._action_max = environment.action_spec().maximum
      self._action_all = self._get_action_list()

  def _get_action_list(self):
      return list(np.linspace(self._action_min, self._action_max, num=self._num_bins).transpose())

  def action_spec(self):
      act_shape = self._action_min.shape
      return specs.BoundedArray(
        shape=(act_shape[0],),
        dtype=np.int32,
        minimum=0,
        maximum=self._num_bins*act_shape[0]-1,
      )

  def reset(self) -> dm_env.TimeStep:
    # Reset the environment
    timestep = super().reset()
    return timestep

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
      # Step the environment
      action = np.take(self._action_all, action)
      timestep = self._environment.step(action)

      return timestep


def _concat(values: types.NestedArray) -> np.ndarray:
  """Concatenates the leaves of `values` along the leading dimension.
  Treats scalars as 1d arrays and expects that the shapes of all leaves are
  the same except for the leading dimension.
  Args:
    values: the nested arrays to concatenate.
  Returns:
    The concatenated array.
  """
  leaves = list(map(np.atleast_1d, tree.flatten(values)))
  return np.concatenate(leaves)


class CustomConcatObservationWrapper(base.EnvironmentWrapper):
  """Wrapper that concatenates observation fields.
  It takes an environment with nested observations and concatenates the fields
  in a single tensor. The orginial fields should be 1-dimensional.
  Observation fields that are not in name_filter are dropped.
  """

  def __init__(self, environment: dm_env.Environment,
               name_filter: Optional[Sequence[str]] = None):
    """Initializes a new ConcatObservationWrapper.
    Args:
      environment: Environment to wrap.
      name_filter: Sequence of observation names to keep. None keeps them all.
    """
    super().__init__(environment)
    observation_spec = environment.observation_spec()
    if name_filter is None:
      name_filter = list(observation_spec.keys())
    self._obs_names = [x for x in name_filter if x in observation_spec.keys()]

    dummy_obs = utils.zeros_like(observation_spec)
    dummy_obs = self._convert_observation(dummy_obs)
    self._observation_spec = dm_env.specs.BoundedArray(
        shape=dummy_obs.shape,
        dtype=dummy_obs.dtype,
        minimum=-np.inf,
        maximum=np.inf,
        name='state')

  def _convert_observation(self, observation):
    obs = {k: observation[k] for k in self._obs_names}
    return _concat(obs)

  def step(self, action) -> dm_env.TimeStep:
    timestep = self._environment.step(action)
    return timestep._replace(
        observation=self._convert_observation(timestep.observation))

  def reset(self) -> dm_env.TimeStep:
    timestep = self._environment.reset()
    return timestep._replace(
        observation=self._convert_observation(timestep.observation))

  def observation_spec(self) -> types.NestedSpec:
    return self._observation_spec


class DMCVisionWrapper(base.EnvironmentWrapper):

  def __init__(self, environment, domain, action_repeat=1, frame_stack=3, size=(84, 84), camera=None):

      super().__init__(environment)

      self._action_repeat = action_repeat
      self._frame_stack = frame_stack
      self._frames = collections.deque([], maxlen=frame_stack)

      self._size = size
      if not camera:
        camera = dict(quadruped=2).get(domain, 0)
      self._camera = camera

      self._observation_spec = collections.OrderedDict()

      timestep = self.reset()
      observation = timestep.observation
      self._observation_spec = dm_env.specs.BoundedArray(
          shape=observation.shape, 
          dtype=observation.dtype, 
          minimum=0, 
          maximum=255, 
          name='image')

  def _get_obs(self):
    assert len(self._frames) == self._frame_stack
    return np.concatenate(list(self._frames), axis=-1)

  def step(self, action) -> dm_env.TimeStep:

    reward = 0.0
    for _ in range(self._action_repeat):
      timestep = self._environment.step(action)
      reward += timestep.reward
      if timestep.last():
        break

    image = self._render()
    self._frames.append(image)
    timestep = timestep._replace(observation=self._get_obs(), reward=reward)
    
    return timestep

  def reset(self) -> dm_env.TimeStep:
    timestep = self._environment.reset()
    image = self._render()
    for _ in range(self._frame_stack):
      self._frames.append(image)
    timestep = timestep._replace(observation=self._get_obs())
    return timestep

  def discount_spec(self):
    return self._environment.discount_spec()

  def observation_spec(self):
    return self._observation_spec

  def reward_spec(self):
    return self._environment.reward_spec()

  def _render(self):
    return self._environment._physics.render(*self._size, camera_id=self._camera)


class ActionRepeatWrapper(base.EnvironmentWrapper):

  def __init__(self, environment, action_repeat=1):

      super().__init__(environment)

      self._action_repeat = action_repeat

  def step(self, action) -> dm_env.TimeStep:

    reward = 0.0
    for _ in range(self._action_repeat):
      timestep = self._environment.step(action)
      reward += timestep.reward
      if timestep.last():
        break
    
    return timestep


import gym
from acme.wrappers import GymWrapper

class MetaWorldWrapper(GymWrapper):

  def __init__(self, environment: gym.Env, truncated=True, duration=1000):

    self._trunc = truncated
    self._steps = duration
    self._step = None
    super().__init__(environment)

  def reset(self) -> dm_env.TimeStep:
    self._step = 0
    return super().reset()

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    """Steps the environment."""
    if self._reset_next_step:
      return self.reset()

    observation, reward, done, info = self._environment.step(action)

    if done and (self._step+1 < self._steps) and not self._trunc:
      done = False

    if self._step+1 > self._steps-1:
      '''Keep this in mind: https://github.com/rlworkgroup/metaworld/issues/236'''
      done = True

    self._reset_next_step = done

    self._step += 1

    if done:
      truncated = info.get('TimeLimit.truncated', False)
      if truncated:
        return dm_env.truncation(reward, observation)
      return dm_env.termination(reward, observation)
    return dm_env.transition(reward, observation)
