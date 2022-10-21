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

"""DQN agent implementation."""

import copy

from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.agents import agent
from agents.dqn.actors import CustomDiscreteFeedForwardActor
from agents.dqn import learning
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import loggers
import reverb
import sonnet as snt
import tensorflow as tf

from functools import partial

from misc.utils import epsilon_greedy as epsilon_greedy_base


class DQN(agent.Agent):
  """DQN agent.

  This implements a single-process DQN agent. This is a simple Q-learning
  algorithm that inserts N-step transitions into a replay buffer, and
  periodically updates its policy by sampling these transitions using
  prioritization.
  -----------------------------------------------------------------------
  Code adapted to handle decoupled Q-networks and vision input.
  """

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      network: snt.Module,
      encoder: snt.Module,
      config,
      logger: loggers.Logger = None,
      checkpoint: bool = True,
  ):
    """Initialize the agent.

    Args:
      environment_spec: description of the actions, observations, etc.
      network: the online Q network (the one being optimized)
      encoder: the encoder used for vision inputs
      config: agent hyperparameters
      logger: logger object to be used by learner.
      checkpoint: boolean indicating whether to checkpoint the learner.
    """

    # Get parameter values
    batch_size = config.batch_size
    prefetch_size = config.prefetch_size
    target_update_period = config.target_update_period
    samples_per_insert = config.samples_per_insert
    min_replay_size = config.min_replay_size
    max_replay_size = config.max_replay_size
    importance_sampling_exponent = config.importance_sampling_exponent
    priority_exponent = config.priority_exponent
    n_step = config.adder_n_step
    epsilon = config.epsilon
    learning_rate = config.learning_rate
    discount = config.discount

    # Create a replay server to add data to. This uses no limiter behavior in
    # order to allow the Agent interface to handle it.
    replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Prioritized(priority_exponent),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_size,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=adders.NStepTransitionAdder.signature(environment_spec))
    self._server = reverb.Server([replay_table], port=None)

    # The adder is used to insert observations into replay.
    address = f'localhost:{self._server.port}'
    adder = adders.NStepTransitionAdder(
        client=reverb.Client(address),
        n_step=n_step,
        discount=discount)

    # The dataset provides an interface to sample from replay.
    replay_client = reverb.TFClient(address)
    dataset = datasets.make_reverb_dataset(
        server_address=address,
        batch_size=batch_size,
        prefetch_size=prefetch_size)

    # Use constant 0.05 epsilon greedy policy by default.
    if epsilon is None:
      epsilon = 0.05
    epsilon = tf.Variable(epsilon, trainable=False)

    epsilon_greedy = partial(epsilon_greedy_base, config.decouple)

    network_combined = []
    if config.use_pixels:
      assert encoder
      network_combined.append(encoder)
    network_combined.append(network)

    policy_network = snt.Sequential([
        *network_combined,
        lambda q: epsilon_greedy(tf.maximum(*q), epsilon=epsilon),
    ])

    eval_policy_network = snt.Sequential([
        *network_combined,
        lambda q: epsilon_greedy(tf.maximum(*q), epsilon=0.0),
    ])

    # Create a target network.
    target_network = copy.deepcopy(network)

    # Ensure that we create the variables before proceeding (maybe not needed).
    obs_spec = environment_spec.observations
    if config.use_pixels:
      emb_spec = tf2_utils.create_variables(encoder, [obs_spec])
    else:
      emb_spec = obs_spec

    tf2_utils.create_variables(network, [emb_spec])
    tf2_utils.create_variables(target_network, [emb_spec])

    # Create the actor which defines how we take actions.
    actor = CustomDiscreteFeedForwardActor(policy_network, adder)

    self.eval_actor = CustomDiscreteFeedForwardActor(eval_policy_network)

    # The learner updates the parameters (and initializes them).
    learner = learning.DQNLearner(
        config=config,
        network=network,
        target_network=target_network,
        encoder=encoder,
        discount=discount,
        importance_sampling_exponent=importance_sampling_exponent,
        learning_rate=learning_rate,
        target_update_period=target_update_period,
        dataset=dataset,
        replay_client=replay_client,
        logger=logger,
        checkpoint=checkpoint)

    if checkpoint:
      self._checkpointer = tf2_savers.Checkpointer(
          directory=config.learner_dir,
          objects_to_save=learner.state,
          subdirectory='dqn_learner',
          time_delta_minutes=60.,
          add_uid=False)
    else:
      self._checkpointer = None

    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=max(batch_size, min_replay_size),
        observations_per_step=float(batch_size) / samples_per_insert)

  def update(self):
    super().update()
    if self._checkpointer is not None:
      self._checkpointer.save()
