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

"""DQN learner implementation."""

import time
from typing import Dict, List

import acme
from acme.adders import reverb as adders
from acme.tf import losses
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf

from functools import partial
from misc.utils import double_qlearning as double_qlearning_base
from misc.augmentation import random_shift


class DQNLearner(acme.Learner, tf2_savers.TFSaveable):
  """DQN learner.

  This is the learning component of a DQN agent. It takes a dataset as input
  and implements update functionality to learn from this dataset. Optionally
  it takes a replay client as well to allow for updating of priorities.
  """

  def __init__(
      self,
      config,
      network: snt.Module,
      target_network: snt.Module,
      encoder: snt.Module,
      discount: float,
      importance_sampling_exponent: float,
      learning_rate: float,
      target_update_period: int,
      dataset: tf.data.Dataset,
      huber_loss_parameter: float = 1.,
      replay_client: reverb.TFClient = None,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
      checkpoint: bool = True,
      snapshot: bool = False,
  ):
    """Initializes the learner.

    Args:
      config: the agent hyperparameters
      network: the online Q network (the one being optimized)
      target_network: the target Q critic (which lags behind the online net).
      encoder: the encoder used for vision experiments
      discount: discount to use for TD updates.
      importance_sampling_exponent: power to which importance weights are raised
        before normalizing.
      learning_rate: learning rate for the q-network update.
      target_update_period: number of learner steps to perform before updating
        the target networks.
      dataset: dataset to learn from, whether fixed or from a replay buffer (see
        `acme.datasets.reverb.make_dataset` documentation).
      huber_loss_parameter: Quadratic-linear boundary for Huber loss.
      replay_client: client to replay to allow for updating priorities.
      counter: Counter object for (potentially distributed) counting.
      logger: Logger object for writing logs to.
      checkpoint: boolean indicating whether to checkpoint the learner.
      snapshot: boolean indicating whether to snapshot the learner (adds UID to dir).
    """

    # Internalise agent components (replay buffer, networks, optimizer).
    self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types
    self._network = network
    self._target_network = target_network
    self._optimizer = snt.optimizers.Adam(learning_rate)
    self._replay_client = replay_client

    self._encoder = encoder
    self._optimizer_encoder = snt.optimizers.Adam(learning_rate)

    self._use_pixels = config.use_pixels
    self._action_repeat = config.action_repeat
    self._num_pixels = config.num_pixels
    self._pad_size = config.pad_size
    self._aug_fct = partial(random_shift, self._pad_size)

    # Internalise the hyperparameters.
    self._discount = discount
    self._target_update_period = target_update_period
    self._importance_sampling_exponent = importance_sampling_exponent
    self._huber_loss_parameter = huber_loss_parameter

    self._clipping = config.clip_gradients
    self._clip_norm = config.clip_gradients_norm

    # Learner state.
    self._variables: List[List[tf.Tensor]] = [network.trainable_variables]
    if self._use_pixels:
      self._variables.append(encoder.trainable_variables)
    self._num_steps = tf.Variable(0, dtype=tf.int32)

    # Internalise logging/counting objects.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

    # Create a snapshotter object.
    if snapshot:
      self._snapshotter = tf2_savers.Snapshotter(
          objects_to_save={'network': network}, time_delta_minutes=60., directory=config.learner_dir)
    else:
      self._snapshotter = None

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

    self._decouple = config.decouple
    self._use_double_q = config.use_double_q
    self._double_qlearning = partial(double_qlearning_base, config.decouple)

  @tf.function
  def _step(self) -> Dict[str, tf.Tensor]:
    """Do a step of SGD and update the priorities."""

    # Pull out the data needed for updates/priorities.
    inputs = next(self._iterator)
    o_tm1, a_tm1, r_t, d_t, o_t, _ = inputs.data
    keys, probs = inputs.info[:2]

    if self._use_pixels:
      # Augment
      o_tm1 = self._aug_fct(tf.cast(o_tm1, dtype=tf.float32))
      o_t = self._aug_fct(tf.cast(o_t, dtype=tf.float32))
      # Encode
      o_t = self._encoder(o_t)

    with tf.GradientTape(persistent=True) as tape:

      if self._use_pixels:
        o_tm1 = self._encoder(o_tm1)

      # Evaluate the networks.
      q1_tm1, q2_tm1 = self._network(o_tm1)
      q1_t_value, q2_t_value = self._target_network(o_t)
      q1_t_selector, q2_t_selector = self._network(o_t)

      q_t_value = 0.5 * (q1_t_value + q2_t_value)

      # The rewards and discounts have to have the same type as network values.
      r_t = tf.cast(r_t, q1_tm1.dtype)
      d_t = tf.cast(d_t, q1_tm1.dtype) * tf.cast(self._discount, q1_tm1.dtype)

      if self._decouple:
        r_t = tf.tile(tf.expand_dims(r_t, -1), [1, a_tm1.shape[-1]])
        d_t = tf.tile(tf.expand_dims(d_t, -1), [1, a_tm1.shape[-1]])

      _, extra1 = self._double_qlearning(q1_tm1, a_tm1, r_t, d_t, q_t_value,
                                  q1_t_selector)
      _, extra2 = self._double_qlearning(q2_tm1, a_tm1, r_t, d_t, q_t_value,
                                  q2_t_selector)

      loss1 = losses.huber(extra1.td_error, self._huber_loss_parameter)
      loss2 = losses.huber(extra2.td_error, self._huber_loss_parameter)
      
      if self._use_double_q:
        loss = loss1 + loss2
      else:
        loss = loss1

      # Get the importance weights.
      importance_weights = 1. / probs  # [B]
      importance_weights **= self._importance_sampling_exponent
      importance_weights /= tf.reduce_max(importance_weights)

      # Reweight.
      loss *= tf.cast(importance_weights, loss.dtype)
      loss = tf.reduce_mean(loss, axis=[0])  # []

    # Do a step of SGD.
    gradients = tape.gradient(loss, self._network.trainable_variables)
    gradnorm = tf.zeros(())
    if self._clipping:
      gradients, gradnorm = tf.clip_by_global_norm(gradients, self._clip_norm)
      gradients = tuple(gradients)
    self._optimizer.apply(gradients, self._network.trainable_variables)

    if self._use_pixels:
      gradients_enc = tape.gradient(loss, self._encoder.trainable_variables)
      if self._clipping:
        gradients_enc, _ = tf.clip_by_global_norm(gradients_enc, self._clip_norm)
      self._optimizer_encoder.apply(gradients_enc, self._encoder.trainable_variables)

    del tape

    # Update the priorities in the replay buffer.
    if self._replay_client:
      priorities1 = tf.cast(tf.abs(extra1.td_error), tf.float64)
      priorities2 = tf.cast(tf.abs(extra2.td_error), tf.float64)
      if self._use_double_q:
        priorities = 0.5 * (priorities1 + priorities2)
      else:
        priorities = priorities1
      self._replay_client.update_priorities(
          table=adders.DEFAULT_PRIORITY_TABLE, keys=keys, priorities=priorities)

    # Periodically update the target network.
    if tf.math.mod(self._num_steps, self._target_update_period) == 0:
      for src, dest in zip(self._network.variables,
                           self._target_network.variables):
        dest.assign(src)
    self._num_steps.assign_add(1)

    # Report loss & statistics for logging.
    fetches = {
        'loss': loss,
        'norm': gradnorm,
    }

    return fetches

  def step(self):
    # Do a batch of SGD.
    result = self._step()

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Update our counts and record it.
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    result.update(counts)

    # Snapshot and attempt to write logs.
    if self._snapshotter is not None:
      self._snapshotter.save()
    self._logger.write(result)

  def get_variables(self, names: List[str]) -> List[np.ndarray]:
    return tf2_utils.to_numpy(self._variables)

  @property
  def state(self):
    """Returns the stateful parts of the learner for checkpointing."""
    return {
        'network': self._network,
        'target_network': self._target_network,
        'optimizer': self._optimizer,
        'num_steps': self._num_steps
    }
