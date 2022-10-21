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

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


class AttrDict(dict):

  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__


# Adpated from Dreamer
def args_type(default):
  if isinstance(default, bool):
    return lambda x: bool(['False', 'True'].index(x))
  if isinstance(default, int):
    return lambda x: float(x) if ('e' in x or '.' in x) else int(x)
  return type(default)


def encode_gif(frames, fps):
  """Taken from Dreamer agent"""
  from subprocess import Popen, PIPE
  h, w, c = frames[0].shape
  pxfmt = {1: 'gray', 3: 'rgb24'}[c]
  cmd = ' '.join([
      f'ffmpeg -y -f rawvideo -vcodec rawvideo',
      f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
      f'[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
      f'-r {fps:.02f} -f gif -'])
  proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
  for image in frames:
    proc.stdin.write(image.tostring())
  out, err = proc.communicate()
  if proc.returncode:
    raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
  del proc
  return out


from trfl import base_ops
from trfl import indexing_ops
import collections

DoubleQExtra = collections.namedtuple(
    "double_qlearning_extra", ["target", "td_error", "best_action"])

def double_qlearning(
    decouple, q_tm1, a_tm1, r_t, pcont_t, q_t_value, q_t_selector,
    name="DoubleQLearning"):
  """Implements the double Q-learning loss as a TensorFlow op.

  The loss is `0.5` times the squared difference between `q_tm1[a_tm1]` and
  the target `r_t + pcont_t * q_t_value[argmax q_t_selector]`.

  See "Double Q-learning" by van Hasselt.
  (https://papers.nips.cc/paper/3964-double-q-learning.pdf).

  Args:
    q_tm1: Tensor holding Q-values for first timestep in a batch of
      transitions, shape `[B x num_actions]`.
    a_tm1: Tensor holding action indices, shape `[B]`.
    r_t: Tensor holding rewards, shape `[B]`.
    pcont_t: Tensor holding pcontinue values, shape `[B]`.
    q_t_value: Tensor of Q-values for second timestep in a batch of transitions,
      used to estimate the value of the best action, shape `[B x num_actions]`.
    q_t_selector: Tensor of Q-values for second timestep in a batch of
      transitions used to estimate the best action, shape `[B x num_actions]`.
    name: name to prefix ops created within this op.

  Returns:
    A namedtuple with fields:

    * `loss`: a tensor containing the batch of losses, shape `[B]`.
    * `extra`: a namedtuple with fields:
        * `target`: batch of target values for `q_tm1[a_tm1]`, shape `[B]`
        * `td_error`: batch of temporal difference errors, shape `[B]`
        * `best_action`: batch of greedy actions wrt `q_t_selector`, shape `[B]`
  """
  action_shape = 1
  if decouple:
    action_shape += 1
  
  # Rank and compatibility checks.
  base_ops.wrap_rank_shape_assert(
      [[q_tm1, q_t_value, q_t_selector], [a_tm1, r_t, pcont_t]], [action_shape+1, action_shape], name)

  # double Q-learning op.
  with tf.name_scope(
      name, values=[q_tm1, a_tm1, r_t, pcont_t, q_t_value, q_t_selector]):

    # Build target and select head to update.
    best_action = tf.argmax(q_t_selector, action_shape, output_type=tf.int32)
    double_q_bootstrapped = indexing_ops.batched_index(q_t_value, best_action)
    target = tf.stop_gradient(r_t + pcont_t * double_q_bootstrapped)
    qa_tm1 = indexing_ops.batched_index(q_tm1, a_tm1)

    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
    td_error = target - qa_tm1
    if decouple:
      td_error = tf.reduce_mean(td_error, axis=[-1])

    loss = 0.5 * tf.square(td_error)
    return base_ops.LossOutput(
        loss, DoubleQExtra(target, td_error, best_action))


def epsilon_greedy(decouple, action_values, epsilon):
  """Computes an epsilon-greedy distribution over actions.

  This returns a categorical distribution over a discrete action space. It is
  assumed that the trailing dimension of `action_values` is of length A, i.e.
  the number of actions. It is also assumed that actions are 0-indexed.

  This policy does the following:

  - With probability 1 - epsilon, take the action corresponding to the highest
  action value, breaking ties uniformly at random.
  - With probability epsilon, take an action uniformly at random.

  Args:
    action_values: A Tensor of action values with any rank >= 1 and dtype float.
      Shape can be flat ([A]), batched ([B, A]), a batch of sequences
      ([T, B, A]), and so on.
    epsilon: A scalar Tensor (or Python float) with value between 0 and 1.
    legal_actions_mask: An optional one-hot tensor having the shame shape and
      dtypes as `action_values`, defining the legal actions:
      legal_actions_mask[..., a] = 1 if a is legal, 0 otherwise.
      If not provided, all actions will be considered legal and
      `tf.ones_like(action_values)`.

  Returns:
    policy: tfp.distributions.Categorical distribution representing the policy.
  """
  with tf.name_scope("epsilon_greedy", values=[action_values, epsilon]):

    # Convert inputs to Tensors if they aren't already.
    action_values = tf.convert_to_tensor(action_values)
    epsilon = tf.convert_to_tensor(epsilon, dtype=action_values.dtype)

    # We compute the action space dynamically.
    num_actions = tf.cast(tf.shape(action_values)[-1], action_values.dtype)
    if decouple:
      num_actions *= tf.cast(tf.shape(action_values)[-2], action_values.dtype)

    # Dithering action distribution.
    dither_probs = 1 / num_actions * tf.ones_like(action_values)

    # Greedy action distribution, breaking ties uniformly at random.
    max_value = tf.reduce_max(action_values, axis=-1, keepdims=True)
    greedy_probs = tf.cast(tf.equal(action_values, max_value),
                           action_values.dtype)
    greedy_probs /= tf.reduce_sum(greedy_probs, axis=-1, keepdims=True)

    # Epsilon-greedy action distribution.
    probs = epsilon * dither_probs + (1 - epsilon) * greedy_probs

    # Make the policy object.
    policy = tfp.distributions.Categorical(probs=probs)

  return policy
