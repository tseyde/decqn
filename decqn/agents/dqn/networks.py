import sonnet as snt
from acme.tf import networks

import tensorflow as tf

uniform_initializer = tf.initializers.VarianceScaling(
    distribution='uniform', mode='fan_out', scale=0.333)


def get_output_dimensions(config, action_spec):
  out_dim = None
  if config.decouple:
    out_dim = config.num_bins * action_spec.shape[0]
  else:
    out_dim = action_spec.maximum+1
  assert out_dim
  return out_dim


def get_critic_network(config, action_spec):

  out_dim = get_output_dimensions(config, action_spec)

  layers = []

  if config.use_pixels:
    assert not config.use_residual
    layers.append(snt.nets.MLP([*config.layer_size_network, out_dim]))

  else:
    layers.append(snt.Flatten())
    if config.use_residual:
      layers.extend(
        [
          networks.LayerNormAndResidualMLP(config.layer_size_network[0], num_blocks=1),
          tf.nn.elu,
          snt.nets.MLP([out_dim], w_init=uniform_initializer,),
        ]
      )
    else:
      layers.append(networks.LayerNormMLP([*config.layer_size_network, out_dim]))
  
  if config.decouple:
    layers.append(snt.Reshape(output_shape=(-1, config.num_bins)))

  return  snt.Sequential(layers)


class CriticDQN(snt.Module):

  def __init__(self, config, action_spec):
    super().__init__(name='critic_network')

    self._use_double_q = config.use_double_q
    self._use_pixels = config.use_pixels

    self._q1_network = get_critic_network(config, action_spec)
    self._q2_network = get_critic_network(config, action_spec)

  def __call__(self, inputs):
    q1 = self._q1_network(inputs)
    q2 = self._q2_network(inputs)

    if self._use_double_q:
      return q1, q2
    else:
      return q1, q1
