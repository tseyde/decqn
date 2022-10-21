import sonnet as snt
from sonnet.src import base
import tensorflow as tf
from acme.tf import networks


class VisionEncoder(base.Module):
  '''Based on the architecture used by DrQ-v2.
  https://github.com/facebookresearch/drqv2
  '''

  def __init__(self, config, shape: int = 84, name='dmc_encoder'):
      super().__init__(name)

      self._shape = shape

      self._conv = snt.Sequential([
        snt.Conv2D(32, [3, 3], [2, 2]),
        tf.nn.relu,
        snt.Conv2D(32, [3, 3], [1, 1]),
        tf.nn.relu,
        snt.Conv2D(32, [3, 3], [1, 1]),
        tf.nn.relu,
        snt.Conv2D(32, [3, 3], [1, 1]),
        tf.nn.relu,
        snt.Flatten(),
    ])

      self._network = snt.Sequential([
          self._conv,
          networks.LayerNormMLP([config.layer_size_bottleneck], activate_final=True),
      ])

  def __call__(self, observations) -> tf.Tensor:
      observations = observations / 255 - 0.5
      return self._network(observations)

