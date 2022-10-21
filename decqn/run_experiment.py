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

"""Example for running DecQN on DeepMind Control Suite tasks."""

import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import datetime
import argparse

import tensorflow as tf
import numpy as np

from dm_control import suite
import dm_env
from acme import specs, wrappers
from acme.utils import paths

from agents import dqn
from misc.utils import args_type
from misc.loggers import make_custom_logger
from environment_loop import EnvironmentLoop
from config_default import ConfigDefault


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def make_network(
    action_spec: specs.BoundedArray,
    config: argparse.Namespace,
):
  """Creates Q-network and maybe encoder used by the agent."""

  from agents.dqn.networks import CriticDQN
  network = CriticDQN(config, action_spec)
  encoder = None
  if config.use_pixels:
    from misc.networks import VisionEncoder
    encoder = VisionEncoder(config, shape=config.num_pixels)
  
  return {
      'network': network,
      'encoder': encoder,
  }


def generate_directories(config):
  ADD_UID = False
  config.environ_dir = paths.process_path(config.logdir, 'environment_loop', add_uid=ADD_UID)
  config.eval_dir = paths.process_path(config.logdir, 'evaluation_loop', add_uid=ADD_UID)
  config.learner_dir = paths.process_path(config.logdir, 'learner_loop', add_uid=ADD_UID)
  config.image_dir = paths.process_path(config.logdir, 'image_loop', add_uid=ADD_UID)
  if config.log_video:
    config.video_dir = paths.process_path(config.logdir, 'videos', add_uid=ADD_UID)
  else:
    config.video_dir = None
  config.episode_dir = paths.process_path(config.logdir, 'episodes', add_uid=ADD_UID)

  return config


def process_config(config):
  if "ball_in_cup" in config.task:
    domain_name, task_name = config.task.rsplit('_', 1)
  elif "point_mass" in config.task or "CMU" in config.task:
    domain_name, task_name = config.task.rsplit('_', 1)
  else:
    domain_name, task_name = config.task.split('_', 1)

  config.domain_name = domain_name
  config.task_name = task_name

  logdir = config.logdir + '/' 
  if config.device == 'local' and not task_name in config.logdir:
    dateid = datetime.datetime.now().strftime("%Y%m%d")
    timeid = datetime.datetime.now().strftime("%H%M%S")
    logdir = logdir + config.algorithm + '/' + config.task + '/' + dateid + '/' + timeid + '/'
  else:
    config.log_video = False
  config.logdir = logdir

  return config


def save_config_file(config, environ_dir):
  directory = environ_dir.replace('/environment_loop', '')
  with open(str(directory) + '/config.txt', 'w') as file:
    for (key, value) in config.__dict__.items():
      if isinstance(value, str):
        file.write(key + '=\'' + str(value) + '\'\n')
      else:
        file.write(key + '=' + str(value) + '\n')


def main(config):

  config.decouple = True if 'decqn' in config.algorithm else False
  config.use_pixels = True if 'vis' in config.algorithm else False

  if 'vis' in config.algorithm and config.use_residual:
    config.use_residual = False
    print('Residual architecture not applicable to pixel-based control.')

  if config.debug:
    print("-------------------------------------")
    print("------- RUNNING IN DEBUG MODE -------")
    print("-------------------------------------")
    tf.config.experimental_run_functions_eagerly(True)

  os.environ['MUJOCO_GL'] = 'egl' if config.device == 'local' else "osmesa"

  # Config
  config.log_video = True
  config = process_config(config)
  config = generate_directories(config)
  save_config_file(config, config.environ_dir)

  def make_environment(domain_name: str = 'walker',
                     task_name: str = 'walk') -> dm_env.Environment:
    """Creates a control suite environment."""
    import random
    random.seed(config.seed)

    
    environment = suite.load(domain_name, task_name, task_kwargs=dict(random=config.seed))
    config.original_action_spec = environment.action_spec().replace(dtype=np.float32)
    if config.decouple:
        from wrappers import DecoupledDiscreteWrapper
        environment = DecoupledDiscreteWrapper(environment, config)
    else:
        from wrappers import DiscreteWrapper
        environment = DiscreteWrapper(environment, config)
    if config.use_pixels:
        from wrappers import DMCVisionWrapper
        environment = DMCVisionWrapper(
            environment, 
            domain_name, 
            action_repeat=config.action_repeat,
            size=(config.num_pixels, config.num_pixels)
          )
    else:
        from wrappers import CustomConcatObservationWrapper
        environment = CustomConcatObservationWrapper(environment)

    environment = wrappers.SinglePrecisionWrapper(environment)
    return environment

  # Create an environment and grab the spec.
  environment = make_environment(domain_name=config.domain_name, task_name=config.task_name)
  environment_spec = specs.make_environment_spec(environment)

  networks = make_network(environment_spec.actions, config)

  # Construct the agent.
  agent_logger = make_custom_logger(
    directory=config.learner_dir,
    log_tensorboard=True,
    log_terminal=False,
    log_csv=True
  )

  agent = dqn.DQN(
      config=config,
      environment_spec=environment_spec, 
      network=networks['network'],
      encoder=networks['encoder'],
      logger=agent_logger,
  )

  # Run the environment loop.
  env_logger = make_custom_logger(
    directory=config.environ_dir,
    log_tensorboard=True,
    log_terminal=True,
    log_csv=True,
  )
  eval_logger = make_custom_logger(
    directory=config.eval_dir,
    log_tensorboard=True,
    log_terminal=False,
    log_csv=True,
  )
  if config.video_dir:
    video_logger = make_custom_logger(
        directory=config.video_dir,
        log_tensorboard=True,
        log_terminal=False, 
        log_csv=False
      )
  else:
    video_logger = None
  loop = EnvironmentLoop(
    environment, 
    agent, 
    domain=config.domain_name, 
    logger=env_logger, 
    eval_logger=eval_logger,
    video_logger=video_logger, 
    eval_every=20,
    eval_video=False)
  loop.run(num_episodes=config.num_episodes)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  config = ConfigDefault().get_config()

  config.debug = False

  # ### Environment ###
  config.task = 'cartpole_swingup'
  config.num_episodes = 1000
  config.seed = 0

  # ### Algorithm ###
  config.algorithm = 'decqnvis'
  config.num_bins = 2

  config.use_double_q = True
  config.use_residual = True
  config.adder_n_step = 3
  config.batch_size = 256
  config.epsilon = 0.10

  if 'vis' in config.algorithm:
    config.layer_size_network = [1024, 1024]
  else:
    config.layer_size_network = [512, 512]


  for key, value in config.items():
    if type(value) is list:
      parser.add_argument(f'--{key}', nargs='+', type=args_type(value[0]), default=value)
    else:
      parser.add_argument(f'--{key}', type=args_type(value), default=value)
  
  main(parser.parse_args())