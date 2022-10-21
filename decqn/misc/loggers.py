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

import os
import time
import csv
import abc
from typing import Any, Callable, Optional, Sequence, Mapping

from acme.utils.loggers import base

from acme.utils import paths 

from absl import logging

import numpy as np
import tensorflow as tf

import tensorflow.compat.v1 as tf1
from misc.utils import encode_gif


LoggingData = Mapping[str, Any]


class CustomBaseLogger(abc.ABC):
  """A logger has a `write` method."""

  @abc.abstractmethod
  def write(
    self, 
    data: LoggingData, 
    log_csv: bool = True,
    log_terminal: bool = True,
    log_tb: bool = True,
    increment: bool = True,
  ):
    """Writes `data` to destination (file, terminal, database, etc)."""


def make_custom_logger(
    directory: str,
    time_delta: float = 1.0,
    log_tensorboard=False,
    log_terminal=False,
    log_csv=False
) -> base.Logger:
  """ Logs to tensorboard and terminal """

  label = directory.split('/')[-1]

  # Create tensorboard logger w/ optional terminal and csv logger
  logger = []
  if log_terminal:
    logger.append(CustomTerminalLogger(label=label, time_delta=time_delta))
  if log_csv:
    logger.append(CustomCSVLogger(directory=directory))
  if log_tensorboard:
    logger.append(CustomTFSummaryLogger(logdir=directory))
  
  # Post-process
  logger = CustomDispatcher(logger)
  logger = CustomNoneFilter(logger)
  logger = CustomTimeFilter(logger, time_delta)
  return logger


class CustomCSVLogger(CustomBaseLogger):
  """Log to CSV file."""

  _open = open

  def __init__(self,
               directory: str,
               time_delta: float = 0.):
    self._file_path = os.path.join(directory, 'logs.csv')
    logging.info('Logging to %s', self._file_path)
    self._time = time.time()
    self._time_delta = time_delta
    self._header_exists = False

  def write(
    self, 
    data: base.LoggingData, 
    log_csv: bool = True,
    log_terminal: bool = False,
    log_tb: bool = False,
    increment: bool = True,
  ):
    """Writes a `data` into a row of comma-separated values."""

    if log_csv:
      # Only log if `time_delta` seconds have passed since last logging event.
      now = time.time()
      if now - self._time < self._time_delta:
        return
      self._time = now

      # Append row to CSV.
      with self._open(self._file_path, mode='a') as f:
        data = base.to_numpy(data)
        keys = sorted(data.keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        if not self._header_exists:
          # Only write the column headers once.
          writer.writeheader()
          self._header_exists = True
        writer.writerow(data)

  @property
  def file_path(self) -> str:
    return self._file_path


def _format_key(key: str) -> str:
  """Internal function for formatting keys."""
  return key.replace('_', ' ').title()


def _format_value(value: Any) -> str:
  """Internal function for formatting values."""
  value = base.to_numpy(value)
  if isinstance(value, (float, np.number)):
    return f'{value:0.3f}'
  return f'{value}'


def serialize(values: base.LoggingData) -> str:
  """Converts `values` to a pretty-printed string.

  This takes a dictionary `values` whose keys are strings and returns a
  a formatted string such that each key, value pair is separated by ' = ' and
  each entry is separated by ' | '. The keys are sorted alphabetically to ensure
  a consistent order, and snake case is split into words.

  For example:

      values = {'a': 1, 'b' = 2.33333333, 'c': 'hello', 'big_value': 10}
      # Returns 'A = 1 | B = 2.333 | Big Value = 10 | C = hello'
      values_string = serialize(values)

  Args:
    values: A dictionary with string keys.

  Returns:
    A formatted string.
  """
  return ' | '.join(f'{_format_key(k)} = {_format_value(v)}'
                    for k, v in sorted(values.items()))


class CustomTerminalLogger(CustomBaseLogger):
  """Logs to terminal."""

  def __init__(
      self,
      label: str = '',
      print_fn: Callable[[str], None] = print,
      serialize_fn: Callable[[base.LoggingData], str] = serialize,
      time_delta: float = 0.0,
  ):
    """Initializes the logger.

    Args:
      label: label string to use when logging.
      print_fn: function to call which acts like print.
      serialize_fn: function to call which transforms values into a str.
      time_delta: How often (in seconds) to write values. This can be used to
        minimize terminal spam, but is 0 by default---ie everything is written.
    """

    self._print_fn = print_fn
    self._serialize_fn = serialize_fn
    self._label = label and f'[{_format_key(label)}] '
    self._time = time.time()
    self._time_delta = time_delta

  def write(
    self, 
    values: base.LoggingData, 
    log_csv: bool = False,
    log_terminal: bool = True,
    log_tb: bool = False,
    increment: bool = True):
    if log_terminal:
      now = time.time()
      if (now - self._time) > self._time_delta:
        self._print_fn(f'{self._label}{self._serialize_fn(values)}')
        self._time = now
  

class CustomDispatcher(CustomBaseLogger):
  """Writes data to multiple `Logger` objects."""

  def __init__(
      self, to: Sequence[base.Logger],
      serialize_fn: Optional[Callable[[base.LoggingData], str]] = None):
    """Initialize `Dispatcher` connected to several `Logger` objects."""
    self._to = to
    self._serialize_fn = serialize_fn

  def write(
    self, 
    values: base.LoggingData, 
    log_csv: bool = True,
    log_terminal: bool = True,
    log_tb: bool = True,
    increment: bool = True):
    """Writes `values` to the underlying `Logger` objects."""
    if self._serialize_fn:
      values = self._serialize_fn(values)
    for logger in self._to:
      logger.write(
        values, 
        log_csv=log_csv, 
        log_terminal=log_terminal, 
        log_tb=log_tb,
        increment=increment
      )

  
class CustomNoneFilter(CustomBaseLogger):
  """Logger which writes to another logger, filtering any `None` values."""

  def __init__(self, to: CustomBaseLogger):
    """Initializes the logger.

    Args:
      to: A `Logger` object to which the current object will forward its results
        when `write` is called.
    """
    self._to = to

  def write(
    self, 
    values: base.LoggingData, 
    log_csv: bool = True,
    log_terminal: bool = True,
    log_tb: bool = True,
    increment: bool = True,
  ):
    values = {k: v for k, v in values.items() if v is not None}
    self._to.write(
      values, 
      log_csv=log_csv, 
      log_terminal=log_terminal, 
      log_tb=log_tb,
      increment=increment
    )


class CustomTimeFilter(CustomBaseLogger):
  """Logger which writes to another logger at a given time interval."""

  def __init__(self, to: CustomBaseLogger, time_delta: float):
    """Initializes the logger.

    Args:
      to: A `Logger` object to which the current object will forward its results
        when `write` is called.
      time_delta: How often to write values out in seconds.
    """
    self._to = to
    self._time = time.time()
    self._time_delta = time_delta

  def write(
    self, 
    values: base.LoggingData, 
    log_csv: bool = True,
    log_terminal: bool = True,
    log_tb: bool = True,
    increment: bool = True,
  ):
    now = time.time()
    if (now - self._time) > self._time_delta:
      self._to.write(
        values, 
        log_csv=log_csv, 
        log_terminal=log_terminal, 
        log_tb=log_tb,
        increment=increment
      )
      self._time = now


def _format_key_tb(key: str) -> str:
  """Internal function for formatting keys in Tensorboard format."""
  return key.title().replace('_', '')


class CustomTFSummaryLogger(CustomBaseLogger):
  """Logs to a tf.summary created in a given logdir.

  If multiple TFSummaryLogger are created with the same logdir, results will be
  categorized by labels.
  """

  def __init__(
      self,
      logdir: str,
      label: str = 'Logs',
  ):
    """Initializes the logger.

    Args:
      logdir: directory to which we should log files.
      label: label string to use when logging. Default to 'Logs'.
    """
    self._time = time.time()
    self.label = label
    self._iter = 0
    self.summary = tf.summary.create_file_writer(logdir)

  def write(
    self, 
    values: base.LoggingData, 
    log_csv: bool = True,
    log_terminal: bool = True,
    log_tb: bool = True,
    increment: bool = True,
  ):
    with self.summary.as_default():
      if log_tb:
        for key, value in values.items():
          if not tf.is_tensor(value):
            value = tf.convert_to_tensor(value)
            
          if len(value.shape)==0:
            tf.summary.scalar(
                f'{self.label}/{_format_key_tb(key)}',
                value,
                step=self._iter)

          elif len(value.shape)==4 and value.shape[0]==1:
            tf.summary.image(
                f'{self.label}/{_format_key_tb(key)}',
                value,
                step=self._iter)

          elif len(value.shape)==4:
            T, H, W, C = value.shape
            summary = tf1.Summary()
            image = tf1.Summary.Image(height=H, width=T * W, colorspace=C)
            image.encoded_image_string = encode_gif(value.numpy(), fps=15)
            summary.value.add(tag=f'{self.label}/{_format_key_tb(key)}', image=image)
            tf.summary.experimental.write_raw_pb(summary.SerializeToString(), self._iter)

        if increment:
          self._iter += 1
