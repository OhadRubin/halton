"""Hyperparameter sweeps with Halton sequences of quasi-random numbers.

Based off the algorithms described in https://arxiv.org/abs/1706.03200. Inspired
by the code in
https://github.com/google/uncertainty-baselines/blob/master/uncertainty_baselines/halton.py
written by the same authors.
"""

import collections
import functools
import itertools
import math
from typing import Any, Callable, Dict, List, Sequence, Text, Tuple, Union

from absl import logging
from numpy import random
import contextlib
import numpy as np

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
        
_SweepSequence = List[Dict[Text, Any]]
_GeneratorFn = Callable[[float], Tuple[Text, float]]


def generate_primes(n: int) -> List[int]:
  """Generate primes less than `n` (except 2) using the Sieve of Sundaram."""
  half_m1 = int((n - 2) / 2)
  sieve = [0] * (half_m1 + 1)
  for outer in range(1, half_m1 + 1):
    inner = outer
    while outer + inner + 2 * outer * inner <= half_m1:
      sieve[outer + inner + (2 * outer * inner)] = 1
      inner += 1
  return [2 * i + 1 for i in range(1, half_m1 + 1) if sieve[i] == 0]


def _is_prime(n: int) -> bool:
  """Check if `n` is a prime number."""
  return all(n % i != 0 for i in range(2, int(n**0.5) + 1)) and n != 2


def _generate_dim(num_samples: int,
                  base: int,
                  per_dim_shift: bool,
                  shuffled_seed_sequence: List[int]) -> List[float]:
  """Generate `num_samples` from a Van der Corput sequence with base `base`.

  Args:
    num_samples: int, the number of samples to generate.
    base: int, the base for the Van der Corput sequence. Must be prime.
    per_dim_shift: boolean, if true then each dim in the sequence is shifted by
      a random float (and then passed through fmod(n, 1.0) to keep in the range
      [0, 1)).
    shuffled_seed_sequence: An optional list of length `base`, used as the input
      sequence to generate samples. Useful for deterministic testing.

  Returns:
    A shuffled Van der Corput sequence of length `num_samples`, and optionally a
    shift added to each dimension.

  Raises:
    ValueError: if `base` is negative or not prime.
  """
  if base < 0 or not _is_prime(base):
    raise ValueError('Each Van der Corput sequence requires a prime `base`, '
                     f'received {base}.')

  rng = random.RandomState(base)
  if shuffled_seed_sequence is None:
    shuffled_seed_sequence = list(range(1, base))
    # np.random.RandomState uses MT19937 (see
    # https://numpy.org/devdocs/reference/random/legacy.html#numpy.random.RandomState).
    rng.shuffle(shuffled_seed_sequence)
    shuffled_seed_sequence = [0] + shuffled_seed_sequence

  # Optionally generate a random float in the range [0, 1) to shift this
  # dimension by.
  dim_shift = rng.random_sample() if per_dim_shift else None

  dim_sequence = []
  for i in range(1, num_samples + 1):
    num = 0.
    denominator = base
    while i:
      num += shuffled_seed_sequence[i % base] / denominator
      denominator *= base
      i //= base
    if per_dim_shift:
      num = math.fmod(num + dim_shift, 1.0)
    dim_sequence.append(num)
  return dim_sequence


Matrix = List[List[int]]


def generate_sequence(num_samples: int,
                      num_dims: int,
                      skip: int = 100,
                      per_dim_shift: bool = True,
                      shuffle_sequence: bool = True,
                      primes: Sequence[int] = None,
                      shuffled_seed_sequence: Matrix = None) -> Matrix:
  """Generate `num_samples` from a Halton sequence of dimension `num_dims`.

  Each dimension is generated independently from a shuffled Van der Corput
  sequence with a different base prime, and an optional shift added. The
  generated points are, by default, shuffled before returning.

  Args:
    num_samples: int, the number of samples to generate.
    num_dims: int, the number of dimensions per generated sample.
    skip: non-negative int, if positive then a sequence is generated and the
      first `skip` samples are discarded in order to avoid unwanted
      correlations.
    per_dim_shift: boolean, if true then each dim in the sequence is shifted by
      a random float (and then passed through fmod(n, 1.0) to keep in the range
      [0, 1)).
    shuffle_sequence: boolean, if true then shuffle the sequence before
      returning.
    primes: An optional sequence (of length `num_dims`) of prime numbers to use
      as the base for the Van der Corput sequence for each dimension. Useful for
      deterministic testing.
    shuffled_seed_sequence: An optional list of length `num_dims`, with each
      element being a sequence of length `primes[d]`, used as the input sequence
      to the Van der Corput sequence for each dimension. Useful for
      deterministic testing.

  Returns:
    A shuffled Halton sequence of length `num_samples`, where each sample has
    `num_dims` dimensions, and optionally a shift added to each dimension.

  Raises:
    ValueError: if `skip` is negative.
    ValueError: if `primes` is provided and not of length `num_dims`.
    ValueError: if `shuffled_seed_sequence` is provided and not of length
      `num_dims`.
    ValueError: if `shuffled_seed_sequence[d]` is provided and not of length
      `primes[d]` for any d in range(num_dims).
  """
  if skip < 0:
    raise ValueError(f'Skip must be non-negative, received: {skip}.')

  if primes is not None and len(primes) != num_dims:
    raise ValueError(
        'If passing in a sequence of primes it must be the same length as '
        f'num_dims={num_dims}, received {primes} (len {len(primes)}).')

  if shuffled_seed_sequence is not None:
    if len(shuffled_seed_sequence) != num_dims:
      raise ValueError(
          'If passing in `shuffled_seed_sequence` it must be the same length '
          f'as num_dims={num_dims}, received {shuffled_seed_sequence} '
          f'(len {len(shuffled_seed_sequence)}).')
    for d in range(num_dims):
      if len(shuffled_seed_sequence[d]) != primes[d]:
        raise ValueError(
            'If passing in `shuffled_seed_sequence` it must have element `{d}` '
            'be a sequence of length `primes[{d}]`={expected}, received '
            '{actual} (len {length})'.format(
                d=d,
                expected=primes[d],
                actual=shuffled_seed_sequence[d],
                length=shuffled_seed_sequence[d]))

  if primes is None:
    primes = []
    prime_attempts = 1
    while len(primes) < num_dims + 1:
      primes = generate_primes(1000 * prime_attempts)
      prime_attempts += 1
    primes = primes[-num_dims - 1:-1]

  # Skip the first `skip` points in the sequence because they can have unwanted
  # correlations.
  num_samples += skip

  halton_sequence = []
  for d in range(num_dims):
    if shuffled_seed_sequence is None:
      dim_shuffled_seed_sequence = None
    else:
      dim_shuffled_seed_sequence = shuffled_seed_sequence[d]
    dim_sequence = _generate_dim(
        num_samples=num_samples,
        base=primes[d],
        shuffled_seed_sequence=dim_shuffled_seed_sequence,
        per_dim_shift=per_dim_shift)
    dim_sequence = dim_sequence[skip:]
    halton_sequence.append(dim_sequence)

  # Transpose the 2-D list to be shape [num_samples, num_dims].
  halton_sequence = list(zip(*halton_sequence))

  # Shuffle the sequence.
  if shuffle_sequence:
    random.shuffle(halton_sequence)
  return halton_sequence


def _generate_double_point(name: Text,
                           min_val: float,
                           max_val: float,
                           scaling: Text,
                           halton_point: float) -> Tuple[str, float]:
  """Generate a float hyperparameter value from a Halton sequence point."""
  if scaling not in ['linear', 'log']:
    raise ValueError(
        'Only log or linear scaling is supported for floating point '
        f'parameters. Received {scaling}.')
  if scaling == 'log':
    # To transform from [0, 1] to [min_val, max_val] on a log scale we do:
    # min_val * exp(x * log(max_val / min_val)).
    rescaled_value = (
        min_val * math.exp(halton_point * math.log(max_val / min_val)))
  else:
    rescaled_value = halton_point * (max_val - min_val) + min_val
  return name, rescaled_value


def _generate_discrete_point(name: str,
                             feasible_points: Sequence[Any],
                             halton_point: float) -> Any:
  """Generate a discrete hyperparameter value from a Halton sequence point."""
  index = int(math.floor(halton_point * len(feasible_points)))
  return name, feasible_points[index]


_DiscretePoints = collections.namedtuple('_DiscretePoints', 'feasible_points')



def discrete(feasible_points: Sequence[Any]) -> _DiscretePoints:
  return _DiscretePoints(feasible_points)


def interval(start: int, end: int) -> Tuple[int, int]:
  return start, end


def loguniform(name: Text, range_endpoints: Tuple[int, int]) -> _GeneratorFn:
  min_val, max_val = range_endpoints
  return functools.partial(_generate_double_point,
                           name,
                           min_val,
                           max_val,
                           'log')


def uniform(
    name: Text, search_points: Union[_DiscretePoints,
                                     Tuple[int, int]]) -> _GeneratorFn:
  if isinstance(search_points, _DiscretePoints):
    return functools.partial(_generate_discrete_point,
                             name,
                             search_points.feasible_points)

  min_val, max_val = search_points
  return functools.partial(_generate_double_point,
                           name,
                           min_val,
                           max_val,
                           'linear')


def product(sweeps: Sequence[_SweepSequence]) -> _SweepSequence:
  """Cartesian product of a list of hyperparameter generators."""
  # A List[Dict] of hyperparameter names to sweep values.
  hyperparameter_sweep = []
  for hyperparameter_index in range(len(sweeps)):
    hyperparameter_sweep.append([])
    # Keep iterating until the iterator in sweep() ends.
    sweep_i = sweeps[hyperparameter_index]
    for point_index in range(len(sweep_i)):
      hyperparameter_name, value = list(sweep_i[point_index].items())[0]
      hyperparameter_sweep[-1].append((hyperparameter_name, value))
  return list(map(dict, itertools.product(*hyperparameter_sweep)))


def sweep(name, feasible_points: Sequence[Any]) -> _SweepSequence:
  return [{name: x} for x in feasible_points.feasible_points]


def zipit(generator_fns_or_sweeps: Sequence[Union[_GeneratorFn,
                                                  _SweepSequence]],
          length: int) -> _SweepSequence:
  """Zip together a list of hyperparameter generators.

  Args:
    generator_fns_or_sweeps: A sequence of either:
      - Generator functions that accept a Halton sequence point and return a
      quasi-ranom sample, such as those returned by halton.uniform() or
      halton.loguniform()
      - Lists of dicts with one key/value such as those returned by
      halton.sweep()
      We need to support both of these (instead of having halton.sweep() return
      a list of generator functions) so that halton.sweep() can be used directly
      as a list.
    length: the number of hyperparameter points to generate. If any of the
      elements in generator_fns_or_sweeps are sweep lists, and their length is
      less than `length`, the sweep generation will be terminated and will be
      the same length as the shortest sweep sequence.

  Returns:
    A list of dictionaries, one for each trial, with a key for each unique
    hyperparameter name from generator_fns_or_sweeps.
  """
  halton_sequence = generate_sequence(
      num_samples=length, num_dims=len(generator_fns_or_sweeps))
  # A List[Dict] of hyperparameter names to sweep values.
  hyperparameter_sweep = []
  for trial_index in range(length):
    hyperparameter_sweep.append({})
    for hyperparameter_index in range(len(generator_fns_or_sweeps)):
      halton_point = halton_sequence[trial_index][hyperparameter_index]
      if callable(generator_fns_or_sweeps[hyperparameter_index]):
        generator_fn = generator_fns_or_sweeps[hyperparameter_index]
        hyperparameter_name, value = generator_fn(halton_point)
      else:
        sweep_list = generator_fns_or_sweeps[hyperparameter_index]
        if trial_index > len(sweep_list):
          break
        hyperparameter_point = sweep_list[trial_index]
        hyperparameter_name, value = list(hyperparameter_point.items())[0]
      hyperparameter_sweep[trial_index][hyperparameter_name] = value
  return hyperparameter_sweep


_DictSearchSpace = Dict[str, Dict[str, Union[str, float, Sequence]]]
_ListSearchSpace = List[Dict[str, Union[str, float, Sequence]]]


def generate_search(search_space: Union[_DictSearchSpace, _ListSearchSpace],
                    num_trials: int) -> List[collections.namedtuple]:
  """Generate a random search with the given bounds and scaling.

  Args:linear
    search_space: A dict where the keys are the hyperparameter names, and the
      values are a dict of:
        - {"min": x, "max", y, "scaling": z} where x and y are floats and z is
        one of "linear" or "log"
        - {"feasible_points": [...]} for discrete hyperparameters.
      Alternatively, it can be a list of dict where keys are the hyperparameter
      names, and the values are hyperparameters.
    num_trials: the number of hyperparameter points to generate.

  Returns:
    A list of length `num_trials` of namedtuples, each of which has attributes
    corresponding to the given hyperparameters, and values randomly sampled.
  """
  if isinstance(search_space, dict):
    all_hyperparameter_names = list(search_space.keys())
  elif isinstance(search_space, list):
    assert len(search_space) > 0
    all_hyperparameter_names = list(search_space[0].keys())
  else:
    raise AttributeError('tuning_search_space should either be a dict or list.')

  named_tuple_class = collections.namedtuple('Hyperparameters',
                                             all_hyperparameter_names)

  if isinstance(search_space, dict):
    hyperparameter_generators = []
    for name, space in search_space.items():
      if 'feasible_points' in space:  # Discrete search space.
        generator_fn = uniform(name, discrete(space['feasible_points']))
      else:  # Continuous space.
        if space['scaling'] == 'log':
          generator_fn = loguniform(name, interval(space['min'], space['max']))
        else:
          generator_fn = uniform(name, interval(space['min'], space['max']))
      hyperparameter_generators.append(generator_fn)
    return [
        named_tuple_class(**p)
        for p in zipit(hyperparameter_generators, num_trials)
    ]
  else:
    hyperparameters = []
    updated_num_trials = min(num_trials, len(search_space))
    if num_trials != len(search_space):
      logging.info(f'--num_tuning_trials was set to {num_trials}, but '
                   f'{len(search_space)} trial(s) found in the JSON file. '
                   f'Updating --num_tuning_trials to {updated_num_trials}.')
    for trial in search_space:
      hyperparameters.append(named_tuple_class(**trial))
    return hyperparameters[:updated_num_trials]
  
import tempfile

def format_value(x):
  if isinstance(x, float):
    return '{:.2e}'.format(x)
  return x
def format_hp(hp):
  return {k:format_value(v) for k,v in hp._asdict().items()}
import json
def print_param(search_space,*,idx,num_trials,verbose=False,**kwargs):
  np.random.seed(0)
  hp_list = generate_search(search_space=search_space,num_trials=num_trials)
  if verbose:
    for x in hp_list:
      print(format_hp(x))
  hp_list = format_hp(hp_list[idx])
  with open("hp.txt","w") as f:
      json.dump(hp_list,f)

import click

def add_feasible_points(input_dict):
    return {k:{'feasible_points': v} for k,v in input_dict.items()}


import os
import tempfile


def handle_opt(options):
    options = options.split("\n")
    options = [x for x in options if x]
    options = [x.strip().split(",") for x in options]
    options = [list(x) for x in options]
    ARG_mapping = {}
    val_mapping = {}
    for opt in options:    
        ARG,arg,default_val = map(str.strip,opt)
        ARG_mapping[arg] = ARG
        val_mapping[arg] = default_val
    return ARG_mapping,val_mapping

def parse_config(config):
    v_opt,q_opt = config.strip().split("---")
    ARG_mapping,val_mapping = handle_opt(f"{v_opt}\n{q_opt}")
    wandb_args = list(handle_opt(v_opt)[0].keys())
    return ARG_mapping,val_mapping,wandb_args


def set_environment_variables(config,hp_path):
    ARG_mapping, val_mapping, wandb_args = parse_config(config)
    ARG_VARS = {}
    with open(hp_path) as f:
        state = json.load(f)
    for param_name, default_value in val_mapping.items():
        if param_name in state:
            value = state[param_name]
        else:
          value = os.environ.get(f'_{param_name}', default_value)
        if param_name in wandb_args:
            state['WANDB_NAME'] = f"{state['WANDB_NAME']}_{param_name}{value}"
        state[param_name] = value
        ARG_VARS[ARG_mapping[param_name]] = value
    ARG_VARS["WANDB_NAME"] = state['WANDB_NAME']

    # Create a temporary file named {WANDB_NAME}.sh
    with tempfile.NamedTemporaryFile(mode='w', prefix=f"{state['WANDB_NAME']}_", suffix='.sh', delete=False) as temp_file:
        temp_path = temp_file.name

        # Write the environment variables to the temporary file
        for var_name, var_value in ARG_VARS.items():
            temp_file.write(f'export {var_name}={var_value}\n')

    print(temp_path)
  
def grid_search(search_space,*,idx,**kwargs):
  sweep_list = [sweep(name,_DiscretePoints(value['feasible_points'])) for name, value in search_space.items()]
  all_hp_list = product(sweep_list)
  hp_list = all_hp_list[idx]
  with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
    temp_path = temp_file.name
    json.dump(hp_list,temp_file)
  print(temp_path)

import os

class SearchSpace:
    def __init__(self,wandb_name):
        self.SEARCH_SPACES = {}
        self.IDX = {}
        self.wandb_name = wandb_name
    def add(self,search_space,*args):
        search_space["WANDB_NAME"] = [self.wandb_name]
        for i,arg in enumerate(args):
            self.SEARCH_SPACES[arg] = search_space
            self.IDX[arg] = i
    def get(self,node_id):
        return dict(search_space=add_feasible_points(self.SEARCH_SPACES[node_id]),idx=self.IDX[node_id])

from pathlib import Path
@click.command()
@click.argument('param_name', type=str)
@click.argument('default_value',default="",type=str)
@click.argument('add_to_wandb',type=int,default=0)
def echo_param(param_name,default_value,add_to_wandb):
  if Path("hp.txt").exists():
    with open("hp.txt","r") as f:
      hp = json.load(f)
  else:
    hp = {}
  if param_name=="WANDB_NAME":
    assert default_value==""
    print(hp['WANDB_NAME'])
  else:
    assert default_value!=""
    if param_name in hp:
      value = hp[param_name]
      add_to_wandb = True
    else:
      value = os.environ.get(f'_{param_name}',default_value)
    if add_to_wandb:
      WANDB_NAME = [hp['WANDB_NAME']] if "WANDB_NAME" in hp else []
      printable_value = value
      WANDB_NAME.append(f"{param_name}{printable_value}")
      hp['WANDB_NAME'] = "_".join(WANDB_NAME)
      with open("hp.txt","w") as f:
        json.dump(hp,f)
    print(value)



if __name__=="__main__":
  echo_param()
