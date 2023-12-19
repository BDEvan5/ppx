from typing import NamedTuple, Optional, Dict
import jax.numpy as jnp
import chex
from flax import struct
from gymnax.environments import environment, spaces


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LearnerState(NamedTuple):
    network_params: chex.Array
    opt_state: chex.Array
    env_state: LogEnvState
    last_observation: chex.Array
    rng_key: chex.Array




class ExperimentOutput(NamedTuple):
    learner_state: LearnerState
    episodes_info: Dict[str, chex.Array]
    total_loss: Optional[chex.Array] = None
    value_loss: Optional[chex.Array] = None
    loss_actor: Optional[chex.Array] = None
    entropy: Optional[chex.Array] = None


