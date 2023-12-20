from typing import NamedTuple, Optional, Dict, Any, Callable, Tuple
from flax.core.frozen_dict import FrozenDict
import jax.numpy as jnp
import chex
from flax import struct
from gymnax.environments import environment, spaces
import distrax

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

class EvalState(NamedTuple):
    """State of the evaluator."""

    key: chex.PRNGKey
    env_state: Any
    # timestep: TimeStep
    last_observation: chex.Array
    step_count_: chex.Numeric
    return_: chex.Numeric
    done: bool


NetworkApply = Callable[[FrozenDict, chex.Array], Tuple[distrax.Distribution, chex.Array]]
LearnerFn = Callable[[LearnerState], ExperimentOutput]
EvalFn = Callable[[FrozenDict, chex.PRNGKey], ExperimentOutput]


