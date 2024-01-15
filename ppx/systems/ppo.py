import distrax
import optax
import gymnax
import time
import hydra
import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments.environment import Environment, EnvParams, EnvState
from optax._src.base import OptState
from flax.training.train_state import TrainState
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.core.frozen_dict import FrozenDict

from typing import Dict, Sequence, Tuple, Callable, Any
from rich.pretty import pprint
from colorama import Fore, Style
from omegaconf import DictConfig, OmegaConf

from ppx.wrappers import LogWrapper, FlattenObservationWrapper
from ppx.types import Transition, LearnerState, ExperimentOutput, NetworkApply, LearnerFn
from ppx.logger import logger_setup
from ppx.evaluator import evaluator_setup


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[distrax.Categorical, jnp.ndarray]:
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


def get_learner_fn(
    env: Environment,
    env_params: EnvParams,
    network_apply_fn: NetworkApply,
    update_fn: optax.TransformUpdateFn,
    config: Dict,
) -> Callable[[LearnerState], ExperimentOutput]:
    # TRAINING LOOP
    def _update_step(learner_state: LearnerState, _: Any) -> Tuple[LearnerState, Tuple]:
        # COLLECT TRAJECTORIES
        def _env_step(
            learner_state: LearnerState, _: Any
        ) -> Tuple[LearnerState, Transition]:
            network_params, opt_states, env_state, last_obs, rng = learner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = network_apply_fn(network_params, last_obs)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["num_envs"])
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(rng_step, env_state, action, env_params)
            transition = Transition(
                done, action, value, reward, log_prob, last_obs, info
            )
            learner_state = LearnerState(
                network_params, opt_states, env_state, obsv, rng
            )
            return learner_state, transition

        # COLLECT A TRAJECTORY BATCH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config["rollout_length"]
        )

        # CALCULATE ADVANTAGE
        network_params, opt_states, env_state, obsv, rng = learner_state
        _, last_val = network_apply_fn(network_params, obsv)

        def _calculate_gae(traj_batch: Transition, last_val: jnp.ndarray) -> Tuple:
            def _get_advantages(
                gae_and_next_value: Tuple, transition: Transition
            ) -> Tuple:
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config["gamma"] * next_value * (1 - done) - value
                gae = delta + config["gamma"] * config["gae_lambda"] * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(update_state: Tuple, _: Any) -> Tuple[Tuple, Tuple]:
            def _update_minbatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                traj_batch, advantages, targets = batch_info
                network_params, opt_states = train_state

                def _loss_fn(
                    params: FrozenDict,
                    opt_states: OptState,
                    traj_batch: Transition,
                    gae: chex.Array,
                    targets: chex.Array,
                ) -> Tuple[chex.Array, Tuple]:
                    # RERUN NETWORK
                    pi, value = network_apply_fn(params, traj_batch.obs)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["clip_eps"], config["clip_eps"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8) # Advantage normalisation
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config["clip_eps"],
                            1.0 + config["clip_eps"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    total_loss = (
                        loss_actor
                        + config["vf_coef"] * value_loss
                        - config["ent_coef"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy)

                # CALCULATE LOSSSES
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                loss_info, grads = grad_fn(
                    network_params, opt_states, traj_batch, advantages, targets
                )
                # UPDATE NETWORK PARAMETERS
                network_updates, new_opt_state = update_fn(grads, opt_states)
                new_network_params = optax.apply_updates(
                    network_params, network_updates
                )

                return (new_network_params, new_opt_state), loss_info

            (
                network_params,
                opt_states,
                traj_batch,
                advantages,
                targets,
                rng,
            ) = update_state
            rng, _rng = jax.random.split(rng)
            # Batching and Shuffling
            batch_size = config["rollout_length"] * config["num_envs"]
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            # Mini-batch Updates
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [config["num_minibatches"], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            (new_network_params, new_opt_states), total_loss = jax.lax.scan(
                _update_minbatch, (network_params, opt_states), minibatches
            )
            update_state = (
                new_network_params,
                new_opt_states,
                traj_batch,
                advantages,
                targets,
                rng,
            )

            return update_state, total_loss

        # Updating Training State and Metrics:
        update_state = (
            network_params,
            opt_states,
            traj_batch,
            advantages,
            targets,
            rng,
        )
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["ppo_epochs"]
        )
        network_params, opt_states, traj_batch, advantages, targets, rng = update_state
        metric = traj_batch.info

        # Debugging mode
        if config.get("DEBUG"):

            def callback(info):
                return_values = info["episode_return"][info["returned_episode"]]
                timesteps = (
                    info["timestep"][info["returned_episode"]] * config["num_envs"]
                )
                for t in range(len(timesteps)):
                    print(
                        f"Global step={timesteps[t]}, episodic return={return_values[t]}"
                    )

            jax.debug.callback(callback, metric)

        learner_state = LearnerState(network_params, opt_states, env_state, obsv, rng)
        return learner_state, (metric, loss_info)

    def learner_fn(learner_state: LearnerState) -> ExperimentOutput:
        learner_state, (traj_info, loss_info) = jax.lax.scan(
            _update_step, learner_state, None, config["num_updates_per_eval"]
        )

        total_loss, (value_loss, loss_actor, entropy) = loss_info
        return ExperimentOutput(
            learner_state, traj_info, total_loss, value_loss, loss_actor, entropy
        )

    return learner_fn


def learner_setup(
    config: Dict, rng: chex.Array, env: Environment, env_params: EnvParams
) -> Tuple[LearnerFn, LearnerState, ActorCritic]:
    def linear_schedule(count: int) -> float:
        frac = (
            1.0
            - (count // (config["num_minibatches"] * config["ppo_epochs"]))
            / config["num_updates"]
        )
        return config["learning_rate"] * frac

    # INIT NETWORK
    network = ActorCritic(
        env.action_space(env_params).n, activation=config["ACTIVATION"]
    )

    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params = network.init(_rng, init_x)
    if config["ANNEAL_LR"]:
        optimiser = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        optimiser = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(config["learning_rate"], eps=1e-5),
        )

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["num_envs"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

    opt_states = optimiser.init(network_params)
    learner_state = LearnerState(network_params, opt_states, env_state, obsv, rng)

    learn = get_learner_fn(env, env_params, network.apply, optimiser.update, config)

    return learn, learner_state, network


def run_experiment(config: Dict) -> None:
    rng = jax.random.PRNGKey(config["seed"])
    config["num_updates"] = int(config['total_training_steps'] // (config['rollout_length'] * config['num_envs']))
    config["num_updates_per_eval"] = config["num_updates"] // config["num_evaluation"]
    steps_per_rollout = (
        config["rollout_length"] * config["num_updates_per_eval"] * config["num_envs"]
    )
    log = logger_setup(config)

    env, env_params = gymnax.make(config["env_name"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    learn, learner_state, network = learner_setup(config, rng, env, env_params)
    network_params, opt_states, env_state, obsv, rng = learner_state
    rng, _rng = jax.random.split(rng)
    rng, eval_rng = jax.random.split(rng)

    evaluator, absolute_metric_evaluator = evaluator_setup(
        env, env_params, network, config
    )

    for i in range(config["num_evaluation"]):
        start_time = time.time()
        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)
        elapsed_time = time.time() - start_time
        learner_output.episodes_info["steps_per_second"] = (
            steps_per_rollout / elapsed_time
        )
        log(learner_output, steps_per_rollout * (i + 1), trainer_metric=True)

        start_time = time.time()
        evaluator_output = evaluator(learner_state.network_params, eval_rng)
        jax.block_until_ready(evaluator_output)

        elapsed_time = time.time() - start_time
        evaluator_output.episodes_info["steps_per_second"] = (
            steps_per_rollout / elapsed_time
        )
        log(evaluator_output, steps_per_rollout * (i + 1), eval_step=i)

        learner_state = learner_output.learner_state

    start_time = time.time()
    evaluator_output = absolute_metric_evaluator(learner_state.network_params, eval_rng)
    jax.block_until_ready(evaluator_output)

    elapsed_time = time.time() - start_time
    evaluator_output.episodes_info["steps_per_second"] = (
        steps_per_rollout / elapsed_time
    )
    log(evaluator_output, steps_per_rollout * (i + 1), absolute_metric=True)


@hydra.main(
    config_path="../configs", config_name="default_ppo.yaml", version_base="1.2"
)
def hydra_entry_point(cfg: DictConfig) -> None:
    """Experiment entry point."""
    # Convert config to python dict.
    cfg: Dict = OmegaConf.to_container(cfg, resolve=True)

    print(f"{Fore.YELLOW}{Style.BRIGHT}Starting PPO experiment{Style.RESET_ALL}")
    # Run experiment.
    run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}PPO experiment completed{Style.RESET_ALL}")


if __name__ == "__main__":
    hydra_entry_point()
