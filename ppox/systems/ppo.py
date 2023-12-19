from typing import Dict
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
import optax
import gymnax
import time 
import hydra

from rich.pretty import pprint
from colorama import Fore, Style
from omegaconf import DictConfig, OmegaConf

from ppox.wrappers import LogWrapper, FlattenObservationWrapper
from ppox.types import Transition, LearnerState
from ppox.network import ActorCritic



def get_learner_fn(env, env_params, network_apply_fn, update_fn, config):
    # TRAIN LOOP
    # train_state, env_state, last_obs, rng = runner_state
    
    def _update_step(learner_state, unused):
        # COLLECT TRAJECTORIES
        def _env_step(learner_state, unused):
            network_params, opt_states, env_state, last_obs, rng = learner_state

            # train_state, env_state, last_obs, rng = runner_state

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
            # runner_state = (train_state, env_state, obsv, rng)
            learner_state = LearnerState(network_params, opt_states, env_state, obsv, rng)
            return learner_state, transition

        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config["rollout_length"]
        )

        # CALCULATE ADVANTAGE
        network_params, opt_states, env_state, obsv, rng = learner_state
        # train_state, env_state, last_obs, rng = runner_state
        _, last_val = network_apply_fn(network_params, obsv)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config["gamma"] * next_value * (1 - done) - value
                gae = (
                    delta
                    + config["gamma"] * config["gae_lambda"] * (1 - done) * gae
                )
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
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                traj_batch, advantages, targets = batch_info
                network_params, opt_states = train_state

                def _loss_fn(params, opt_states, traj_batch, gae, targets):
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
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
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

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    network_params, opt_states, traj_batch, advantages, targets
                )
                # train_state = train_state.apply_gradients(grads=grads)
                network_updates, new_opt_state = update_fn(grads, opt_states)
                new_network_params = optax.apply_updates(network_params, network_updates)

                new_train_state = (new_network_params, new_opt_state)
                #TODO: add more detailed loss information here

                return new_train_state, total_loss

            network_params, opt_states, traj_batch, advantages, targets, rng = update_state
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
            update_state = (new_network_params, new_opt_states, traj_batch, advantages, targets, rng)
            #TODO: I do not think that update_state is the correct class to use
            return update_state, total_loss
        # Updating Training State and Metrics:
        update_state = (network_params, opt_states, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["ppo_epochs"]
        )
        # train_state = update_state[0]
        network_params, opt_states, traj_batch, advantages, targets, rng = update_state
        metric = traj_batch.info
        # rng = update_state[-1]
        
        # Debugging mode
        if config.get("DEBUG"):
            def callback(info):
                return_values = info["returned_episode_returns"][info["returned_episode"]]
                timesteps = info["timestep"][info["returned_episode"]] * config["num_envs"]
                for t in range(len(timesteps)):
                    print(f"Global step={timesteps[t]}, episodic return={return_values[t]}")
            jax.debug.callback(callback, metric)

        learner_state = LearnerState(network_params, opt_states, env_state, obsv, rng)
        # runner_state = (train_state, env_state, last_obs, rng)
        return learner_state, metric

    def learner_fn(learner_state):
        learner_state, metric = jax.lax.scan(
            _update_step, learner_state, None, config["num_updates_per_eval"]
        )

        return learner_state, metric

    return learner_fn


def learner_setup(config, rng, env, env_params):

    def linear_schedule(count):
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
    # train_state = TrainState.create(
    #     apply_fn=network.apply,
    #     params=network_params,
    #     tx=tx,
    # )

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["num_envs"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

    # runner_state = (train_state, env_state, obsv, rng)
    opt_states = optimiser.init(network_params)
    learner_state = LearnerState(network_params, opt_states, env_state, obsv, rng)

    learn = get_learner_fn(env, env_params, network.apply, optimiser.update, config)
    # _update_step = learn(env, network, runner_state, config, env_params, rng)

    return learn, learner_state #TODO: when I add a separate evaluator, then return the network so that I can use it for evaluation


def run_experiment(config):
    rng = jax.random.PRNGKey(config["seed"])
    config["num_updates_per_eval"] = config["num_updates"] // config["num_evaluation"]

    env, env_params = gymnax.make(config["env_name"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    learn, learner_state = learner_setup(config, rng, env, env_params)

    network_params, opt_states, env_state, obsv, rng = learner_state
    rng, _rng = jax.random.split(rng)

    for i in range(config["num_evaluation"]):
        start_time = time.time()
        learner_state, metric = learn(learner_state)
        jax.block_until_ready(learner_state)

        elapsed_time = time.time() - start_time
        print(f"Eval batch {i} completed in {elapsed_time} ")
        #TODO: add code to run evaluation here


    return {"runner_state": learner_state, "metrics": metric}



@hydra.main(config_path="../configs", config_name="default_ppo.yaml", version_base="1.2")
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