import jax
import jax.numpy as jnp
from ppx.types import ExperimentOutput, EvalState



def get_ff_evaluator_fn(env, env_params, apply_fn, config: dict, eval_multiplier: int = 1):
    """Get the evaluator function for feedforward networks.

    Args:
        env (Environment): An evironment isntance for evaluation.
        apply_fn (callable): Network forward pass method.
        config (dict): Experiment configuration.
        eval_multiplier (int): A scalar that will increase the number of evaluation
            episodes by a fixed factor. The reason for the increase is to enable the
            computation of the `absolute metric` which is a metric computed and the end
            of training by rolling out the policy which obtained the greatest evaluation
            performance during training for 10 times more episodes than were used at a
            single evaluation step.
    """

    def eval_one_episode(params, init_eval_state):
        """Evaluate one episode. It is vectorized over the number of evaluation episodes."""

        def _env_step(eval_state):
            """Step the environment."""
            # PRNG keys.
            rng, env_state, last_observation, step_count_, return_, done = eval_state

            # Select action.
            rng, _rng = jax.random.split(rng)
            rng, rng_step = jax.random.split(rng)
            pi, _values = apply_fn(params, last_observation)

            if config["evaluation_greedy"]:
                action = pi.mode()
            else:
                action = pi.sample(seed=_rng)

            # Step environment.
            last_observation, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)

            # Log episode metrics.
            return_ += reward
            step_count_ += 1
            eval_state = EvalState(rng, env_state, last_observation, step_count_, return_, done)
            return eval_state

        def not_done(carry):
            """Check if the episode is done."""
            done = carry[-1]
            is_not_done: bool = ~done
            return is_not_done

        final_state = jax.lax.while_loop(not_done, _env_step, init_eval_state)

        eval_metrics = {
            "episode_return": final_state.return_,
            "episode_length": final_state.step_count_,
        }
        return eval_metrics

    def evaluator_fn(trained_params, rng) -> ExperimentOutput[EvalState]:
        """Evaluator function."""

        # Initialise environment states and timesteps.
        eval_batch = (config["num_eval_episodes"]) * eval_multiplier

        env_rngs = jax.random.split(rng, eval_batch)
        obsvs, env_states = jax.vmap(env.reset, in_axes=(0, None))(env_rngs, env_params)

        step_rngs = jax.random.split(rng, eval_batch)
        eval_state = EvalState(
            step_rngs, env_states, obsvs, 0, jnp.zeros_like(env_states.env_state.time), False
        )
        eval_metrics = jax.vmap(
            eval_one_episode,
            in_axes=(None, EvalState(0, 0, 0, None, None, None)),
            axis_name="eval_batch",
        )(trained_params, eval_state)

        return ExperimentOutput(episodes_info=eval_metrics, learner_state=eval_state)

    return evaluator_fn


def evaluator_setup(env, env_params, network, config):
    vmapped_eval_network_apply_fn = jax.vmap(
            network.apply,
            in_axes=(None, 0),
        )
    vmapped_eval_network_apply_fn = network.apply
    evaluator = get_ff_evaluator_fn(
        env, env_params,
        vmapped_eval_network_apply_fn,
        config,
    )
    absolute_metric_evaluator = get_ff_evaluator_fn(
        env, env_params,
        vmapped_eval_network_apply_fn,
        config,
        10,
    )

    return evaluator, absolute_metric_evaluator


