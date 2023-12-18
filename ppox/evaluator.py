



def get_ff_evaluator_fn(
    env: Environment, apply_fn: ActorApply, config: dict, eval_multiplier: int = 1
) -> EvalFn:
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

    def eval_one_episode(params: FrozenDict, init_eval_state: EvalState) -> Dict:
        """Evaluate one episode. It is vectorized over the number of evaluation episodes."""

        def _env_step(eval_state: EvalState) -> EvalState:
            """Step the environment."""
            # PRNG keys.
            rng, env_state, last_timestep, step_count_, return_ = eval_state

            # Select action.
            rng, _rng = jax.random.split(rng)
            pi = apply_fn(params, last_timestep.observation)

            if config["arch"]["evaluation_greedy"]:
                action = pi.mode()
            else:
                action = pi.sample(seed=_rng)

            # Step environment.
            env_state, timestep = env.step(env_state, action)

            # Log episode metrics.
            return_ += timestep.reward
            step_count_ += 1
            eval_state = EvalState(rng, env_state, timestep, step_count_, return_)
            return eval_state

        def not_done(carry: Tuple) -> bool:
            """Check if the episode is done."""
            timestep = carry[2]
            is_not_done: bool = ~timestep.last()
            return is_not_done

        final_state = jax.lax.while_loop(not_done, _env_step, init_eval_state)

        eval_metrics = {
            "episode_return": final_state.return_,
            "episode_length": final_state.step_count_,
        }
        return eval_metrics

    def evaluator_fn(trained_params: FrozenDict, rng: chex.PRNGKey) -> ExperimentOutput[EvalState]:
        """Evaluator function."""

        # Initialise environment states and timesteps.
        n_devices = len(jax.devices())

        eval_batch = (config["arch"]["num_eval_episodes"] // n_devices) * eval_multiplier

        rng, *env_rngs = jax.random.split(rng, eval_batch + 1)
        env_states, timesteps = jax.vmap(env.reset)(
            jnp.stack(env_rngs),
        )
        # Split rngs for each core.
        rng, *step_rngs = jax.random.split(rng, eval_batch + 1)
        # Add dimension to pmap over.
        step_rngs = jnp.stack(step_rngs).reshape(eval_batch, -1)

        eval_state = EvalState(
            step_rngs, env_states, timesteps, 0, jnp.zeros_like(timesteps.reward)
        )
        eval_metrics = jax.vmap(
            eval_one_episode,
            in_axes=(None, EvalState(0, 0, 0, None, None)),
            axis_name="eval_batch",
        )(trained_params, eval_state)

        return ExperimentOutput(episodes_info=eval_metrics, learner_state=eval_state)

    return evaluator_fn


def evaluator_setup(eval_env, rng_e):







    return evaluator, absolute_metric_evaluator, eval_env, rng_e