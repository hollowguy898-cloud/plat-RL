"""
Training loop skeleton for CelesteKnight-XLand.
This is a minimal PPO-style loop scaffold using pure JAX.

For production, integrate PureJaxRL or your preferred RL library.
"""

from typing import Any, Tuple

import jax
from jax import Array
import jax.numpy as jnp
import chex

from jax2d_env import GameConfig, make_env_fns


def init_agent_params(rng: chex.PRNGKey, obs_dim: int, action_dim: int) -> dict:
    """
    Initialize dummy agent parameters.
    Replace with your actual model (haiku, flax, etc.).
    """
    rng1, rng2 = jax.random.split(rng)
    # Dummy linear policy & value heads
    params = {
        "policy_w": jax.random.normal(rng1, (obs_dim, action_dim)) * 0.01,
        "value_w": jax.random.normal(rng2, (obs_dim, 1)) * 0.01,
    }
    return params


def policy_fn(obs: chex.Array, params: dict) -> Tuple[chex.Array, jnp.ndarray]:
    """
    Dummy policy: linear projection + softmax over actions.
    obs: (obs_dim,)
    Returns: (action_logits, log_prob)
    """
    logits = obs @ params["policy_w"]  # (action_dim,)
    probs = jax.nn.softmax(logits)
    action = jnp.argmax(logits)  # greedy
    log_prob = jnp.log(probs[action] + 1e-8)
    return action, log_prob


def value_fn(obs: chex.Array, params: dict) -> jnp.ndarray:
    """
    Dummy value function: linear regression.
    obs: (obs_dim,)
    Returns: scalar value
    """
    return (obs @ params["value_w"]).squeeze()


def collect_rollout(
    env_states: Any,
    obs: Any,
    params: dict,
    v_step,
    cfg: GameConfig,
    n_steps: int,
    rng: chex.PRNGKey,
) -> Tuple[Any, Any, Array, Array, chex.PRNGKey]:
    """
    Collect n_steps of experience from vectorized envs.
    Returns: (states, observations, rewards, dones, rng)
    """

    trajectory_rewards = []
    trajectory_dones = []
    env_state = env_states
    obs_state = obs
    local_rng = rng

    for _ in range(n_steps):
        local_rng, step_rng = jax.random.split(local_rng)
        batch_size = jnp.shape(obs_state)[0]
        actions = jax.random.randint(step_rng, (batch_size, 5), minval=-1, maxval=2).astype(jnp.float32)

        env_state, obs_state, rewards, dones, _ = v_step(env_state, actions)
        trajectory_rewards.append(rewards)
        trajectory_dones.append(dones)

    trajectory_rewards = jnp.stack(trajectory_rewards)
    trajectory_dones = jnp.stack(trajectory_dones)

    return env_state, obs_state, trajectory_rewards, trajectory_dones, local_rng


def train_step(
    params: dict,
    trajectories: dict,
    learning_rate: float = 1e-3,
) -> dict:
    """
    Dummy training step. In production, implement proper PPO/ES update.
    """
    # For now, just return params unchanged
    # TODO: compute loss, gradients, update params
    return params


def _batch_where(mask: chex.Array, new: chex.Array, old: chex.Array) -> chex.Array:
    if new.ndim == 1:
        return jnp.where(mask, new, old)
    return jnp.where(mask.reshape((-1,) + (1,) * (new.ndim - 1)), new, old)


def main():
    """Main training loop."""
    print("=" * 80)
    print("CelesteKnight-XLand Training Loop Skeleton")
    print("=" * 80)

    # =========================================================================
    # Setup
    # =========================================================================
    cfg = GameConfig()
    n_envs = 256
    n_steps_per_rollout = 50
    n_rollouts = 10

    print(f"\n[CONFIG]")
    print(f"  Game mode: {cfg.mode}")
    print(f"  Num parallel envs: {n_envs}")
    print(f"  Steps per rollout: {n_steps_per_rollout}")
    print(f"  Total rollouts: {n_rollouts}")

    # Initialize vectorized env
    v_reset, v_step = make_env_fns(cfg)
    print(f"\n[INIT] Creating {n_envs} parallel environments...")
    rng = jax.random.PRNGKey(0)
    keys = jax.random.split(rng, n_envs)
    env_states, obs = v_reset(keys)
    print(f"  ✓ obs shape: {obs.shape}")
    print(f"  ✓ env batch ready")

    # Initialize agent
    obs_dim = obs.shape[-1]
    action_dim = 5
    params = init_agent_params(jax.random.PRNGKey(1), obs_dim, action_dim)
    print(f"\n[AGENT] Initialized dummy agent")
    print(f"  obs_dim: {obs_dim}")
    print(f"  action_dim: {action_dim}")

    # =========================================================================
    # Training Loop
    # =========================================================================
    print(f"\n[TRAIN] Starting {n_rollouts} rollouts...\n")

    cumulative_reward = jnp.zeros(n_envs)

    for rollout in range(n_rollouts):
        # Collect experience
        env_states, obs, rewards, dones, rng = collect_rollout(
            env_states, obs, params, v_step, cfg, n_steps_per_rollout, rng
        )
        cumulative_reward += rewards.sum(axis=0)  # sum over time steps

        # Training step (dummy)
        trajectories = {"rewards": rewards, "dones": dones, "obs": obs}
        params = train_step(params, trajectories)

        # Stats
        mean_cumul_reward = cumulative_reward.mean()
        max_cumul_reward = cumulative_reward.max()
        episode_resets = dones.sum()

        print(f"Rollout {rollout + 1:3d} / {n_rollouts}  |  "
              f"cumul_reward (mean): {mean_cumul_reward:8.2f}  |  "
              f"cumul_reward (max): {max_cumul_reward:8.2f}  |  "
              f"resets: {episode_resets:4.0f}")

        # Reset finished envs
        last_step = dones[jnp.shape(dones)[0] - 1]
        reset_mask = last_step > 0.5  # last step dones
        if bool(reset_mask.sum() > 0):
            new_keys = jax.random.split(jax.random.PRNGKey(rollout), n_envs)
            new_env_states, new_obs = v_reset(new_keys)
            # Blend: keep running, reset finished
            env_states = jax.tree_util.tree_map(
                lambda old, new: _batch_where(reset_mask, new, old),
                env_states, new_env_states
            )
            obs = _batch_where(reset_mask, new_obs, obs)
            cumulative_reward = jnp.where(reset_mask, 0., cumulative_reward)

    print(f"\n[DONE] Training complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
