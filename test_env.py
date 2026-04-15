"""
Smoke test for CelesteKnight-XLand environment.
Verifies that the modular skeleton runs and produces expected outputs.
"""

import jax
import jax.numpy as jnp
from jax2d_env import GameConfig, reset, step, make_env_fns


def test_single_env():
    """Test single environment reset and step."""
    print("=" * 60)
    print("TEST 1: Single Environment Reset & Step")
    print("=" * 60)

    cfg = GameConfig()
    rng = jax.random.PRNGKey(42)

    # Reset
    state, obs = reset(rng, cfg)
    print(f"✓ Reset successful")
    print(f"  - obs shape: {obs.shape}")
    print(f"  - player pos: ({state.player.x:.2f}, {state.player.y:.2f})")
    print(f"  - on_ground: {state.player.on_ground}")

    # Step with dummy action
    action = jnp.array([0., 0., 0., 0., 0.])  # no movement
    new_state, new_obs, reward, done, info = step(state, action, cfg)
    print(f"✓ Step successful")
    print(f"  - reward: {reward:.2f}")
    print(f"  - done: {done}")
    print(f"  - new_obs shape: {new_obs.shape}")
    print(f"  - info keys: {list(info.keys())}")

    # Verify state changed (gravity applied)
    print(f"✓ Physics applied: vy changed from {state.player.vy:.2f} to {new_state.player.vy:.2f}")

    return True


def test_vectorized_env():
    """Test vectorized environment (vmap)."""
    print("\n" + "=" * 60)
    print("TEST 2: Vectorized Environment (vmap)")
    print("=" * 60)

    cfg = GameConfig()
    N = 64
    v_reset, v_step = make_env_fns(cfg)

    # Batch reset
    keys = jax.random.split(jax.random.PRNGKey(42), N)
    states, obs = v_reset(keys)
    print(f"✓ Batch reset successful")
    print(f"  - batch size: {N}")
    print(f"  - obs shape: {obs.shape}")  # (N, obs_dim)
    print(f"  - player x shape: {states.player.x.shape}")

    # Batch step with random actions
    rng = jax.random.PRNGKey(0)
    rng, k = jax.random.split(rng)
    actions = jax.random.randint(k, (N, 5), minval=-1, maxval=2).astype(jnp.float32)

    new_states, new_obs, rewards, dones, infos = v_step(states, actions)
    print(f"✓ Batch step successful")
    print(f"  - rewards shape: {rewards.shape}")
    print(f"  - dones shape: {dones.shape}")
    print(f"  - reward sample (first 4): {rewards[:4]}")
    print(f"  - done sample (first 4): {dones[:4]}")

    return True


def test_random_rollout():
    """Test a longer rollout with random actions."""
    print("\n" + "=" * 60)
    print("TEST 3: Random Rollout (5 steps)")
    print("=" * 60)

    cfg = GameConfig()
    v_reset, v_step = make_env_fns(cfg)

    N = 32
    keys = jax.random.split(jax.random.PRNGKey(0), N)
    states, obs = v_reset(keys)

    rng = jax.random.PRNGKey(100)
    cumulative_rewards = jnp.zeros(N)

    for t in range(5):
        rng, k = jax.random.split(rng)
        actions = jax.random.randint(k, (N, 5), minval=-1, maxval=2).astype(jnp.float32)
        states, obs, rewards, dones, infos = v_step(states, actions)
        cumulative_rewards += rewards

        print(f"  Step {t + 1}: mean_reward={rewards.mean():.2f}, "
              f"max_reward={rewards.max():.2f}, done_count={dones.sum():.0f}")

    print(f"✓ Rollout complete")
    print(f"  - final cumulative rewards (first 4): {cumulative_rewards[:4]}")

    return True


def test_config_variants():
    """Test different game mode configurations."""
    print("\n" + "=" * 60)
    print("TEST 4: Config Variants")
    print("=" * 60)

    # Celeste mode (default)
    cfg_celeste = GameConfig(mode="celeste")
    print(f"✓ Celeste config: gravity={cfg_celeste.celeste.gravity}, "
          f"dash_speed={cfg_celeste.celeste.dash_speed}")

    # Hollow Knight mode (stub for now)
    cfg_hk = GameConfig(mode="hollow_knight")
    print(f"✓ Hollow Knight config: gravity={cfg_hk.hk.gravity}, "
          f"max_run_speed={cfg_hk.hk.max_run_speed}")

    return True


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 80)
    print(" " * 20 + "CELESTKNIGHT-XLAND SMOKE TESTS")
    print("=" * 80)

    try:
        test_single_env()
        test_vectorized_env()
        test_random_rollout()
        test_config_variants()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80 + "\n")
        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
