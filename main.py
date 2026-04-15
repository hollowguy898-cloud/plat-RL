#!/usr/bin/env python3
"""
CelesteKnight-XLand Main Entry Point

Usage:
  python main.py test           # Run smoke tests
  python main.py train          # Run training skeleton
  python main.py demo [steps]   # Run interactive demo
"""

import sys
import argparse
import jax
import jax.numpy as jnp

from jax2d_env import GameConfig, reset, step, make_env_fns


def cmd_test():
    """Run smoke tests."""
    print("Running smoke tests...")
    print("(Run: python test_env.py for detailed output)\n")

    import subprocess
    result = subprocess.run([sys.executable, "test_env.py"])
    return result.returncode


def cmd_train():
    """Run training skeleton."""
    print("Running training loop skeleton...")
    print("(Run: python training_skeleton.py for detailed output)\n")

    import subprocess
    result = subprocess.run([sys.executable, "training_skeleton.py"])
    return result.returncode


def cmd_demo(steps: int = 10):
    """Run a quick interactive demo."""
    print("=" * 80)
    print("CelesteKnight-XLand Demo")
    print("=" * 80)
    print(f"Running {steps} steps in 64 parallel environments\n")

    cfg = GameConfig(mode="celeste")
    v_reset, v_step = make_env_fns(cfg)

    N = 64
    keys = jax.random.split(jax.random.PRNGKey(0), N)
    states, obs = v_reset(keys)

    print(f"✓ Initialized {N} parallel environments")
    print(f"  - obs shape: {obs.shape}")
    print(f"  - player positions (first 4): {states.player.x[:4]}")

    rng = jax.random.PRNGKey(100)
    total_rewards = jnp.zeros(N)

    print(f"\nRunning {steps} steps...\n")

    for t in range(steps):
        rng, k = jax.random.split(rng)
        actions = jax.random.randint(k, (N, 5), minval=-1, maxval=2).astype(jnp.float32)

        states, obs, rewards, dones, info = v_step(states, actions)
        total_rewards += rewards

        # Reset finished episodes
        reset_mask = dones > 0.5
        if reset_mask.sum() > 0:
            new_keys = jax.random.split(jax.random.PRNGKey(t), N)
            new_states, new_obs = v_reset(new_keys)
            states = jax.tree_map(
                lambda old, new: jnp.where(reset_mask[:, None], new, old),
                states, new_states
            )
            obs = jnp.where(reset_mask[:, None], new_obs, obs)
            total_rewards = jnp.where(reset_mask, 0., total_rewards)

        print(f"Step {t+1:2d} | mean_reward: {rewards.mean():7.2f} | "
              f"max_reward: {rewards.max():7.2f} | resets: {dones.sum():3.0f}")

    print(f"\n✓ Demo complete!")
    print(f"  Final avg cumulative reward: {total_rewards.mean():.2f}")
    print("=" * 80 + "\n")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="CelesteKnight-XLand RL Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py test              # Run smoke tests
  python main.py train             # Run training loop
  python main.py demo              # Run 10-step demo
  python main.py demo --steps 100  # Run 100-step demo
        """,
    )

    parser.add_argument(
        "command",
        choices=["test", "train", "demo"],
        help="Command to run",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of steps for demo (default: 10)",
    )

    args = parser.parse_args()

    if args.command == "test":
        return cmd_test()
    elif args.command == "train":
        return cmd_train()
    elif args.command == "demo":
        return cmd_demo(args.steps)


if __name__ == "__main__":
    sys.exit(main())
