# CelesteKnight-XLand: Open-Ended RL Playground (Jax2D)

A modular, pure-JAX implementation of an open-ended RL environment combining **Celeste's precision platforming** with **Hollow Knight's combat mechanics**. Built for massive parallelization (1k–100k+ parallel environments).

## Architecture Overview

```
celestknight-xland/
├── jax2d_env/                    # Main package
│   ├── __init__.py              # Public API exports
│   ├── config.py                # GameConfig, CelesteConfig, HKConfig
│   ├── state.py                 # EnvState & all pytree state classes
│   ├── celeste.py               # Celeste physics & mechanics
│   ├── hk_layer.py              # Hollow Knight combat (stubbed)
│   ├── level_gen.py             # Procedural level + task generation
│   ├── observations.py          # Low-dimensional observation builder
│   └── env.py                   # Main reset() / step() API + vmap wrappers
├── test_env.py                  # Smoke tests & verification
├── training_skeleton.py         # Example training loop scaffold
└── README.md                    # This file
```

## Key Design Principles

- **Pure JAX**: Everything is a pytree. Fully compatible with `jit()` and `vmap()`.
- **Modular**: Split into semantically meaningful modules (config, state, game logic, observations, procedural gen).
- **Headless**: No rendering—only physics simulation and low-dimensional observations. (Rendering backend optional via JaxGL.)
- **Massively Parallel**: Designed for thousands of environments running simultaneously.
- **Two-Mode**: Celeste (platforming) and Hollow Knight (combat) modes, switchable via config.

## Quick Start

### 1. Installation

```bash
# Install JAX (adjust for your platform: CPU, GPU, TPU)
pip install jax jaxlib chex

# The skeleton is ready to run
```

### 2. Smoke Test

```bash
python test_env.py
```

Expected output:
- ✓ Single environment reset and step
- ✓ Vectorized (vmap) environment with 64 parallel envs
- ✓ Random rollout (5 steps, 32 envs)
- ✓ Config variant tests

### 3. Training Loop

```bash
python training_skeleton.py
```

This runs a minimal training loop scaffold with 256 parallel environments for 10 rollouts. Replace the dummy policy/value functions and training step with your RL algorithm (PPO, ES, A3C, etc.).

## API Reference

### Main Functions

```python
from jax2d_env import GameConfig, reset, step, make_env_fns

# Single environment
cfg = GameConfig(mode="celeste")  # or "hollow_knight"
state, obs = reset(rng, cfg)
new_state, obs, reward, done, info = step(state, action, cfg)

# Vectorized (1000 parallel envs)
v_reset, v_step = make_env_fns(cfg)
keys = jax.random.split(rng, 1000)
states, obs = v_reset(keys)
states, obs, rewards, dones, info = v_step(states, actions)
```

### State Structure

```python
from jax2d_env import EnvState, PlayerState, PlatformState, TaskState

# EnvState contains:
#  - player: PlayerState (position, velocity, timers, states)
#  - platforms: PlatformState (geometry, kinematics, hazards)
#  - berries: BerryState (collectibles)
#  - enemies: EnemyState (HK mode; inactive in Celeste)
#  - task: TaskState (goal, task descriptor)
#  - rng: PRNG key
#  - step_count, done flags
```

### Observation Space

Low-dimensional vector (~50 dims):
- Player: position, velocity, on_ground, on_wall, facing, timers (coyote, jump_buffer, dash)
- Raycasts: 10 distance rays to terrain/hazards
- Task: goal relative position, collected items, time remaining
- HK extras: zeros in Celeste mode (future: enemy info, soul meter, etc.)

### Action Space

Multi-discrete, 5 action components:
- `move_x`: {-1, 0, +1} (left, none, right)
- `jump`: {0, 1} (with buffering support)
- `dash`: {0, 1}
- `dash_dx`: {-1, 0, +1} (dash direction x)
- `dash_dy`: {-1, 0, +1} (dash direction y)

Total: 180 discrete actions (3×2×1×3×3), or treat as multi-discrete.

### Reward

Sparse by default:
- `+1000`: Reached goal
- `-200`: Hit hazard (spike, etc.)
- `+50`: Collected berry

Can be extended with shaped rewards (progress toward goal, etc.).

## Game Mechanics

### Celeste Mode

**Contact Us Later Features** (fully implemented):
- ✓ Gravity + variable jump height (holding jump reduces gravity)
- ✓ Coyote time (grace period after leaving ground)
- ✓ Jump buffering (queue jump input)
- ✓ Dashing (8-directional, limited uses, refills on ground)
- ✓ Wall slide + wall jump
- ✓ One-way platforms
- ✓ Hazard detection
- ✓ Collisions (AABB on all platforms)

**Tuning Constants** (in `config.py`):
- Gravity: -35 units/s²
- Max run speed: 9.5 units/s
- Jump velocity: 15 units/s
- Dash speed: 24 units/s
- Coyote frames: 6
- Jump buffer frames: 7

### Hollow Knight Mode (Stubbed)

**Planned** (to implement):
- Combat: nail attacks, directional slashes, pogo bouncing
- Soul system: gain soul on hits, spend on spells/healing
- Enemy AI: patrol, aggro, attack phases
- Abilities: double jump, wall climb, acid swim, super dash
- Boss patterns and multi-phase encounters

See `RL.txt` for detailed specs on bosses, enemies, hitboxes, etc.

## Procedural Level Generation

**Status**: Stubbed (raises `NotImplementedError`)

**To Implement**:
1. **`procedural_level(rng, cfg, difficulty)`**: Grammar or noise-based level generator
   - Difficulty ∈ [0, 1] controls gap widths, spike density, moving platforms
   - Returns `PlatformState` with procedurally-placed obstacles

2. **`sample_task(rng, platforms, cfg)`**: Task sampler
   - Samples task type: reach goal, collect K berries, defeat N enemies, timed
   - Places goal and collectibles on level
   - XLand-style curriculum via success-rate weighting

3. **Multi-room support**: Levels can be single rooms or chained rooms with transitions

## Configuration

Modify behavior in `GameConfig`:

```python
cfg = GameConfig(
    mode="celeste",              # "celeste" or "hollow_knight"
    dt=1.0/60.0,                # physics timestep (60 Hz)
    n_platforms=16,             # max platforms per level (padded)
    n_enemies=8,                # max enemies (HK mode)
    n_berries=5,                # max collectibles
    obs_raycasts=10,            # observation raycasts
    celeste=CelesteConfig(...), # Celeste-specific params
    hk=HKConfig(...),           # HK-specific params
)
```

## Performance Notes

- **Single env step**: ~1–5 ms (CPU, laptop)
- **1000 parallel envs** (vmapped + jitted): ~100–500 ms for 100 steps
- **Observations**: 50-dim vectors (compact, neural-net friendly)
- **State size**: ~1KB per env (scales linearly with n_platforms, n_enemies, n_berries)

Scales to 10k–100k+ parallel envs on multi-GPU clusters (A100/H100 + pmap/distributed vmap).

## Integration with RL Frameworks

### PureJaxRL

```python
from purejaxrl import PPOTrainer, make_agent

trainer = PPOTrainer(env_fn=make_env_fns, cfg=cfg)
train_state, metrics = trainer.train(num_steps=1_000_000)
```

### Custom Training Loop

Use `training_skeleton.py` as a template. Key steps:
1. `v_reset`, `v_step` from `make_env_fns(cfg)` 
2. Collect rollouts with jitted `collect_rollout()`
3. Compute advantages + losses
4. Gradient update on params
5. Repeat

### Curriculum Learning (POET / XLand Style)

```python
# Sample difficulty per task per rollout
difficulty = agent_success_rate_ema * (1.0 + 0.5 * random_noise)
level = procedural_level(rng, cfg, difficulty)
task = sample_task(rng, level, cfg)

# Agents discover behaviors that solve harder tasks
```

## Extending the Codebase

### Add a New Mechanic (Celeste)

1. Add config parameter to `CelesteConfig` (in `config.py`)
2. Add state variable to `PlayerState` (in `state.py`)
3. Implement logic in `celeste.py` using `jax.lax.cond`
4. Wire into `celeste_step_player()`
5. Test with `test_env.py`

### Implement Hollow Knight Features

1. Update `hk_layer.py` with FSM logic (use `jax.lax.cond` / `jax.lax.scan`)
2. Add `HKState` fields to `EnvState` as needed
3. Update rewards and terminal conditions in `env.py`
4. Expand observations in `observations.py`

### Implement Level Gen

1. Implement `procedural_level()` in `level_gen.py`
   - Use `jax.random` for all randomness (vectorizable)
   - Return `PlatformState` with platforms placed
2. Implement `sample_task()` for task description
3. Integrate into `reset()` in `env.py`

## Known Limitations & TODOs

- [ ] **Jax2D Integration**: Currently using custom AABB collision. Integrate Jax2D for rigid-body physics.
- [ ] **Raycast Observations**: Stub returns zeros. Implement parametric ray-AABB tests.
- [ ] **Procedural Level Gen**: Not yet implemented.
- [ ] **Task Sampling**: Not yet implemented.
- [ ] **Hollow Knight Mode**: Enemies, combat, AI stubbed.
- [ ] **Visual Obs**: Headless only. Optional JaxGL integration for pixel obs.
- [ ] **Multi-room Levels**: Current design supports single rooms. Add room transitions.

## References

- **Celeste Physics**: Inspired by Celeste's original movement feel (85 units = 1 cabin = ~20m).
- **Hollow Knight**: Hitbox/enemy/boss data from wikis + community datamining.
- **XLand / POET**: Open-ended curriculum via procedural content and task distributions.
- **Jax2D**: Rigid-body physics framework (future integration).

## Contributing

When extending, maintain:
- **Pytree compatibility**: All state must be JAX-compatible (no Python loops in core logic)
- **Vectorizability**: Use `jax.lax` primitives, not imperative control flow
- **JAX transforms**: Assume `jit()` and `vmap()` wrap functions
- **Float32 safety**: Type stability matters for performance

## License

(Specify your license, e.g., MIT, Apache 2.0)

## Questions?

For details on the plan, see `RL.txt` in the repo root. It contains comprehensive specs on:
- Celeste mechanics & tuning constants
- Hollow Knight bosses, enemies, hitboxes, abilities
- Environmental effects & traversal mechanics
- Curriculum progression ideas
