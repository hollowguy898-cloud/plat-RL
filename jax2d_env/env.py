"""
Main environment API: reset() and step() for Jax2D RL playground.
Fully jit + vmap compatible.
"""

from functools import partial
from typing import Tuple, Dict, Any
import jax
import jax.numpy as jnp
import chex

from .config import GameConfig
from .state import (
    EnvState, PlayerState, PlatformState, BerryState, EnemyState, TaskState
)
from .level_gen import make_static_level, procedural_level, sample_task, _sample_enemy_types
from .celeste import celeste_step_player
from .hk_layer import hk_step_enemies, hk_resolve_nail
from .observations import make_observation


def reset(rng: chex.PRNGKey, cfg: GameConfig) -> Tuple[EnvState, chex.Array]:
    """
    Initialise env state. Returns (state, obs).
    Fully jit + vmap compatible.
    """
    rng, rng_level, rng_berries, rng_task, rng_player = jax.random.split(rng, 5)

    platforms = procedural_level(rng_level, cfg, cfg.difficulty)

    berry_x = jax.random.uniform(rng_berries, (cfg.n_berries,), minval=4.0, maxval=cfg.celeste.world_w - 4.0)
    berry_y = jax.random.uniform(rng_berries, (cfg.n_berries,), minval=2.5, maxval=cfg.celeste.world_h - 5.0)
    berries = BerryState(
        x=berry_x,
        y=berry_y,
        collected=jnp.zeros(cfg.n_berries, dtype=bool),
    )

    rng, rng_enemies = jax.random.split(rng)
    if cfg.mode == "hollow_knight":
        enemy_type, enemy_hp, enemy_active = _sample_enemy_types(rng_enemies, cfg, cfg.difficulty)
        enemy_x = jax.random.uniform(rng_enemies, (cfg.n_enemies,), minval=4.0, maxval=cfg.celeste.world_w - 4.0)
        enemy_y = jax.random.uniform(rng_enemies, (cfg.n_enemies,), minval=1.5, maxval=cfg.celeste.world_h - 3.5)
        enemy_x = jnp.where(enemy_active, enemy_x, 0.0)
        enemy_y = jnp.where(enemy_active, enemy_y, 0.0)
    else:
        enemy_type = jnp.zeros(cfg.n_enemies, dtype=jnp.int32)
        enemy_hp = jnp.zeros(cfg.n_enemies, dtype=jnp.int32)
        enemy_active = jnp.zeros(cfg.n_enemies, dtype=bool)
        enemy_x = jnp.zeros(cfg.n_enemies)
        enemy_y = jnp.zeros(cfg.n_enemies)

    enemies = EnemyState(
        x=enemy_x,
        y=enemy_y,
        vx=jnp.zeros(cfg.n_enemies),
        vy=jnp.zeros(cfg.n_enemies),
        hp=enemy_hp,
        enemy_type=enemy_type,
        ai_phase=jnp.zeros(cfg.n_enemies, dtype=jnp.int32),
        active=enemy_active,
    )

    task = sample_task(rng_task, platforms, cfg)

    player = PlayerState(
        x=jnp.array(1., dtype=jnp.float32), y=jnp.array(2., dtype=jnp.float32),
        vx=jnp.array(0., dtype=jnp.float32), vy=jnp.array(0., dtype=jnp.float32),
        on_ground=jnp.array(0., dtype=jnp.float32),
        on_wall=jnp.array(0., dtype=jnp.float32),
        facing=jnp.array(1., dtype=jnp.float32),
        coyote_timer=jnp.array(0., dtype=jnp.float32),
        jump_buffer_timer=jnp.array(0., dtype=jnp.float32),
        dash_remaining=jnp.array(cfg.celeste.dash_max, dtype=jnp.float32),
        dash_active=jnp.array(0., dtype=jnp.float32),
        dash_timer=jnp.array(0., dtype=jnp.float32),
        dash_vx=jnp.array(0., dtype=jnp.float32),
        dash_vy=jnp.array(0., dtype=jnp.float32),
        half_grav_active=jnp.array(0., dtype=jnp.float32),
        hp=jnp.array(3., dtype=jnp.float32),
        soul=jnp.array(0., dtype=jnp.float32),
        nail_cooldown=jnp.array(0., dtype=jnp.float32),
    )

    state = EnvState(
        player=player, platforms=platforms, berries=berries,
        enemies=enemies, task=task, rng=rng,
        step_count=jnp.array(0., dtype=jnp.float32),
        done=jnp.array(0., dtype=jnp.float32),
    )

    obs = make_observation(state, cfg)
    return state, obs


def step(
    state:  EnvState,
    action: chex.Array,     # (5,) — see celeste_step_player
    cfg:    GameConfig,
) -> Tuple[EnvState, chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
    """
    One environment step. Returns (new_state, obs, reward, done, info).
    Fully jit + vmap compatible.
    """
    dt = cfg.dt

    # -----------------------------------------------------------------------
    # 1. Celeste player physics
    # -----------------------------------------------------------------------
    new_player, hit_hazard = celeste_step_player(
        state.player, state.platforms, action, cfg, dt
    )

    # -----------------------------------------------------------------------
    # 2. Kinematic platforms update (TODO: integrate Jax2D scene.step here)
    # -----------------------------------------------------------------------
    new_platforms = state.platforms._replace(
        x=state.platforms.x + state.platforms.vx * dt,
        y=state.platforms.y + state.platforms.vy * dt,
    )

    # -----------------------------------------------------------------------
    # 2b. Hollow Knight enemy update
    # -----------------------------------------------------------------------
    rng, next_rng = jax.random.split(state.rng)
    if cfg.mode == "hollow_knight":
        new_enemies = hk_step_enemies(state.enemies, new_player, cfg, rng, dt)
        new_player, new_enemies = hk_resolve_nail(new_player, new_enemies, action, cfg, dt)
    else:
        new_enemies = state.enemies

    enemy_kills = jnp.sum((state.enemies.active & jnp.logical_not(new_enemies.active)).astype(jnp.float32))

    # -----------------------------------------------------------------------
    # 3. Berry collection
    # -----------------------------------------------------------------------
    p = new_player
    hw, hh = cfg.celeste.player_w / 2., cfg.celeste.player_h / 2.
    berry_collected = jnp.logical_and(
        jnp.logical_and(
            jnp.abs(state.berries.x - p.x) < hw + 0.5,
            jnp.abs(state.berries.y - p.y) < hh + 0.5,
        ),
        jnp.logical_not(state.berries.collected),
    )
    new_berries = state.berries._replace(
        collected=jnp.logical_or(state.berries.collected, berry_collected)
    )
    n_collected = jnp.sum(berry_collected).astype(jnp.float32)

    # -----------------------------------------------------------------------
    # 4. Goal check
    # -----------------------------------------------------------------------
    goal_dist = jnp.sqrt(
        (p.x - state.task.goal_x)**2 + (p.y - state.task.goal_y)**2
    )
    reached_goal = (goal_dist < 1.5).astype(jnp.float32)

    # -----------------------------------------------------------------------
    # 5. Reward shaping
    # -----------------------------------------------------------------------
    reward = (
          reached_goal      * 1000.
        - hit_hazard        * 200.
        + n_collected       * 50.
        + enemy_kills       * 100.
    )

    # -----------------------------------------------------------------------
    # 6. Task state update
    # -----------------------------------------------------------------------
    defeated_enemies = jnp.sum((state.enemies.active & jnp.logical_not(new_enemies.active)).astype(jnp.float32))
    new_task = state.task._replace(
        collected      = state.task.collected + n_collected + jnp.where(state.task.task_type == 2.0, defeated_enemies, 0.0),
        time_remaining = jnp.maximum(state.task.time_remaining - dt, 0.),
    )

    # -----------------------------------------------------------------------
    # 7. Done conditions
    # -----------------------------------------------------------------------
    max_steps = cfg.hk.max_steps if cfg.mode == "hollow_knight" else cfg.celeste.max_steps
    timeout   = (state.step_count + 1.) >= max_steps
    died = (new_player.hp <= 0.).astype(jnp.float32)
    done = (reached_goal > 0.) | (hit_hazard > 0.) | (died > 0.) | timeout

    # -----------------------------------------------------------------------
    # 8. Assemble new state
    # -----------------------------------------------------------------------
    new_state = EnvState(
        player    = new_player,
        platforms = new_platforms,
        berries   = new_berries,
        enemies   = new_enemies,
        task      = new_task,
        rng       = next_rng,
        step_count= state.step_count + 1.,
        done      = done.astype(jnp.float32),
    )

    obs  = make_observation(new_state, cfg)
    info = {
        "reached_goal": reached_goal,
        "hit_hazard": hit_hazard,
        "enemy_kills": enemy_kills,
        "player_hp": new_player.hp,
    }

    return new_state, obs, reward, done.astype(jnp.float32), info


def make_env_fns(cfg: GameConfig):
    """
    Returns (v_reset, v_step) vmapped over leading batch axis.

    Usage:
        v_reset, v_step = make_env_fns(cfg)
        keys = jax.random.split(jax.random.PRNGKey(0), N_ENVS)
        states, obs = v_reset(keys)           # (N, ...) batched
        states, obs, rew, done, info = v_step(states, actions)
    """
    _reset = jax.jit(partial(reset, cfg=cfg))
    _step  = jax.jit(partial(step,  cfg=cfg))

    v_reset = jax.vmap(_reset)
    v_step  = jax.vmap(_step)

    return v_reset, v_step
