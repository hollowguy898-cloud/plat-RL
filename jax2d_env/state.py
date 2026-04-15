"""
Core state dataclasses (all pytree-compatible for JAX vmap/jit).
"""

from typing import NamedTuple
import chex
from jax import Array


class PlatformState(NamedTuple):
    """Padded array of platforms (static + kinematic)."""
    x:          Array   # (n_platforms,) float32 — center x
    y:          Array   # (n_platforms,) float32 — center y
    w:          Array   # (n_platforms,) float32 — half-width
    h:          Array   # (n_platforms,) float32 — half-height
    vx:         Array   # (n_platforms,) float32 — kinematic vel x
    vy:         Array   # (n_platforms,) float32 — kinematic vel y
    one_way:    Array   # (n_platforms,) bool
    hazard:     Array   # (n_platforms,) bool   — spike or acid platforms
    hazard_type:Array   # (n_platforms,) int32 — hazard type index
    active:     Array   # (n_platforms,) bool   — mask for vmap


class BerryState(NamedTuple):
    """Collectible berries/strawberries."""
    x:         Array   # (n_berries,) float32
    y:         Array   # (n_berries,) float32
    collected: Array   # (n_berries,) bool


class EnemyState(NamedTuple):
    """Hollow Knight enemies (padded)."""
    x:        Array   # (n_enemies,)
    y:        Array   # (n_enemies,)
    vx:       Array
    vy:       Array
    hp:       Array   # int
    enemy_type: Array   # (n_enemies,) int — type index from EnemyType
    ai_phase: Array   # int — FSM: 0=patrol 1=aggro 2=attack 3=recover
    active:   Array   # bool mask


class PlayerState(NamedTuple):
    """Player character state (fully JAX-compatible)."""
    x:    Array
    y:    Array
    vx:   Array
    vy:   Array

    # Celeste-specific timers / counters (all float for JAX compat)
    on_ground:          Array   # bool-as-float  0. or 1.
    on_wall:            Array   # -1. left / 0. none / 1. right
    facing:             Array   # -1. or 1.
    coyote_timer:       Array   # frames remaining
    jump_buffer_timer:  Array   # frames remaining
    dash_remaining:     Array   # uses remaining (int-as-float)
    dash_active:        Array   # bool: currently dashing
    dash_timer:         Array   # seconds remaining in dash
    dash_vx:            Array   # locked dash velocity
    dash_vy:            Array
    half_grav_active:   Array   # bool

    # HK extras (zero in Celeste mode)
    hp:                 Array   # player health (masks)
    soul:               Array   # HK soul meter
    nail_cooldown:      Array   # seconds


class TaskState(NamedTuple):
    """XLand-style task descriptor."""
    # Task type encoded as one-hot or int
    task_type:      Array   # 0=reach_goal 1=collect_k_berries 2=defeat_enemies 3=timed
    target_count:   Array   # berries / enemies needed
    collected:      Array   # running tally
    time_remaining: Array   # seconds
    goal_x:         Array
    goal_y:         Array


class EnvState(NamedTuple):
    """
    Single pytree holding all env state.
    Fully jit + vmap compatible.
    """
    player:     PlayerState
    platforms:  PlatformState
    berries:    BerryState
    enemies:    EnemyState
    task:       TaskState
    rng:        chex.PRNGKey
    step_count: Array       # int-as-float
    done:       Array       # bool
