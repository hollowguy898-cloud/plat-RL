"""
Low-dimensional observation builder for RL agents.
"""

import jax.numpy as jnp
import chex

from .config import GameConfig
from .state import EnvState


def _cast_rays(
    ox: chex.Array,
    oy: chex.Array,
    angles: chex.Array,     # (n_rays,)
    platforms,
    max_dist: float = 20.,
) -> chex.Array:
    """
    Ray-AABB distance queries for active platforms.
    Returns normalized distances in [0, 1].
    """
    n_rays = jnp.shape(angles)[0]
    # Ray directions
    dx = jnp.cos(angles)  # (n_rays,)
    dy = jnp.sin(angles)

    left  = platforms.x - platforms.w
    right = platforms.x + platforms.w
    bottom = platforms.y - platforms.h
    top   = platforms.y + platforms.h

    # Expand dimensions for batch ray-platform intersection
    dx = dx[:, None]
    dy = dy[:, None]
    left = left[None, :]
    right = right[None, :]
    bottom = bottom[None, :]
    top = top[None, :]

    inv_dx = jnp.where(dx == 0.0, 1e6, 1.0 / dx)
    inv_dy = jnp.where(dy == 0.0, 1e6, 1.0 / dy)

    tx1 = (left - ox) * inv_dx
    tx2 = (right - ox) * inv_dx
    ty1 = (bottom - oy) * inv_dy
    ty2 = (top - oy) * inv_dy

    tmin = jnp.maximum(jnp.minimum(tx1, tx2), jnp.minimum(ty1, ty2))
    tmax = jnp.minimum(jnp.maximum(tx1, tx2), jnp.maximum(ty1, ty2))

    hit = (tmax >= jnp.maximum(tmin, 0.0)) & platforms.active[None, :]
    t_hit = jnp.where(hit, tmin, max_dist * 10.0)
    nearest = jnp.min(t_hit, axis=1)
    dist = jnp.clip(nearest / max_dist, 0.0, 1.0)
    dist = jnp.where(dist == (max_dist * 10.0) / max_dist, 1.0, dist)
    return dist


def make_observation(state: EnvState, cfg: GameConfig) -> chex.Array:
    """
    Low-dimensional observation vector (~50 dims):
      [0:2]   player pos (normalised to world bounds)
      [2:4]   player vel (normalised)
      [4]     on_ground
      [5]     on_wall
      [6]     coyote_timer / coyote_frames
      [7]     jump_buffer_timer / buffer_frames
      [8]     dash_remaining / dash_max
      [9]     dash_active
      [10]    facing
      [11:21] raycasts (normalised)
      [21:24] goal relative pos + distance (normalised)
      [24:30] nearest 2 berries: rel_x, rel_y, collected (×3 each, so 6 values)
      [30:33] task: type, target_count, time_remaining
      [33:50] HK extras (hp, soul, nearest enemy ×2, nail_cd) — zeros in Celeste
    """
    p = state.player
    t = state.task
    cc = cfg.celeste

    ray_angles = jnp.linspace(0., 2. * jnp.pi, cfg.obs_raycasts, endpoint=False)
    rays = _cast_rays(p.x, p.y, ray_angles, state.platforms)

    # Relative goal vector
    gx = (t.goal_x - p.x) / cfg.celeste.world_w
    gy = (t.goal_y - p.y) / cfg.celeste.world_h
    g_dist = jnp.sqrt(gx**2 + gy**2)

    # Nearest berry (simplified — just first 2 uncollected)
    bx = (state.berries.x - p.x) / cfg.celeste.world_w
    by = (state.berries.y - p.y) / cfg.celeste.world_h
    bm = jnp.logical_not(state.berries.collected).astype(jnp.float32)

    # Nearest active enemies for Hollow Knight mode
    enemy_dx = (state.enemies.x - p.x) / cfg.celeste.world_w
    enemy_dy = (state.enemies.y - p.y) / cfg.celeste.world_h
    enemy_dist = jnp.sqrt(enemy_dx**2 + enemy_dy**2 + 1e-8)
    enemy_penalty = jnp.where(state.enemies.active, enemy_dist, 1e6)
    enemy_order = jnp.argsort(enemy_penalty)
    nearest_idx = enemy_order[:2]
    nearest_rel = jnp.concatenate([enemy_dx[nearest_idx], enemy_dy[nearest_idx]])
    nearest_active = state.enemies.active[nearest_idx].astype(jnp.float32)

    obs = jnp.concatenate([
        jnp.array([p.x / cc.world_w, p.y / cc.world_h]),                # 0:2
        jnp.array([p.vx / cc.max_run_speed, p.vy / cc.jump_vel]),       # 2:4
        jnp.array([p.on_ground, p.on_wall / 1., p.facing]),             # 4:7 (but only 2 more values)
        jnp.array([p.coyote_timer / cc.coyote_frames,
                   p.jump_buffer_timer / cc.jump_buffer_frames,
                   p.dash_remaining / cc.dash_max,
                   p.dash_active]),                                      # 7:11
        rays,                                                            # 11:21
        jnp.array([gx, gy, g_dist]),                                     # 21:24
        jnp.concatenate([bx[:2], by[:2], bm[:2]]),                      # 24:30
        jnp.array([t.task_type / 3., t.target_count / 5.,
                   t.time_remaining / (cc.max_steps * cfg.dt)]),        # 30:33
        jnp.concatenate([
            jnp.array([p.hp / 5., p.soul / cfg.hk.soul_max]),
            nearest_rel[:4],
            nearest_active,
            jnp.array([p.nail_cooldown / cfg.hk.nail_cooldown]),
            jnp.zeros(8),
        ]),
    ])
    return obs
