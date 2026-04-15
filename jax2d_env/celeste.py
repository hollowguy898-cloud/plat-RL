"""
Celeste-specific physics and mechanics on top of Jax2D.
"""

from typing import Tuple
import jax
import jax.numpy as jnp
from jax import Array
from jax import lax
import chex

from .config import CelesteConfig, GameConfig
from .state import PlayerState, PlatformState


def _select_player(cond, true_player: PlayerState, false_player: PlayerState) -> PlayerState:
    return jax.tree_util.tree_map(
        lambda t, f: jnp.where(cond, t, f),
        true_player,
        false_player,
    )


def _apply_gravity(player: PlayerState, cfg: CelesteConfig, dt: float) -> PlayerState:
    """Standard + half-gravity at jump peak."""
    use_half = (player.half_grav_active > 0.) & (player.vy > 0.) & (player.vy < cfg.half_grav_threshold)
    grav = jnp.where(use_half, cfg.gravity * 0.5, cfg.gravity)
    new_vy = jnp.clip(player.vy + grav * dt, cfg.terminal_fall, 100.)
    return player._replace(vy=new_vy)


def _apply_horizontal(
    player: PlayerState,
    move_x: chex.Array,      # -1 / 0 / +1
    cfg: CelesteConfig,
    dt: float,
) -> PlayerState:
    """Ground / air acceleration + friction."""
    accel = jnp.where(player.on_ground > 0., cfg.ground_accel, cfg.air_accel)
    fric  = jnp.where(player.on_ground > 0., cfg.ground_friction, cfg.air_friction)

    target_vx = move_x * cfg.max_run_speed
    # Accelerate toward target
    diff = target_vx - player.vx
    delta = jnp.clip(diff, -accel * dt, accel * dt)
    friction = player.vx - jnp.sign(player.vx) * jnp.minimum(jnp.abs(player.vx), fric * dt)
    new_vx = jnp.where(move_x == 0., friction, player.vx + delta)
    facing = jnp.where(move_x != 0., move_x, player.facing)
    return player._replace(vx=new_vx, facing=facing)


def _try_jump(
    player: PlayerState,
    jump_pressed: Array,  # bool-as-float
    cfg: CelesteConfig,
    dt: float,
) -> PlayerState:
    """
    Attempt a jump using coyote time or jump buffer.
    Wall jump handled separately.
    """
    # Decrement buffer timer regardless
    buf = jnp.maximum(player.jump_buffer_timer - 1., 0.)
    # Queue buffer if pressed this frame
    buf = jnp.where(jump_pressed > 0., float(cfg.jump_buffer_frames), buf)

    can_jump = ((player.on_ground > 0.) | (player.coyote_timer > 0.)) & (buf > 0.)

    jumped = player._replace(
        vy=cfg.jump_vel,
        on_ground=0.,
        coyote_timer=0.,
        jump_buffer_timer=0.,
        half_grav_active=1.,
    )
    player = player._replace(jump_buffer_timer=buf)

    return player._replace(
        vy=jnp.where(can_jump, jumped.vy, player.vy),
        on_ground=jnp.where(can_jump, jumped.on_ground, player.on_ground),
        coyote_timer=jnp.where(can_jump, jumped.coyote_timer, player.coyote_timer),
        jump_buffer_timer=jnp.where(can_jump, jumped.jump_buffer_timer, player.jump_buffer_timer),
        half_grav_active=jnp.where(can_jump, jumped.half_grav_active, player.half_grav_active),
    )


def _try_wall_jump(
    player: PlayerState,
    jump_pressed: Array,
    cfg: CelesteConfig,
) -> PlayerState:
    """Angled kick off wall."""
    on_wall = player.on_wall != 0.
    can_wall_jump = on_wall & (jump_pressed > 0.) & (player.on_ground == 0.)
    kicked = player._replace(
        vx=-player.on_wall * 10.5,
        vy=cfg.jump_vel * 0.9,
        on_wall=0.,
        half_grav_active=1.,
    )
    return player._replace(
        vx=jnp.where(can_wall_jump, kicked.vx, player.vx),
        vy=jnp.where(can_wall_jump, kicked.vy, player.vy),
        on_wall=jnp.where(can_wall_jump, kicked.on_wall, player.on_wall),
        half_grav_active=jnp.where(can_wall_jump, kicked.half_grav_active, player.half_grav_active),
    )


def _try_dash(
    player: PlayerState,
    dash_pressed: Array,
    dash_dir_x: Array,
    dash_dir_y: Array,
    cfg: CelesteConfig,
) -> PlayerState:
    """
    8-directional dash — sets velocity to dash_speed in chosen direction.
    Refills on ground touch (handled in collision step).
    """
    can_dash = (dash_pressed > 0.) & (player.dash_remaining > 0.) & (player.dash_active == 0.)

    # Normalize direction; default to facing if no input
    raw_dx = jnp.where(dash_dir_x == 0., player.facing, dash_dir_x)
    norm   = jnp.sqrt(raw_dx**2 + dash_dir_y**2 + 1e-8)
    ndx    = raw_dx / norm
    ndy    = dash_dir_y / norm

    dashed = player._replace(
        vx=ndx * cfg.dash_speed,
        vy=ndy * cfg.dash_speed,
        dash_remaining=player.dash_remaining - 1.,
        dash_active=1.,
        dash_timer=cfg.dash_duration,
        dash_vx=ndx * cfg.dash_speed,
        dash_vy=ndy * cfg.dash_speed,
    )

    return player._replace(
        vx=jnp.where(can_dash, dashed.vx, player.vx),
        vy=jnp.where(can_dash, dashed.vy, player.vy),
        dash_remaining=jnp.where(can_dash, dashed.dash_remaining, player.dash_remaining),
        dash_active=jnp.where(can_dash, dashed.dash_active, player.dash_active),
        dash_timer=jnp.where(can_dash, dashed.dash_timer, player.dash_timer),
        dash_vx=jnp.where(can_dash, dashed.dash_vx, player.dash_vx),
        dash_vy=jnp.where(can_dash, dashed.dash_vy, player.dash_vy),
    )


def _tick_dash(player: PlayerState, cfg: CelesteConfig, dt: float) -> PlayerState:
    """Advance dash timer; clear dash_active when expired."""
    new_timer = jnp.maximum(player.dash_timer - dt, 0.)
    still_dashing = new_timer > 0.
    # While dashing, lock velocity to dash vel (no air control)
    new_vx = jnp.where(still_dashing, player.dash_vx, player.vx)
    new_vy = jnp.where(still_dashing, player.dash_vy, player.vy)
    return player._replace(
        dash_active=jnp.where(still_dashing, 1., 0.),
        dash_timer=new_timer,
        vx=new_vx, vy=new_vy,
    )


def _tick_coyote(player: PlayerState, left_ground_this_frame: float | Array) -> PlayerState:
    """Start or decrement coyote timer."""
    new_timer = jnp.where(
        left_ground_this_frame > 0.,
        player.coyote_timer,
        jnp.maximum(player.coyote_timer - 1., 0.),
    )
    return player._replace(coyote_timer=new_timer)


def _resolve_collisions_celeste(
    player: PlayerState,
    platforms: PlatformState,
    cfg: CelesteConfig,
) -> PlayerState:
    """
    Post-physics AABB collision resolution.
    One-way platforms: only collide from above + downward velocity.
    Returns updated player with on_ground / on_wall flags.
    """
    # --- AABB overlap for all active platforms (vectorised) ---
    px, py = player.x, player.y
    hw, hh = cfg.player_w / 2., cfg.player_h / 2.

    # Player AABB
    p_left  = px - hw;  p_right  = px + hw
    p_bot   = py - hh;  p_top    = py + hh

    plat_left  = platforms.x - platforms.w
    plat_right = platforms.x + platforms.w
    plat_bot   = platforms.y - platforms.h
    plat_top   = platforms.y + platforms.h

    overlap_x = (p_right > plat_left) & (p_left < plat_right)
    overlap_y = (p_top   > plat_bot ) & (p_bot  < plat_top )
    overlap   = overlap_x & overlap_y & platforms.active

    # One-way: ignore if approaching from below
    prev_bot_above = (py - hh - player.vy / 60.) >= plat_top  # crude: was above last frame
    one_way_skip   = platforms.one_way & jnp.logical_not(prev_bot_above)
    overlap        = overlap & jnp.logical_not(one_way_skip)

    # Penetration depths
    pen_top   = plat_top  - p_bot   # +ve means floor collision
    pen_bot   = p_top     - plat_bot
    pen_right = plat_right - p_left
    pen_left  = p_right   - plat_left

    # Minimum penetration axis per platform
    min_pen = jnp.minimum(jnp.minimum(pen_top, pen_bot), jnp.minimum(pen_right, pen_left))
    is_floor = (pen_top == min_pen) & (player.vy <= 0.)
    is_ceil  = (pen_bot  == min_pen) & (player.vy >= 0.)
    is_right_wall = (pen_right == min_pen) & (player.vx <= 0.)
    is_left_wall  = (pen_left  == min_pen) & (player.vx >= 0.)

    any_floor = jnp.any(overlap & is_floor)
    any_ceil  = jnp.any(overlap & is_ceil)

    # Wall detection (for wall slide / wall jump)
    any_right_wall = jnp.any(overlap & is_right_wall)
    any_left_wall  = jnp.any(overlap & is_left_wall)
    on_wall = jnp.where(any_right_wall, -1., jnp.where(any_left_wall, 1., 0.))

    # Resolve push-out (largest single axis among colliding platforms)
    push_y = jnp.sum(jnp.where(overlap & is_floor, pen_top, 0.)) \
           - jnp.sum(jnp.where(overlap & is_ceil,  pen_bot, 0.))
    push_x = jnp.sum(jnp.where(overlap & is_right_wall, -pen_right, 0.)) \
           + jnp.sum(jnp.where(overlap & is_left_wall,   pen_left,  0.))

    new_py = py + push_y
    new_px = px + push_x
    new_vy = jnp.where(any_floor | any_ceil, 0., player.vy)
    new_vx = jnp.where(any_right_wall | any_left_wall, 0., player.vx)

    # Refill dash on ground touch
    new_dash = jnp.where(any_floor, float(cfg.dash_max), player.dash_remaining)

    # Coyote: give grace frames when leaving ground
    left_ground = (player.on_ground > 0.) & jnp.logical_not(any_floor)
    new_coyote  = jnp.where(left_ground, float(cfg.coyote_frames), player.coyote_timer)

    # Wall slide
    wall_slide = jnp.logical_not(any_floor) & (on_wall != 0.) & (new_vy < 0.)
    new_vy_ws  = jnp.where(wall_slide, jnp.maximum(new_vy, cfg.wall_slide_speed), new_vy)

    return player._replace(
        x=new_px, y=new_py,
        vx=new_vx, vy=new_vy_ws,
        on_ground=jnp.where(any_floor, 1., 0.),
        on_wall=on_wall,
        coyote_timer=new_coyote,
        dash_remaining=new_dash,
    )


def _check_hazards(player: PlayerState, platforms: PlatformState, cfg: CelesteConfig) -> Array:
    """Returns 1. if player overlaps any hazard platform (spike → death)."""
    hw, hh = cfg.player_w / 2., cfg.player_h / 2.
    p_left  = player.x - hw;  p_right = player.x + hw
    p_bot   = player.y - hh;  p_top   = player.y + hh

    overlap = (
        (p_right > platforms.x - platforms.w) &
        (p_left  < platforms.x + platforms.w) &
        (p_top   > platforms.y - platforms.h) &
        (p_bot   < platforms.y + platforms.h) &
        platforms.hazard & platforms.active
    )
    return jnp.any(overlap).astype(jnp.float32)


def celeste_step_player(
    player:       PlayerState,
    platforms:    PlatformState,
    action:       Array,      # shape (5,) — see action space below
    cfg:          GameConfig,
    dt:           float,
) -> Tuple[PlayerState, Array]:
    """
    Full Celeste player update for one timestep.
    Returns (new_player, hit_hazard).

    Action vector layout (multi-discrete):
      action[0]: move_x  ∈ {-1, 0, 1}
      action[1]: jump    ∈ {0, 1}
      action[2]: dash    ∈ {0, 1}
      action[3]: dash_dx ∈ {-1, 0, 1}   (direction)
      action[4]: dash_dy ∈ {-1, 0, 1}
    """
    cc = cfg.celeste

    move_x  = action[0]
    jump    = action[1]
    dash    = action[2]
    dash_dx = action[3]
    dash_dy = action[4]

    # 1. Gravity
    p = _apply_gravity(player, cc, dt)

    # 2. Horizontal
    # Skip horizontal control while dashing
    horizontal = _apply_horizontal(p, move_x, cc, dt)
    p = _select_player(p.dash_active > 0., p, horizontal)

    # 3. Dash (takes priority over jump this frame if both pressed)
    p = _try_dash(p, dash, dash_dx, dash_dy, cc)

    # 4. Jump (coyote + buffer)
    jump_state = _try_jump(p, jump, cc, dt)
    p = _select_player(p.dash_active > 0., p, jump_state)

    # 5. Wall jump
    wall_jump_state = _try_wall_jump(p, jump, cc)
    p = _select_player(p.dash_active > 0., p, wall_jump_state)

    # 6. Integrate position (simple Euler; Jax2D scene.step() replaces this)
    p = p._replace(x=p.x + p.vx * dt, y=p.y + p.vy * dt)

    # 7. Collision resolution (post-physics)
    p = _resolve_collisions_celeste(p, platforms, cc)

    # 8. Tick dash state machine
    p = _tick_dash(p, cc, dt)

    # 9. Tick coyote
    p = _tick_coyote(p, left_ground_this_frame=0.)

    # 10. Hazard check
    hit_hazard = _check_hazards(p, platforms, cc)

    return p, hit_hazard
