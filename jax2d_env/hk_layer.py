"""
Hollow Knight combat layer.
"""

from typing import Tuple
import jax
import jax.numpy as jnp
import chex

from .config import GameConfig, EnemyType, ENEMY_STATS
from .state import PlayerState, EnemyState

# Lookup arrays for simple vectorized behavior
_ENEMY_SPEED = jnp.array([ENEMY_STATS[EnemyType(i)]["speed"] for i in range(len(EnemyType))], dtype=jnp.float32)
_ENEMY_DAMAGE = jnp.array([ENEMY_STATS[EnemyType(i)]["damage"] for i in range(len(EnemyType))], dtype=jnp.float32)
_ENEMY_POGO = jnp.array([ENEMY_STATS[EnemyType(i)]["pogo_bonus"] for i in range(len(EnemyType))], dtype=jnp.float32)

GROUND_TYPES = jnp.array([
    EnemyType.CRAWLID,
    EnemyType.HUSK_WANDERER,
    EnemyType.HUSK_GUARD,
    EnemyType.MOSS_KNIGHT,
    EnemyType.BALDUR,
    EnemyType.BROKEN_VESSEL,
    EnemyType.DIRT_CARVER,
    EnemyType.TRAITOR_LORD,
    EnemyType.PRIMAL_ASPID,
], dtype=jnp.int32)
FLYER_TYPES = jnp.array([
    EnemyType.GRUZZER,
    EnemyType.VENGEFLY,
    EnemyType.ASPID_HATCHLING,
    EnemyType.MOSSFLY,
    EnemyType.CRYSTAL_HUNTER,
    EnemyType.NOSK,
    EnemyType.HIVE,
], dtype=jnp.int32)
HOPPER_TYPES = jnp.array([
    EnemyType.TIKTIK,
    EnemyType.MANTIS,
    EnemyType.PRIMAL_ASPID,
], dtype=jnp.int32)


def hk_step_enemies(
    enemies:  EnemyState,
    player:   PlayerState,
    cfg:      GameConfig,
    rng:      chex.PRNGKey,
    dt:       float,
) -> EnemyState:
    """
    Update enemies using simple type-specific movement and aggro logic.
    """
    active = enemies.active
    etype = enemies.enemy_type
    speed = _ENEMY_SPEED[etype]

    dx = player.x - enemies.x
    dy = player.y - enemies.y
    dist = jnp.sqrt(dx**2 + dy**2 + 1e-8)
    direction_x = dx / jnp.maximum(dist, 1e-4)
    direction_y = dy / jnp.maximum(dist, 1e-4)
    aggro = active & (dist < 10.0)

    is_flyer = jnp.isin(etype, FLYER_TYPES)
    is_hopper = jnp.isin(etype, HOPPER_TYPES)

    patrol_dir = jnp.sign(enemies.vx + 1e-6)
    patrol_dir = jnp.where(patrol_dir == 0.0, 1.0, patrol_dir)
    patrol_vx = patrol_dir * speed * 0.35
    patrol_vx = jnp.where(enemies.x < 3.0, jnp.abs(patrol_vx), patrol_vx)
    patrol_vx = jnp.where(enemies.x > cfg.celeste.world_w - 3.0, -jnp.abs(patrol_vx), patrol_vx)

    chase_vx = direction_x * speed * jnp.where(is_flyer, 0.75, 0.65)
    chase_vy = direction_y * speed * jnp.where(is_flyer, 0.55, 0.0)
    hop_vy = jnp.where(is_hopper & aggro & (jnp.abs(dx) < 2.0), 3.0, 0.0)

    vx = jnp.where(aggro, chase_vx, patrol_vx)
    vy = jnp.where(is_flyer & aggro, chase_vy, 0.0)
    vy = jnp.where(hop_vy > 0.0, hop_vy, vy)

    new_x = enemies.x + vx * dt
    new_y = enemies.y + vy * dt
    new_x = jnp.clip(new_x, 1.0, cfg.celeste.world_w - 1.0)
    new_y = jnp.clip(new_y, 1.0, cfg.celeste.world_h - 1.0)

    new_phase = jnp.where(aggro, 1, 0).astype(jnp.int32)

    return enemies._replace(
        x=jnp.where(active, new_x, enemies.x),
        y=jnp.where(active, new_y, enemies.y),
        vx=jnp.where(active, vx, enemies.vx),
        vy=jnp.where(active, vy, enemies.vy),
        ai_phase=jnp.where(active, new_phase, enemies.ai_phase),
    )


def hk_resolve_nail(
    player:  PlayerState,
    enemies: EnemyState,
    action:  chex.Array,
    cfg:     GameConfig,
    dt:      float,
) -> Tuple[PlayerState, EnemyState]:
    """
    Resolve nail attacks, enemy damage, and contact-based player damage.
    """
    attack = action[2] > 0.5
    attack_dx = action[3]
    attack_dy = action[4]
    attack_dx = jnp.where(attack_dx == 0.0, player.facing, attack_dx)
    attack_dir_norm = jnp.sqrt(attack_dx**2 + attack_dy**2 + 1e-8)
    attack_dx = attack_dx / attack_dir_norm
    attack_dy = attack_dy / attack_dir_norm

    raw_dx = enemies.x - player.x
    raw_dy = enemies.y - player.y
    dist = jnp.sqrt(raw_dx**2 + raw_dy**2 + 1e-8)
    in_range = dist <= cfg.hk.nail_range
    dot = (raw_dx * attack_dx + raw_dy * attack_dy) / jnp.maximum(dist, 1e-4)
    target_mask = enemies.active & in_range & (dot >= 0.3)

    hit_dist = jnp.where(target_mask, dist, 1e6)
    hit_index = jnp.argmin(hit_dist)
    hit_valid = jnp.any(target_mask)
    hit_mask = jnp.arange(cfg.n_enemies, dtype=jnp.int32) == hit_index
    hit_mask = hit_mask & target_mask

    pogo_mask = hit_mask & (attack_dy < -0.5) & (raw_dy < -0.2)
    damage_amount = cfg.hk.nail_damage * jnp.where(pogo_mask, 1.5, 1.0)

    enemy_damage = jnp.where(hit_mask, damage_amount.astype(jnp.int32), 0)
    new_hp = enemies.hp - enemy_damage
    new_active = enemies.active & (new_hp > 0)

    bounced = jnp.any(pogo_mask)
    new_vy = jnp.where(bounced, cfg.hk.pogo_bounce_vel, player.vy)
    new_soul = player.soul + cfg.hk.soul_per_hit * hit_valid

    cooldown = jnp.maximum(player.nail_cooldown - dt, 0.0)
    cooldown = jnp.where(attack & (player.nail_cooldown <= 0.0), cfg.hk.nail_cooldown, cooldown)

    contact_radius = 1.2
    contact = enemies.active & (dist < contact_radius)
    contact_damage_by_type = jnp.take(_ENEMY_DAMAGE, enemies.enemy_type)
    contact_damage = jnp.sum(contact_damage_by_type * contact.astype(jnp.float32) * 0.05)
    new_hp_player = jnp.maximum(player.hp - contact_damage, 0.0)

    new_enemies = enemies._replace(
        hp=new_hp,
        active=new_active,
    )
    new_player = player._replace(
        vy=new_vy,
        soul=jnp.minimum(new_soul, cfg.hk.soul_max),
        nail_cooldown=cooldown,
        hp=new_hp_player,
    )
    return new_player, new_enemies
