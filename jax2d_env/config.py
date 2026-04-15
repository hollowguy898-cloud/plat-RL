"""
Game configuration classes for Celeste and Hollow Knight modes.
Includes all constants, enemy stats, boss data, and environment effects.
"""

from typing import NamedTuple, Dict
import enum


# ============================================================================
# CELESTE CONFIGURATION
# ============================================================================

class CelesteConfig(NamedTuple):
    """Celeste-specific physics and mechanics parameters."""
    # World (1 unit ≈ 1 Celeste tile)
    gravity:            float = -35.0       # units/s²
    terminal_fall:      float = -20.0       # units/s (normal)
    terminal_fall_fast: float = -24.0       # units/s (holding down)
    fast_fall_mult:     float = 1.3

    # Horizontal movement
    max_run_speed:      float = 9.5         # units/s
    ground_accel:       float = 18.0        # units/s² (snappy)
    air_accel:          float = 11.0
    ground_friction:    float = 24.0        # decel when no input (instant stop)
    air_friction:       float = 6.0

    # Jumping (variable height via half-gravity)
    jump_vel:           float = 15.0        # units/s (initial velocity)
    jump_height_short:  float = 4.5         # short tap (~0.3s hold)
    jump_height_tall:   float = 7.5         # full hold
    coyote_frames:      int   = 6           # grace period after leaving ground (~0.1s)
    jump_buffer_frames: int   = 7           # queue jump input (~0.12s)
    half_grav_threshold: float = 4.0        # upward vel below which half-grav applies
    half_grav_mult:     float = 0.5         # gravity reduction while holding jump

    # Dashing (the main mechanic)
    dash_speed:         float = 24.0        # units/s (instant velocity set)
    dash_duration:      float = 0.15        # seconds before momentum decay
    dash_max:           int   = 1           # uses before ground refill
    dash_recharge_ground: bool = True       # refill on ground contact
    dash_recharge_walls: bool = True        # refill on wall cling

    # Wall mechanics
    wall_slide_speed:   float = -2.5        # slow fall against wall
    wall_jump_x:        float = 17.0        # horizontal burst
    wall_jump_y:        float = 16.0        # vertical burst

    # World bounds
    world_w:            float = 60.0        # ~40–80 units typical
    world_h:            float = 40.0        # ~25–60 units typical

    # Player hitbox (capsule feel)
    player_w:           float = 0.9
    player_h:           float = 1.5

    # Episode
    max_steps:          int   = 1200        # 20 s @ 60 Hz


# ============================================================================
# HOLLOW KNIGHT CONFIGURATION
# ============================================================================

class HKConfig(NamedTuple):
    """Hollow Knight mode configuration."""
    gravity:            float = -30.0
    max_run_speed:      float = 8.0
    player_h:           float = 1.6
    player_w:           float = 1.0

    # Nail (base attack)
    nail_range:         float = 2.5
    nail_cooldown:      float = 0.25        # seconds
    nail_damage:        int   = 5           # base damage
    nail_knockback:     float = 8.0         # impulse

    # Pogo (downward nail attack on enemy)
    pogo_active:        bool  = True
    pogo_bounce_vel:    float = 15.0        # upward velocity on pogo

    # Soul system
    soul_max:           int   = 99          # 3 vessels × 33 each
    soul_per_hit:       int   = 11          # on nail attack
    soul_cost_heal:     int   = 33          # to recover 1 mask
    soul_cost_spell:    int   = 33          # per spell cast

    # Spells (base 3)
    spell_names:        tuple = ("Vengeful Spirit", "Desolate Dive", "Howling Wraiths")
    spell_cooldown:     float = 0.5         # seconds
    spell_speed:        float = 20.0        # projectile speed

    # Double jump / Monarch Wings
    double_jump_enabled: bool = False       # curriculum unlock

    # Super dash / Crystal Heart
    super_dash_enabled: bool = False
    super_dash_speed:   float = 30.0

    # Acid immunity / Isma's Tear
    acid_immunity:      bool  = False
    acid_swim_speed:    float = 6.0

    # Shade Cloak (dash upgrade)
    invuln_dash:        bool  = False

    # Enemy spawning
    max_enemies:        int   = 12
    max_bosses:         int   = 1

    # Episode
    max_steps:          int   = 2400        # 40 s @ 60 Hz


# ============================================================================
# ENEMY TYPES & STATS
# ============================================================================

class EnemyType(enum.IntEnum):
    """Enemy type IDs for lookup."""
    CRAWLID = 0
    GRUZZER = 1
    TIKTIK = 2
    HUSK_WANDERER = 3
    ASPID_HATCHLING = 4
    VENGEFLY = 5
    MOSS_KNIGHT = 6
    HUSK_GUARD = 7
    BALDUR = 8
    MOSSFLY = 9
    AMBLOOM = 10
    FUNGLING = 11
    MANTIS = 12
    CRYSTAL_CRAWLER = 13
    CRYSTAL_HUNTER = 14
    FLUKEFEY = 15
    FLUKEMUNGA = 16
    DUNG = 17
    INFECTED = 18
    BROKEN_VESSEL = 19
    DIRT_CARVER = 20
    WEAVER = 21
    NOSK = 22
    PRIMAL_ASPID = 23
    HIVE = 24
    TRAITOR_LORD = 25


# Enemy base stats: (health, damage, speed, speed_class)
ENEMY_STATS: Dict[EnemyType, Dict] = {
    EnemyType.CRAWLID: {
        "health": 10,
        "damage": 4,
        "speed": 4.0,
        "speed_class": "slow",
        "pogo_bonus": 1.2,
    },
    EnemyType.GRUZZER: {
        "health": 8,
        "damage": 4,
        "speed": 6.0,
        "speed_class": "medium",
        "pogo_bonus": 1.2,
    },
    EnemyType.TIKTIK: {
        "health": 8,
        "damage": 5,
        "speed": 5.0,
        "speed_class": "medium",
        "pogo_bonus": 1.2,
    },
    EnemyType.HUSK_WANDERER: {
        "health": 20,
        "damage": 8,
        "speed": 5.0,
        "speed_class": "medium",
        "pogo_bonus": 1.0,
    },
    EnemyType.ASPID_HATCHLING: {
        "health": 5,
        "damage": 4,
        "speed": 7.0,
        "speed_class": "fast",
        "pogo_bonus": 1.2,
    },
    EnemyType.VENGEFLY: {
        "health": 10,
        "damage": 6,
        "speed": 10.0,
        "speed_class": "fast",
        "pogo_bonus": 1.2,
    },
    EnemyType.MOSS_KNIGHT: {
        "health": 50,
        "damage": 12,
        "speed": 7.0,
        "speed_class": "fast",
        "pogo_bonus": 1.0,
    },
    EnemyType.HUSK_GUARD: {
        "health": 60,
        "damage": 12,
        "speed": 6.0,
        "speed_class": "medium",
        "pogo_bonus": 0.9,
    },
    EnemyType.BALDUR: {
        "health": 18,
        "damage": 8,
        "speed": 3.5,
        "speed_class": "slow",
        "pogo_bonus": 1.1,
    },
    EnemyType.MOSSFLY: {
        "health": 10,
        "damage": 5,
        "speed": 5.5,
        "speed_class": "medium",
        "pogo_bonus": 1.1,
    },
    EnemyType.AMBLOOM: {
        "health": 12,
        "damage": 6,
        "speed": 4.5,
        "speed_class": "medium",
        "pogo_bonus": 1.0,
    },
    EnemyType.FUNGLING: {
        "health": 14,
        "damage": 6,
        "speed": 4.0,
        "speed_class": "slow",
        "pogo_bonus": 1.1,
    },
    EnemyType.MANTIS: {
        "health": 45,
        "damage": 9,
        "speed": 8.0,
        "speed_class": "fast",
        "pogo_bonus": 1.0,
    },
    EnemyType.CRYSTAL_CRAWLER: {
        "health": 40,
        "damage": 8,
        "speed": 3.5,
        "speed_class": "slow",
        "pogo_bonus": 1.0,
    },
    EnemyType.CRYSTAL_HUNTER: {
        "health": 35,
        "damage": 9,
        "speed": 5.0,
        "speed_class": "medium",
        "pogo_bonus": 1.0,
    },
    EnemyType.FLUKEFEY: {
        "health": 110,
        "damage": 12,
        "speed": 2.8,
        "speed_class": "slow",
        "pogo_bonus": 0.9,
    },
    EnemyType.FLUKEMUNGA: {
        "health": 140,
        "damage": 14,
        "speed": 3.0,
        "speed_class": "slow",
        "pogo_bonus": 0.9,
    },
    EnemyType.DUNG: {
        "health": 55,
        "damage": 10,
        "speed": 3.5,
        "speed_class": "slow",
        "pogo_bonus": 0.9,
    },
    EnemyType.INFECTED: {
        "health": 8,
        "damage": 5,
        "speed": 2.5,
        "speed_class": "slow",
        "pogo_bonus": 1.2,
    },
    EnemyType.BROKEN_VESSEL: {
        "health": 120,
        "damage": 12,
        "speed": 4.0,
        "speed_class": "medium",
        "pogo_bonus": 0.9,
    },
    EnemyType.DIRT_CARVER: {
        "health": 45,
        "damage": 9,
        "speed": 4.5,
        "speed_class": "medium",
        "pogo_bonus": 1.0,
    },
    EnemyType.WEAVER: {
        "health": 35,
        "damage": 8,
        "speed": 5.0,
        "speed_class": "medium",
        "pogo_bonus": 1.0,
    },
    EnemyType.NOSK: {
        "health": 180,
        "damage": 14,
        "speed": 5.5,
        "speed_class": "fast",
        "pogo_bonus": 0.8,
    },
    EnemyType.PRIMAL_ASPID: {
        "health": 70,
        "damage": 10,
        "speed": 6.5,
        "speed_class": "fast",
        "pogo_bonus": 0.9,
    },
    EnemyType.HIVE: {
        "health": 80,
        "damage": 10,
        "speed": 7.0,
        "speed_class": "fast",
        "pogo_bonus": 0.9,
    },
    EnemyType.TRAITOR_LORD: {
        "health": 95,
        "damage": 11,
        "speed": 6.5,
        "speed_class": "fast",
        "pogo_bonus": 0.9,
    },
}


# ============================================================================
# BOSS TYPES & PHASES
# ============================================================================

class BossType(enum.IntEnum):
    """Boss type IDs."""
    FALSE_KNIGHT = 0
    GRUZ_MOTHER = 1
    HORNET_PROTECTOR = 2
    MANTIS_LORDS = 3
    SOUL_MASTER = 4
    CRYSTAL_GUARDIAN = 5
    GRIMM = 6
    PURE_VESSEL = 7


# Boss base stats: (health, damage, speed, num_phases)
BOSS_STATS: Dict[BossType, Dict] = {
    BossType.FALSE_KNIGHT: {
        "health": 260,
        "health_per_phase": [260, 260, 260],  # 3 phases
        "damage": 10,
        "speed": 3.0,
        "speed_class": "slow",
    },
    BossType.GRUZ_MOTHER: {
        "health": 650,
        "damage": 8,
        "speed": 5.0,
        "speed_class": "medium_slow",
    },
    BossType.HORNET_PROTECTOR: {
        "health": 900,
        "damage": 10,
        "speed": 12.0,
        "speed_class": "very_fast",
    },
    BossType.MANTIS_LORDS: {
        "health": 450,  # per lord
        "damage": 12,
        "speed": 10.0,
        "speed_class": "fast",
    },
    BossType.SOUL_MASTER: {
        "health": 750,
        "health_per_phase": [750, 800],  # 2 phases, second more aggressive
        "damage": 8,
        "speed": 8.0,
        "speed_class": "fast",
    },
    BossType.CRYSTAL_GUARDIAN: {
        "health": 400,
        "damage": 10,
        "speed": 4.0,
        "speed_class": "slow",
    },
    BossType.GRIMM: {
        "health": 1000,
        "health_per_phase": [1000, 1200],
        "damage": 12,
        "speed": 14.0,
        "speed_class": "very_fast",
    },
    BossType.PURE_VESSEL: {
        "health": 1500,
        "damage": 14,
        "speed": 13.0,
        "speed_class": "very_fast",
    },
}


# ============================================================================
# ENVIRONMENTAL EFFECTS
# ============================================================================

class HazardType(enum.IntEnum):
    """Hazard/environmental effect types."""
    SPIKE = 0
    ACID_POOL = 1
    BOUNCY_MUSHROOM = 2
    MOVING_SAW = 3
    WIND = 4
    DARKNESS = 5
    ACID_RAIN = 6
    THORN = 7


# Hazard properties
HAZARD_PROPS: Dict[HazardType, Dict] = {
    HazardType.SPIKE: {
        "damage": 200,  # instant death or big penalty
        "damage_type": "contact",
        "effect": "death",
    },
    HazardType.ACID_POOL: {
        "damage": 50,  # damage per second
        "damage_type": "dot",
        "duration": None,  # continuous while in pool
        "can_swim": False,  # until ability unlocked
    },
    HazardType.BOUNCY_MUSHROOM: {
        "damage": 0,
        "bounce_vel": 15.0,  # upward impulse when pogo on top
        "effect": "bounce",
    },
    HazardType.MOVING_SAW: {
        "damage": 100,
        "damage_type": "contact",
        "effect": "death",
    },
    HazardType.WIND: {
        "force": 3.0,  # units/s² horizontal push
        "effect": "push",
        "direction": 1,  # or -1
    },
    HazardType.DARKNESS: {
        "raycast_range_mult": 0.3,  # observation raycasts shortened
        "effect": "visibility",
    },
    HazardType.ACID_RAIN: {
        "damage": 30,  # per second if exposed
        "damage_type": "dot",
    },
    HazardType.THORN: {
        "damage": 100,
        "damage_type": "contact",
    },
}


# ============================================================================
# BIOMES / LEVEL GENERATION
# ============================================================================

class BiomeType(enum.IntEnum):
    """Biome/area types for procedural generation."""
    FORGOTTEN_CROSSROADS = 0
    GREENPATH = 1
    FUNGAL_WASTES = 2
    CITY_OF_TEARS = 3
    CRYSTAL_PEAK = 4
    ROYAL_WATERWAYS = 5
    ANCIENT_BASIN = 6
    DEEPNEST = 7
    KINGDOM_EDGE = 8
    QUEENS_GARDENS = 9
    HOWLING_CLIFFS = 10
    FOG_CANYON = 11
    RESTING_GROUNDS = 12
    ABYSS = 13
    HIVE = 14
    WHITE_PALACE = 15


# Biome properties
BIOME_PROPS: Dict[BiomeType, Dict] = {
    BiomeType.FORGOTTEN_CROSSROADS: {
        "difficulty_tier": "early",
        "width_range": (30, 50),
        "height_range": (25, 40),
        "enemy_pool": [
            EnemyType.CRAWLID,
            EnemyType.GRUZZER,
            EnemyType.TIKTIK,
            EnemyType.HUSK_WANDERER,
        ],
        "hazards": [HazardType.SPIKE],
        "platforms_density": 0.6,
        "gravity_mult": 1.0,
    },
    BiomeType.GREENPATH: {
        "difficulty_tier": "early",
        "width_range": (40, 60),
        "height_range": (30, 50),
        "enemy_pool": [
            EnemyType.VENGEFLY,
            EnemyType.MOSS_KNIGHT,
            EnemyType.ASPID_HATCHLING,
        ],
        "hazards": [HazardType.SPIKE, HazardType.ACID_POOL],
        "platforms_density": 0.7,
        "gravity_mult": 1.0,
    },
    BiomeType.FUNGAL_WASTES: {
        "difficulty_tier": "early_mid",
        "width_range": (40, 70),
        "height_range": (30, 60),
        "enemy_pool": [
            EnemyType.MOSS_KNIGHT,
            EnemyType.TIKTIK,
            EnemyType.HUSK_GUARD,
        ],
        "hazards": [HazardType.SPIKE, HazardType.BOUNCY_MUSHROOM],
        "platforms_density": 0.6,
        "gravity_mult": 1.0,
    },
    BiomeType.CITY_OF_TEARS: {
        "difficulty_tier": "mid",
        "width_range": (50, 80),
        "height_range": (35, 70),
        "enemy_pool": [
            EnemyType.HUSK_GUARD,
            EnemyType.MOSS_KNIGHT,
        ],
        "hazards": [HazardType.SPIKE, HazardType.MOVING_SAW],
        "platforms_density": 0.7,
        "gravity_mult": 1.0,
    },
    BiomeType.CRYSTAL_PEAK: {
        "difficulty_tier": "mid",
        "width_range": (50, 80),
        "height_range": (35, 70),
        "enemy_pool": [
            EnemyType.HUSK_GUARD,
            EnemyType.MOSS_KNIGHT,
        ],
        "hazards": [HazardType.SPIKE, HazardType.MOVING_SAW],
        "platforms_density": 0.65,
        "gravity_mult": 1.0,
    },
    BiomeType.ROYAL_WATERWAYS: {
        "difficulty_tier": "mid",
        "width_range": (50, 80),
        "height_range": (35, 70),
        "enemy_pool": [
            EnemyType.HUSK_GUARD,
            EnemyType.MOSS_KNIGHT,
        ],
        "hazards": [HazardType.SPIKE, HazardType.ACID_POOL],
        "platforms_density": 0.6,
        "gravity_mult": 1.0,
    },
    BiomeType.DEEPNEST: {
        "difficulty_tier": "late",
        "width_range": (60, 100),
        "height_range": (40, 80),
        "enemy_pool": [
            EnemyType.MOSS_KNIGHT,
            EnemyType.HUSK_GUARD,
        ],
        "hazards": [HazardType.SPIKE, HazardType.DARKNESS],
        "platforms_density": 0.5,
        "gravity_mult": 1.0,
    },
    BiomeType.KINGDOM_EDGE: {
        "difficulty_tier": "late",
        "width_range": (70, 120),
        "height_range": (45, 90),
        "enemy_pool": [
            EnemyType.MOSS_KNIGHT,
            EnemyType.HUSK_GUARD,
        ],
        "hazards": [HazardType.SPIKE, HazardType.WIND],
        "platforms_density": 0.5,
        "gravity_mult": 1.0,
    },
    BiomeType.QUEENS_GARDENS: {
        "difficulty_tier": "late",
        "width_range": (70, 120),
        "height_range": (45, 90),
        "enemy_pool": [
            EnemyType.MOSS_KNIGHT,
            EnemyType.TRAITOR_LORD,
        ],
        "hazards": [HazardType.SPIKE, HazardType.THORN],
        "platforms_density": 0.65,
        "gravity_mult": 1.0,
    },
    BiomeType.HOWLING_CLIFFS: {
        "difficulty_tier": "late_optional",
        "width_range": (60, 100),
        "height_range": (35, 70),
        "enemy_pool": [
            EnemyType.PRIMAL_ASPID,
            EnemyType.HIVE,
        ],
        "hazards": [HazardType.WIND],
        "platforms_density": 0.55,
        "gravity_mult": 1.0,
    },
    BiomeType.FOG_CANYON: {
        "difficulty_tier": "transitional",
        "width_range": (50, 80),
        "height_range": (30, 60),
        "enemy_pool": [
            EnemyType.NOSK,
            EnemyType.INFECTED,
        ],
        "hazards": [HazardType.DARKNESS],
        "platforms_density": 0.5,
        "gravity_mult": 1.0,
    },
    BiomeType.RESTING_GROUNDS: {
        "difficulty_tier": "mid_late",
        "width_range": (40, 70),
        "height_range": (25, 55),
        "enemy_pool": [
            EnemyType.INFECTED,
            EnemyType.FUNGLING,
        ],
        "hazards": [HazardType.DARKNESS],
        "platforms_density": 0.45,
        "gravity_mult": 1.0,
    },
    BiomeType.ABYSS: {
        "difficulty_tier": "very_late",
        "width_range": (80, 140),
        "height_range": (50, 100),
        "enemy_pool": [
            EnemyType.NOSK,
            EnemyType.INFECTED,
        ],
        "hazards": [HazardType.DARKNESS, HazardType.ACID_POOL],
        "platforms_density": 0.4,
        "gravity_mult": 1.0,
    },
    BiomeType.HIVE: {
        "difficulty_tier": "late_optional",
        "width_range": (45, 75),
        "height_range": (35, 70),
        "enemy_pool": [
            EnemyType.HIVE,
            EnemyType.ASPID_HATCHLING,
        ],
        "hazards": [HazardType.BOUNCY_MUSHROOM],
        "platforms_density": 0.6,
        "gravity_mult": 1.0,
    },
    BiomeType.WHITE_PALACE: {
        "difficulty_tier": "special",
        "width_range": (55, 90),
        "height_range": (40, 80),
        "enemy_pool": [],
        "hazards": [HazardType.MOVING_SAW, HazardType.SPIKE],
        "platforms_density": 0.75,
        "gravity_mult": 1.0,
    },
}


# ============================================================================
# MAIN GAME CONFIG
# ============================================================================

class GameConfig(NamedTuple):
    """Main game configuration combining both modes."""
    mode:               str         = "celeste"    # "celeste" | "hollow_knight"
    dt:                 float       = 1.0 / 60.0  # physics timestep

    # Entity slots (padded for vmap)
    n_platforms:        int         = 20          # platform slots
    n_enemies:          int         = 12          # enemy slots
    n_bosses:           int         = 1           # boss slots
    n_berries:          int         = 8           # collectible slots
    n_npcs:             int         = 3           # NPC trigger zones
    n_hazards:          int         = 16          # environmental hazards

    # Observations
    obs_raycasts:       int         = 10          # distance rays
    obs_dim:            int         = 50          # total obs dimension

    # Configs
    celeste:              CelesteConfig = CelesteConfig()
    hk:                   HKConfig      = HKConfig()

    # Biome / generation
    default_biome:      BiomeType   = BiomeType.FORGOTTEN_CROSSROADS
    difficulty:         float       = 0.5         # 0.0 (easy) to 1.0 (hard)
