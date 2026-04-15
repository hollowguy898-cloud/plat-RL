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

class CelesteBossType(enum.IntEnum):
    """Celeste boss/chase encounters used for mode-specific tasks."""
    OSHIRO = 0
    BADELINE_CHASE = 1
    BADELINE_BOSS = 2


CELESTE_BOSS_STATS: Dict[CelesteBossType, Dict] = {
    CelesteBossType.OSHIRO: {
        "health": 700,
        "damage": 1,  # Celeste uses death/reset; this is a normalized RL damage unit
        "speed": 11.0,
        "speed_class": "fast",
        "phases": 1,
    },
    CelesteBossType.BADELINE_CHASE: {
        "health": 900,
        "damage": 1,
        "speed": 13.0,
        "speed_class": "very_fast",
        "phases": 2,
    },
    CelesteBossType.BADELINE_BOSS: {
        "health": 1100,
        "damage": 1,
        "speed": 14.0,
        "speed_class": "very_fast",
        "phases": 3,
    },
}


class BossType(enum.IntEnum):
    """Boss type IDs."""
    FALSE_KNIGHT = 0
    GRUZ_MOTHER = 1
    VENGEFLY_KING = 2
    HORNET_PROTECTOR = 3
    MASSIVE_MOSS_CHARGER = 4
    MANTIS_LORDS = 5
    SOUL_MASTER = 6
    DUNG_DEFENDER = 7
    CRYSTAL_GUARDIAN = 8
    BROKEN_VESSEL = 9
    BROODING_MAWLEK = 10
    HIVE_KNIGHT = 11
    TRAITOR_LORD = 12
    THE_COLLECTOR = 13
    NOSK = 14
    UUMUU = 15
    WATCHER_KNIGHTS = 16
    HORNET_SENTINEL = 17
    FLUKEMARM = 18
    TROUPE_MASTER_GRIMM = 19
    NIGHTMARE_KING_GRIMM = 20
    PURE_VESSEL = 21
    THE_HOLLOW_KNIGHT = 22
    THE_RADIANCE = 23
    ABSOLUTE_RADIANCE = 24
    FAILED_CHAMPION = 25
    LOST_KIN = 26
    WINGED_NOSK = 27
    SISTERS_OF_BATTLE = 28
    GOD_TAMER = 29
    OBBLOBBLES = 30


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
    BossType.VENGEFLY_KING: {
        "health": 450,
        "damage": 8,
        "speed": 11.0,
        "speed_class": "fast",
    },
    BossType.HORNET_PROTECTOR: {
        "health": 900,
        "damage": 10,
        "speed": 12.0,
        "speed_class": "very_fast",
    },
    BossType.MASSIVE_MOSS_CHARGER: {
        "health": 480,
        "damage": 9,
        "speed": 10.0,
        "speed_class": "fast",
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
    BossType.DUNG_DEFENDER: {
        "health": 800,
        "damage": 9,
        "speed": 7.5,
        "speed_class": "medium",
    },
    BossType.CRYSTAL_GUARDIAN: {
        "health": 400,
        "damage": 10,
        "speed": 4.0,
        "speed_class": "slow",
    },
    BossType.BROKEN_VESSEL: {
        "health": 1200,
        "damage": 10,
        "speed": 8.0,
        "speed_class": "medium_fast",
    },
    BossType.BROODING_MAWLEK: {
        "health": 1050,
        "damage": 10,
        "speed": 7.0,
        "speed_class": "medium",
    },
    BossType.HIVE_KNIGHT: {
        "health": 900,
        "damage": 11,
        "speed": 11.0,
        "speed_class": "fast",
    },
    BossType.TRAITOR_LORD: {
        "health": 1200,
        "damage": 13,
        "speed": 13.0,
        "speed_class": "very_fast",
    },
    BossType.THE_COLLECTOR: {
        "health": 1100,
        "damage": 10,
        "speed": 11.0,
        "speed_class": "fast",
    },
    BossType.NOSK: {
        "health": 900,
        "damage": 12,
        "speed": 12.0,
        "speed_class": "fast",
    },
    BossType.UUMUU: {
        "health": 700,
        "damage": 8,
        "speed": 3.5,
        "speed_class": "slow",
    },
    BossType.WATCHER_KNIGHTS: {
        "health": 450,  # per knight
        "damage": 10,
        "speed": 11.5,
        "speed_class": "fast",
    },
    BossType.HORNET_SENTINEL: {
        "health": 1400,
        "damage": 12,
        "speed": 14.0,
        "speed_class": "very_fast",
    },
    BossType.FLUKEMARM: {
        "health": 650,
        "damage": 9,
        "speed": 3.0,
        "speed_class": "slow",
    },
    BossType.TROUPE_MASTER_GRIMM: {
        "health": 1000,
        "health_per_phase": [1000, 1200],
        "damage": 12,
        "speed": 14.0,
        "speed_class": "very_fast",
    },
    BossType.NIGHTMARE_KING_GRIMM: {
        "health": 1600,
        "damage": 14,
        "speed": 16.0,
        "speed_class": "extremely_fast",
    },
    BossType.PURE_VESSEL: {
        "health": 1500,
        "damage": 14,
        "speed": 13.0,
        "speed_class": "very_fast",
    },
    BossType.THE_HOLLOW_KNIGHT: {
        "health": 1700,
        "damage": 14,
        "speed": 10.0,
        "speed_class": "medium_fast",
    },
    BossType.THE_RADIANCE: {
        "health": 2200,
        "damage": 16,
        "speed": 15.0,
        "speed_class": "extremely_fast",
    },
    BossType.ABSOLUTE_RADIANCE: {
        "health": 2600,
        "damage": 18,
        "speed": 16.0,
        "speed_class": "extremely_fast",
    },
    BossType.FAILED_CHAMPION: {
        "health": 900,
        "damage": 14,
        "speed": 6.0,
        "speed_class": "medium",
    },
    BossType.LOST_KIN: {
        "health": 1300,
        "damage": 13,
        "speed": 10.0,
        "speed_class": "fast",
    },
    BossType.WINGED_NOSK: {
        "health": 1100,
        "damage": 13,
        "speed": 13.0,
        "speed_class": "very_fast",
    },
    BossType.SISTERS_OF_BATTLE: {
        "health": 450,  # per sister
        "damage": 13,
        "speed": 14.0,
        "speed_class": "very_fast",
    },
    BossType.GOD_TAMER: {
        "health": 1400,
        "damage": 12,
        "speed": 10.5,
        "speed_class": "fast",
    },
    BossType.OBBLOBBLES: {
        "health": 850,  # per obblobble
        "damage": 11,
        "speed": 8.0,
        "speed_class": "medium_fast",
    },
}


# ============================================================================
# NPC TYPES (CELESTE + HOLLOW KNIGHT)
# ============================================================================

class CelesteNpcType(enum.IntEnum):
    """Major Celeste NPC/event trigger types."""
    THEO = 0
    GRANNY = 1
    OSHIRO = 2
    BADELINE = 3
    PAYPHONE = 4


class HollowKnightNpcType(enum.IntEnum):
    """Major Hollow Knight NPC types used for dialogue/reward triggers."""
    ELDERBUG = 0
    CORNIFER = 1
    ISELDA = 2
    QUIRREL = 3
    HORNET = 4
    ZOTE = 5
    BRETTA = 6
    SALUBRA = 7
    SEER = 8
    NAILSMITH = 9
    SLY = 10
    LEG_EATER = 11
    MILLIBELLE = 12
    TUK = 13
    DIVINE = 14
    BRUMM = 15
    MIDWIFE = 16
    CLOTH = 17
    DUNG_DEFENDER = 18
    WHITE_LADY = 19


CELESTE_NPC_PROFILES: Dict[CelesteNpcType, Dict] = {
    CelesteNpcType.THEO: {"role": "story", "reward": 25},
    CelesteNpcType.GRANNY: {"role": "story", "reward": 20},
    CelesteNpcType.OSHIRO: {"role": "story_boss_intro", "reward": 30},
    CelesteNpcType.BADELINE: {"role": "story_rival", "reward": 30},
    CelesteNpcType.PAYPHONE: {"role": "story_event", "reward": 35},
}


HK_NPC_PROFILES: Dict[HollowKnightNpcType, Dict] = {
    HollowKnightNpcType.ELDERBUG: {"role": "town_dialogue", "geo_reward": 0},
    HollowKnightNpcType.CORNIFER: {"role": "map_vendor", "geo_reward": 10},
    HollowKnightNpcType.ISELDA: {"role": "shopkeeper", "geo_reward": 0},
    HollowKnightNpcType.QUIRREL: {"role": "story_helper", "geo_reward": 15},
    HollowKnightNpcType.HORNET: {"role": "rival_story", "geo_reward": 0},
    HollowKnightNpcType.ZOTE: {"role": "comic_relief", "geo_reward": 5},
    HollowKnightNpcType.BRETTA: {"role": "town_story", "geo_reward": 5},
    HollowKnightNpcType.SALUBRA: {"role": "charm_vendor", "geo_reward": 0},
    HollowKnightNpcType.SEER: {"role": "essence_rewards", "geo_reward": 10},
    HollowKnightNpcType.NAILSMITH: {"role": "upgrade_vendor", "geo_reward": 0},
    HollowKnightNpcType.SLY: {"role": "shopkeeper", "geo_reward": 0},
    HollowKnightNpcType.LEG_EATER: {"role": "fragile_charms", "geo_reward": 0},
    HollowKnightNpcType.MILLIBELLE: {"role": "banker", "geo_reward": 0},
    HollowKnightNpcType.TUK: {"role": "rancid_egg_vendor", "geo_reward": 0},
    HollowKnightNpcType.DIVINE: {"role": "grimm_troupe", "geo_reward": 0},
    HollowKnightNpcType.BRUMM: {"role": "grimm_troupe", "geo_reward": 0},
    HollowKnightNpcType.MIDWIFE: {"role": "deepnest_story", "geo_reward": 5},
    HollowKnightNpcType.CLOTH: {"role": "combat_helper", "geo_reward": 10},
    HollowKnightNpcType.DUNG_DEFENDER: {"role": "friendly_boss_npc", "geo_reward": 10},
    HollowKnightNpcType.WHITE_LADY: {"role": "late_story", "geo_reward": 20},
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
    DIRTMOUTH = 16
    COLOSSEUM_OF_FOOLS = 17
    GODHOME = 18


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
    BiomeType.DIRTMOUTH: {
        "difficulty_tier": "hub",
        "width_range": (25, 40),
        "height_range": (15, 25),
        "enemy_pool": [],
        "hazards": [],
        "platforms_density": 0.2,
        "gravity_mult": 1.0,
    },
    BiomeType.COLOSSEUM_OF_FOOLS: {
        "difficulty_tier": "late_optional",
        "width_range": (45, 75),
        "height_range": (35, 65),
        "enemy_pool": [
            EnemyType.PRIMAL_ASPID,
            EnemyType.HUSK_GUARD,
            EnemyType.MOSS_KNIGHT,
        ],
        "hazards": [HazardType.SPIKE, HazardType.THORN],
        "platforms_density": 0.5,
        "gravity_mult": 1.0,
    },
    BiomeType.GODHOME: {
        "difficulty_tier": "boss_rush",
        "width_range": (40, 70),
        "height_range": (30, 55),
        "enemy_pool": [],
        "hazards": [HazardType.SPIKE],
        "platforms_density": 0.4,
        "gravity_mult": 1.0,
    },
}


# ============================================================================
# MAP COVERAGE (CANONICAL AREA/CHAPTER TABLES)
# ============================================================================

class HollowKnightMapArea(enum.IntEnum):
    """Major Hollow Knight map regions (core + optional/DLC)."""
    DIRTMOUTH = 0
    FORGOTTEN_CROSSROADS = 1
    GREENPATH = 2
    FUNGAL_WASTES = 3
    CITY_OF_TEARS = 4
    CRYSTAL_PEAK = 5
    ROYAL_WATERWAYS = 6
    ANCIENT_BASIN = 7
    DEEPNEST = 8
    KINGDOMS_EDGE = 9
    QUEENS_GARDENS = 10
    HOWLING_CLIFFS = 11
    FOG_CANYON = 12
    RESTING_GROUNDS = 13
    THE_ABYSS = 14
    THE_HIVE = 15
    COLOSSEUM_OF_FOOLS = 16
    WHITE_PALACE = 17
    GODHOME = 18


HK_MAP_PROPS: Dict[HollowKnightMapArea, Dict] = {
    HollowKnightMapArea.DIRTMOUTH: {"tier": "hub", "focus": "safe_zone"},
    HollowKnightMapArea.FORGOTTEN_CROSSROADS: {"tier": "early", "focus": "starter_combat_platforming"},
    HollowKnightMapArea.GREENPATH: {"tier": "early", "focus": "vertical_platforming_acid"},
    HollowKnightMapArea.FUNGAL_WASTES: {"tier": "early_mid", "focus": "bounce_pogo_groups"},
    HollowKnightMapArea.CITY_OF_TEARS: {"tier": "mid", "focus": "dense_patrols_vertical_routes"},
    HollowKnightMapArea.CRYSTAL_PEAK: {"tier": "mid", "focus": "laser_hazards_minecart_routes"},
    HollowKnightMapArea.ROYAL_WATERWAYS: {"tier": "mid", "focus": "acid_dot_tight_corridors"},
    HollowKnightMapArea.ANCIENT_BASIN: {"tier": "mid_late", "focus": "deep_shafts_infection"},
    HollowKnightMapArea.DEEPNEST: {"tier": "late", "focus": "ambush_darkness_tunnels"},
    HollowKnightMapArea.KINGDOMS_EDGE: {"tier": "late", "focus": "long_gaps_wind"},
    HollowKnightMapArea.QUEENS_GARDENS: {"tier": "late", "focus": "thorn_precision_combat"},
    HollowKnightMapArea.HOWLING_CLIFFS: {"tier": "late_optional", "focus": "wind_exposure"},
    HollowKnightMapArea.FOG_CANYON: {"tier": "transitional", "focus": "floating_jelly_platforms"},
    HollowKnightMapArea.RESTING_GROUNDS: {"tier": "mid_late", "focus": "dream_events_low_combat"},
    HollowKnightMapArea.THE_ABYSS: {"tier": "very_late", "focus": "void_darkness_endgame"},
    HollowKnightMapArea.THE_HIVE: {"tier": "late_optional", "focus": "aggressive_swarm_vertical"},
    HollowKnightMapArea.COLOSSEUM_OF_FOOLS: {"tier": "arena", "focus": "wave_survival"},
    HollowKnightMapArea.WHITE_PALACE: {"tier": "special", "focus": "precision_saw_platforming"},
    HollowKnightMapArea.GODHOME: {"tier": "special", "focus": "boss_rush_pantheons"},
}


class CelesteMapArea(enum.IntEnum):
    """Celeste chapters/sub-areas used for map-faithful task sampling."""
    PROLOGUE = 0
    CH1_FORSAKEN_CITY = 1
    CH2_OLD_SITE = 2
    CH3_CELESTIAL_RESORT = 3
    CH4_GOLDEN_RIDGE = 4
    CH5_MIRROR_TEMPLE = 5
    CH6_REFLECTION = 6
    CH7_SUMMIT = 7
    CH8_CORE = 8
    CH9_FAREWELL = 9
    EPILOGUE = 10


CELESTE_MAP_PROPS: Dict[CelesteMapArea, Dict] = {
    CelesteMapArea.PROLOGUE: {"tier": "intro", "focus": "basic_run_jump"},
    CelesteMapArea.CH1_FORSAKEN_CITY: {"tier": "early", "focus": "dash_intro_strawberries"},
    CelesteMapArea.CH2_OLD_SITE: {"tier": "early", "focus": "dream_blocks_badeline_chase"},
    CelesteMapArea.CH3_CELESTIAL_RESORT: {"tier": "mid", "focus": "moving_obstacles_oshiro_pressure"},
    CelesteMapArea.CH4_GOLDEN_RIDGE: {"tier": "mid", "focus": "wind_forces_cloud_jumps"},
    CelesteMapArea.CH5_MIRROR_TEMPLE: {"tier": "mid", "focus": "seekers_tight_routes"},
    CelesteMapArea.CH6_REFLECTION: {"tier": "mid_late", "focus": "feather_bumpers_badeline_boss"},
    CelesteMapArea.CH7_SUMMIT: {"tier": "late", "focus": "full_mechanics_gauntlet"},
    CelesteMapArea.CH8_CORE: {"tier": "late", "focus": "hot_cold_state_switching"},
    CelesteMapArea.CH9_FAREWELL: {"tier": "expert", "focus": "ultra_precision_stamina_control"},
    CelesteMapArea.EPILOGUE: {"tier": "story", "focus": "non_combat_transition"},
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
