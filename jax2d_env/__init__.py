"""
CelesteKnight-XLand: Pure JAX open-ended RL playground.
Combines Celeste-style platforming with Hollow Knight mechanics.
"""

from .config import GameConfig, CelesteConfig, HKConfig
from .state import (
    EnvState, PlayerState, PlatformState, BerryState, EnemyState, TaskState
)
from .env import reset, step, make_env_fns
from .observations import make_observation

__all__ = [
    "GameConfig",
    "CelesteConfig",
    "HKConfig",
    "EnvState",
    "PlayerState",
    "PlatformState",
    "BerryState",
    "EnemyState",
    "TaskState",
    "reset",
    "step",
    "make_env_fns",
    "make_observation",
]
