from __future__ import annotations

import copy
import enum
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from uncertain_worms.structs import (
    Environment,
    Heuristic,
    InitialModel,
    Observation,
    ObservationModel,
    RewardModel,
    State,
    TransitionModel,
)



# -----------------------------------------------------------------------------#
# Domain parameters (feel free to tweak)                                       #
# -----------------------------------------------------------------------------#
NUM_ROCKS   = 2
GRID_SIZE   = 5
ROCK_POSITIONS = [(1, 1), (1, 4)]   # len == NUM_ROCKS

# -----------------------------------------------------------------------------#
# Actions                                                                      #
# -----------------------------------------------------------------------------#
class RockSampleActions(enum.IntEnum):
    """They can be added directly to the state position."""
    MOVE_NORTH  = 0
    MOVE_SOUTH  = 1
    MOVE_EAST   = 2
    MOVE_WEST   = 3
    SAMPLE      = 4
    EXIT        = 5

    CHECK_ROCK_0 = 6
    CHECK_ROCK_1 = 7


CHECK_ACTIONS: List[int] = [
    RockSampleActions.CHECK_ROCK_0,
    RockSampleActions.CHECK_ROCK_1,
]

# -----------------------------------------------------------------------------#
# Observation                                                                  #
# -----------------------------------------------------------------------------#
@dataclass
class RockSampleObservation(Observation):
    """Observation after an action.

    Always embeds the rover pose (x, y).  
    For sensor actions:
        * ``rock_idx``  – index of inspected rock
        * ``is_good``   – noisy reading (True = GOOD, False = BAD)
    For all other actions both fields are ``None``.
    """
    x: int
    y: int
    rock_idx: int | None
    is_good: bool | None

# -----------------------------------------------------------------------------#
# State                                                                        #
# -----------------------------------------------------------------------------#
@dataclass
class RockSampleState(State):
    """Full underlying state (fully observable to the simulator)."""
    x: int
    y: int
    rocks: Tuple[bool, ...]   # immutable tuple of good/bad flags

    # --- Equality / hashing --------------------------------------------------
    def __eq__(self, other: object) -> bool:                  # type: ignore[override]
        return (
            isinstance(other, RockSampleState)
            and self.x == other.x
            and self.y == other.y
            and self.rocks == other.rocks
        )

    # Convenience -------------------------------------------------------------
    def at_rock(self) -> int | None:
        """Return the index of the rock at the agent's (x,y) or ``None``."""
        try:
            return ROCK_POSITIONS.index((self.x, self.y))
        except ValueError:
            return None