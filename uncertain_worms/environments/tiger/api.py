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


# --------------------------- Enums & Dataclasses -----------------------------#
class TigerObservationEnum(enum.IntEnum):
    """Possible observations the agent can receive."""
    HEAR_LEFT = 0, HEAR_RIGHT = 1, NONE = 2

class TigerActions(enum.IntEnum):
    """Agent actions in the classic Tiger problem."""
    OPEN_LEFT = 0
    OPEN_RIGHT = 1
    LISTEN = 2

@dataclass(frozen=True)
class TigerObservation(Observation):
    """Observation dataclass."""
    obs: int # 0 = hear left, 1 = hear right, 2 = none

@dataclass(frozen=True)
class TigerState(State):
    """Underlying hidden state: tiger behind LEFT (0) or RIGHT (1) door."""
    tiger_location: int  # 0 = left, 1 = right