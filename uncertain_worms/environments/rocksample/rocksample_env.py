from __future__ import annotations
"""
RockSample POMDP implementation (grid-world navigation + information gathering).

* Dataclass State / Observation
* `*_model_gen` functions returning callables
* Optional heuristic
* `RockSampleEnv` Environment subclass with `reset`, `step`, `visualize_episode`

Key domain specifics
--------------------
* The action space depends on the number of rocks (k); we build an IntEnum
  dynamically in ``RockSampleEnv.__init__`` so callers can still pass an ``int``.
* The observation *always* contains the rover position (x, y).  
  For sensor actions it additionally carries the rock index inspected and the
  noisy Good/Bad reading.
* State encodes the rover position plus a tuple of Good/Bad flags, one per rock.

Reward scheme (canonical benchmark settings)
--------------------------------------------
* Move actions            …   0
* Check_i (sensor)        …  -1
* Sample on good rock     … +10
* Sample on bad rock      … -10
* Exit (leave right edge) … +10 and terminates episode
* Discounting / truncation handled by the surrounding algorithm.
"""

import enum
import math
import random
from dataclasses import dataclass
from copy import deepcopy
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

    # --- (de)serialisation helpers ------------------------------------------
    def encode(self) -> "RockSampleObservation":             # type: ignore[override]
        return self

    @staticmethod
    def decode(obj: "RockSampleObservation") -> "RockSampleObservation":
        return RockSampleObservation(
            x=obj.x, y=obj.y, rock_idx=obj.rock_idx, is_good=obj.is_good
        )

    # --- Equality / hashing --------------------------------------------------
    def __eq__(self, other: object) -> bool:                 # type: ignore[override]
        return (
            isinstance(other, RockSampleObservation)
            and (self.x, self.y, self.rock_idx, self.is_good)
            == (other.x, other.y, other.rock_idx, other.is_good)
        )

    def __hash__(self) -> int:                               # type: ignore[override]
        return hash((self.x, self.y, self.rock_idx, self.is_good))

    def __repr__(self) -> str:                               # type: ignore[override]
        if self.rock_idx is None:
            return f"Obs(x={self.x},y={self.y}, NULL)"
        label = "GOOD" if self.is_good else "BAD"
        return f"Obs(x={self.x}, y={self.y}, rock={self.rock_idx}, reading={label})"


# -----------------------------------------------------------------------------#
# State                                                                        #
# -----------------------------------------------------------------------------#
@dataclass
class RockSampleState(State):
    """Full underlying state (fully observable to the simulator)."""
    x: int
    y: int
    rocks: Tuple[bool, ...]   # immutable tuple of good/bad flags

    # --- (de)serialisation helpers ------------------------------------------
    def encode(self) -> "RockSampleState":                    # type: ignore[override]
        return self

    @staticmethod
    def decode(obj: "RockSampleState") -> "RockSampleState":
        return RockSampleState(x=obj.x, y=obj.y, rocks=obj.rocks)

    # --- Equality / hashing --------------------------------------------------
    def __eq__(self, other: object) -> bool:                  # type: ignore[override]
        return (
            isinstance(other, RockSampleState)
            and self.x == other.x
            and self.y == other.y
            and self.rocks == other.rocks
        )

    def __hash__(self) -> int:                                # type: ignore[override]
        return hash((self.x, self.y, self.rocks))

    # Convenience -------------------------------------------------------------
    def at_rock(self) -> int | None:
        """Return the index of the rock at the agent's (x,y) or ``None``."""
        try:
            return ROCK_POSITIONS.index((self.x, self.y))
        except ValueError:
            return None

    def __repr__(self) -> str:                                # type: ignore[override]
        rock_str = ",".join("G" if g else "B" for g in self.rocks)
        return f"RockSampleState(x={self.x}, y={self.y}, rocks=[{rock_str}])"


# -----------------------------------------------------------------------------#
# Model generators                                                             #
# -----------------------------------------------------------------------------#
def rocksample_empty_state_gen() -> RockSampleState:
    return RockSampleState(-1, -1, tuple([False] * NUM_ROCKS))

def rocksample_empty_observation_gen() -> RockSampleObservation:
    return RockSampleObservation(-1, -1, None, None)

# --- Initial model -----------------------------------------------------------
def rocksample_initial_model_gen(p_good: float = 0.5) -> InitialModel:
    def initial_model(_: Any) -> RockSampleState:             # type: ignore[override]
        rocks = tuple(random.random() < p_good for _ in ROCK_POSITIONS)
        return RockSampleState(
            random.randint(0, GRID_SIZE - 1),
            random.randint(0, GRID_SIZE - 1),
            rocks,
        )
    return initial_model

# --- Transition model --------------------------------------------------------
def rocksample_transition_model_gen() -> TransitionModel:
    """Deterministic motion & sampling."""
    def transition_model(state: RockSampleState, action: enum.IntEnum) -> RockSampleState:  # type: ignore[override]
        x, y = state.x, state.y
        rocks = list(state.rocks)

        # Movement ------------------------------------------------------------
        if action == RockSampleActions.MOVE_NORTH and y < GRID_SIZE - 1:
            y += 1
        elif action == RockSampleActions.MOVE_SOUTH and y > 0:
            y -= 1
        elif action == RockSampleActions.MOVE_EAST and x < GRID_SIZE - 1:
            x += 1
        elif action == RockSampleActions.MOVE_WEST and x > 0:
            x -= 1
        # SAMPLE --------------------------------------------------------------
        elif action == RockSampleActions.SAMPLE:
            idx = state.at_rock()
            if idx is not None:
                rocks[idx] = False
        # CHECK_i and EXIT do **not** change state ---------------------------

        return RockSampleState(x, y, tuple(rocks))
    return transition_model

# --- Observation model -------------------------------------------------------
HALF_EFFICIENCY_DISTANCE = 20.0  # distance at which sensor accuracy is 75 %

def rocksample_observation_model_gen() -> ObservationModel:
    def _sensor_accuracy(distance: float) -> float:
        return 0.5 + 0.5 * 2 ** (-distance / HALF_EFFICIENCY_DISTANCE)

    def observation_model(                         # type: ignore[override]
        state: RockSampleState, action: int, _unused: RockSampleObservation
    ) -> RockSampleObservation:
        # Sensor action ------------------------------------------------------
        if action in CHECK_ACTIONS:
            rock_idx = CHECK_ACTIONS.index(action)
            rock_x, rock_y = ROCK_POSITIONS[rock_idx]
            p_correct = _sensor_accuracy(math.dist((state.x, state.y), (rock_x, rock_y)))
            reading_is_good = (
                state.rocks[rock_idx]
                if random.random() < p_correct
                else not state.rocks[rock_idx]
            )
            return RockSampleObservation(state.x, state.y, rock_idx, reading_is_good)

        # All non-sensor actions --------------------------------------------
        return RockSampleObservation(state.x, state.y, None, None)

    return observation_model

# --- Reward model ------------------------------------------------------------
def rocksample_reward_model_gen() -> RewardModel:
    def reward_model(
        state: RockSampleState, action: enum.IntEnum, next_state: RockSampleState
    ) -> Tuple[float, bool]:                                # type: ignore[override]
        reward = 0.0
        done = False

        if action in (
            RockSampleActions.MOVE_NORTH,
            RockSampleActions.MOVE_SOUTH,
            RockSampleActions.MOVE_EAST,
            RockSampleActions.MOVE_WEST,
        ):
            reward = 0.0

        elif action == RockSampleActions.SAMPLE:
            idx = state.at_rock()
            if idx is not None:
                reward = 10.0 if state.rocks[idx] else -10.0

        elif action in CHECK_ACTIONS:
            reward = -1.0

        elif action == RockSampleActions.EXIT and state.x == GRID_SIZE - 1:
            reward = 10.0
            done = True

        # Terminate if rover has moved off the right edge --------------------
        if next_state.x == GRID_SIZE:
            done = True

        return reward, done
    return reward_model

# --- Optional heuristic ------------------------------------------------------
def rocksample_heuristic_gen() -> Heuristic:
    def heuristic(state: RockSampleState) -> float:           # type: ignore[override]
        return 0.0
    return heuristic

# -----------------------------------------------------------------------------#
# Environment                                                                  #
# -----------------------------------------------------------------------------#
class RockSampleEnv(Environment):
    """Parametrised RockSample POMDP environment compatible with *uncertain_worms*."""
    def __init__(self, max_steps: int = 100, **kwargs: Any) -> None:
        super().__init__(max_steps=max_steps)

        self.k = len(ROCK_POSITIONS)

        # Hard-coded model instances -----------------------------------------
        self._initial_model     = rocksample_initial_model_gen()
        self._transition_model  = rocksample_transition_model_gen()
        self._observation_model = rocksample_observation_model_gen()
        self._reward_model      = rocksample_reward_model_gen()

        self._current_state: RockSampleState | None = None

    # --------------------------------------------------------------------- #
    # Environment API                                                       #
    # --------------------------------------------------------------------- #
    def reset(self, seed: int | None = None) -> RockSampleState:
        if seed is not None:
            random.seed(seed)
        self.current_step = 0
        init_state = self._initial_model(None)                # type: ignore[arg-type]
        self._current_state = init_state
        return init_state

    def step(
        self, action: int
    ) -> Tuple[
        RockSampleObservation, RockSampleState, float, bool, bool, Dict[str, Any]
    ]:
        assert self._current_state is not None, "Environment not reset!"

        # 1) Transition ------------------------------------------------------
        next_state = self._transition_model(self._current_state, action)

        # 2) Observation -----------------------------------------------------
        obs = self._observation_model(next_state, action, None)

        # 3) Reward / termination -------------------------------------------
        reward, done = self._reward_model(self._current_state, action, next_state)

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        if not done and not truncated:
            self._current_state = next_state

        info: Dict[str, Any] = {}
        return obs, next_state, reward, done, truncated, info

    # --------------------------------------------------------------------- #
    # Convenience accessors                                                 #
    # --------------------------------------------------------------------- #
    @property
    def current_state(self) -> RockSampleState | None:        # type: ignore[override]
        return self._current_state

    # --------------------------------------------------------------------- #
    # Visualisation (optional)                                              #
    # --------------------------------------------------------------------- #
    def visualize_episode(
        self,
        states: List[RockSampleState],
        observations: List[RockSampleObservation],
        actions: List[int],
        episode_num: int,
    ) -> None:
        print(f"\nEpisode {episode_num}:")
        for t, (s, o) in enumerate(zip(states, observations)):
            print(f"  t={t}, state={s}, obs={o}")
            if t < len(actions):
                print("    Action:", RockSampleActions(actions[t]).name)
        print()

# -----------------------------------------------------------------------------#
# Quick manual test                                                            #
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    env = RockSampleEnv(max_steps=50)
    state = env.reset(seed=8)
    print("Initial state:", state)

    done = truncated = False
    states: List[RockSampleState] = [state]
    observations: List[RockSampleObservation] = []
    actions: List[int] = []

    for step_i in range(20):
        action_int = random.choice(list(RockSampleActions)).value
        obs, s_next, r, done, truncated, _ = env.step(action_int)
        print(
            f"Step {step_i:2d} | Act={RockSampleActions(action_int).name:12s} "
            f"| R={r:5.1f} | done={done} | s'={s_next} | o={obs}"
        )

        states.append(s_next)
        observations.append(obs)
        actions.append(action_int)
        if done or truncated:
            break

    env.visualize_episode(states, observations, actions, episode_num=1)
