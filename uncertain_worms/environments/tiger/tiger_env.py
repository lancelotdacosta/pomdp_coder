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
class TigerActions(enum.IntEnum):
    """Agent actions in the classic Tiger problem."""

    OPEN_LEFT = 0
    OPEN_RIGHT = 1
    LISTEN = 2


class TigerObservationEnum(enum.IntEnum):
    """Possible observations the agent can receive."""

    HEAR_LEFT = 0
    HEAR_RIGHT = 1
    NONE = 2  # Returned after opening either door


@dataclass(frozen=True)
class TigerObservation(Observation):
    """Observation dataclass."""

    obs: int

    # Standard boilerplate for hashing / encoding
    def encode(self) -> TigerObservation:
        return copy.deepcopy(self)

    @staticmethod
    def decode(obj: TigerObservation) -> TigerObservation:
        return copy.deepcopy(obj)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TigerObservation) and self.obs == other.obs

    def __hash__(self) -> int:
        return hash(self.obs)

    def __repr__(self) -> str:
        return f"TigerObservation(obs={str(self.obs)})"


@dataclass(frozen=True)
class TigerState(State):
    """Underlying hidden state: tiger behind LEFT (0) or RIGHT (1) door."""

    tiger_location: int  # 0 = left, 1 = right

    def encode(self) -> TigerState:
        return copy.deepcopy(self)

    @staticmethod
    def decode(obj: TigerState) -> TigerState:
        return copy.deepcopy(obj)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, TigerState)
            and self.tiger_location == other.tiger_location
        )

    def __hash__(self) -> int:
        return hash(self.tiger_location)

    def __repr__(self) -> str:
        loc = "LEFT" if self.tiger_location == 0 else "RIGHT"
        return f"TigerState(tiger_location={str(loc)})"


# --------------------------- POMDP Component Gens --------------------------- #
def tiger_initial_model_gen() -> InitialModel:
    """Uniformly draw tiger location at episode start."""

    def initial_model(_: TigerState | None) -> TigerState:
        return TigerState(random.choice([0, 1]))

    return initial_model


def tiger_empty_state_gen() -> TigerState:
    """Empty state for the environment (not used)."""
    return TigerState(-1)


def tiger_empty_observation_gen() -> TigerObservation:
    return TigerObservation(TigerObservationEnum.NONE)


def tiger_transition_model_gen() -> TransitionModel:
    """LISTEN keeps state fixed.

    Opening either door randomizes tiger’s new location (standard
    benchmark).
    """

    def transition_model(state: TigerState, action: TigerActions) -> TigerState:
        return copy.deepcopy(state)

    return transition_model


def tiger_observation_model_gen(listen_correct_prob: float = 0.85) -> ObservationModel:
    """Action‑conditioned observation model:

    • LISTEN → noisy information about tiger location • OPEN_* →
    deterministic NONE observation (no information)
    """

    def observation_model(
        state: TigerState, action: TigerActions, empty_obs: TigerObservation
    ) -> TigerObservation:
        if action != TigerActions.LISTEN:
            return TigerObservation(TigerObservationEnum.NONE)

        correct = random.random() < listen_correct_prob
        if state.tiger_location == 0:  # tiger left
            return TigerObservation(
                TigerObservationEnum.HEAR_LEFT
                if correct
                else TigerObservationEnum.HEAR_RIGHT
            )
        else:  # tiger right
            return TigerObservation(
                TigerObservationEnum.HEAR_RIGHT
                if correct
                else TigerObservationEnum.HEAR_LEFT
            )

    return observation_model


# Rewards (standard: +1 for gold, 0 for listen, 0 or ‑1 can also be used)
LISTEN_REWARD = 0.0
TIGER_REWARD = 0.0  # reward when eaten (often −100; left at 0 for demo)
NO_TIGER_REWARD = 1.0  # reward when gold obtained


def tiger_reward_model_gen() -> RewardModel:
    def reward_model(
        state: TigerState, action: TigerActions, _next_state: TigerState
    ) -> Tuple[float, bool]:
        if action == TigerActions.LISTEN:
            return LISTEN_REWARD, False

        ate_by_tiger = (
            action == TigerActions.OPEN_LEFT and state.tiger_location == 0
        ) or (action == TigerActions.OPEN_RIGHT and state.tiger_location == 1)
        reward = TIGER_REWARD if ate_by_tiger else NO_TIGER_REWARD
        return reward, True  # episode ends after opening

    return reward_model


def tiger_heuristic_gen() -> Heuristic:
    return lambda _s: 0.0


# ------------------------------ Environment ---------------------------------#
class TigerEnv(Environment):
    """Classic Tiger POMDP with action‑conditioned observations.

    Observation model: f(state, action) → TigerObservation
    """

    def __init__(self, max_steps: int = 100, **kwargs: Any):
        super().__init__(max_steps=max_steps)
        self._initial_model = tiger_initial_model_gen()
        self._transition_model = tiger_transition_model_gen()
        self._observation_model = tiger_observation_model_gen(0.85)
        self._reward_model = tiger_reward_model_gen()
        self.empty_observation = TigerObservation(TigerObservationEnum.NONE)
        self._current_state: TigerState | None = None

    # ------------- Gym‑like API ---------------- #
    def reset(self, seed: int = 0) -> TigerState:
        random.seed(seed)
        self.current_step = 0
        self._current_state = self._initial_model(None)
        assert self._current_state is not None, "Initial state must be set."
        return self._current_state

    def step(
        self, action: int
    ) -> Tuple[TigerObservation, TigerState, float, bool, bool, Dict[str, Any]]:
        act_enum = TigerActions(action)
        assert self._current_state is not None, "Call reset() first."

        # Transition → Observation → Reward
        next_state = self._transition_model(self._current_state, act_enum)
        obs = self._observation_model(next_state, act_enum, self.empty_observation)
        reward, done = self._reward_model(self._current_state, act_enum, next_state)

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        if not done and not truncated:
            self._current_state = next_state

        return obs, next_state, reward, done, truncated, {}

    # ------------- Optional helpers ------------- #
    @property
    def current_state(self) -> TigerState | None:
        return self._current_state

    def visualize_episode(
        self,
        states: List[TigerState],
        observations: List[TigerObservation],
        actions: List[TigerActions],
        episode_num: int,
    ) -> None:
        print(f"\nEpisode {episode_num} trace:")
        for t, (s, o) in enumerate(zip(states, observations)):
            act_str = f", action={str(actions[t])}" if t < len(actions) else ""
            print(f"  t={t}: state={s}, obs={o}{act_str}")
        print()


# ------------------------------ Demo runner ----------------------------------#
if __name__ == "__main__":
    env = TigerEnv(max_steps=10)
    state = env.reset(seed=42)
    print("Initial:", state)
    states = [state]
    observations = []
    actions = []
    done = truncated = False

    for step_i in range(5):
        action = random.choice(list(TigerActions))
        obs, nxt, reward, done, truncated, _ = env.step(action)
        print(
            f"Step {step_i}: {action.name}, r={reward}, s'={nxt}, o={obs}, done={done}"
        )
        states.append(nxt)
        observations.append(obs)
        actions.append(action)
        if done or truncated:
            break

    env.visualize_episode(states, observations, actions, episode_num=1)