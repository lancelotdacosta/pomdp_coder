import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from uncertain_worms.policies.base_policy import Policy
from uncertain_worms.structs import ActType, ObsType, ReplayBuffer, StateType
from uncertain_worms.utils import PROJECT_ROOT


class BehaviorCloning(Policy[StateType, ActType, ObsType]):
    def __init__(
        self,
        *args: Any,
        dataset_path: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            actions: A list of all possible actions.
            database: A mapping from histories (tuples of observations) to actions.
                      If no database is provided, an empty dict is used.
        """
        super().__init__(*args, **kwargs)
        self.observation_hist: List[ObsType] = []

        # Train
        train_replay_buffer = ReplayBuffer[StateType, ActType, ObsType].load_from_file(
            os.path.join(PROJECT_ROOT, dataset_path)
        )
        self.init_types(train_replay_buffer)
        self.database = self._build_database_from_replay_buffer(train_replay_buffer)

    def _build_database_from_replay_buffer(
        self, replay_buffer: ReplayBuffer
    ) -> Dict[Tuple[ObsType, ...], ActType]:
        """Builds a mapping from observation histories to actions using the
        replay buffer.

        For each episode, for every action taken, we use the preceding
        observations as the history. If multiple actions are found for
        the same history, the majority vote is used.
        """
        temp_db: Dict[Tuple[ObsType, ...], List[ActType]] = defaultdict(list)
        for episode in replay_buffer.episodes:
            # We assume each episode has attributes `previous_observations` and `actions`.
            # The observation history for an action is taken as the observations leading up to it.
            for i, action in enumerate(episode.actions):
                # Create a tuple of observations from the start of the episode up to this step.
                history = tuple(episode.next_observations[:i])
                temp_db[history].append(action)

        # For each history, choose the most common (majority) action.
        final_db: Dict[Tuple[ObsType, ...], ActType] = {}
        for history, actions in temp_db.items():
            majority_action = Counter(actions).most_common(1)[0][0]
            final_db[history] = majority_action
        return final_db

    def reset(self) -> None:
        """Resets the observation history."""
        self.observation_hist = []

    def get_next_action(self, obs: Optional[ObsType]) -> ActType:
        """Given a new observation, update the history and select an action.

        If the full history exists in the database, return its action.
        Otherwise, select a random action from the available actions.
        """
        if obs is not None:
            self.observation_hist.append(obs)

        history_key = tuple(self.observation_hist)
        if history_key in self.database:
            return self.database[history_key]
        else:
            print("History not found in database. Choosing random action.")
            return random.choice(self.actions)
