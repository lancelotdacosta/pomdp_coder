from __future__ import annotations

import heapq
import itertools
import logging
import random
import traceback
from typing import Any, Dict, List, Optional, Tuple

from uncertain_worms.planners.base_planner import FullyObservablePlanner
from uncertain_worms.structs import ActType, StateType

log = logging.getLogger(__name__)


class AStarPlanner(FullyObservablePlanner[StateType, ActType]):
    """A* planner that searches for a transition yielding a positive reward.

    Instead of requiring a heuristic or goal function, this planner uses:
      - A default heuristic that always returns 0.
      - A goal test that considers any transition with immediate reward > 0 as reaching a goal.
    The cost for a step is defined as the negative of the immediate reward, so that
    transitions yielding higher reward have lower cost.
    Parameters:
      - transition_model: Function that, given a state and an action, returns the next state.
      - reward_model: Function that, given a state, action, and next state, returns (reward, terminated).
      - actions: List of available actions.
      - max_steps: Maximum number of steps to search.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def plan_next_action(
        self, start_state: StateType, max_steps: int
    ) -> Tuple[ActType, Dict]:
        """Performs an A* search starting from `start_state` until a transition
        produces an immediate reward > 0.

        Returns the first action of the computed path. If no such
        transition is found within max_steps, a random action is
        returned along with an error message.
        """
        # Each entry in the open set is a tuple:
        # (f, counter, g, current_state, parent_state, action_taken, steps)
        # where:
        #   g = cost-so-far (cumulative negative reward),
        #   f = g + heuristic(current_state),
        #   counter is a tie-breaker to avoid comparing states directly.
        counter = itertools.count()
        open_set: List[
            Tuple[
                float,
                int,
                float,
                StateType,
                Optional[StateType],
                Optional[ActType],
                int,
            ]
        ] = []
        start_entry = (
            self.heuristic(start_state),
            next(counter),
            0.0,
            start_state,
            None,
            None,
            0,
        )
        heapq.heappush(open_set, start_entry)

        # For path reconstruction.
        came_from: Dict[StateType, Tuple[StateType, ActType]] = {}
        cost_so_far: Dict[StateType, float] = {start_state: 0.0}
        num_popped = 0
        while open_set:
            (
                _,
                _,
                current_g,
                current_state,
                parent,
                action_taken,
                steps,
            ) = heapq.heappop(open_set)
            num_popped += 1
            if steps >= max_steps:
                continue

            # Expand current_state using all possible actions.
            for action in self.actions:
                try:
                    next_state = self.transition_model(current_state, action)
                except Exception as e:
                    error_message = {"transition_model": traceback.format_exc()}
                    log.info(error_message)
                    continue  # Skip this action if transition fails.

                try:
                    immediate_reward, terminated = self.reward_model(
                        current_state, action, next_state
                    )
                    # Define step cost as the negative of the immediate reward.
                    step_cost = 1 - immediate_reward
                except Exception as e:
                    error_message = {"reward_model": traceback.format_exc()}
                    log.info(error_message)
                    continue

                new_cost = current_g + step_cost

                # Check the goal condition: any transition with immediate_reward > 0 is considered a goal.
                if immediate_reward > 0:
                    came_from[next_state] = (current_state, action)
                    # Reconstruct the path from start_state to next_state.
                    path: List[ActType] = []
                    current_node = next_state
                    while current_node != start_state:
                        if current_node not in came_from:
                            break  # This should not happen.
                        prev_state, act = came_from[current_node]
                        path.append(act)
                        current_node = prev_state

                    path.reverse()
                    if path:
                        log.info("A* found path to goal")
                        return path[0], {}
                    else:
                        error = {"A*": "No action found during path reconstruction."}
                        log.info(error)
                        return random.choice(self.actions), error

                if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                    cost_so_far[next_state] = new_cost
                    priority = new_cost + self.heuristic(next_state)
                    heapq.heappush(
                        open_set,
                        (
                            priority,
                            next(counter),
                            new_cost,
                            next_state,
                            current_state,
                            action,
                            steps + 1,
                        ),
                    )
                    came_from[next_state] = (current_state, action)

        # If no goal state was found within the search limit, return a random action.
        error = {"A*": "No path found with positive reward within max_steps."}
        log.info(error)
        return random.choice(self.actions), error
