from __future__ import annotations

import heapq
import itertools
import logging
import random
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Tuple

import numpy as np

from uncertain_worms.planners.base_planner import FullyObservablePlanner
from uncertain_worms.structs import ActType, StateType

log = logging.getLogger(__name__)

NUM_ROLLOUTS = 2  # Start with determinisic
ACTION_COST = 1


def rollout_fn(fn: Callable, inputs: List[Any]) -> Dict[Any, float]:
    counts = Counter([fn(*inputs) for _ in range(NUM_ROLLOUTS)])
    total = sum(counts.values())
    prob_dist = {item: count / total for item, count in counts.items()}
    return prob_dist


@dataclass
class SearchNode(Generic[StateType, ActType]):
    ...


@dataclass
class ActionNode(SearchNode[StateType, ActType]):
    state: StateType
    action: ActType

    def __hash__(self) -> int:
        return hash((self.state, self.action))


@dataclass
class StateNode(SearchNode[StateType, ActType]):
    state: StateType
    goal: bool = False
    terminal: bool = False

    def __hash__(self) -> int:
        return hash((self.state, self.goal))


class FO_DAStar(FullyObservablePlanner[StateType, ActType]):
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
        super(FO_DAStar, self).__init__(*args, **kwargs)

    def plan_next_action(
        self, start_state: StateType, max_steps: int
    ) -> Tuple[ActType, Dict]:
        """Performs an A* search starting from `start_state` until a transition
        produces an immediate reward > 0.

        Returns the first action of the computed path. If no such
        transition is found within max_steps, a random action is
        returned along with an error message.
        """
        counter = itertools.count()
        open_set: List[Tuple[float, int, float, SearchNode, int]] = []
        start_node = StateNode[StateType, ActType](start_state)
        start_entry = (
            self.heuristic(start_state),
            next(counter),
            0.0,
            start_node,
            0,
        )
        heapq.heappush(open_set, start_entry)

        # For path reconstruction.
        came_from: Dict[SearchNode, SearchNode] = {}
        cost_so_far: Dict[SearchNode, float] = {start_node: 0.0}
        num_popped = 0
        while open_set:
            (_, _, current_g, current_node, steps) = heapq.heappop(open_set)
            num_popped += 1
            if steps >= max_steps:
                continue

            neighbors: List[Tuple[SearchNode, float]] = []
            if isinstance(current_node, StateNode):
                if current_node.terminal and not current_node.goal:
                    continue
                # Check the goal condition: any transition with immediate_reward > 0 is considered a goal.
                if current_node.goal:
                    # Reconstruct the path from start_state to next_state.
                    path: List[ActType] = []
                    while current_node != start_node:
                        assert current_node in came_from
                        came_from_act = came_from[current_node]
                        assert isinstance(came_from_act, ActionNode)
                        path.append(came_from_act.action)
                        prev_state_node = came_from[came_from_act]
                        current_node = prev_state_node

                    path.reverse()
                    if path:
                        log.info("A* found path to goal")
                        print(num_popped)
                        return path[0], {}
                    else:
                        error = {"A*": "No action found during path reconstruction."}
                        print(num_popped)
                        log.info(error)
                        return random.choice(self.actions), error

                # We can take actions from a state node
                for action in self.actions:
                    neighbors.append(
                        (
                            ActionNode(current_node.state, action),
                            current_g + ACTION_COST,
                        )
                    )

            elif isinstance(current_node, ActionNode):
                try:
                    outcome_states = rollout_fn(
                        self.transition_model, [current_node.state, current_node.action]
                    )
                except Exception as e:
                    error_message = {"transition_model": traceback.format_exc()}
                    log.info(error_message)
                    continue  # Skip this action if transition fails.

                try:
                    for outcome_state, prob in outcome_states.items():
                        immediate_reward, terminal = self.reward_model(
                            current_node.state, current_node.action, outcome_state
                        )
                        goal = immediate_reward > 0
                        cost = current_g - np.log(prob)
                        next_state = StateNode[StateType, ActType](
                            outcome_state, goal, terminal=terminal
                        )
                        neighbors.append((next_state, cost))

                except Exception as e:
                    error_message = {"reward_model": traceback.format_exc()}
                    log.info(error_message)
                    continue

            for neighbor, new_cost in neighbors:
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    if isinstance(neighbor, StateNode):
                        priority = new_cost + self.heuristic(neighbor.state)
                        next_steps = steps + 1
                    else:
                        priority = new_cost
                        next_steps = steps

                    heapq.heappush(
                        open_set,
                        (
                            priority,
                            next(counter),
                            new_cost,
                            neighbor,
                            next_steps,
                        ),
                    )
                    came_from[neighbor] = current_node

        # If no goal state was found within the search limit, return a random action.
        error = {"A*": "No path found with positive reward within max_steps."}
        log.info(error)
        return random.choice(self.actions), error
