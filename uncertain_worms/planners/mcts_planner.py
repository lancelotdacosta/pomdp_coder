from __future__ import annotations

import logging
import math
import random

# import traceback
from typing import Any, Dict, Generic, List, Optional, Tuple

import numpy as np

from uncertain_worms.planners.base_planner import FullyObservablePlanner
from uncertain_worms.structs import (
    ActType,
    Heuristic,
    RewardModel,
    StateType,
    TransitionModel,
)

log = logging.getLogger(__name__)
CATCH_ERRORS = True


class MCTSNode(Generic[StateType, ActType]):
    def __init__(
        self,
        state: StateType,
        actions: List[ActType],
        action_taken: Optional[ActType],
        terminal: bool,
        steps: int = 0,
        max_steps: int = 0,
        parent: Optional[MCTSNode] = None,
    ) -> None:
        self.c = 1.0
        self.state = state
        self.parent = parent
        self.children: Dict[ActType, MCTSNode] = {}
        self.visits = 0
        self.value = 0.0
        self.action_taken = action_taken
        self.terminal = terminal
        self.actions = actions
        self.steps = steps
        self.max_steps = max_steps

    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(self.actions)

    def best_child(self, heuristic: Heuristic, exploration: bool = True) -> MCTSNode:
        # Use UCB1 formula to balance exploration and exploitation
        if exploration:
            weights = [
                (child.value / (child.visits + 1e-6))  # Avoid division by zero
                + (
                    self.c
                    * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
                )
                - (heuristic(child.state))  # Incorporate heuristic
                for _, child in self.children.items()
            ]
        else:
            weights = [child.visits for _, child in self.children.items()]

        return list(self.children.values())[np.argmax(weights)]

    def expand(
        self, transition_model: TransitionModel, reward_model: RewardModel
    ) -> Tuple[Optional[MCTSNode], Dict]:
        untried_actions = [
            action for action in self.actions if action not in self.children
        ]
        action = random.choice(untried_actions)

        try:
            next_state = transition_model(self.state, action)

        except Exception as e:
            # error_message = {"transition_model": traceback.format_exc()}
            error_message = {"transition_model": f"{type(e).__name__}: {str(e)}"}
            log.info(error_message)
            if not CATCH_ERRORS:
                raise e
            return None, error_message

        try:
            _, terminated = reward_model(self.state, action, next_state)
        except Exception as e:
            # error_message = {"reward_model": traceback.format_exc()}
            error_message = {"reward_model": f"{type(e).__name__}: {str(e)}"}
            log.info(error_message)

            if not CATCH_ERRORS:
                raise e
            return None, error_message

        # Increment steps and check termination condition for max_steps
        next_steps = self.steps + 1
        if next_steps >= self.max_steps:
            terminated = True

        child_node = MCTSNode[StateType, ActType](
            next_state,
            actions=self.actions,
            action_taken=action,
            terminal=terminated,
            steps=next_steps,
            max_steps=self.max_steps,
            parent=self,
        )
        self.children[action] = child_node
        return child_node, {}

    def update(self, reward: float) -> None:
        self.visits += 1
        self.value += (reward - self.value) / self.visits  # Running average

    def is_terminal(self) -> bool:
        return self.terminal


ACTION_PENALTY = -0.01


class MCTS(FullyObservablePlanner[StateType, ActType]):
    def __init__(
        self,
        *args: Any,
        depth: int = 10,
        iterations: int = 1000,
        **kwargs: Any,
    ):
        self.depth = depth
        self.iterations = iterations

        super(MCTS, self).__init__(*args, **kwargs)

    def simulate(self, node: MCTSNode, max_steps: int) -> Tuple[float, Dict]:
        # Perform a random simulation from this node
        state = node.state
        total_reward = 0.0
        steps = node.steps

        # Simulate until a terminal condition, fixed depth, or max_steps is reached
        for _ in range(self.depth):
            if steps >= max_steps:
                break

            action = random.choice(self.actions)

            try:
                next_state = self.transition_model(state, action)

            except Exception as e:
                if not CATCH_ERRORS:
                    raise e
                # error_message = {"transition_model": traceback.format_exc()}
                error_message = {"transition_model": f"{type(e).__name__}: {str(e)}"}
                log.info(error_message)
                return 0, error_message

            try:
                reward, terminated = self.reward_model(state, action, next_state)
                reward += ACTION_PENALTY
            except Exception as e:
                if not CATCH_ERRORS:
                    raise e
                # error_message = {"reward_model": traceback.format_exc()}
                error_message = {"reward_model": f"{type(e).__name__}: {str(e)}"}
                log.info(error_message)
                return 0, error_message

            # Incorporate heuristic into the reward estimate
            reward -= self.heuristic(next_state)

            total_reward += reward
            steps += 1

            if terminated:
                break

            state = next_state

        return total_reward, {}

    def backpropagate(self, node: Optional[MCTSNode], reward: float) -> None:
        while node is not None:
            node.update(reward)
            node = node.parent

    def plan_next_action(
        self, start_state: StateType, max_steps: int
    ) -> Tuple[ActType, Dict]:
        root_node = MCTSNode[StateType, ActType](
            start_state,
            actions=self.actions,
            action_taken=None,
            terminal=False,
            steps=0,
            max_steps=max_steps,
            parent=None,
        )

        for _ in range(self.iterations):
            node = root_node

            # Selection
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.best_child(heuristic=self.heuristic)

            # Expansion
            if not node.is_fully_expanded():
                expanded_node, error = node.expand(
                    self.transition_model, self.reward_model
                )
                if len(error) > 0 or expanded_node is None:
                    return random.choice(self.actions), error
                else:
                    node = expanded_node

            # Simulation
            reward, error = self.simulate(node, max_steps)

            if len(error) > 0:
                return random.choice(self.actions), error

            # Backpropagation
            self.backpropagate(node, reward)

        action_taken = root_node.best_child(
            exploration=False, heuristic=self.heuristic
        ).action_taken

        if action_taken is None:
            raise NotImplementedError
        else:
            return action_taken, {}
