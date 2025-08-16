from __future__ import annotations

import copy
import logging
import os
import random
import time
from typing import Any, Dict, Generic, List, Optional, Tuple, Union

from pyvis.network import Network  # type: ignore

from uncertain_worms.planners.base_planner import PartiallyObservablePlanner
from uncertain_worms.structs import ActType, BeliefType, ObsType, StateType
from uncertain_worms.utils import get_log_dir

log = logging.getLogger(__name__)


class HistoryNode(Generic[ActType, ObsType]):
    def __init__(self) -> None:
        self.action_edges: Dict[ActType, ActionEdge] = {}
        self.N: int = 0  # Visit count


class ActionEdge(Generic[ObsType]):
    def __init__(self) -> None:
        self.observation_edges: Dict[ObsType, HistoryNode] = {}
        self.N: int = 0  # Visit count
        self.Q: float = 0.0  # Estimated Q-value for this action


def visualize_search_tree(
    root: "HistoryNode[ActType, ObsType]", filename: str = "search_tree.html"
) -> None:
    net: Network = Network(height="750px", width="100%", directed=True)
    added_nodes: Dict[int, bool] = {}

    def traverse(
        node: Union["HistoryNode[Any, Any]", "ActionEdge[Any]"],
        parent_id: Optional[int] = None,
        edge_label: str = "",
    ) -> None:
        node_id: int = id(node)
        if node_id not in added_nodes:
            if isinstance(node, HistoryNode):
                label: str = f"History Node\nN: {node.N}"
                color: str = "purple" if parent_id is None else "blue"
                if not node.action_edges:
                    color = "red"  # Terminal node
            else:  # ActionEdge
                value: float = node.Q / node.N if node.N > 0 else 0.0
                label = f"Action Edge\nN: {node.N}\nQ: {node.Q:.2f}\nValue: {value:.2f}"
                color = "green"
                if not node.observation_edges:
                    color = "red"  # Terminal node

            net.add_node(node_id, label=label, color=color)
            added_nodes[node_id] = True

        if parent_id is not None:
            net.add_edge(parent_id, node_id, label=edge_label)

        if isinstance(node, HistoryNode):
            for action, action_edge in node.action_edges.items():
                action_label = f"Action: {action}"
                traverse(action_edge, parent_id=node_id, edge_label=action_label)
        else:  # ActionEdge
            for obs, history_node in node.observation_edges.items():
                obs_label = f"Obs: {obs}"
                traverse(history_node, parent_id=node_id, edge_label=obs_label)

    traverse(root)
    net.write_html(filename)


class POMCP(PartiallyObservablePlanner[StateType, ActType, ObsType, BeliefType]):
    def __init__(
        self,
        empty_observation: ObsType,
        *args: Any,
        num_simulations: int = 1000,
        exploration_constant: float = 0.0001,
        **kwargs: Any,
    ) -> None:
        self.empty_observation = empty_observation
        self.num_simulations = num_simulations
        self.c = exploration_constant  # UCB exploration constant
        super(POMCP, self).__init__(*args, **kwargs)

    def ucb(
        self,
        action: ActType,
        action_edge: ActionEdge,
        parent_node: HistoryNode,
        state: StateType,
    ) -> float:
        """
        Standard UCB1 formula for action selection:
          Q(s,a) / N(s,a) + c * sqrt(log(N(s)) / N(s,a))
        """
        if action_edge.N == 0:
            return float("inf")  # Encourage exploring unvisited actions
        exploitation = action_edge.Q / action_edge.N
        exploration = self.c * ((parent_node.N**0.5) / (1 + action_edge.N))
        return exploitation + exploration

    def simulate(
        self,
        state: StateType,
        history_node: HistoryNode,
        horizon: int,
        actions: List[ActType],
        steps: int,
        max_steps: int,
    ) -> float:
        """Simulate a trajectory and update the search tree."""
        # If we've hit the horizon or max_steps, no further reward is gained.
        if horizon == 0 or steps >= max_steps:
            return 0.0

        # If this history node hasn't been expanded yet, expand it with action edges
        if not history_node.action_edges:
            for action in actions:
                history_node.action_edges[action] = ActionEdge()
            rollout_reward = self.rollout(state, horizon, steps, max_steps)
            history_node.N += 1
            return rollout_reward

        # Node is expanded; pick action via UCB
        best_action = max(
            actions,
            key=lambda a: self.ucb(
                a,
                history_node.action_edges[a],
                history_node,
                state,
            ),
        )
        action_edge = history_node.action_edges[best_action]

        # Transition
        next_state = self.transition_model(copy.deepcopy(state), best_action)
        observation = self.observation_model(
            copy.deepcopy(next_state), best_action, self.empty_observation
        )

        # Get or create next node
        if observation not in action_edge.observation_edges:
            action_edge.observation_edges[observation] = HistoryNode()
        next_history_node = action_edge.observation_edges[observation]

        # Immediate reward + check for terminal
        reward, done = self.reward_model(state, best_action, next_state)
        if done:
            total_reward = reward
        else:
            future_reward = self.simulate(
                next_state,
                next_history_node,
                horizon - 1,
                actions,
                steps + 1,
                max_steps,
            )
            total_reward = reward + future_reward

        # Backpropagation
        action_edge.N += 1
        action_edge.Q += (total_reward - action_edge.Q) / action_edge.N
        history_node.N += 1

        return total_reward

    def rollout(
        self, state: StateType, horizon: int, steps: int, max_steps: int
    ) -> float:
        """Random (or heuristic) rollout from the given state, used after
        expansion."""
        total_reward = 0.0  # Start at 0 (no extra heuristic offset)
        current_state = copy.deepcopy(state)
        current_steps = steps

        for _ in range(horizon):
            if current_steps >= max_steps:
                break

            action = random.choice(self.actions)
            next_state = self.transition_model(current_state, action)
            reward, done = self.reward_model(current_state, action, next_state)
            total_reward += reward
            if done:
                break

            current_state = next_state
            current_steps += 1

        return total_reward

    def plan_next_action(
        self, belief: BeliefType, max_steps: int
    ) -> Tuple[ActType, Dict]:
        # Create root node
        root = HistoryNode[ActType, ObsType]()

        # Build the search tree via simulations
        for i in range(self.num_simulations):
            state = belief.sample()
            if state is None:
                return random.choice(self.actions), {
                    "observation_model": "Impossible observation",
                }
            self.simulate(
                state, root, max_steps, self.actions, steps=0, max_steps=max_steps
            )

        # Choose best action by average Q
        best_action = None
        best_value = float("-inf")
        for action in self.actions:
            edge = root.action_edges.get(action)
            if edge and edge.N > 0:
                avg_q = edge.Q / edge.N
                if avg_q > best_value:
                    best_value = avg_q
                    best_action = action

        if best_action is None:
            best_action = random.choice(self.actions)

        # Visualize the tree
        visualize_search_tree(
            root,
            filename=os.path.join(
                get_log_dir(), f"pomcp_search_tree_{time.time()}.html"
            ),
        )

        return best_action, {}
