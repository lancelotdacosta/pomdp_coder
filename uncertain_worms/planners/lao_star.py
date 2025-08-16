from __future__ import annotations

import logging
import random
from collections import Counter
from copy import deepcopy
from typing import Any, Callable, Dict, Generic, List, Tuple

import numpy as np

from uncertain_worms.planners.base_planner import FullyObservablePlanner
from uncertain_worms.structs import ActType, StateType

log = logging.getLogger(__name__)

NUM_ROLLOUTS = 2  # Start with determinisic


def rollout_fn(fn: Callable, inputs: List[Any]) -> Dict[Any, float]:
    counts = Counter([fn(*inputs) for _ in range(NUM_ROLLOUTS)])
    total = sum(counts.values())
    prob_dist = {item: count / total for item, count in counts.items()}
    return prob_dist


class LAOStar(FullyObservablePlanner[StateType, ActType], Generic[StateType, ActType]):
    def __init__(
        self,
        *args: Any,
        epsilon: float = 1e-3,
        n_iter: int = 1000,
        **kwargs: Any,
    ) -> None:
        """LAO* planner that incrementally builds an explicit graph.

        - actions: available actions
        - transition_model: generative model: (s, a) -> next_state
        - reward_model: returns (reward, terminated) for (s, a, next_state)
        - heuristic: function mapping state to estimated cost (assumed 0 for goal states)
        - epsilon: convergence threshold for value iteration
        - n_iter: maximum number of iterations for value iteration
        """
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.logger = log

    # --- Helper Functions ---
    @staticmethod
    def flatten(l: List[List[Any]]) -> List[Any]:
        return [x for sublist in l for x in sublist]

    def get_unexpanded_states(
        self, bpsg: Dict[StateType, Dict], explicit_graph: Dict[StateType, Dict]
    ) -> List[StateType]:
        return [
            s
            for s in bpsg
            if not explicit_graph[s]["expanded"] and not explicit_graph[s]["goal"]
        ]

    def add_state_graph(
        self, s: StateType, graph: Dict[StateType, Dict], goal: bool
    ) -> Dict[StateType, Dict]:
        new_graph = deepcopy(graph)
        new_graph[s] = {"Adj": [], "expanded": False, "goal": goal}
        return new_graph

    def expand_state(
        self,
        s: StateType,
        explicit_graph: Dict[StateType, Dict],
    ) -> Dict[StateType, Dict]:
        """Expand state 's' in the explicit graph (mdp).

        This routine adds outcome distributions and updates the
        adjacency list. It does not update the best partial solution
        graph (bpsg), which is handled later.
        """
        if explicit_graph[s]["goal"]:
            raise ValueError(f"State {s} can't be expanded because it is a goal state")

        new_explicit_graph = deepcopy(explicit_graph)
        for a in self.actions:
            dist = rollout_fn(self.transition_model, [s, a])
            for sp, p in dist.items():
                rew_dist = rollout_fn(self.reward_model, [s, a, sp])
                goal = list(rew_dist.keys())[0][0] > 0.0
                if sp not in new_explicit_graph:
                    new_explicit_graph = self.add_state_graph(
                        sp, new_explicit_graph, goal
                    )
                new_explicit_graph[s]["Adj"].append({"name": sp, "A": {a: p}})

        new_explicit_graph[s]["expanded"] = True

        return new_explicit_graph

    def __find_ancestors(
        self, s: StateType, bpsg: Dict[StateType, Dict], visited: set
    ) -> List[StateType]:
        direct_ancestors = [
            s_
            for s_ in bpsg
            if s_ != s
            and s_ not in visited
            and any(child["name"] == s for child in bpsg[s_]["Adj"])
        ]
        result = list(direct_ancestors)
        for a in direct_ancestors:
            if a not in visited:
                result += self.__find_ancestors(a, bpsg, visited.union({a}))
        return result

    def find_ancestors(
        self, s: StateType, bpsg: Dict[StateType, Dict]
    ) -> List[StateType]:
        return self.__find_ancestors(s, bpsg, set())

    def find_reachable(
        self, s: StateType, a: ActType, mdp: Dict[StateType, Dict]
    ) -> List[Dict]:
        """Return the list of objects in mdp[s]["Adj"] that contain action 'a'
        in their 'A' field."""
        return [obj for obj in mdp[s]["Adj"] if a in obj["A"]]

    def dfs_visit(
        self,
        i: int,
        colors: List[str],
        d: List[int],
        f: List[int],
        time: List[int],
        S: List[StateType],
        V_i: Dict[StateType, int],
        mdp: Dict[StateType, Dict],
    ) -> None:
        colors[i] = "g"
        time[0] += 1
        d[i] = time[0]
        s = S[i]
        for s_obj in mdp[s]["Adj"]:
            s_next = s_obj["name"]
            j = V_i[s_next]
            if colors[j] == "w":
                self.dfs_visit(j, colors, d, f, time, S, V_i, mdp)
        colors[i] = "b"
        time[0] += 1
        f[i] = time[0]

    def find_unreachable(
        self, s0: StateType, mdp: Dict[StateType, Dict]
    ) -> List[StateType]:
        S = list(mdp.keys())
        len_s = len(S)
        V_i = {S[i]: i for i in range(len_s)}
        colors = ["w"] * len_s
        d = [-1] * len_s
        f = [-1] * len_s
        time = [0]
        self.dfs_visit(V_i[s0], colors, d, f, time, S, V_i, mdp)
        return [S[i] for i, c in enumerate(colors) if c != "b"]

    def bellman(
        self,
        V: List[float],
        V_i: Dict[StateType, int],
        pi: List[Any],
        A: List[ActType],
        Z: List[StateType],
        mdp: Dict[StateType, Dict],
        c: float = 1,
        gamma: float = 0.9,
    ) -> Tuple[np.ndarray, List[Any]]:
        V_ = np.array(V)
        for s in Z:
            actions_results = []
            for a in A:
                reachable = self.find_reachable(s, a, mdp)
                c_ = 0 if mdp[s]["goal"] else c
                total = sum(V[V_i[obj["name"]]] * obj["A"][a] for obj in reachable)
                actions_results.append(c_ + gamma * total)
            i_min = int(np.argmin(actions_results))
            pi[int(V_i[s])] = A[int(i_min)]
            V_[int(V_i[s])] = actions_results[int(i_min)]
        return V_, pi

    def value_iteration(
        self,
        V: List[float],
        V_i: Dict[StateType, int],
        pi: List[Any],
        A: List[ActType],
        Z: List[StateType],
        mdp: Dict[StateType, Dict],
        c: float = 1,
        epsilon: float = 1e-3,
        n_iter: int = 1000,
        gamma: float = 0.9,
    ) -> Tuple[np.ndarray, List[Any]]:
        i = 1
        while True:
            V_, pi = self.bellman(V, V_i, pi, A, Z, mdp, c, gamma=gamma)
            if i == n_iter or np.linalg.norm(V_ - np.array(V), np.inf) < epsilon:
                break
            V = V_.tolist()
            i += 1
        return V_, pi

    def update_action_partial_solution(
        self,
        s: StateType,
        s0: StateType,
        V_i: Dict[StateType, int],
        pi: List[Any],
        bpsg: Dict[StateType, Dict],
        explicit_graph: Dict[StateType, Dict],
    ) -> Dict[StateType, Dict]:
        bpsg_ = deepcopy(bpsg)
        states = [s]
        while states:
            current = states.pop()
            a_current = pi[int(V_i[current])]
            s_obj = bpsg_.get(current, {"Adj": []})
            s_obj["Adj"] = []
            reachable = self.find_reachable(current, a_current, explicit_graph)
            for obj in reachable:
                s_next = obj["name"]
                s_obj["Adj"].append(
                    {"name": s_next, "A": {a_current: obj["A"][a_current]}}
                )
                if s_next not in bpsg_:
                    bpsg_ = self.add_state_graph(
                        s_next, bpsg_, explicit_graph[s_next]["goal"]
                    )
                    bpsg_[current] = s_obj
                    if explicit_graph[s_next]["expanded"]:
                        states.append(s_next)
        unreachable = self.find_unreachable(s0, bpsg_)
        for s_ in unreachable:
            if s_ in bpsg_:
                bpsg_.pop(s_)
        return bpsg_

    def update_partial_solution(
        self,
        pi: List[Any],
        V_i: Dict[StateType, int],
        s0: StateType,
        S: List[StateType],
        bpsg: Dict[StateType, Dict],
        mdp: Dict[StateType, Dict],
    ) -> Dict[StateType, Dict]:
        bpsg_ = deepcopy(bpsg)
        for s, a in zip(S, pi):
            if s not in bpsg_:
                continue
            s_obj = bpsg_[s]
            if len(s_obj.get("Adj", [])) == 0:
                if a is not None:
                    bpsg_ = self.update_action_partial_solution(
                        s, s0, V_i, pi, bpsg_, mdp
                    )
            else:
                best_current_action = next(iter(s_obj["Adj"][0]["A"].keys()))
                if a is not None and best_current_action != a:
                    bpsg_ = self.update_action_partial_solution(
                        s, s0, V_i, pi, bpsg_, mdp
                    )
        return bpsg_

    def convergence_test(
        self,
        V: List[float],
        V_i: Dict[StateType, int],
        pi: List[Any],
        A: List[ActType],
        Z: List[StateType],
        mdp: Dict[StateType, Dict],
        c: float = 1,
        epsilon: float = 1e-3,
        gamma: float = 1,
    ) -> Tuple[np.ndarray, List[Any], bool]:
        V_new, pi_new = self.value_iteration(
            deepcopy(V),
            V_i,
            deepcopy(pi),
            A,
            Z,
            mdp,
            c,
            epsilon,
            self.n_iter,
            gamma=gamma,
        )
        policy_stable = all(pi[int(V_i[s])] == pi_new[int(V_i[s])] for s in Z)
        value_diff = max(abs(V_new[int(V_i[s])] - V[int(V_i[s])]) for s in Z)
        converged = policy_stable and (value_diff < epsilon)
        return V_new, pi_new, converged

    def plan_next_action(
        self, current_state: StateType, max_steps: int
    ) -> Tuple[ActType, Dict]:
        # Initialize mdp, bpsg, V, etc.
        bpsg: Dict[StateType, Dict] = {}
        V: Dict[StateType, float] = {}
        bpsg[current_state] = {"Adj": [], "goal": False, "expanded": False}
        explicit_graph = deepcopy(bpsg)
        V[current_state] = self.heuristic(current_state)
        S: List[StateType] = list(explicit_graph.keys())
        V_i: Dict[StateType, int] = {s: i for i, s in enumerate(S)}
        pi: List[Any] = [None] * len(S)
        V_vec = np.array([V[s] for s in S])

        unexpanded = self.get_unexpanded_states(bpsg, explicit_graph)
        while True:
            while unexpanded:
                s = unexpanded[0]
                explicit_graph = self.expand_state(s, explicit_graph)

                # --- NEW: Update S, V, V_i, and pi with any new states ---
                new_states = [ns for ns in explicit_graph if ns not in S]
                if new_states:
                    for ns in new_states:
                        V[ns] = self.heuristic(ns)
                    S = list(explicit_graph.keys())
                    V_i = {s: i for i, s in enumerate(S)}
                    V_vec = np.array([V[s] for s in S])
                    # Extend the policy list for new states:
                    pi.extend([None] * (len(S) - len(pi)))
                # -------------------------------------------------------

                Z = self.find_ancestors(s, bpsg) + [s]
                V_vec, pi = self.value_iteration(
                    V_vec.tolist(),
                    V_i,
                    pi,
                    self.actions,
                    Z,
                    explicit_graph,
                    epsilon=self.epsilon,
                )
                V = {s: V_vec[V_i[s]] for s in S}
                bpsg = self.update_partial_solution(
                    pi, V_i, current_state, S, bpsg, explicit_graph
                )
                unexpanded = self.get_unexpanded_states(bpsg, explicit_graph)

            bpsg_states = [s for s in bpsg if not explicit_graph[s]["goal"]]
            V_vec, pi, converged = self.convergence_test(
                V_vec.tolist(),
                V_i,
                pi,
                self.actions,
                bpsg_states,
                explicit_graph,
                epsilon=self.epsilon,
            )
            if converged:
                break
            bpsg = self.update_partial_solution(
                pi, V_i, current_state, S, bpsg, explicit_graph
            )
            unexpanded = self.get_unexpanded_states(bpsg, explicit_graph)

        start_index = int(V_i[current_state])
        best_action = pi[start_index]
        if best_action is None:
            best_action = random.choice(self.actions)
        return best_action, {}

    def is_terminal(self, state: StateType) -> bool:
        """Assume a state is terminal (goal) if its heuristic value is zero."""
        return self.heuristic(state) == 0
