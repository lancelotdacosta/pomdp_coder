# type:ignore
import json
import os
from collections import deque
from typing import Optional

import rerun as rr

from uncertain_worms.environments.spot.spot_env import (
    PROJECT_ROOT,
    SpotActions,
    SpotEnvironment,
    SpotState,
    compute_frustum,
)


def bfs_collect_states(
    initial_state: SpotState, transition_model, max_depth: Optional[int] = None
) -> set:
    """Performs BFS from the given initial state using the provided transition
    model. It returns a set of all reachable states.

    Args:
        initial_state (SpotState): The starting state.
        transition_model (function): Function that maps (state, action) -> next_state.
        max_depth (Optional[int]): If provided, limits the search depth.

    Returns:
        A set of SpotState objects representing all reachable states.
    """
    visited = set()
    queue = deque()

    # Start with the initial state at depth 0.
    queue.append((initial_state, 0))
    visited.add(initial_state)

    while queue:
        current_state, depth = queue.popleft()
        # If a depth limit is provided, do not explore further beyond it.
        if max_depth is not None and depth >= max_depth:
            continue

        # Iterate over all possible actions.
        for action in SpotActions:
            next_state = transition_model(current_state, action)
            # Only add new states that haven't been visited.
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, depth + 1))

    return visited


# Example usage within your main code:
if __name__ == "__main__":
    # Set up your environment as before.
    env = SpotEnvironment(
        graphnav="spot_room_graphnav",
        # graphnav="open_area_graphnav",
        render_pygame=True,
        real_spot=False,
        fully_obs=False,
    )
    env.reset()

    # Use the current state of the environment as the BFS starting point.
    initial_state = env.current_state

    # Collect all reachable states (for example, limit to depth 10 to avoid long runtime).
    reachable_states = bfs_collect_states(
        initial_state, env.transition_model, max_depth=float("inf")
    )

    for si, state in enumerate(reachable_states):
        state.visibility_grid.clear_voxels_within_frustum(state.camera_pose)
        rr.set_time_sequence("timeline", si)
        env.plot_voxel_grid()

    env.plot_voxel_grid()

    print(f"Number of reachable states:", len(reachable_states))

    # Construct the path to the graphnav folder.
    graphnav_folder = os.path.join(
        PROJECT_ROOT, "environments/spot/world_scans", env.graphnav
    )
    json_file_path = os.path.join(graphnav_folder, "unviewed_voxels.json")

    # Write the list to the JSON file.
    with open(json_file_path, "w") as f:
        json.dump(state.visibility_grid.get_voxels().tolist(), f)

    print(f"Saved unviewed voxel indices to {json_file_path}")
