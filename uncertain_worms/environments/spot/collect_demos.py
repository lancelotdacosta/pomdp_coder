# type:ignore
"""collect_spot_demos.py.

This script allows a user to manually control a Spot agent via keyboard input to collect demos.
The controls are:
    w: move forward
    s: move backward
    a: move left
    d: move right
    q: rotate left
    e: rotate right
    i: arm stow
    j: arm left
    l: arm right
    k: arm down
    p: pickup
    f: finish the current episode

After collecting the defined number of demo episodes, the demos are saved to a replay buffer file.
Additionally, the rerun state is updated after each step for visualization/debug purposes.
"""

import os
import random

# Import the SpotEnvironment and Actions enum.
# (Assumes these are available in your uncertain_worms package.)
from uncertain_worms.environments.spot.spot_env import SpotActions, SpotEnvironment
from uncertain_worms.structs import Observation, ReplayBuffer, State
from uncertain_worms.utils import PROJECT_ROOT


def main():
    # Set the number of demos and maximum steps per episode.
    num_demos = 10

    # Define the graph navigation file (for example, "spot_room_graphnav").
    # graphnav = "spot_room_graphnav"
    # fixed_object_names = ["cabinet1", "cabinet2", "cabinet3"]
    # max_steps = 60  # Adjust as needed.

    graphnav = "open_area_graphnav"
    fixed_object_names = [
        "small-table-1",
        "small-table-2",
        "small-table-3",
        "small-table-4",
        "small-table-5",
    ]
    max_steps = 120  # Adjust as needed.

    # Initialize an empty replay buffer.
    replay_buffer = ReplayBuffer[State, int, Observation]()

    # Create the Spot environment.
    # Set render_pygame to False to disable the pygame window during demo collection;
    # set real_spot to False to run in simulation.
    env = SpotEnvironment(
        graphnav=graphnav,
        max_steps=max_steps,
        fully_obs=False,
        render_pygame=False,
        real_spot=False,
        fixed_object_names=fixed_object_names,
    )

    # Define key-to-action mapping for the Spot environment.
    key_action_mapping = {
        "w": SpotActions.move_forward,
        "s": SpotActions.move_backward,
        "a": SpotActions.move_left,
        "d": SpotActions.move_right,
        "q": SpotActions.rotate_left,
        "e": SpotActions.rotate_right,
        "p": SpotActions.pickup,
        "f": "finish",
    }

    print("Spot demo collection started.")
    print("Controls:")
    print("  w: move forward")
    print("  s: move backward")
    print("  a: move left")
    print("  d: move right")
    print("  q: rotate left")
    print("  e: rotate right")
    print("  i: arm stow")
    print("  j: arm left")
    print("  l: arm right")
    print("  k: arm down")
    print("  p: pickup")
    print("  f: finish the current episode")

    for demo in range(num_demos):
        print(f"\n=== Starting demo episode {demo + 1}/{num_demos} ===")
        # Reset the environment with a random seed.
        initial_state = env.reset(seed=random.randint(0, 10000))

        print("Initial state:")
        print(initial_state)

        # Initialize episode tracking.
        previous_state = initial_state
        terminated = False
        step = 0

        while not terminated:
            user_input = (
                input("Enter action (w/s/a/d/q/e/i/j/l/k/p/f): ").strip().lower()
            )
            if user_input not in key_action_mapping:
                print("Invalid input. Please try again.")
                continue
            if key_action_mapping[user_input] == "finish":
                print("Finishing current episode.")
                break

            action = key_action_mapping[user_input]
            next_obs, next_state, reward, terminated, truncated, info = env.step(action)
            print(
                f"Step {step}: Action {action}, Reward: {reward}, Terminated: {terminated}"
            )
            print("Next state:")
            print(next_state)
            print("Next observation:")
            print(next_obs)

            # Append the transition to the replay buffer.
            replay_buffer.append_episode_step(
                previous_state,
                next_state,
                action,
                next_obs,
                reward,
                terminated,
            )
            previous_state = next_state
            step += 1

            # # Update the rerun state (if available) for visualization/debug purposes.
            # if env.rerun_spot is not None:
            #     env.update_rerun_state(env.rerun_spot, env.current_state, step)

            if terminated or truncated:
                break

        # Finish the current episode in the replay buffer.
        replay_buffer.wrap_up_episode()
        print(f"Demo episode {demo + 1} finished.\n")

    # Define a filename and save the replay buffer.
    save_path = os.path.join(
        PROJECT_ROOT, "environments/spot/trajectory_data", graphnav + ".pkl"
    )
    replay_buffer.save_to_file(save_path)
    print(f"Demos saved to {save_path}")


if __name__ == "__main__":
    main()
