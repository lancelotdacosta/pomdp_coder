import os
import pickle

from uncertain_worms.structs import ReplayBuffer

if __name__ == "__main__":
    fn = "/Users/aidancurtis/uncertain-world-models/uncertain_worms/environments/spot/trajectory_data/spot_room_graphnav.pkl"
    # Load the pickle file as replay buffer
    with open(fn, "rb") as f:
        replay_buffer = pickle.load(f)

    print([replay_buffer.episodes[i].previous_states[0].ons for i in range(10)])

    replay_buffer.episodes = replay_buffer.episodes[5:] + replay_buffer.episodes[:5]

    # Save the modified replay buffer back to a pickle file
    with open(fn, "wb") as f:
        pickle.dump(replay_buffer, f)

    print("Saved")
