# type: ignore

import os
import shutil
from pathlib import Path

import yaml

OUTPUT_DIR = "experiments"


MAX_STEPS = {
    "MiniGrid-Empty-5x5-v0": 100,
    "CornerGoalRandom-Empty-10x10-v0": 100,
    "MyMiniGrid-LavaWall-v0": 100,
    "MyMiniGrid-FourRooms-v0": 100,
    "MyUnlockEnv-v0": 100,
    "tiger": 20,
    "rocksample": 20,
}

MG_ENV_NAMES = [
    "MiniGrid-Empty-5x5-v0",
    "CornerGoalRandom-Empty-10x10-v0",
    "MyMiniGrid-LavaWall-v0",
    "MyMiniGrid-FourRooms-v0",
    "MyUnlockEnv-v0",
]

TOY_ENV_NAMES = ["tiger", "rocksample"]

ENV_NAMES = MG_ENV_NAMES + TOY_ENV_NAMES

def create_seeded_yamls(exp_yaml, env_name, seeds):
    input_yaml_path = os.path.join(os.path.dirname(__file__), "approaches", exp_yaml)
    # Read the input yaml file
    with open(input_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    data["save_log"] = True
    data["num_episodes"] = 10
    data["max_steps"] = MAX_STEPS[env_name]
    total = 0
    # Get the base filename without extension
    base_name = Path(input_yaml_path).stem

    # if "Unlock" not in env_name:
    #     continue
    data["name"] = exp_yaml.replace(".yaml", "") + f"__envname_{env_name}"
    # Create N copies with different seeds
    for seed in seeds:
        # Add or update the seed in the yaml data
        if isinstance(data, dict):
            data["seed"] = seed
            data["env_name"] = env_name

        # Create new filename with seed
        new_filename = f"{base_name}__seed_{seed}__envname_{env_name}.yaml"
        output_path = os.path.join(OUTPUT_DIR, new_filename)

        total += 1
        # Write the new yaml file
        with open(output_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    print(f"Created {total} yaml files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    """Currently, we run experiments by simply copying existing yaml templates
    and modifying the seed.

    Eventually, we will have more dimensions along which we vary the
    experiments
    """

    # Remove output directory if it exists and create it fresh
    if Path(OUTPUT_DIR).exists():
        shutil.rmtree(OUTPUT_DIR)
    Path(OUTPUT_DIR).mkdir()

    seeds = [0, 1, 2, 3, 4]  # Number of copies you want to create
        
    # for env_name in MG_ENV_NAMES:    
    #     # create_seeded_yamls("direct/direct_llm_po_agent.yaml", env_name, seeds)
    #     # create_seeded_yamls("hardcoded/hardcoded_po_planning_agent.yaml", env_name, seeds)
        
    #     create_seeded_yamls("ours/llm_TROI_po_planning_agent.yaml", env_name, seeds)
    #     create_seeded_yamls("online_ours/online_llm_TROI_po_planning_agent.yaml", env_name, seeds)
    #     create_seeded_yamls("offline_ours/offline_llm_TROI_po_planning_agent.yaml", env_name, seeds)

    #     # create_seeded_yamls("behavior_cloning/bc_po_agent.yaml", env_name, seeds)
    #     # create_seeded_yamls("random/random_po_agent.yaml", env_name, seeds)
    #     # create_seeded_yamls("tabular/tabular_TROI_po_planning_agent.yaml", env_name, seeds)

    # Toys
    for env_name in TOY_ENV_NAMES:
        create_seeded_yamls(f"direct/{env_name}_direct_llm_po_agent.yaml", env_name, seeds)
        create_seeded_yamls(f"hardcoded/{env_name}_hardcoded.yaml", env_name, seeds)

        create_seeded_yamls(f"ours/{env_name}_llm_TROI_po_planning_agent.yaml", env_name, seeds)
        create_seeded_yamls(f"online_ours/{env_name}_online_TROI_po_planning_agent.yaml", env_name, seeds)
        create_seeded_yamls(f"offline_ours/{env_name}_offline_TROI_po_planning_agent.yaml", env_name, seeds)

        create_seeded_yamls(f"behavior_cloning/{env_name}_bc_po_agent.yaml", env_name, seeds)
        create_seeded_yamls(f"random/{env_name}_random.yaml", env_name, seeds)
        create_seeded_yamls(f"tabular/{env_name}_tabular_TROI_po_planning_agent.yaml", env_name, seeds)
