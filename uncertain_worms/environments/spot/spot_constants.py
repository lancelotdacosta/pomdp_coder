import numpy as np

# --- New robot dimensions and rotation angle ---
ROBOT_LENGTH = 1.2  # Length of the robot (meters)
ROBOT_WIDTH = 0.45  # Width of the robot (meters)

ROTATION_ANGLE = [i * np.pi / 4.0 for i in range(8)]  # Rotation angles in radians
VOXEL_SIZE = 0.1  # Adjust as needed
FOCAL_LENGTH = 800
IMG_SIZE = (480, 640)
NAVIGATION_STEP_SIZE = 5
PADDING_VIS_ABOVE = 0.25
PADDING_VIS_BELOW = 0.1
POINT_RADIUS = 0.1
OCCUPANCY_RESOLUTION = 0.05
FRUSTUM_DEPTH = 3.0
PICKUP_RADIUS = 1.5
SPOT_IP = "192.168.80.3"

ARM_LIMITS = {
    "arm0.sh0": (-2.61799387799149441136, 3.14159265358979311599),
    "arm0.sh1": (-3.14159265358979311599, 0.52359877559829881565),
    "arm0.hr0": (-1e6, 1e6),
    "arm0.el0": (0, 3.14159265358979311599),
    "arm0.el1": (-2.79252680319092716487, 2.79252680319092716487),
    "arm0.wr0": (-1.83259571459404613236, 1.83259571459404613236),
    "arm0.wr1": (-2.87979326579064354163, 2.87979326579064354163),
    "arm0.f1x": (-1.57, 0.0),
}

ARM_CONFS = {
    # "ARM_STOW": {
    #     "arm0.sh0": 0.0,
    #     "arm0.sh1": -2.6,
    #     "arm0.hr0": 0.0,
    #     "arm0.el0": 3.129352569580078,
    #     "arm0.el1": 1.5654863119125366,
    #     "arm0.wr0": 0.0,
    #     "arm0.wr1": -1.5699973106384277,
    #     "arm0.f1x": 0.0,
    # },
    "ARM_STOW": {  # To look above the chairs in open area
        "arm0.sh0": 0.0,
        "arm0.sh1": -2.1,
        "arm0.hr0": 0.0,
        "arm0.el0": 2.5,
        "arm0.el1": 1.5654863119125366,
        "arm0.wr0": 0.0,
        "arm0.wr1": -1.5699973106384277,
        "arm0.f1x": 0.0,
    },
    "ARM_DOWN": {
        "arm0.sh0": 0.0,
        "arm0.sh1": -np.pi + 1.5,
        "arm0.hr0": 0.0,
        "arm0.el0": np.pi - 2.0,
        "arm0.el1": 0.0,
        "arm0.wr0": 1.83,
        "arm0.wr1": 0.0,
        "arm0.f1x": 0.0,
    },
    # For open area
    "ARM_LEFT": {
        "arm0.sh0": np.pi / 4.0,
        "arm0.sh1": -np.pi + 1.5,
        "arm0.hr0": 0.0,
        "arm0.el0": np.pi - 2.0,
        "arm0.el1": 0.0,
        "arm0.wr0": 1.23,
        "arm0.wr1": 0.0,
        "arm0.f1x": 0.0,
    },
    "ARM_RIGHT": {
        "arm0.sh0": -np.pi / 4.0,
        "arm0.sh1": -np.pi + 1.5,
        "arm0.hr0": 0.0,
        "arm0.el0": np.pi - 2.0,
        "arm0.el1": 0.0,
        "arm0.wr0": 1.23,
        "arm0.wr1": 0.0,
        "arm0.f1x": 0.0,
    },
}


CAMERA_INTRINSICS = {
    "frontleft": np.array(
        [[330.1010437, 0, 306.50964355], [0, 329.146698, 237.60688782], [0, 0, 1]]
    ),
    "frontright": np.array(
        [[330.21432495, 0, 315.07336426], [0, 329.23696899, 240.25497437], [0, 0, 1]]
    ),
    "left": np.array(
        [[329.43948364, 0, 317.31555176], [0, 328.62130737, 244.0030365], [0, 0, 1]]
    ),
    "right": np.array(
        [[329.60354614, 0, 313.66186523], [0, 328.55203247, 235.0330658], [0, 0, 1]]
    ),
    "back": np.array(
        [[329.12207031, 0, 316.90737915], [0, 328.21459961, 242.75737], [0, 0, 1]]
    ),
    "hand": np.array([[552.02910122, 0, 320], [0, 552.02910122, 240], [0, 0, 1]]),
}


def assert_joint_limits() -> None:
    for conf_name, conf in ARM_CONFS.items():
        for joint, value in conf.items():
            lower, upper = ARM_LIMITS[joint]
            assert lower <= value <= upper, (
                f"Configuration '{conf_name}' for joint '{joint}' is out of limits: "
                f"{value} not in [{lower}, {upper}]"
            )
    print("All joint configurations are within limits.")


assert_joint_limits()

CAMERA_POSES = {
    "ARM_STOW": (
        (0.5194184184074402, 0.02031540311872959, 0.17494438588619232),
        (
            0.5395318865776062,
            -0.5365196466445923,
            0.4593556225299835,
            -0.4583059251308441,
        ),
    ),
    "ARM_DOWN": (
        (0.642764151096344, 0.020205000415444374, 0.6565937399864197),
        (
            0.6743023991584778,
            -0.674302339553833,
            0.21287639439105988,
            -0.21287639439105988,
        ),
    ),
    "ARM_LEFT": (
        (0.5699304342269897, 0.30650463700294495, 0.7110267281532288),
        (
            -0.7594740986824036,
            0.31458449363708496,
            -0.2179063856601715,
            0.52607262134552,
        ),
    ),
    "ARM_RIGHT": (
        (0.5985046625137329, -0.27793043851852417, 0.7110267281532288),
        (
            -0.31458449363708496,
            0.7594741582870483,
            -0.5260725617408752,
            0.2179064154624939,
        ),
    ),
}

DEFAULT_CONF = {
    "x": 0,
    "y": 0,
    "z": 0,
    "theta": 0,
    "fl.hx": 0.0,
    "fl.hy": 0.8,
    "fl.kn": -1.6,
    "fr.hx": 0.0,
    "fr.hy": 0.8,
    "fr.kn": -1.6,
    "hl.hx": 0.0,
    "hl.hy": 0.8,
    "hl.kn": -1.6,
    "hr.hx": 0.0,
    "hr.hy": 0.8,
    "hr.kn": -1.6,
} | ARM_CONFS["ARM_DOWN"]

JOINT_NAMES = list(DEFAULT_CONF.keys())
