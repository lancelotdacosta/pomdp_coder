# type: ignore
import copy
import logging
import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

import uncertain_worms.environments.spot.pb_utils as pbu
from uncertain_worms.environments.spot.spot_constants import *
from uncertain_worms.structs import Observation, State

log = logging.getLogger(__name__)


NAVIGATION_STEP_SIZE = 5  # size of each step in the navigation
FRUSTUM_DEPTH = 3.0
ROTATION_ANGLE = [i * np.pi / 4.0 for i in range(8)]  # Angles for the robot to rotate
PICKUP_DISTANCE_THRESHOLD = 2.0  # Adjust this value as needed


class SpotActions(IntEnum):
    move_left = 0
    move_right = 1
    move_forward = 2
    move_backward = 3
    rotate_left = 4
    rotate_right = 5
    pickup = 6  # pick up the object if the object is in the camera's view


ARM_CONF = "ARM_STOW"


@dataclass
class AABB:
    lower: List[float, float, float]
    upper: List[float, float, float]


def pose_to_se2(pose):
    return [pose[0][0], pose[0][1], pbu.euler_from_quat(pose[1])[2]]


def se2_to_pose(se2):
    return pbu.Pose(point=pbu.Point(x=se2[0], y=se2[1]), euler=pbu.Euler(yaw=se2[2]))


def transformation_matrix(
    translation: NDArray[np.float64], quat: NDArray[np.float64]
) -> NDArray[np.float64]:
    r = R.from_quat(quat)
    rotation_matrix = r.as_matrix()
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T

class SpotActions(IntEnum):
    move_left = 0
    move_right = 1
    move_forward = 2
    move_backward = 3
    rotate_left = 4
    rotate_right = 5
    arm_stow = 6
    arm_left = 7
    arm_right = 8
    arm_down = 9
    pickup = 10  # pick up the object if the object is in the camera's view


@dataclass
class SceneObject:
    name: str
    location: List[int]
    aabb: AABB = None

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.location)))

    def __eq__(self, other: Any) -> bool:
        return hash(other) == hash(self)

    def __repr__(self) -> str:
        return (
            'SceneObject(name="'
            + str(self.name)
            + '", location='
            + str(self.location)
            + ", aabb="
            + str(self.aabb)
            + ")"
        )


@dataclass
class SpotState(State):
    body_location: List[
        int
    ]  # x voxel index, y voxel index, rotation index into ROTATION_ANGLE
    occupancy_grid: OccupancyGrid
    visibility_grid: VisibilityGrid
    movable_objects: List[SceneObject] = field(default_factory=list)
    fixed_objects: List[SceneObject] = field(default_factory=list)
    carry_object: Optional[SceneObject] = None
    ons: List[Tuple[str, str]] = field(
        default_factory=list
    )  # what object is on what other object

    def __repr__(self) -> str:
        return f"SpotState(body_location={self.body_location}, movable_objects={str([o for o in self.movable_objects])}, carry_object={self.carry_object}, ons={self.ons}, fixed_objects={str([o for o in self.fixed_objects])})"

    @property
    def camera_pose(self):
        return pbu.multiply(
            se2_to_pose(self.occupancy_grid.to_world(self.body_location)),
            CAMERA_POSES[ARM_CONF],
        )

@dataclass
class SpotObservation(Observation):
    body_location: List[
        int
    ]  # x voxel index, y voxel index, rotation index into ROTATION_ANGLE
    visible_movable_objects: List[SceneObject] = field(default_factory=list)
    carry_object: Optional[SceneObject] = None

    def __repr__(self) -> str:
        return f"SpotObservation(body_location={self.body_location}, carry_object={self.carry_object}, visible_movable_objects={str([o for o in self.visible_movable_objects])})"

    @property
    def camera_pose(self):
        return CAMERA_POSES[ARM_CONF]


class OccupancyGrid:
    def check_collision(self, body_location: Tuple[int, int, int]) -> bool:
        """Returns the collision result for the given robot body location."""
        ...

    def from_world(
        self, world_state: Tuple[float, float, float]
    ) -> Tuple[int, int, int]:
        """Converts a world state (x, y, theta) into a discrete occupancy grid
        state (row, col, theta_index)."""
        ...

    def to_world(
        self, occupancy_grid_state: Tuple[int, int, int]
    ) -> Tuple[float, float, float]:
        """Converts an occupancy grid state (row, col, theta_index) to world
        coordinates."""
        ...

    @property
    def grid_size(self) -> Tuple[int, int]:
        """Returns the size of the occupancy grid."""
        ...


class VisibilityGrid:
    def from_world(
        self, world_state: Tuple[float, float, float]
    ) -> Tuple[int, int, int]:
        """Converts a world coordinates (x, y, theta) into a discrete
        visibility grid state (row, col, theta_index)."""
        ...

    def to_world(self, grid_state: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Converts a visibility grid state (row, col, theta_index) to world
        coordinates."""
        ...

    def get_voxels_above_aabb(self, aabb: AABB) -> NDArray[np.int64]:
        """Returns the indices of voxels in the visibility grid whose centers
        are directly above the given aabb.

        IMPORTANT: You must use this function to get the location of a goal that is on top of a fixed object.
        """
        ...

    @property
    def grid_size(self) -> Tuple[int, int, int]:
        """Returns the size of the occupancy grid."""
        ...
