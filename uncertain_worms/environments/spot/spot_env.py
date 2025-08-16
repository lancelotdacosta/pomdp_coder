# type: ignore
from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import dotenv
import numpy as np
import open3d as o3d  # type:ignore
import rerun as rr
import yaml  # type:ignore
import zmq
from bosdyn.client import math_helpers  # type:ignore
from bosdyn.client.robot_command import blocking_stand  # type:ignore
from numpy.typing import NDArray
from PIL import Image
from rerun_robotics import load_spot  # type:ignore
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R  # type:ignore

import uncertain_worms.environments.spot.pb_utils as pbu
from uncertain_worms.environments.spot.spot_constants import *
from uncertain_worms.environments.spot.spot_utils.controllers.arm_control import (
    grasp,
    move_arm,
    move_arm_relative,
    open_gripper,
)
from uncertain_worms.environments.spot.spot_utils.controllers.startup import (
    RobotClient,
    go_home,
    navigate_to_absolute_pose,
    setup_robot,
)
from uncertain_worms.environments.spot.spot_utils.perception.capture import capture_rgbd
from uncertain_worms.environments.spot.spot_utils.structures.robot import (
    ArmJointPositions,
)
from uncertain_worms.structs import (
    ActType,
    Environment,
    InitialModel,
    Observation,
    ObservationModel,
    RewardModel,
    State,
    TransitionModel,
)
from uncertain_worms.utils import PROJECT_ROOT, get_log_dir

log = logging.getLogger(__name__)


def transformation_matrix(
    translation: NDArray[np.float64], quat: NDArray[np.float64]
) -> NDArray[np.float64]:
    r = R.from_quat(quat)
    rotation_matrix = r.as_matrix()
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T


def pose_to_se2(pose):
    return [pose[0][0], pose[0][1], pbu.euler_from_quat(pose[1])[2]]


def se2_to_pose(se2):
    return pbu.Pose(point=pbu.Point(x=se2[0], y=se2[1]), euler=pbu.Euler(yaw=se2[2]))


PICKUP_DISTANCE_THRESHOLD = 2.0  # Adjust this value as needed


@dataclass
class AABB:
    lower: List[float, float, float]
    upper: List[float, float, float]


class SpotActions(IntEnum):
    move_left = 0
    move_right = 1
    move_forward = 2
    move_backward = 3
    rotate_left = 4
    rotate_right = 5
    pickup = 6  # pick up the object if the object is in the camera's view


ARM_CONF = "ARM_STOW"


class OccupancyGrid:
    def __init__(
        self,
        points: NDArray[np.float64],
        min_bound,
        max_bound,
        home_pose,
        min_height: float = -0.2,
        max_height: float = 0.5,
        sample_resolution: float = OCCUPANCY_RESOLUTION,
    ) -> None:
        self.sample_resolution = sample_resolution
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.home_pose = home_pose
        # Filter points based on height
        valid_points = points[
            (points[:, 2] >= min_height) & (points[:, 2] <= max_height)
        ]
        points_2d = valid_points[:, :2]

        grid_width = int(np.ceil((self.max_bound[0] - self.min_bound[0]) / VOXEL_SIZE))
        grid_height = int(np.ceil((self.max_bound[1] - self.min_bound[1]) / VOXEL_SIZE))

        # Create occupancy grid (1 indicates an obstacle)
        self.grid = np.zeros((grid_height, grid_width), dtype=int)
        cols = ((points_2d[:, 0] - self.min_bound[0]) / VOXEL_SIZE).astype(int)
        rows = ((points_2d[:, 1] - self.min_bound[1]) / VOXEL_SIZE).astype(int)
        self.grid[rows, cols] = 1

        # Precompute the collision cache for all configurations
        self.collision_cache = {}
        self.precompute_collision_cache()

    @property
    def grid_size(self) -> Tuple[int, int]:
        """Returns the size of the occupancy grid."""
        return self.grid.shape

    def __deepcopy__(self, memo):
        return self

    def from_world(
        self, world_state: Tuple[float, float, float]
    ) -> Tuple[int, int, int]:
        """Converts a world state (x, y, theta) into a discrete grid state
        (row, col, theta_index).

        The (row, col) is determined by which grid cell the (x, y) falls
        into. The theta_index is chosen as the index of the closest
        angle in ROTATION_ANGLE.
        """
        x, y, theta = world_state

        # Compute grid indices from world coordinates.
        col = int((x - self.min_bound[0]) / VOXEL_SIZE)
        row = int((y - self.min_bound[1]) / VOXEL_SIZE)

        # Determine the discrete orientation index by choosing the angle in ROTATION_ANGLE closest to theta.
        # We take care of angle wrap-around by using a modulo operation.
        differences = np.array(
            [abs(((theta - a + np.pi) % (2 * np.pi)) - np.pi) for a in ROTATION_ANGLE]
        )
        theta_index = int(np.argmin(differences))

        return row, col, theta_index

    def to_world(self, grid_state: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Converts a grid state (row, col, theta_index) to world coordinates.

        The (x, y) coordinate is computed as the center of the
        corresponding grid cell.
        """
        row, col, theta_index = grid_state
        x = self.min_bound[0] + (col + 0.5) * VOXEL_SIZE
        y = self.min_bound[1] + (row + 0.5) * VOXEL_SIZE
        return x, y, ROTATION_ANGLE[theta_index]

    def _compute_collision(self, grid_state: Tuple[int, int, int]) -> bool:
        """Computes whether a given configuration (grid state) is in collision.

        This method implements the same logic as your original
        check_collision.
        """
        x, y, theta = self.to_world(grid_state)
        half_length = ROBOT_LENGTH / 2.0
        half_width = ROBOT_WIDTH / 2.0

        xs = np.arange(-half_length, half_length, self.sample_resolution)
        ys = np.arange(-half_width, half_width, self.sample_resolution)
        # Generate local robot points for sampling
        local_points = np.array([[xi, yi] for xi in xs for yi in ys])
        # Build rotation matrix
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        # Transform robot points to world frame
        world_points = (rot @ local_points.T).T + np.array([x, y])

        # Check if any robot point is out-of-bounds.
        in_bounds = (
            (world_points[:, 0] >= self.min_bound[0])
            & (world_points[:, 0] <= self.max_bound[0])
            & (world_points[:, 1] >= self.min_bound[1])
            & (world_points[:, 1] <= self.max_bound[1])
        )
        if not np.all(in_bounds):
            return True  # Collision because part of the robot is out-of-bounds

        # Map the world points to grid indices
        grid_min = self.min_bound[:2]
        grid_height, grid_width = self.grid.shape
        rows = ((world_points[:, 1] - grid_min[1]) / VOXEL_SIZE).astype(int)
        cols = ((world_points[:, 0] - grid_min[0]) / VOXEL_SIZE).astype(int)
        # Clip indices to grid boundaries
        rows = np.clip(rows, 0, grid_height - 1)
        cols = np.clip(cols, 0, grid_width - 1)
        # Check if any of the points hit an obstacle
        collisions = self.grid[rows, cols] == 1
        return np.any(collisions)

    def precompute_collision_cache(self):
        """Iterates over all discrete grid states (each cell and each rotation
        angle) and precomputes the collision result."""
        grid_height, grid_width = self.grid.shape
        num_angles = len(ROTATION_ANGLE)
        for row in range(grid_height):
            for col in range(grid_width):
                for theta_index in range(num_angles):
                    state = (row, col, theta_index)
                    self.collision_cache[state] = self._compute_collision(state)

    def check_collision(self, grid_state: Tuple[int, int, int]) -> bool:
        """Returns the precomputed collision result for the given grid state.

        If the state is not found in the cache (which should not occur
        if precomputation was done correctly), it defaults to returning
        True.
        """
        return self.collision_cache.get(tuple(grid_state), True)


class VisibilityGrid:
    def __init__(
        self,
        min_bound: NDArray[np.float64],
        max_bound: NDArray[np.float64],
        points_array: NDArray[np.float64],
        colors_array: NDArray[np.float64],
        unviewable_voxels: Optional[NDArray[np.float64]] = None,
    ) -> None:
        self.points_array = points_array
        self.colors_array = colors_array
        """Initializes a 3D occupancy grid covering the extent of the point
        cloud and clears voxels containing initial points."""
        self.min_bound = np.array(min_bound)
        self.max_bound = np.array(max_bound)

        # Compute number of voxels along each axis.
        self.num_voxels = np.ceil(
            (self.max_bound - self.min_bound) / VOXEL_SIZE
        ).astype(int)

        # Create a 3D boolean array; True indicates the voxel is occupied.
        self.grid = np.ones(self.num_voxels, dtype=bool)

        # Clear initial points from the grid
        self.clear_voxels_below_floor(points_array)
        if unviewable_voxels is not None:
            for unviewable_voxel in unviewable_voxels:
                self.grid[*unviewable_voxel] = False

        self.initial_grid = copy.deepcopy(self.grid)

    @property
    def grid_size(self) -> Tuple[int, int, int]:
        """Returns the size of the occupancy grid."""
        return self.grid.shape

    def get_voxels_above_aabb(self, aabb: AABB) -> NDArray[np.int64]:
        """Returns the indices of voxels in the visibility grid whose centers
        are directly above the given OOBB."""

        # Unpack the axis-aligned bounding box corners.
        min_corner, max_corner = aabb.lower, aabb.upper
        # Use the maximum z value as the top of the OOBB.
        top_z = max_corner[2]

        # Get all indices of voxels that are currently "active" in the grid.
        voxel_indices = np.argwhere(self.grid)  # shape (n, 3)
        # Convert voxel indices to world coordinates (centers).
        voxel_centers = self.to_world(voxel_indices)  # shape (n, 3)

        # Create masks for the x and y coordinates to be within the oobb's footprint.
        mask_x = (voxel_centers[:, 0] >= min_corner[0]) & (
            voxel_centers[:, 0] <= max_corner[0]
        )
        mask_y = (voxel_centers[:, 1] >= min_corner[1]) & (
            voxel_centers[:, 1] <= max_corner[1]
        )
        # And for the z coordinate to be above the top of the OOBB.
        mask_z = voxel_centers[:, 2] > top_z

        # Combine the masks.
        mask = mask_x & mask_y & mask_z
        above_voxels = voxel_indices[mask]

        return above_voxels

    def reset(self):
        self.grid = copy.deepcopy(self.initial_grid)

    def __deepcopy__(self, memo):
        return self

    def closest_unviewed_voxel(self, voxel) -> Optional[List[int]]:
        # Compute the distance to each unviewed voxel
        unviewed_voxels = np.argwhere(self.grid)
        if len(unviewed_voxels) == 0:
            return None
        distances = np.linalg.norm(unviewed_voxels - voxel, axis=1)
        # Find the index of the closest unviewed voxel

        closest_index = np.argmin(distances)
        return unviewed_voxels[closest_index]

    def populate_grid(
        self, initial_model: InitialModel, empty_state: SpotState
    ) -> None:
        spot_states = [initial_model(empty_state) for _ in range(20000)]

        self.grid[:] = False
        for spot_state in spot_states:
            self.grid[*spot_state.movable_objects[0].location] = True

    def clear_voxels_below_floor(self, points: NDArray[np.float64]) -> None:
        """Clears voxels below the floor (lowest z point) of the pointcloud and
        ensures only a single layer of voxels exists above the terrain.

        Args:
            points: Nx3 array of points representing the terrain/floor.
        """
        # Convert points to voxel indices
        voxel_indices = self.from_world(points)

        # Create a 2D grid to store the highest z-value for each (x,y) position
        max_z_grid = np.full((self.num_voxels[0], self.num_voxels[1]), -1, dtype=int)

        # Process only the points that are within the voxel grid boundaries
        for i, (x, y, z) in enumerate(voxel_indices):
            # Check if the point is within the voxel grid boundaries
            if (
                0 <= x < self.num_voxels[0]
                and 0 <= y < self.num_voxels[1]
                and 0 <= z < self.num_voxels[2]
            ):
                # Update the max z value for this (x,y) position
                if z > max_z_grid[x, y]:
                    max_z_grid[x, y] = z
            # Handle points below the grid (z < 0) but within x,y bounds
            elif 0 <= x < self.num_voxels[0] and 0 <= y < self.num_voxels[1] and z < 0:
                # Mark these positions with a special value (e.g., -2) to indicate floor below grid
                if (
                    max_z_grid[x, y] == -1
                ):  # Only update if not already set by an in-bounds point
                    max_z_grid[x, y] = -2

        for x in range(self.num_voxels[0]):
            for y in range(self.num_voxels[1]):
                max_z = max_z_grid[x, y]
                if max_z >= 0:
                    # Clear voxels below and including the floor voxel
                    self.grid[x, y, : max_z + 1] = False
                    # Clear voxels above the layer immediately above the floor
                    if max_z + 1 < self.num_voxels[2]:
                        self.grid[x, y, max_z + 2 :] = False
                        # Activate the voxel immediately above the floor
                        self.grid[x, y, max_z + 1] = True
                elif max_z == -2:
                    # For floor points below the grid: set bottom layer active
                    self.grid[x, y, 1:] = False
                    self.grid[x, y, 0] = True
                else:  # max_z == -1, meaning no valid points found at this (x, y)
                    self.grid[x, y, :] = False
                    # Set the default voxel at the bottom of the grid
                    self.grid[x, y, 0] = True

    def clear_voxels_containing_points(self, points: NDArray[np.float64]) -> None:
        points = np.asarray(points)
        voxel_indices = self.from_world(points)

        offsets = np.array(
            [
                [dx, dy, dz]
                for dx in range(-1, 2)
                for dy in range(-1, 2)
                for dz in range(-1, 2)
            ]
        ) * int(np.ceil(POINT_RADIUS / VOXEL_SIZE))

        expanded_voxel_indices = voxel_indices[:, None, :] + offsets[None, :, :]
        expanded_voxel_indices = expanded_voxel_indices.reshape(-1, 3)

        valid = (
            (expanded_voxel_indices[:, 0] >= 0)
            & (expanded_voxel_indices[:, 0] < self.num_voxels[0])
            & (expanded_voxel_indices[:, 1] >= 0)
            & (expanded_voxel_indices[:, 1] < self.num_voxels[1])
            & (expanded_voxel_indices[:, 2] >= 0)
            & (expanded_voxel_indices[:, 2] < self.num_voxels[2])
        )
        expanded_voxel_indices = expanded_voxel_indices[valid]

        if expanded_voxel_indices.shape[0] == 0:
            return

        flat_indices = np.ravel_multi_index(expanded_voxel_indices.T, self.num_voxels)
        self.grid.flat[flat_indices] = False

    def clear_voxels_within_frustum(
        self,
        camera_pose_world: Tuple[
            Tuple[float, float, float], Tuple[float, float, float, float]
        ],
    ) -> List[np.ndarray]:
        """Clears voxels within the camera frustum."""

        # Grab indices of voxels that are currently “unviewed”.
        occupied_indices = np.argwhere(self.grid)  # (N, 3)
        if occupied_indices.size == 0:
            return []

        # Convert those indices to world-frame centres.
        voxel_centres_w = self.to_world(occupied_indices)  # (N, 3)

        # Use the existing visibility test point-by-point.
        visible_mask = np.array(
            [is_object_visible(camera_pose_world, p) for p in voxel_centres_w],
            dtype=bool,
        )

        # Any voxel whose centre is visible gets cleared.
        indices_to_clear = occupied_indices[visible_mask]
        if indices_to_clear.size:  # only mutate if there’s something to clear
            self.grid[
                indices_to_clear[:, 0],
                indices_to_clear[:, 1],
                indices_to_clear[:, 2],
            ] = False

        return indices_to_clear

    def get_voxels(self) -> NDArray[np.float64]:
        return np.argwhere(self.grid)

    def from_world(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        indices = np.floor((points - self.min_bound) / VOXEL_SIZE).astype(int)
        return indices

    def to_world(self, voxels):
        return np.array(voxels) * VOXEL_SIZE + self.min_bound + VOXEL_SIZE / 2.0

    def get_points(self) -> NDArray[np.float64]:
        occupied_indices = np.argwhere(self.grid)
        centers = occupied_indices * VOXEL_SIZE + self.min_bound + VOXEL_SIZE / 2.0
        return centers


def compute_frustum(
    camera_name: str, camera_pose: Tuple[Any], depth: float = FRUSTUM_DEPTH
) -> Tuple[NDArray[np.float64], List[List[int]]]:
    height, width = IMG_SIZE
    fx, fy, cx, cy = (
        CAMERA_INTRINSICS[camera_name][0, 0],
        CAMERA_INTRINSICS[camera_name][1, 1],
        CAMERA_INTRINSICS[camera_name][0, 2],
        CAMERA_INTRINSICS[camera_name][1, 2],
    )
    frustum_corners = np.array(
        [
            [(0 - cx) * depth / fx, (0 - cy) * depth / fy, depth],
            [(width - cx) * depth / fx, (0 - cy) * depth / fy, depth],
            [(width - cx) * depth / fx, (height - cy) * depth / fy, depth],
            [(0 - cx) * depth / fx, (height - cy) * depth / fy, depth],
            [0, 0, 0],
        ]
    )
    tform = transformation_matrix(*camera_pose)
    frustum_corners = (tform[:3, :3] @ frustum_corners.T).T + tform[:3, 3]
    frustum_lines = [
        [4, 0],
        [4, 1],
        [4, 2],
        [4, 3],
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
    ]
    lines = [[frustum_corners[l] for l in line] for line in frustum_lines]
    return frustum_corners, lines


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

    def encode(self) -> Any:
        return self

    @classmethod
    def decode(cls: Type["SpotState"], encoded: Any) -> "SpotState":
        return encoded

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.body_location),
                tuple(self.movable_objects),
                tuple(self.fixed_objects),
                self.carry_object,
            )
        )

    @property
    def camera_pose(self):
        return pbu.multiply(
            se2_to_pose(self.occupancy_grid.to_world(self.body_location)),
            CAMERA_POSES[ARM_CONF],
        )

    def __eq__(self, other: Any) -> bool:
        return hash(other) == hash(self)


@dataclass
class SpotObservation(Observation):
    body_location: List[
        int
    ]  # x voxel index, y voxel index, rotation index into ROTATION_ANGLE
    visible_movable_objects: List[SceneObject] = field(default_factory=list)
    carry_object: Optional[SceneObject] = None

    def __repr__(self) -> str:
        return f"SpotObservation(body_location={self.body_location}, carry_object={self.carry_object}, visible_movable_objects={str([o for o in self.visible_movable_objects])})"

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.body_location),
                tuple(self.visible_movable_objects),
                self.carry_object,
            )
        )

    def __eq__(self, other: Any) -> bool:
        return hash(other) == hash(self)

    def encode(self) -> Any:
        return self

    @property
    def camera_pose(self):
        return CAMERA_POSES[ARM_CONF]

    @classmethod
    def decode(cls: Type["SpotObservation"], encoded: Any) -> "SpotObservation":
        return encoded


def spot_empty_observation_gen() -> SpotObservation:
    return SpotObservation(
        body_location=[0, 0, 0],
        visible_movable_objects=[],
    )


def spot_empty_state_gen(graphnav: str) -> SpotState:
    point_cloud = o3d.io.read_point_cloud(
        os.path.join(
            PROJECT_ROOT, "environments/spot/world_scans", graphnav, "pointcloud.ply"
        )
    )
    gn_path = Path(
        os.path.join(PROJECT_ROOT, "environments/spot/world_scans", graphnav)
    )
    with open(os.path.join(gn_path, "metadata.yaml"), "r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)
    home_pose_data = metadata["spot-home-pose"]
    room_pose_data = metadata["spot-room-pose"]

    room_pose = pbu.Pose(
        pbu.Point(x=room_pose_data["x"], y=room_pose_data["y"], z=room_pose_data["z"]),
        pbu.Euler(yaw=room_pose_data["angle"]),
    )
    home_pose = pbu.Pose(
        pbu.Point(x=home_pose_data["x"], y=home_pose_data["y"]),
        pbu.Euler(yaw=home_pose_data["angle"]),
    )
    T = transformation_matrix(*room_pose)
    point_cloud.transform(T)
    unfiltered_points = np.asarray(point_cloud.points)
    unfiltered_colors = np.asarray(point_cloud.colors)

    metadata["spot-room-bounds"]["max_z"]

    # Filter by max z
    bounds = metadata["spot-room-bounds"]

    # Create a mask that checks x, y, and z simultaneously
    mask = (
        (unfiltered_points[:, 0] >= bounds["min_x"])
        & (unfiltered_points[:, 0] <= bounds["max_x"])
        & (unfiltered_points[:, 1] >= bounds["min_y"])
        & (unfiltered_points[:, 1] <= bounds["max_y"])
        & (unfiltered_points[:, 2] >= bounds["min_z"])
        & (unfiltered_points[:, 2] <= bounds["max_z"])
    )

    # Apply the mask to both points and colors arrays
    points_array = unfiltered_points[mask]
    colors_array = unfiltered_colors[mask]
    min_bound = points_array.min(axis=0)
    min_bound[2] = min_bound[2] + PADDING_VIS_BELOW
    max_bound = points_array.max(axis=0)
    max_bound[2] = max_bound[2] - PADDING_VIS_ABOVE

    unviewed_voxels_path = os.path.join(gn_path, "unviewed_voxels.json")
    if os.path.exists(unviewed_voxels_path):
        with open(unviewed_voxels_path, "r", encoding="utf-8") as f:
            unviewed_voxels = json.load(f)
    else:
        unviewed_voxels = None

    visibility_grid = VisibilityGrid(
        min_bound, max_bound, points_array, colors_array, unviewed_voxels
    )

    fixed_objects = []
    if "fixed-objects" in metadata:
        fixed_datas = metadata["fixed-objects"]
        for fixed_data in fixed_datas:
            bb_points_array = np.array(fixed_data["points"])
            oobb = pbu.bounding_box(bb_points_array)
            fixed_objects.append(
                SceneObject(
                    fixed_data["name"],
                    visibility_grid.from_world(oobb.pose[0]),
                    oobb.aabb,
                )
            )

    occupancy_grid = OccupancyGrid(points_array, min_bound, max_bound, home_pose)

    body_location = list(occupancy_grid.from_world(pose_to_se2(home_pose)))
    print("body_location: ", body_location)

    return SpotState(
        body_location=body_location,
        occupancy_grid=occupancy_grid,
        visibility_grid=visibility_grid,
        fixed_objects=fixed_objects,
    )


def spot_hardcoded_initial_model_gen(
    fixed_object_names: Optional[List[str]] = None,
) -> InitialModel:
    def spot_hardcoded_initial_model(empty_state: SpotState) -> SpotState:
        if fixed_object_names is None:
            unviewed_voxels = copy.deepcopy(empty_state.visibility_grid.get_voxels())
            ons = []
        else:
            fixed_selection = random.choice(
                [n for n in empty_state.fixed_objects if n.name in fixed_object_names]
            )
            unviewed_voxels = empty_state.visibility_grid.get_voxels_above_aabb(
                fixed_selection.aabb
            ).tolist()
            ons = [("goal", fixed_selection.name)]
        # Generate a random goal position within the x-y bounds and a z between the bounds.
        rand_index = np.random.choice(len(unviewed_voxels))
        goal_x, goal_y, goal_z = unviewed_voxels[rand_index]
        goal_obj = SceneObject(name="goal", location=[goal_x, goal_y, goal_z])
        return SpotState(
            body_location=list(empty_state.body_location),
            occupancy_grid=empty_state.occupancy_grid,
            visibility_grid=empty_state.visibility_grid,
            fixed_objects=empty_state.fixed_objects,
            movable_objects=[goal_obj],
            ons=ons,
        )

    return spot_hardcoded_initial_model


def is_object_visible(camera_pose_world: Tuple[Any], object_pos_3d: np.ndarray) -> bool:
    # Convert object world position to homogeneous coords
    object_hom = np.array([object_pos_3d[0], object_pos_3d[1], object_pos_3d[2], 1.0])

    # Transform from world -> camera
    tform = transformation_matrix(*camera_pose_world)  # 4×4
    object_cam = np.linalg.inv(tform) @ object_hom  # now in camera frame
    z_cam = object_cam[2]

    # Must be in front & within FRUSTUM_DEPTH
    if z_cam <= 0 or z_cam > FRUSTUM_DEPTH:
        return False

    # Project to image plane using camera intrinsics
    proj = CAMERA_INTRINSICS["hand"] @ object_cam[:3]
    proj = proj / proj[2]  # (x', y', 1)
    px, py = proj[0], proj[1]

    height, width = IMG_SIZE
    # Must lie in valid image coordinates

    padding_width = 0.1 * width
    padding_height = 0.0 * height
    if not (
        padding_width <= px < width - padding_width
        and padding_height <= py < height - padding_height
    ):
        return False

    return True


def spot_hardcoded_observation_model_gen() -> ObservationModel:
    def spot_hardcoded_observation_model(
        state: SpotState, action: int, empty_obs: SpotObservation
    ) -> SpotObservation:
        visible_objects = []

        # For each object, check if is_object_visible under the *current* camera pose
        for obj in state.movable_objects:
            obj_xyz = state.visibility_grid.to_world(
                obj.location
            )  # object location in 3D
            # state.camera_pose is a property: it gives (trans, quat) in world
            if is_object_visible(state.camera_pose, obj_xyz):
                visible_objects.append(obj)

        return SpotObservation(
            body_location=state.body_location,
            visible_movable_objects=visible_objects,
            carry_object=state.carry_object,
        )

    return spot_hardcoded_observation_model


def move_robot(state: SpotState, action: int) -> List[int, int, int]:
    """Moves the robot in the specified direction and returns the new body
    location."""
    proposed_body_location = copy.deepcopy(state.body_location)
    current_theta = ROTATION_ANGLE[proposed_body_location[2]]

    if action == SpotActions.move_forward:
        dx = np.cos(current_theta) * NAVIGATION_STEP_SIZE
        dy = np.sin(current_theta) * NAVIGATION_STEP_SIZE
        proposed_body_location[1] += int(dx)
        proposed_body_location[0] += int(dy)
    elif action == SpotActions.move_backward:
        dx = -np.cos(current_theta) * NAVIGATION_STEP_SIZE
        dy = -np.sin(current_theta) * NAVIGATION_STEP_SIZE
        proposed_body_location[1] += int(dx)
        proposed_body_location[0] += int(dy)
    elif action == SpotActions.move_left:
        dx = -np.sin(current_theta) * NAVIGATION_STEP_SIZE
        dy = np.cos(current_theta) * NAVIGATION_STEP_SIZE
        proposed_body_location[1] += int(dx)
        proposed_body_location[0] += int(dy)
    elif action == SpotActions.move_right:
        dx = np.sin(current_theta) * NAVIGATION_STEP_SIZE
        dy = -np.cos(current_theta) * NAVIGATION_STEP_SIZE
        proposed_body_location[1] += int(dx)
        proposed_body_location[0] += int(dy)
    elif action == SpotActions.rotate_left:
        proposed_body_location[2] = (proposed_body_location[2] + 1) % len(
            ROTATION_ANGLE
        )
    elif action == SpotActions.rotate_right:
        proposed_body_location[2] = (proposed_body_location[2] - 1) % len(
            ROTATION_ANGLE
        )
    return proposed_body_location


def pickup(state: SpotState) -> Tuple[List[Tuple[str, str]], Optional[SceneObject]]:
    """Pick up an object if the robot is not already carrying one."""
    new_ons = state.ons
    new_carry_object = state.carry_object
    if new_carry_object is None:
        camera_pose_world = pbu.multiply(
            se2_to_pose(state.occupancy_grid.to_world(state.body_location)),
            CAMERA_POSES[ARM_CONF],
        )
        for obj in state.movable_objects:
            obj_xyz = state.visibility_grid.to_world(obj.location)
            if is_object_visible(camera_pose_world, obj_xyz):
                distance = np.linalg.norm(
                    np.array(camera_pose_world[0]) - np.array(obj_xyz)
                )
                if distance < PICKUP_DISTANCE_THRESHOLD:
                    new_carry_object = obj
                    new_ons = [o for o in state.ons if o[0] != obj.name]
                    break  # Only pickup the first qualified object

    return new_ons, new_carry_object


def spot_hardcoded_transition_model_gen() -> TransitionModel:
    def spot_hardcoded_transition_model(state: SpotState, action: int) -> SpotState:
        proposed_body_location = list(state.body_location)
        ons = state.ons
        # Copying the object list so that we continue to keep the objects in the scene.
        proposed_objects = list(state.movable_objects)
        proposed_carry_object = state.carry_object
        proposed_body_location = move_robot(state, action)

        if action == SpotActions.pickup:
            ons, proposed_carry_object = pickup(state)

        # Collision check: if there is no collision, update the state
        if not state.occupancy_grid.check_collision(tuple(proposed_body_location)):
            return SpotState(
                body_location=list(proposed_body_location),
                occupancy_grid=state.occupancy_grid,
                visibility_grid=state.visibility_grid,
                movable_objects=proposed_objects,
                fixed_objects=state.fixed_objects,
                carry_object=proposed_carry_object,
                ons=ons,
            )
        # If there is a collision, remain in the same state.
        return state

    return spot_hardcoded_transition_model


def spot_hardcoded_reward_model_gen() -> RewardModel:
    def spot_hardcoded_reward_model(
        state: SpotState, action: int, next_state: SpotState
    ) -> Tuple[float, bool]:
        # If next_state.carry_object is not None, we have picked up something
        if next_state.carry_object is not None:
            return (1.0, True)
        return (0.0, False)

    return spot_hardcoded_reward_model


class SpotEnvironment(Environment):
    def __init__(
        self,
        graphnav: str,
        *args: Any,
        fully_obs: bool = True,
        real_spot: bool = False,
        live_rerun: bool = True,
        fixed_object_names: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.real_spot = real_spot
        self.fully_obs = fully_obs
        self.default_conf = DEFAULT_CONF
        self.graphnav = graphnav
        self.empty_state: SpotState = spot_empty_state_gen(self.graphnav)
        self.empty_obs: SpotObservation = spot_empty_observation_gen()
        self.steps: int = 0
        self.live_rerun = live_rerun
        self.fixed_object_names = fixed_object_names

        self.transition_model = spot_hardcoded_transition_model_gen()
        self.reward_model = spot_hardcoded_reward_model_gen()
        self.observation_model = spot_hardcoded_observation_model_gen()
        self.initial_model = spot_hardcoded_initial_model_gen(
            fixed_object_names=self.fixed_object_names
        )

        self.rerun_spot = None
        self.cleared_belief = False

    def reset(self, seed: int = 0) -> State:
        log.info("Resetting environment")
        self.found_goal = False
        self.empty_state.visibility_grid.reset()
        self.steps = 0

        self.starting_state: SpotState = self.initial_model(self.empty_state)

        if self.live_rerun:
            self.rerun_spot = self.init_rerun_state()

        if self.real_spot:
            # If on the real robot, move the goal out of the way so it explores everywhere
            self.starting_state.movable_objects[0].location = [0, 0, 100]

        self.current_state = self.starting_state

        if self.real_spot:
            env_file = os.path.join(PROJECT_ROOT, "..", ".env")
            dotenv.load_dotenv(env_file, override=True)
            self.real_robot_client = setup_robot(
                graphnav=self.graphnav, spot_ip=SPOT_IP
            )
            blocking_stand(self.real_robot_client.command_client, timeout_sec=10)
            go_home(self.real_robot_client)
            open_gripper(self.real_robot_client)
            self.move_arm_state(ARM_CONF)
            self.robot_state_thread = None

            self.iserver_context = zmq.Context()
            self.iserver_socket = self.iserver_context.socket(zmq.REQ)
            self.iserver_socket.connect("tcp://128.30.227.100:5555")

        obs = self.obs_from_state(self.current_state, action=None)

        if self.real_spot:
            self.real_robot_client.localizer.localize()
            if self.robot_state_thread is not None:
                log.info("Stopping robot state thread")
                self.stop_event.set()
                self.robot_state_thread.join()
                log.info("Done stopping robot state thread")

            log.info("Starting new robot state thread")
            self.stop_event = threading.Event()
            self.robot_state_thread = threading.Thread(
                target=track_robot_state,
                args=(
                    self.real_robot_client,
                    self.empty_state.occupancy_grid.home_pose,
                    self.stop_event,
                    self.current_rerun_name,
                ),
            )

            self.robot_state_thread.start()
            print("Sleeping to wait for robot to load into rerun")
            time.sleep(5)
            print("Done sleeping")

        return self.current_state

    def obs_from_state(
        self, state: SpotState, action: Optional[ActType]
    ) -> SpotObservation:
        if self.real_spot:
            camera_results = {}
            # Wait for spot to stop moving to remove motion blur
            time.sleep(0.5)
            # for camera in CAMERAS:
            camera = "hand"
            rgbd = capture_rgbd(self.real_robot_client, camera)
            camera_results[camera] = rgbd
            log_transform_axes(
                [
                    rgbd.frame.position,
                    (
                        rgbd.frame.rotation.x,
                        rgbd.frame.rotation.y,
                        rgbd.frame.rotation.z,
                        rgbd.frame.rotation.w,
                    ),
                ]
            )

            seg_array = rgbd.rgb
            image = Image.fromarray(seg_array)
            image_path = os.path.join(
                get_log_dir(), f"spot_hand_image_step={self.steps}.png"
            )
            image.save(image_path)

            log.info("Sending request to server...")
            message = {"rgbPixels": rgbd.rgb.tolist(), "categories": ["apple"]}
            self.iserver_socket.send_json(message)
            reply = self.iserver_socket.recv_json()

            log.info("Received reply from server: " + str(reply.keys()))
            if "priors" in reply and len(reply["priors"]) > 0:
                log.info("Detected target object. Calculating position...")

                seg_array = np.array(reply["image"]).astype(np.uint8)
                seg_image = Image.fromarray(seg_array)
                seg_image_path = os.path.join(
                    get_log_dir(), f"spot_hand_image_step_seg={self.steps}.png"
                )
                seg_image.save(seg_image_path)

                mask_np = np.array(list(reply["priors"].values())[0])

                lower_bound = 0.05  # 5 cm
                upper_bound = 3.0  # 3 m

                # Compute the masked depth (convert depth to meters and apply the binary mask)
                masked_depth = (rgbd.depth / 1000.0) * mask_np

                # Create a boolean mask for valid depth values within the specified bounds
                valid_depth_mask = (masked_depth >= lower_bound) & (
                    masked_depth <= upper_bound
                )

                # Use the valid_depth_mask to filter the depth values
                depth_values = masked_depth[valid_depth_mask]

                # Compute the mean depth from the valid pixels
                mean_depth = np.mean(depth_values)

                y_indices, x_indices = np.where(mask_np)
                avg_x = np.mean(x_indices)
                avg_y = np.mean(y_indices)

                if action == SpotActions.pickup:
                    # self.move_arm_state("ARM_STOW")
                    grasp(self.real_robot_client, rgbd.rgb_response, [avg_x, avg_y])
                    log.info("Issuing arm move command")
                    move_arm_relative(self.real_robot_client, 0.0, 0.0, 0.15)
                    log.info("Issuing stow command")
                    self.move_arm_state("ARM_STOW")

                if not self.found_goal:
                    camera_matrix = CAMERA_INTRINSICS["hand"]
                    fx = camera_matrix[0, 0]
                    fy = camera_matrix[1, 1]
                    cx = camera_matrix[0, 2]
                    cy = camera_matrix[1, 2]

                    if not np.isnan(mean_depth):
                        # Back-project from pixel coordinates to 3D point
                        X = (avg_x - cx) * mean_depth / fx
                        Y = (avg_y - cy) * mean_depth / fy
                        Z = mean_depth

                        print(
                            f"Back-projected coordinates: X={X}, Y={Y}, Z={Z}, mean_depth={mean_depth}"
                        )

                        world_T_target = pbu.multiply(
                            state.camera_pose, pbu.Pose(point=[X, Y, Z])
                        )

                        print("world_T_target: ", world_T_target)

                        goal_voxel = self.empty_state.visibility_grid.from_world(
                            world_T_target[0]
                        )
                        print("Goal voxel pre filtering: ", goal_voxel)
                        goal_voxel = (
                            self.empty_state.visibility_grid.closest_unviewed_voxel(
                                goal_voxel
                            )
                        )
                        if goal_voxel is None:
                            log.info("No unviewed voxels found")
                        else:
                            print(f"Goal voxel: {goal_voxel}")
                            self.found_goal = True
                            for o in self.current_state.movable_objects:
                                if o.name == "goal":
                                    o.location = goal_voxel

                            for o in state.movable_objects:
                                if o.name == "goal":
                                    o.location = goal_voxel
                                    break

            else:
                log.info("No target object detected")

            if self.live_rerun:
                rr.log("image", rr.Image(seg_array))
                rr.log("depth_image", rr.Image(rgbd.depth))

        next_obs: SpotObservation = self.observation_model(
            state, action, self.empty_obs
        )

        if self.live_rerun:
            self.update_rerun_state(self.rerun_spot, state, self.steps)

        return next_obs

    def move_arm_state(self, arm_conf: str):
        positions = [
            ARM_CONFS[arm_conf][jn]
            for jn in [
                "arm0.sh0",
                "arm0.sh1",
                "arm0.el0",
                "arm0.el1",
                "arm0.wr0",
                "arm0.wr1",
            ]
        ]
        move_arm(self.real_robot_client, ArmJointPositions.from_list(positions))

    def step(
        self, action: ActType
    ) -> Tuple[Observation, State, float, bool, bool, Dict[str, Any]]:
        # Used for demo setup
        # if(self.steps == 0):
        #     input("Press enter to start")

        self.steps += 1

        next_state: SpotState = self.transition_model(self.current_state, action)
        if self.real_spot:
            world_T_virt_now = math_helpers.SE2Pose(
                *self.empty_state.occupancy_grid.to_world(next_state.body_location)
            )
            navigate_to_absolute_pose(self.real_robot_client, world_T_virt_now)
            self.move_arm_state(ARM_CONF)
            self.real_robot_client.localizer.localize()

        reward, terminated = self.reward_model(self.current_state, action, next_state)
        truncated = False
        if self.steps >= self.max_steps:
            truncated = True
        self.current_state = next_state
        next_obs = self.obs_from_state(self.current_state, action)

        return next_obs, self.current_state, reward, terminated, truncated, {}

    @property
    def current_conf(self) -> Dict[str, float]:
        return self.default_conf | {
            k: v
            for k, v in zip(
                ["x", "y", "theta"],
                self.empty_state.occupancy_grid.to_world(
                    self.current_state.body_location
                ),
            )
        }

    def update_rerun_state(self, spot: Any, state: SpotState, step: int = 0) -> None:
        if not self.cleared_belief:
            self.empty_state.visibility_grid.populate_grid(
                self.initial_model, self.empty_state
            )
            self.cleared_belief = True

        if not self.real_spot:
            updated_conf = self.default_conf | {
                k: v
                for k, v in zip(
                    ["x", "y", "theta"],
                    state.occupancy_grid.to_world(state.body_location),
                )
            }
            updated_conf = updated_conf | ARM_CONFS[ARM_CONF]
            spot.set_joint_positions(
                tuple(updated_conf.setdefault(k, 0.0) for k in JOINT_NAMES)
            )

        frustum_pts, frustum_lines = compute_frustum(
            camera_name="hand", camera_pose=state.camera_pose
        )

        for obj in state.movable_objects:
            if obj.name == "goal":
                print("Goal location: " + str(obj.location))
                if state.carry_object != obj:
                    positions = [
                        self.empty_state.visibility_grid.to_world(obj.location)
                    ]
                    rr.log(
                        "debug/goal",
                        rr.Points3D(
                            radii=[0.1],
                            positions=positions,
                            colors=[[0, 255, 0, 255]],
                        ),
                    )
                else:
                    rr.log(
                        "debug/goal",
                        rr.Points3D(
                            radii=[0.1],
                            positions=state.camera_pose[0],
                            colors=[[0, 255, 0, 255]],
                        ),
                    )

        if not self.fully_obs:
            self.empty_state.visibility_grid.clear_voxels_within_frustum(
                state.camera_pose
            )
            self.plot_voxel_grid()

        if self.real_spot:
            rr.set_time_nanos("timeline", int(time.time() * 1e9))
        else:
            rr.log(
                "debug/frustum_points",
                rr.Points3D(
                    positions=frustum_pts,
                    colors=[[255, 255, 255, 255]] * len(frustum_pts),
                ),
            )
            rr.log(
                "debug/frustum_lines",
                rr.LineStrips3D(
                    frustum_lines, colors=[[255, 255, 255, 255]] * len(frustum_pts)
                ),
            )

            rr.set_time_sequence("timeline", step + 1)

    def plot_voxel_grid(self) -> None:
        voxel_points = self.empty_state.visibility_grid.get_points()
        print("Num voxel points: " + str(len(voxel_points)))
        voxel_colors = [[0, 0, 255] for _ in range(len(voxel_points))]
        rr.log(
            "debug/visibility_grid_points",
            rr.Boxes3D(
                centers=voxel_points,
                half_sizes=[[VOXEL_SIZE / 2.0 for _ in range(3)] for _ in voxel_points],
                colors=voxel_colors,
            ),
        )

    def init_rerun_state(self) -> Any:
        self.current_rerun_name = f"demo_{str(time.time())}"
        rr.init(self.current_rerun_name, spawn=True)
        if self.real_spot:
            rr.set_time_nanos("timeline", int(time.time() * 1e9))
        else:
            rr.set_time_sequence("timeline", 0)
        if not self.fully_obs:
            self.plot_voxel_grid()
        rr.log(
            "debug/full_pointcloud",
            rr.Points3D(
                positions=self.empty_state.visibility_grid.points_array,
                colors=self.empty_state.visibility_grid.colors_array,
            ),
        )
        # for fixed_object in self.starting_state.fixed_objects:
        #     name = fixed_object.name
        #     aabb = fixed_object.aabb
        #     rr.log(
        #         f"debug/{name}",
        #         rr.Boxes3D(
        #             centers=[self.starting_state.visibility_grid.to_world(fixed_object.location)],
        #             half_sizes=[pbu.get_aabb_extent(aabb) / 2.0],
        #             colors=[255, 255, 0],
        #         ),
        #     )

        spot = None
        if not self.real_spot:
            spot = load_spot()
        return spot


def get_robot_pose(robot_state_client: Any) -> Tuple[Any]:
    robot_state = robot_state_client.get_robot_state()
    frame_name = "vision"
    snapshot = robot_state.kinematic_state.transforms_snapshot
    if frame_name in snapshot.child_to_parent_edge_map:
        transform = snapshot.child_to_parent_edge_map[frame_name]
        x = transform.parent_tform_child.position.x
        y = transform.parent_tform_child.position.y
        z = transform.parent_tform_child.position.z
        quat = transform.parent_tform_child.rotation
        return pbu.invert(((x, y, z), (quat.x, quat.y, quat.z, quat.w)))
    else:
        return None, None


def track_robot_state(
    robot_client: RobotClient,
    world_T_robot_start: Tuple[Any],
    stop_event: Any,
    rerun_name: str,
) -> None:
    rr.init(rerun_name, spawn=True)
    spot = load_spot()

    start_x, start_y, start_z = world_T_robot_start[0]
    start_yaw = pbu.euler_from_quat(world_T_robot_start[1])[2]

    world_T_start = pbu.Pose(
        point=pbu.Point(x=start_x, y=start_y, z=start_z),
        euler=pbu.Euler(yaw=start_yaw),
    )

    start_odom = get_robot_pose(robot_client.state_client)
    start_odom_x, start_odom_y, start_odom_z = start_odom[0]
    start_odom_yaw = pbu.euler_from_quat(start_odom[1])[2]

    world_T_odom_start = pbu.Pose(
        point=pbu.Point(x=start_odom_x, y=start_odom_y, z=start_odom_z),
        euler=pbu.Euler(yaw=start_odom_yaw),
    )

    try:
        while not stop_event.is_set():
            current_odom = get_robot_pose(robot_client.state_client)

            current_odom_x, current_odom_y, current_odom_z = current_odom[0]
            current_odom_yaw = pbu.euler_from_quat(current_odom[1])[2]
            world_T_odom_now = pbu.Pose(
                point=pbu.Point(x=current_odom_x, y=current_odom_y, z=current_odom_z),
                euler=pbu.Euler(yaw=current_odom_yaw),
            )

            robot_state = robot_client.state_client.get_robot_state()
            joint_states = robot_state.kinematic_state.joint_states

            odom_start_T_odom_now = pbu.multiply(
                pbu.invert(world_T_odom_start), world_T_odom_now
            )

            world_T_now = pbu.multiply(world_T_start, odom_start_T_odom_now)
            now_yaw = pbu.euler_from_quat(world_T_now[1])[2]
            joint_dict = {
                joint.name: float(joint.position.value) for joint in joint_states
            } | {
                "x": float(world_T_now[0][0]),
                "y": float(world_T_now[0][1]),
                "z": float(world_T_now[0][2]),
                "theta": float(now_yaw),
            }
            rr.set_time_nanos("timeline", int(time.time() * 1e9))
            spot.set_joint_positions(
                tuple(joint_dict.setdefault(k, 0.0) for k in JOINT_NAMES)
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping robot state tracking.")


def log_transform_axes(camera_pose, scale=0.2, prefix="debug/camera_pose"):
    """Logs the transform axes (x, y, z) from a camera pose.

    Args:
        camera_pose: A tuple or list where:
                     - camera_pose[0] is the translation (array-like of 3 numbers).
                     - camera_pose[1] is the quaternion (array-like of 4 numbers in xyzw order).
        scale: Length multiplier for the axes.
        prefix: String prefix for the log channels.
    """
    # Extract translation and quaternion
    translation = np.array(camera_pose[0])
    quaternion = camera_pose[1]

    # Create a quaternion object and get its rotation matrix.
    # Adjust the API call if your quaternion implementation differs.
    rot = R.from_quat(quaternion).as_matrix()

    # Compute the rotated unit axes using the rotation matrix
    x_axis = np.dot(rot, np.array([1, 0, 0]))
    y_axis = np.dot(rot, np.array([0, 1, 0]))
    z_axis = np.dot(rot, np.array([0, 0, 1]))

    # The vector for the arrow is the rotated axis scaled by 'scale'
    x_vector = x_axis * scale
    y_vector = y_axis * scale
    z_vector = z_axis * scale

    # Log each arrow with the origin as the translation and the vector as computed:
    rr.log(
        prefix + "_x",
        rr.Arrows3D(origins=[translation], vectors=[x_vector], colors=[[255, 0, 0]]),
    )  # Red for X-axis
    rr.log(
        prefix + "_y",
        rr.Arrows3D(origins=[translation], vectors=[y_vector], colors=[[0, 255, 0]]),
    )  # Green for Y-axis
    rr.log(
        prefix + "_z",
        rr.Arrows3D(origins=[translation], vectors=[z_vector], colors=[[0, 0, 255]]),
    )  # Blue for Z-axis
