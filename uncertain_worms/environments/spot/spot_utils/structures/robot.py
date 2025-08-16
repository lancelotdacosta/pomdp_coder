"""Data structures for the robot and robot parts."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from bosdyn.api import robot_state_pb2
from bosdyn.api.graph_nav import map_pb2, nav_pb2
from bosdyn.client import ResponseError, TimedOutError, math_helpers
from bosdyn.client.exceptions import ProxyConnectionError, TimedOutError
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot import Robot
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.sdk import Robot, Sdk

NUM_LOCALIZATION_RETRIES = 10
LOCALIZATION_RETRY_WAIT_TIME = 1.0


def get_robot_state(
    robot: Robot, timeout_per_call: float = 20, num_retries: int = 10
) -> robot_state_pb2.RobotState:
    """Get the robot state."""
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    for _ in range(num_retries):
        try:
            robot_state = robot_state_client.get_robot_state(timeout=timeout_per_call)
            return robot_state
        except (TimedOutError, ProxyConnectionError):
            print("WARNING: get robot state failed once, retrying...")
    raise RuntimeError("get_robot_state() failed permanently.")


class LocalizationFailure(Exception):
    """Raised when localization fails."""


class SpotLocalizer:
    """Localizes spot in a previously mapped environment."""

    def __init__(
        self,
        robot: Robot,
        upload_path: Path,
        lease_client: LeaseClient,
        lease_keepalive: LeaseKeepAlive,
    ) -> None:
        self._robot = robot
        self._upload_path = upload_path
        self._lease_client = lease_client
        self._lease_keepalive = lease_keepalive

        # Force trigger timesync.
        self._robot.time_sync.wait_for_sync()

        # Create the client for the Graph Nav main service.
        self.graph_nav_client = self._robot.ensure_client(
            GraphNavClient.default_service_name
        )

        # Upload graph and snapshots on start.
        self._upload_graph_and_snapshots()

        # Initialize robot pose, which will be updated in localize().
        self._robot_pose = math_helpers.SE3Pose(0, 0, 0, math_helpers.Quat())
        # Initialize the robot's position in the map.
        robot_state = get_robot_state(self._robot)
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot
        ).to_proto()
        localization = nav_pb2.Localization()
        for r in range(NUM_LOCALIZATION_RETRIES + 1):
            try:
                self.graph_nav_client.set_localization(
                    initial_guess_localization=localization,
                    ko_tform_body=current_odom_tform_body,
                )
                break
            except (ResponseError, TimedOutError) as e:
                # Retry or fail.
                if r == NUM_LOCALIZATION_RETRIES:
                    msg = f"Localization failed permanently: {e}."
                    logging.warning(msg)
                    raise LocalizationFailure(msg)
                logging.warning("Localization failed once, retrying.")
                time.sleep(LOCALIZATION_RETRY_WAIT_TIME)

        # Run localize once to start.
        self.localize()

    def _upload_graph_and_snapshots(self) -> None:
        """Upload the graph and snapshots to the robot."""
        # pylint: disable=no-member
        logging.info("Loading the graph from disk into local storage...")
        # Load the graph from disk.
        with open(self._upload_path / "graph", "rb") as f:
            data = f.read()
            current_graph = map_pb2.Graph()
            current_graph.ParseFromString(data)
            logging.info(
                f"Loaded graph has {len(current_graph.waypoints)} "
                f"waypoints and {len(current_graph.edges)} edges"
            )
        # Load the waypoint snapshots from disk.
        waypoint_path = self._upload_path / "waypoint_snapshots"
        waypoint_snapshots: Dict[str, map_pb2.WaypointSnapshot] = {}
        for waypoint in current_graph.waypoints:
            with open(waypoint_path / waypoint.snapshot_id, "rb") as f:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(f.read())
                waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
        # Load the edge snapshots from disk.
        edge_path = self._upload_path / "edge_snapshots"
        edge_snapshots: Dict[str, map_pb2.EdgeSnapshot] = {}
        for edge in current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            with open(edge_path / edge.snapshot_id, "rb") as f:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(f.read())
                edge_snapshots[edge_snapshot.id] = edge_snapshot
        # Upload the graph to the robot.
        logging.info("Uploading the graph and snapshots to the robot...")
        true_if_empty = not len(current_graph.anchoring.anchors)
        response = self.graph_nav_client.upload_graph(
            graph=current_graph, generate_new_anchoring=true_if_empty
        )
        # Upload the snapshots to the robot.
        for snapshot_id in response.unknown_waypoint_snapshot_ids:
            waypoint_snapshot = waypoint_snapshots[snapshot_id]
            self.graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
        for snapshot_id in response.unknown_edge_snapshot_ids:
            edge_snapshot = edge_snapshots[snapshot_id]
            self.graph_nav_client.upload_edge_snapshot(edge_snapshot)

    def get_last_robot_pose(self) -> math_helpers.SE3Pose:
        """Get the last estimated robot pose.

        Does not localize.
        """
        return self._robot_pose

    def localize(self, num_retries: int = 10, retry_wait_time: float = 1.0) -> None:
        """Re-localize the robot and return the current SE3Pose of the body.

        It's good practice to call this periodically to avoid drift
        issues. April tags need to be in view.
        """
        try:
            localization_state = self.graph_nav_client.get_localization_state()
            transform = localization_state.localization.seed_tform_body
            if str(transform) == "":
                raise LocalizationFailure("Received empty localization state.")
        except (ResponseError, TimedOutError, LocalizationFailure) as e:
            # Retry or fail.
            if num_retries <= 0:
                msg = f"Localization failed permanently: {e}."
                logging.warning(msg)
                raise LocalizationFailure(msg)
            logging.warning("Localization failed once, retrying.")
            time.sleep(retry_wait_time)
            return self.localize(
                num_retries=num_retries - 1, retry_wait_time=retry_wait_time
            )
        logging.info("Localization succeeded.")
        self._robot_pose = math_helpers.SE3Pose.from_proto(transform)
        return None


@dataclass
class RobotClient:
    """All robot clients packaged into one object."""

    robot: Robot
    sdk: Sdk
    state_client: RobotStateClient = None
    command_client: RobotCommandClient = None
    image_client: ImageClient = None
    manipulation_client: ManipulationApiClient = None
    lease_keepalive: LeaseKeepAlive = None
    localizer: SpotLocalizer = None
    home_pose: math_helpers.SE2Pose = None


@dataclass
class ArmJointPositions:
    """Arm joint positions."""

    sh0: float = 0
    sh1: float = 0
    el0: float = 0
    el1: float = 0
    wr0: float = 0
    wr1: float = 0

    def to_list(self) -> List[float]:
        """Converts the arm joints to a list."""
        return [self.sh0, self.sh1, self.el0, self.el1, self.wr0, self.wr1]

    @staticmethod
    def from_list(pos: List[float]):
        """Converts a list of joint positions into a ArmJointPositions
        object."""
        return ArmJointPositions(*pos)
