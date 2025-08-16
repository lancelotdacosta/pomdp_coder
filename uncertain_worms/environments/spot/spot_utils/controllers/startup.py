"""Functions for starting up and controlling the lease for spot."""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
import yaml
from bosdyn.api import geometry_pb2, robot_state_pb2
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.exceptions import ProxyConnectionError, TimedOutError
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    get_se2_a_tform_b,
)
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot import Robot
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.sdk import Robot

from uncertain_worms.environments.spot.spot_utils.structures.robot import (
    RobotClient,
    SpotLocalizer,
)
from uncertain_worms.utils import PROJECT_ROOT


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


def navigate_to_relative_pose(
    robot: Robot,
    body_tform_goal: math_helpers.SE2Pose,
    max_xytheta_vel: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_xytheta_vel: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
    timeout: float = 20.0,
) -> None:
    """Execute a relative move.

    The pose is dx, dy, dyaw relative to the robot's body.
    """
    # Get the robot's current state.
    robot_state = get_robot_state(robot)
    transforms = robot_state.kinematic_state.transforms_snapshot
    assert str(transforms) != ""

    # We do not want to command this goal in body frame because the body will
    # move, thus shifting our goal. Instead, we transform this offset to get
    # the goal position in the output frame (odometry).
    out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    # Command the robot to go to the goal point in the specified
    # frame. The command will stop at the new position.
    # Constrain the robot not to turn, forcing it to strafe laterally.
    speed_limit = SE2VelocityLimit(
        max_vel=SE2Velocity(
            linear=Vec2(x=max_xytheta_vel[0], y=max_xytheta_vel[1]),
            angular=max_xytheta_vel[2],
        ),
        min_vel=SE2Velocity(
            linear=Vec2(x=min_xytheta_vel[0], y=min_xytheta_vel[1]),
            angular=min_xytheta_vel[2],
        ),
    )
    mobility_params = spot_command_pb2.MobilityParams(vel_limit=speed_limit)

    robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x,
        goal_y=out_tform_goal.y,
        goal_heading=out_tform_goal.angle,
        frame_name=ODOM_FRAME_NAME,
        params=mobility_params,
    )
    cmd_id = robot_command_client.robot_command(
        lease=None, command=robot_cmd, end_time_secs=time.time() + timeout
    )
    start_time = time.perf_counter()
    while (time.perf_counter() - start_time) <= timeout:
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = (
            feedback.feedback.synchronized_feedback.mobility_command_feedback
        )
        if (
            mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING
        ):  # pylint: disable=no-member,line-too-long
            logging.warning("Failed to reach the goal")
            return
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (
            traj_feedback.status == traj_feedback.STATUS_AT_GOAL
            and traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED
        ):
            return
    if (time.perf_counter() - start_time) > timeout:
        logging.warning("Timed out waiting for movement to execute!")


def navigate_to_absolute_pose(
    robot_client: RobotClient,
    target_pose: math_helpers.SE2Pose,
    max_xytheta_vel: Tuple[float, float, float] = (2.0, 2.0, 1.0),
    min_xytheta_vel: Tuple[float, float, float] = (-2.0, -2.0, -1.0),
    timeout: float = 20.0,
) -> None:
    """Move to the absolute SE2 pose."""

    robot_client.localizer.localize()
    current_pose = robot_client.localizer.get_last_robot_pose()
    print("Current robot pose: ", current_pose)
    print("Navigating to absolute pose: ", target_pose)
    robot_pose = current_pose
    robot_se2 = robot_pose.get_closest_se2_transform()
    rel_pose = robot_se2.inverse() * target_pose

    print("Relative pose: ", rel_pose)
    navigate_to_relative_pose(
        robot_client.robot, rel_pose, max_xytheta_vel, min_xytheta_vel, timeout
    )

    robot_client.localizer.localize()
    current_pose = robot_client.localizer.get_last_robot_pose()
    print("New robot pose: ", current_pose.get_closest_se2_transform())

    # input("Good?")


def go_home(
    robot_client: RobotClient,
    max_xytheta_vel: Tuple[float, float, float] = (2.0, 2.0, 1.0),
    min_xytheta_vel: Tuple[float, float, float] = (-2.0, -2.0, -1.0),
    timeout: float = 20.0,
) -> None:
    """Navigate to a known home position (defined in utils.py)."""
    return navigate_to_absolute_pose(
        robot_client,
        target_pose=robot_client.home_pose,
        max_xytheta_vel=max_xytheta_vel,
        min_xytheta_vel=min_xytheta_vel,
        timeout=timeout,
    )


def setup_robot(graphnav: str, spot_ip: str) -> RobotClient:
    """Sets up the robot clients and packages them into a RobotClient class."""
    bosdyn.client.util.setup_logging(False)

    sdk = bosdyn.client.create_standard_sdk("RobotClient")
    robot = sdk.create_robot(spot_ip)

    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    lease_client = robot.ensure_client(
        bosdyn.client.lease.LeaseClient.default_service_name
    )
    lease_client.take()

    state_client: RobotStateClient = robot.ensure_client(
        RobotStateClient.default_service_name
    )
    command_client: RobotCommandClient = robot.ensure_client(
        RobotCommandClient.default_service_name
    )
    image_client = robot.ensure_client(ImageClient.default_service_name)
    manipulation_client = robot.ensure_client(
        ManipulationApiClient.default_service_name
    )
    lease_keepalive = bosdyn.client.lease.LeaseKeepAlive(
        lease_client, must_acquire=True, return_at_exit=True
    )

    robot.power_on(timeout_sec=20)
    gn_path = Path(
        os.path.join(PROJECT_ROOT, "environments/spot/world_scans", graphnav)
    )
    localizer = SpotLocalizer(robot, gn_path, lease_client, lease_keepalive)

    with open(os.path.join(gn_path, "metadata.yaml"), "r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    home_pose_data = metadata["spot-home-pose"]
    home_pose = math_helpers.SE2Pose(
        home_pose_data["x"], home_pose_data["y"], home_pose_data["angle"]
    )

    return RobotClient(
        robot=robot,
        sdk=sdk,
        state_client=state_client,
        command_client=command_client,
        image_client=image_client,
        manipulation_client=manipulation_client,
        lease_keepalive=lease_keepalive,
        localizer=localizer,
        home_pose=home_pose,
    )
