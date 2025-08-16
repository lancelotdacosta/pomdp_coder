# type:ignore
"""Functions for controlling the arm on spot."""

import time
from logging import Logger
from typing import Any, List, Tuple

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
import numpy as np
from bosdyn.api import (
    arm_command_pb2,
    geometry_pb2,
    manipulation_api_pb2,
    robot_command_pb2,
    synchronized_command_pb2,
)
from bosdyn.client.frame_helpers import (
    GRAV_ALIGNED_BODY_FRAME_NAME,
    VISION_FRAME_NAME,
    get_vision_tform_body,
    math_helpers,
)
from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives
from bosdyn.util import duration_to_seconds

from uncertain_worms.environments.spot.spot_utils.perception.capture import capture_rgbd
from uncertain_worms.environments.spot.spot_utils.structures.image import RGBDImage
from uncertain_worms.environments.spot.spot_utils.structures.robot import (
    ArmJointPositions,
    RobotClient,
)


def open_gripper(robot_client: RobotClient) -> None:
    """Helper function to open the robot gripper."""
    # Make the open gripper RobotCommand
    gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)

    # Send the request
    _ = robot_client.command_client.robot_command(gripper_command)
    robot_client.robot.logger.info("Moving arm to position.")


def _print_feedback(feedback_resp: Any, logger: Logger) -> float:
    """Helper function to query for ArmJointMove feedback, and print it to the
    console.

    Returns the time_to_goal value reported in the feedback

    feedback_resp is a protobuf with type Any
    """
    sync_feedback = feedback_resp.feedback.synchronized_feedback
    arm_feedback = sync_feedback.arm_command_feedback
    joint_move_feedback = arm_feedback.arm_joint_move_feedback

    logger.info(f"  planner_status = {joint_move_feedback.planner_status}")
    logger.info(
        f"  time_to_goal = \
            {duration_to_seconds(joint_move_feedback.time_to_goal):.2f} seconds."
    )

    # Query planned_points to determine target pose of arm
    logger.info("  planned_points:")
    for idx, points in enumerate(joint_move_feedback.planned_points):
        pos = points.position
        pos_str = f"sh0 = {pos.sh0.value:.3f}, \
                    sh1 = {pos.sh1.value:.3f}, \
                    el0 = {pos.el0.value:.3f}, \
                    el1 = {pos.el1.value:.3f}, \
                    wr0 = {pos.wr0.value:.3f}, \
                    wr1 = {pos.wr1.value:.3f}"
        logger.info(f"    {idx}: {pos_str}")
    return duration_to_seconds(joint_move_feedback.time_to_goal)


def move_arm(robot_client: RobotClient, arm_pos: ArmJointPositions) -> None:
    """Helper function to move the robot joints to target joint positions."""
    traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
        *arm_pos.to_list()
    )

    arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=[traj_point])
    # Make a RobotCommand

    joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(
        trajectory=arm_joint_traj
    )
    arm_command = arm_command_pb2.ArmCommand.Request(
        arm_joint_move_command=joint_move_command
    )
    sync_arm = synchronized_command_pb2.SynchronizedCommand.Request(
        arm_command=arm_command
    )
    arm_sync_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_arm)
    command = RobotCommandBuilder.build_synchro_command(arm_sync_robot_cmd)

    # Send the request
    cmd_id = robot_client.command_client.robot_command(command)
    robot_client.robot.logger.info("Moving arm to position 1.")

    # Query for feedback to determine how long the goto will take.
    feedback_resp = robot_client.command_client.robot_command_feedback(cmd_id)
    robot_client.robot.logger.info("Feedback for Example 1: single point goto")
    time_to_goal = _print_feedback(feedback_resp, robot_client.robot.logger)
    time.sleep(time_to_goal)


def add_grasp_constraint(grasp, robot_state_client):
    # There are 3 types of constraints:
    #   1. Vector alignment
    #   2. Full rotation
    #   3. Squeeze grasp
    #
    # You can specify more than one if you want and they will be OR'ed together.

    # For these options, we'll use a vector alignment constraint.

    force_top_down_grasp = True
    force_horizontal_grasp = False
    force_45_angle_grasp = False
    force_squeeze_grasp = False

    # force_top_down_grasp = False
    # force_horizontal_grasp = False
    # force_45_angle_grasp = False
    # force_squeeze_grasp = False

    use_vector_constraint = force_top_down_grasp or force_horizontal_grasp

    # Specify the frame we're using.
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector_constraint:
        if force_top_down_grasp:
            # Add a constraint that requests that the x-axis of the gripper is pointing in the
            # negative-z direction in the vision frame.

            # The axis on the gripper is the x-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

            # The axis in the vision frame is the negative z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)

        if force_horizontal_grasp:
            # Add a constraint that requests that the y-axis of the gripper is pointing in the
            # positive-z direction in the vision frame.  That means that the gripper is constrained to be rolled 90 degrees and pointed at the horizon.

            # The axis on the gripper is the y-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)

            # The axis in the vision frame is the positive z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

        # Add the vector constraint to our proto.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            axis_on_gripper_ewrt_gripper
        )
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align_with_ewrt_vo
        )

        # We'll take anything within about 10 degrees for top-down or horizontal grasps.
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17

    elif force_45_angle_grasp:
        # Demonstration of a RotationWithTolerance constraint.  This constraint allows you to
        # specify a full orientation you want the hand to be in, along with a threshold.
        #
        # You might want this feature when grasping an object with known geometry and you want to
        # make sure you grasp a specific part of it.
        #
        # Here, since we don't have anything in particular we want to grasp,  we'll specify an
        # orientation that will have the hand aligned with robot and rotated down 45 degrees as an
        # example.

        # First, get the robot's position in the world.
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(
            robot_state.kinematic_state.transforms_snapshot
        )

        # Rotation from the body to our desired grasp.
        body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)  # 45 degrees
        vision_Q_grasp = vision_T_body.rotation * body_Q_grasp

        # Turn into a proto
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(
            vision_Q_grasp.to_proto()
        )

        # We'll accept anything within +/- 10 degrees
        constraint.rotation_with_tolerance.threshold_radians = 0.17

    elif force_squeeze_grasp:
        # Tell the robot to just squeeze on the ground at the given point.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()


def move_arm_relative(robot_client: RobotClient, x: float, y: float, z: float) -> None:
    state = robot_client.state_client.get_robot_state()
    transforms = state.kinematic_state.transforms_snapshot.child_to_parent_edge_map

    body_T_hand_proto = transforms["hand"].parent_tform_child

    # # duration in seconds
    # seconds = 2
    offset = math_helpers.SE3Pose(x=x, y=y, z=z, rot=math_helpers.Quat(1, 0, 0, 0))

    body_T_hand = math_helpers.SE3Pose.from_proto(body_T_hand_proto)
    body_T_hand = offset * body_T_hand

    arm_command = RobotCommandBuilder.arm_pose_command(
        body_T_hand.x,
        body_T_hand.y,
        body_T_hand.z,
        body_T_hand.rot.w,
        body_T_hand.rot.x,
        body_T_hand.rot.y,
        body_T_hand.rot.z,
        GRAV_ALIGNED_BODY_FRAME_NAME,
        2.0,
    )

    # Make the open gripper RobotCommand
    gripper_command = RobotCommandBuilder.claw_gripper_close_command()

    # Combine the arm and gripper commands into one RobotCommand
    command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

    cmd_id = robot_client.command_client.robot_command(command)
    block_until_arm_arrives(robot_client.command_client, cmd_id, 1.5)


def stow_arm(robot_client: RobotClient) -> None:
    # Build the stow command using RobotCommandBuilder
    stow = RobotCommandBuilder.arm_stow_command()

    # Issue the command via the RobotCommandClient
    stow_command_id = robot_client.command_client.robot_command(stow)
    block_until_arm_arrives(robot_client.command_client, stow_command_id, 1.5)


def grasp(robot_client: RobotClient, image: Any, pixel: List[int]) -> None:
    success = False
    for outer_attempt in range(5):
        pick_vec = geometry_pb2.Vec2(x=pixel[0], y=pixel[1])

        # Build the proto
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole,
        )

        # Optionally add a grasp constraint.  This lets you tell the robot you only want top-down grasps or side-on grasps.
        add_grasp_constraint(grasp, robot_client.state_client)

        # Ask the robot to pick up the object
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(
            pick_object_in_image=grasp
        )

        # Send the request
        cmd_response = robot_client.manipulation_client.manipulation_api_command(
            manipulation_api_request=grasp_request
        )

        # Get feedback from the robot
        attempts = 0
        while attempts < 100:
            attempts += 1

            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id
            )

            # Send the request
            response = (
                robot_client.manipulation_client.manipulation_api_feedback_command(
                    manipulation_api_feedback_request=feedback_request
                )
            )

            print(
                f"Current state: {manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)}"
            )
            print("attempts: ", attempts)

            if (
                response.current_state
                == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED
                or response.current_state
                == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED
            ):
                success = True
                break

            time.sleep(0.25)

        if success:
            break

    robot_client.robot.logger.info("Finished grasp.")


def scan_room(robot_client: RobotClient, num_images: int = 20) -> List[RGBDImage]:
    """Helper function to scan the room using a sequence of RGBD images from
    several arm joint positions that form a circular 360 degree gripper
    path."""
    pos = ArmJointPositions.from_list([0, -2.0, 1.0, 0.0, 1.6, 0.0])
    rgbds = []
    for t in range(num_images):
        edge = np.pi / 8.0
        pos.sh0 = (np.pi * 2 - 2 * edge) * float(t) / float(num_images) - np.pi + edge
        move_arm(robot_client, arm_pos=pos)
        rgbds.append(capture_rgbd(robot_client, camera="hand"))
    return rgbds
