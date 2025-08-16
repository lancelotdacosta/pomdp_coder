# type:ignore
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives

from uncertain_worms.environments.spot.spot_constants import *
from uncertain_worms.environments.spot.spot_utils.controllers.startup import setup_robot


def main():
    real_robot_client = setup_robot(graphnav="spot_room_graphnav", spot_ip=SPOT_IP)
    # robot_state = real_robot_client.state_client.get_robot_state()

    # state = real_robot_client.state_client.get_robot_state()
    # transforms = state.kinematic_state.transforms_snapshot.child_to_parent_edge_map

    # body_T_hand_proto = transforms["hand"].parent_tform_child

    # # # duration in seconds
    # # seconds = 2
    # math_helpers.SE3Pose
    # body_T_hand = math_helpers.SE3Pose.from_proto(body_T_hand_proto)

    # arm_command = RobotCommandBuilder.arm_pose_command(
    #     body_T_hand.x, body_T_hand.y, body_T_hand.z+0.1, body_T_hand.rot.w, body_T_hand.rot.x,
    #     body_T_hand.rot.y, body_T_hand.rot.z, GRAV_ALIGNED_BODY_FRAME_NAME, 3.0)

    # # Make the open gripper RobotCommand
    # gripper_command = RobotCommandBuilder.claw_gripper_close_command()

    # # Combine the arm and gripper commands into one RobotCommand
    # command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)
    # cmd_id = real_robot_client.command_client.robot_command(command)
    # block_until_arm_arrives(real_robot_client.command_client, cmd_id, 1.5)

    # Build the stow command using RobotCommandBuilder
    stow = RobotCommandBuilder.arm_stow_command()

    # Issue the command via the RobotCommandClient
    stow_command_id = real_robot_client.command_client.robot_command(stow)
    block_until_arm_arrives(real_robot_client.command_client, stow_command_id, 1.5)


if __name__ == "__main__":
    main()
