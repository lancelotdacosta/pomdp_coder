"""Scan the scene using only the gripper camera, segment the scene with an
image scene segmentation network, project the segmentations into a 3d
pointcloud, and visualize the segmented pointcloud."""

import os
import time

import bosdyn
from bosdyn.client.image import ImageClient

from uncertain_worms.environments.spot.spot_utils.controllers.startup import (
    DEFAULT_SPOT_IP,
    RobotClient,
)
from uncertain_worms.environments.spot.spot_utils.perception.capture import (
    capture_rgbd,
    save_rgbd,
)

if __name__ == "__main__":
    cameras = ["frontright"]
    print("setting up robot")
    bosdyn.client.util.setup_logging(False)
    sdk = bosdyn.client.create_standard_sdk("RobotClient")
    robot = sdk.create_robot(DEFAULT_SPOT_IP)

    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    save_dir = "./outputs_id"
    os.makedirs(save_dir, exist_ok=True)

    video = False
    input("Ready?")
    while True:
        if not video:
            input("Capture image?")
        time_str = str(time.time())
        for camera_name in cameras:
            image_client = robot.ensure_client(ImageClient.default_service_name)
            robot_client = RobotClient(robot=robot, sdk=sdk, image_client=image_client)
            rgbd = capture_rgbd(robot_client, camera=camera_name)
            save_rgbd(
                rgbd=rgbd,
                save_name=time_str,
                save_dir=os.path.join(save_dir, camera_name),
            )
            print("Captured: " + str(camera_name))
