# type:ignore
import math
import time

import numpy as np
import rerun as rr


def walking_animation(spot_rerun):
    base_conf = {}
    start_time = time.time()
    while True:
        t = time.time() - start_time

        # Move forward: adjust the x-position to simulate forward running
        base_conf["x"] = t * 0.5  # running at 0.5 m/s

        # Add a slight oscillation in orientation (theta) to mimic body motion
        base_conf["theta"] = math.sin(t * 2 * math.pi) * 0.1

        # Use sine waves for a periodic gait.
        # Front-left and hind-right legs are in phase while front-right and hind-left are offset by π.
        leg_amp = 0.2  # amplitude for joint oscillation
        leg_lift = 0.1  # amplitude for vertical movement

        # Update hip extension ("hx") for each leg
        base_conf["fl.hx"] = math.sin(t * 2 * math.pi) * leg_amp
        base_conf["hr.hx"] = math.sin(t * 2 * math.pi) * leg_amp
        base_conf["fr.hx"] = math.sin(t * 2 * math.pi + math.pi) * leg_amp
        base_conf["hl.hx"] = math.sin(t * 2 * math.pi + math.pi) * leg_amp

        # Update vertical position ("hy") to simulate leg lift (oscillating around 0.8)
        base_conf["fl.hy"] = 0.8 + math.sin(t * 2 * math.pi) * leg_lift
        base_conf["hr.hy"] = 0.8 + math.sin(t * 2 * math.pi) * leg_lift
        base_conf["fr.hy"] = 0.8 + math.sin(t * 2 * math.pi + math.pi) * leg_lift
        base_conf["hl.hy"] = 0.8 + math.sin(t * 2 * math.pi + math.pi) * leg_lift

        # Update knee ("kn") joints similarly (offset from a base value)
        base_conf["fl.kn"] = -1.6 + math.sin(t * 2 * math.pi) * 0.2
        base_conf["hr.kn"] = -1.6 + math.sin(t * 2 * math.pi) * 0.2
        base_conf["fr.kn"] = -1.6 + math.sin(t * 2 * math.pi + math.pi) * 0.2
        base_conf["hl.kn"] = -1.6 + math.sin(t * 2 * math.pi + math.pi) * 0.2

        spot_rerun.set_joint_positions(tuple(base_conf.values()))
        time.sleep(0.01)


def explore():
    # Initialize Visibility Grid using the point cloud's extent
    visibility_grid = VisibilityGrid(min_bound, max_bound, voxel_size)

    # Create Perceiver object
    perceiver = Perceiver(points=points_array, colors=colors_array)

    # Camera motion parameters
    num_steps = 50
    radius = 0.1
    height = -0.1
    center = np.array([0, 0, 0])  # Look-at point
    all_points = []
    all_colors = []
    visible_points = []
    visible_colors = []

    theta = 0  # Angle for circular motion

    # Define camera position in world coordinates
    cam_pos = np.array([radius * np.cos(theta), radius * np.sin(theta), height])

    # Compute look-at matrix
    forward = center - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(np.array([0, 0, 1]), forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)

    # Construct camera extrinsics
    camera_pose = np.eye(4)
    camera_pose[:3, 0] = right
    camera_pose[:3, 1] = up
    camera_pose[:3, 2] = -forward
    camera_pose[:3, 3] = cam_pos

    for i in range(num_steps):
        rr.set_time_sequence("timeline", i + 1)

        # Get visible points from the current camera view
        visible_points, visible_colors = perceiver.get_visible_points(
            camera_pose, intrinsics, img_size
        )

        num_candidates = 100
        best_num_voxels = -1
        best_pose = None
        mock_perciever = Perceiver(
            points=np.array(visible_points), colors=np.array(visible_colors)
        )
        for _ in range(num_candidates):
            # Sample a random camera position within the visibility grid
            candidate_cam_pos = np.random.uniform(
                visibility_grid.min_bound, visibility_grid.max_bound
            )

            # Sample random Euler angles (yaw, pitch, roll).
            yaw = np.random.uniform(0, 2 * np.pi)
            # Restrict pitch so the camera is not looking too far up or down.
            pitch = np.random.uniform(-np.pi, np.pi)
            roll = np.random.uniform(-np.pi, np.pi)

            # Compute the rotation matrix from Euler angles.
            Rz = np.array(
                [
                    [np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1],
                ]
            )
            Ry = np.array(
                [
                    [np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)],
                ]
            )
            Rx = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)],
                ]
            )
            R = Rz @ Ry @ Rx

            # Build candidate camera pose (4x4 homogeneous matrix).
            candidate_cam_pose = np.eye(4)
            candidate_cam_pose[:3, :3] = R
            candidate_cam_pose[:3, 3] = candidate_cam_pos

            # Get the candidate's visible points (in camera frame) and colors.
            (
                candidate_visible_points,
                candidate_visible_colors,
            ) = mock_perciever.get_visible_points(
                candidate_cam_pose, intrinsics, img_size
            )
            if candidate_visible_points.shape[0] == 0:
                continue  # skip candidates that see nothing

            # Simulate updating the grid on a temporary copy.
            temp_grid = copy.deepcopy(visibility_grid)
            temp_grid.update_visibility(candidate_cam_pos, candidate_visible_points)
            # Count the number of voxels that were cleared by this candidate.
            num_voxels_updated = np.sum(visibility_grid.grid & (~temp_grid.grid))
            if num_voxels_updated > best_num_voxels:
                best_num_voxels = num_voxels_updated
                best_pose = candidate_cam_pose
                best_transformed_points = candidate_visible_points
                best_visible_colors = candidate_visible_colors

        print("num_voxels_updated: " + str(best_num_voxels))

        if best_pose is None:
            # Fallback: if no candidate saw any points, use a default pose.
            camera_pose = np.eye(4)
            cam_pos = camera_pose[:3, 3]
            visible_points, visible_colors = perceiver.get_visible_points(
                camera_pose, intrinsics, img_size
            )
        else:
            camera_pose = best_pose
            cam_pos = camera_pose[:3, 3]

        # Compute and log the camera frustum for debugging
        frustum_pts, frustum_lines = compute_frustum(camera_pose, intrinsics, img_size)
        rr.log("debug/frustum_points", rr.Points3D(positions=frustum_pts))
        rr.log("debug/frustum_lines", rr.LineStrips3D(frustum_lines))

        # Debug: Print number of visible points in this step
        print(f"Step {i}: {len(visible_points)} points visible.")

        # Update the visibility grid based on the rays from the camera to the visible points
        st = time.time()
        visibility_grid.update_visibility(cam_pos, visible_points)
        print("Update time: " + str(time.time() - st))

        # Visualize the remaining voxels as points (voxel centers)
        voxel_points = visibility_grid.get_points()
        print("Num voxel points: " + str(len(voxel_points)))
        voxel_colors = [[0, 0, 255] for _ in range(len(voxel_points))]  # Blue points

        rr.log(
            "visibility_grid_points",
            rr.Boxes3D(
                centers=voxel_points,
                half_sizes=[[voxel_size / 2.0 for _ in range(3)] for _ in voxel_points],
                colors=voxel_colors,
            ),
        )
        # rr.log("visibility_grid_points", rr.Points3D(positions=voxel_points, colors=voxel_colors, radii=[0.05 for _ in voxel_points]))

        all_points.append(visible_points)
        all_colors.append(visible_colors)

        all_points_stacked = np.vstack(all_points)
        all_colors_stacked = np.vstack(all_colors)
        rr.log(
            "visible_pointcloud",
            rr.Points3D(positions=all_points_stacked, colors=all_colors_stacked),
        )
