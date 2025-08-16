# type:ignore
import argparse
import json
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
from scipy.stats import qmc
from tqdm import tqdm

import uncertain_worms.environments.spot.pb_utils as pbu
from uncertain_worms.environments.spot.spot_env_vg import (
    ARM_CONFS,
    VisibilityGrid,
    check_collision,
    compute_frustum,
    create_navigable_grid,
    load_virtual_map,
    pose_to_se2,
    round_vec,
    se2_to_pose,
)
from uncertain_worms.utils import PROJECT_ROOT

SPOT_ARM_URDF = os.path.join(
    PROJECT_ROOT, "environments/spot/spot_description/arm.urdf"
)


def greedy_set_cover(v1_to_v2, v2):
    """Greedy algorithm for covering all vertices in V2 using vertices from
    V1."""
    cover = set()  # Set of chosen vertices from V1
    uncovered = set(v2)  # Vertices in V2 that are not yet covered

    while uncovered:
        best_v1 = None
        best_covered = set()
        for v1, covers in v1_to_v2.items():
            covered_now = uncovered.intersection(covers)
            if len(covered_now) > len(best_covered):
                best_v1 = v1
                best_covered = covered_now

        if best_v1 is None:
            print(
                "Warning: Not all vertices in V2 can be covered with the given V1 vertices."
            )
            break

        cover.add(best_v1)
        uncovered -= best_covered

    return cover


def interpolate_state(s1, s2, t):
    """Linearly interpolates between two SE(2) states s1 and s2.

    s1 and s2 are tuples (x, y, theta). The angle theta is interpolated
    using the shortest angular difference.
    """
    x = s1[0] + t * (s2[0] - s1[0])
    y = s1[1] + t * (s2[1] - s1[1])
    dtheta = np.arctan2(np.sin(s2[2] - s1[2]), np.cos(s2[2] - s1[2]))
    theta = s1[2] + t * dtheta
    return (x, y, theta)


def edge_is_collision_free(s1, s2, occupancy_grid, min_bound, max_bound, num_steps=20):
    """Checks if a straight-line path (with linear interpolation in SE(2))
    between s1 and s2 is collision free."""
    for t in np.linspace(0, 1, num_steps):
        state = interpolate_state(s1, s2, t)
        if check_collision(occupancy_grid, state, min_bound, max_bound):
            return False
    return True


def get_pitch(point):
    # Here, point = [dx, dy, dz]
    # Pitch is the angle of elevation/depression relative to the horizontal (x-z plane).
    dx, dy, dz = point
    return np.arctan2(dy, np.sqrt(dx**2 + dz**2))


def get_yaw(point):
    # Yaw is measured on the horizontal plane (x-z).
    # It is the angle between the z-axis (forward) and the projection of the vector onto the x-z plane.
    dx, dy, dz = point
    return np.arctan2(dx, dz)


def get_looking_conf(spot_arm, stand_se2, target_point, points_array, client):
    pbu.set_joint_positions(
        spot_arm,
        pbu.joints_from_names(spot_arm, ARM_CONFS["ARM_DOWN"].keys(), client=client),
        ARM_CONFS["ARM_DOWN"].values(),
        client=client,
    )
    pbu.set_pose(spot_arm, se2_to_pose(stand_se2), client=client)

    camera_link = pbu.link_from_name(spot_arm, name="camera_optical", client=client)
    camera_pose = pbu.get_link_pose(
        spot_arm,
        link=camera_link,
        client=client,
    )

    delta_point = np.array(target_point) - np.array(camera_pose[0])
    dx, dy, dz = delta_point
    yaw = np.arctan2(dy, dx)
    pitch = np.arctan2(np.sqrt(dx**2 + dy**2), dz)
    target_camera_pose = pbu.Pose(
        pbu.Point(*camera_pose[0]), pbu.Euler(roll=0, pitch=pitch, yaw=yaw)
    )

    pbu.draw_pose(target_camera_pose, client=client)

    arm_joints = pbu.get_movable_joints(spot_arm, client=client)
    conf = pbu.solve_ik(
        spot_arm,
        link=camera_link,
        target_pose=target_camera_pose,
        joints=arm_joints,
        client=client,
    )

    if conf is None:
        # print("Ik failure")
        return None, None

    # --- Check collision between line and points ---
    # We compute the distance from each environment point (in points_array)
    # to the line segment from the camera to the target. If any point is too close,
    # we consider it a collision.
    camera_pos = np.array(camera_pose[0])
    target_pos = np.array(target_point)
    line_vec = target_pos - camera_pos
    line_len = np.linalg.norm(line_vec)
    if line_len > 0:
        line_dir = line_vec / line_len
    else:
        line_dir = line_vec  # degenerate case

    # Compute projection of each point onto the line (clamped to the segment)
    points = np.array(points_array)
    vec_to_points = points - camera_pos
    proj_lengths = np.dot(vec_to_points, line_dir)
    proj_lengths_clipped = np.clip(proj_lengths, 0, line_len)
    closest_points = camera_pos + np.outer(proj_lengths_clipped, line_dir)
    dists = np.linalg.norm(points - closest_points, axis=1)
    collision_threshold = 0.05  # meters
    if np.any(dists < collision_threshold):
        # print("Collision detected with environment points along the line.")
        return None, None

    # --- Check collision between line and robot body ---

    ray_results = client.rayTest(camera_pos.tolist(), target_point)
    hit_object, hit_link, hit_fraction, hit_pos, hit_normal = ray_results[0]
    if hit_object == spot_arm and hit_link != camera_link:
        # print("Collision detected with robot body along the line.")
        return None, None

    pbu.set_joint_positions(spot_arm, joints=arm_joints, values=conf, client=client)
    camera_pose = pbu.get_link_pose(spot_arm, camera_link, client=client)
    return conf, camera_pose


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the navigation graph generator with a specified config file path."
    )

    random.seed(0)
    np.random.seed(0)

    parser.add_argument(
        "--graphnav", type=str, required=True, help="Path to the config file."
    )
    args = parser.parse_args()

    # Load virtual map and occupancy grid
    (
        min_bound,
        max_bound,
        points_array,
        colors_array,
        room_pose,
        home_pose,
    ) = load_virtual_map(args.graphnav)
    home_se2 = pose_to_se2(home_pose)
    occupancy_grid = create_navigable_grid(points_array)
    print("Occupancy grid shape:", occupancy_grid.shape)  # e.g., (73,73)

    # --- Sample collision-free SE(2) states ---
    num_samples = 1
    nodes = [tuple(round_vec(home_se2))]  # Each node is a tuple (x, y, theta)
    max_attempts = num_samples * 1000
    attempts = 0

    sampler = qmc.Sobol(d=3, scramble=True)
    samples = sampler.random(n=max_attempts)
    scaled_samples = qmc.scale(samples, l_bounds=min_bound, u_bounds=max_bound)

    while len(nodes) < num_samples and attempts < max_attempts:
        x, y, theta = scaled_samples[attempts, :].tolist()
        state = tuple(round_vec([x, y, theta]))
        # Only add state if it is collision free.
        if not check_collision(occupancy_grid, state, min_bound, max_bound):
            nodes.append(state)
        attempts += 1

    print(f"Collected {len(nodes)} collision-free samples from SE(2).")

    # --- Create navigation edges ---
    # We connect nodes that are close enough and if the straight-line path between them is collision free.
    max_edge_distance = 1.0  # Maximum connection distance in meters
    edges = []  # Store edges as pairs of indices into the nodes list

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            s1 = nodes[i]
            s2 = nodes[j]
            # Euclidean distance in the plane (ignoring orientation)
            dist = np.sqrt((s2[0] - s1[0]) ** 2 + (s2[1] - s1[1]) ** 2)
            if dist <= max_edge_distance:
                if edge_is_collision_free(
                    s1, s2, occupancy_grid, min_bound, max_bound, num_steps=20
                ):
                    edges.append((i, j))

    print(f"Formed {len(edges)} navigation edges.")

    # --- Remove vertices not reachable from home_se2 ---
    # Build an undirected graph from the current edges.
    graph = {i: [] for i in range(len(nodes))}
    for i, j in edges:
        graph[i].append(j)
        graph[j].append(i)

    # Perform a breadth-first search (BFS) from home_index.
    reachable = set()
    queue = [0]
    while queue:
        current = queue.pop(0)
        if current in reachable:
            continue
        reachable.add(current)
        for neighbor in graph[current]:
            if neighbor not in reachable:
                queue.append(neighbor)

    print(
        f"Reachable nodes from home_se2: {len(reachable)} out of {len(nodes)} total nodes."
    )

    # Filter nodes and re-map edges to only include reachable vertices.
    new_nodes = []
    mapping = {}
    for i in range(len(nodes)):
        if i in reachable:
            mapping[i] = len(new_nodes)
            new_nodes.append(nodes[i])
    nodes = new_nodes

    new_edges = []
    for i, j in edges:
        if i in reachable and j in reachable:
            new_edges.append((mapping[i], mapping[j]))
    edges = new_edges

    print(
        f"After filtering unreachable nodes, {len(nodes)} nodes remain and {len(edges)} edges remain."
    )
    COMPUTE_GRAPH = True
    if COMPUTE_GRAPH:
        visibility_grid = VisibilityGrid(min_bound, max_bound, points_array)
        client = bc.BulletClient(connection_mode=p.DIRECT)

        spot_arm = pbu.load_pybullet(SPOT_ARM_URDF, client=client)

        distance_threshold = 2.5
        voxels = visibility_grid.get_voxels()
        voxel_points = visibility_grid.voxels_to_points(voxels)
        node_conf_to_voxel = defaultdict(lambda: defaultdict(list))
        total_viewed_voxels = []
        confs = []
        camera_poses = []
        for node_index in tqdm(range(len(nodes))):
            stand_se2 = nodes[node_index]
            dist_sq = (voxel_points[:, 0] - stand_se2[0]) ** 2 + (
                voxel_points[:, 1] - stand_se2[1]
            ) ** 2
            subset_points = voxel_points[dist_sq < distance_threshold**2]
            subset_voxels = voxels[dist_sq < distance_threshold**2]
            target_point_idxs = list(range(subset_points.shape[0]))
            random.shuffle(target_point_idxs)
            for target_point_idx in tqdm(target_point_idxs):
                target_point = subset_points[target_point_idx, :].tolist()
                target_voxel = int(
                    visibility_grid.index_to_flat_index(
                        subset_voxels[target_point_idx, :]
                    )
                )
                if target_voxel in total_viewed_voxels:
                    continue
                conf, camera_pose = get_looking_conf(
                    spot_arm, stand_se2, target_point, points_array, client=client
                )

                if conf is not None:
                    frustum_pts, frustum_lines = compute_frustum(camera_pose)
                    cleared_voxels = visibility_grid.clear_voxels_within_frustum(
                        frustum_pts, clear=False
                    ).tolist()

                    if len(cleared_voxels) > 0:
                        confs.append(conf)
                        camera_poses.append(camera_pose)
                        node_conf_to_voxel[node_index][
                            confs.index(conf)
                        ] += cleared_voxels
                        total_viewed_voxels += cleared_voxels

        print("Grid coverage: " + str(len(total_viewed_voxels) / voxel_points.shape[0]))

        voxel_to_node_conf = defaultdict(list)
        for node, conf_dict in node_conf_to_voxel.items():
            for conf, voxels in conf_dict.items():
                for voxel in voxels:
                    voxel_to_node_conf[voxel].append((node, conf))

        # Save dictionary to a JSON file
        data = {
            "navigable_nodes": nodes,
            "navigable_edges": edges,
            "node_conf_to_voxel": dict(node_conf_to_voxel),
            "voxel_to_node_conf": dict(voxel_to_node_conf),
            "total_viewed_voxels": total_viewed_voxels,
            "confs": confs,
            "camera_poses": camera_poses,
        }

        graph_path = os.path.join(
            PROJECT_ROOT, "environments/spot", args.graphnav, "visibility_graph.json"
        )
        with open(graph_path, "w") as file:
            json.dump(data, file, indent=4)

    else:
        # --- Plotting ---
        plt.figure(figsize=(8, 8))
        # Display occupancy grid. (Using 1 - occupancy_grid to have free space appear bright.)
        plt.imshow(
            1 - occupancy_grid,
            cmap="gray",
            origin="lower",
            extent=[min_bound[0], max_bound[0], min_bound[1], max_bound[1]],
        )
        plt.title("Navigation Graph on Occupancy Grid")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")

        # Plot edges
        for i, j in edges:
            x_vals = [nodes[i][0], nodes[j][0]]
            y_vals = [nodes[i][1], nodes[j][1]]
            plt.plot(x_vals, y_vals, "b-", linewidth=1)

        # Plot nodes
        nodes_xy = np.array([(state[0], state[1]) for state in nodes])
        plt.scatter(nodes_xy[:, 0], nodes_xy[:, 1], c="r", s=30, zorder=5)

        plt.colorbar(label="Occupancy Value")
        plt.show()
