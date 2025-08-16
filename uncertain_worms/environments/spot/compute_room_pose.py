# type:ignore
import logging
import os
from collections import defaultdict

import apriltag
import cv2
import numpy as np
import rerun as rr
from bosdyn.api.graph_nav import map_pb2
from polyform.core.capture_folder import CaptureFolder, Keyframe
from scipy.spatial.transform import Rotation

import uncertain_worms.environments.spot.pb_utils as pbu
from uncertain_worms.utils import PROJECT_ROOT

_log = logging.getLogger(__name__)


def convert_opengl_to_opencv(c2w: np.ndarray):
    """Convert from OpenGL to OpenCV coordinate convention."""
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    c2w = np.matmul(c2w, flip_yz)
    return c2w


def process_frame(frame: Keyframe) -> dict:
    corrected = frame.is_optimized()
    rgb_path = frame.corrected_image_path if corrected else frame.image_path

    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth_path = frame.depth_path
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

    confidence_path = frame.confidence_path
    confidence = cv2.imread(confidence_path)
    confidence_processed = np.full_like(confidence[..., 0], 3, dtype=np.uint8)
    confidence_processed[confidence[..., 0] == 0] = 0
    confidence_processed[confidence[..., 0] == 54] = 1
    confidence_processed[confidence[..., 0] == 255] = 2
    # Check 3 does not exist anymore
    assert (confidence_processed == 3).sum() == 0

    intrinsic = {
        "cx": frame.camera.cx,
        "cy": frame.camera.cy,
        "fx": frame.camera.fx,
        "fy": frame.camera.fy,
        "width": frame.camera.width,
        "height": frame.camera.height,
    }

    c2w = convert_opengl_to_opencv(frame.camera.transform)
    return {
        "rgb": rgb,
        "depth": depth,
        "confidence": confidence_processed,
        "intrinsic": intrinsic,
        "c2w": c2w,
        "rgb_path": rgb_path,
        "depth_path": depth_path,
    }


def get_point_cloud(
    rgb,
    depth,
    fx,
    fy,
    cx,
    cy,
    c2w=None,
    mask=None,
    depth_scale: float = 1000.0,
    depth_trunc: float = 5.0,
):
    """Convert RGB-D image with given camera intrinsics to a point cloud."""
    # We only consider points within the depth truncation
    z = depth / depth_scale
    if mask is not None:
        combined_mask = np.logical_and(z <= depth_trunc, mask)
    else:
        combined_mask = z <= depth_trunc
    z = z[combined_mask].reshape(-1)

    # Get pixel coordinates
    height, width = rgb.shape[:2]
    vu = np.indices((height, width))
    vu = vu[:, combined_mask].reshape(2, -1)
    assert z.shape[0] == vu.shape[1]

    # Compute x, y in camera coordinates
    x = (vu[1] - cx) * z / fx
    y = (vu[0] - cy) * z / fy
    points = np.vstack((x, y, z)).T

    # Transform to world coordinates
    if c2w is not None:
        points = c2w @ np.hstack((points, np.ones((points.shape[0], 1)))).T
        points = points[:3].T

    # Get colors
    colors = rgb[combined_mask].reshape(-1, 3)
    return points, colors


def downsample_point_cloud(points, colors, voxel_size: float):
    """Voxel downsample a point cloud.

    Reference: https://github.com/isl-org/Open3D/blob/d7a2cf608a5e206d8ebc3b78d947c219cd4da8fb/cpp/open3d/geometry/PointCloud.cpp#L354
    """
    assert voxel_size > 0, "voxel_size must be positive"
    voxel_min_bound = points.min(0) - voxel_size * 0.5

    ref_coords = (points - voxel_min_bound) / voxel_size
    voxel_idxs = ref_coords.astype(int)
    voxel_idxs, inverse, counts = np.unique(
        voxel_idxs, axis=0, return_inverse=True, return_counts=True
    )

    voxels = np.zeros((voxel_idxs.shape[0], 3))
    np.add.at(voxels, inverse, points)
    voxels /= counts.reshape(-1, 1)

    voxel_colors = np.zeros((voxel_idxs.shape[0], 3))
    np.add.at(voxel_colors, inverse, colors)
    voxel_colors /= counts.reshape(-1, 1)
    voxel_colors = voxel_colors.astype(np.uint8)
    assert voxel_colors.max() <= 255

    return voxels, voxel_colors


def draw_apriltags(frame, detections):
    for detection in detections:
        # Draw the tag outline
        for i in range(4):
            p1 = tuple(detection.corners[i - 1, :].astype(int))
            p2 = tuple(detection.corners[i, :].astype(int))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

        # Draw the tag ID
        p = tuple(detection.center.astype(int))
        cv2.putText(
            frame,
            str(detection.tag_id),
            p,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    return frame


import numpy as np
from scipy.spatial.transform import Rotation


def average_quaternions(quats: np.ndarray) -> np.ndarray:
    """Compute the average quaternion using the eigenvalue method.

    quats: an (N,4) array of quaternions (in [x, y, z, w] format)
    Returns the average quaternion as a (4,) array.
    """
    # Ensure all quaternions have the same sign (the double-cover issue)
    Q = np.copy(quats)
    for i in range(1, Q.shape[0]):
        if np.dot(Q[0], Q[i]) < 0:
            Q[i] = -Q[i]

    # Build the symmetric accumulator matrix
    A = np.zeros((4, 4))
    for q in Q:
        A += np.outer(q, q)
    A /= Q.shape[0]

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    max_index = np.argmax(eigenvalues)
    avg_quat = eigenvectors[:, max_index]
    avg_quat /= np.linalg.norm(avg_quat)
    return avg_quat


def demo_april_tag(dataset_path: str, viz_pcd: bool = False, max_images: int = 20):
    rr.init("april_tag", spawn=True)
    folder = CaptureFolder(str(dataset_path))
    keyframes = folder.get_keyframes(rotate=True)
    _log.info(
        f"Loaded Polycam scan from {dataset_path} with {len(keyframes)} keyframes"
    )

    # Setup AprilTag detector
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    tag_id_to_poses = defaultdict(list)
    tag_size = 0.146
    voxel_size = 0.025
    points, colors = None, None

    tag_poses = {}

    for idx, raw_frame in enumerate(keyframes):
        if idx > max_images:
            break
        rr.set_time_sequence("frame", idx)
        frame = process_frame(raw_frame)

        intrinsic = frame["intrinsic"]
        width, height = intrinsic["width"], intrinsic["height"]
        rr.log("cam/rgb", rr.Image(frame["rgb"]))

        # Resize depth
        depth = frame["depth"]
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        rr.log("cam/depth", rr.DepthImage(depth, meter=1000.0))

        # Log intrinsic
        cx, cy = intrinsic["cx"], intrinsic["cy"]
        fx, fy = intrinsic["fx"], intrinsic["fy"]
        rr.log(
            "cam",
            rr.Pinhole(
                focal_length=[fx, fy],
                principal_point=[cx, cy],
                width=width,
                height=height,
            ),
        )

        # Log extrinsic
        c2w = frame["c2w"]
        rr.log("cam", rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]))

        # Point cloud - slows down visualization a bit
        if viz_pcd:
            confidence = frame["confidence"]
            confidence = cv2.resize(
                confidence, (width, height), interpolation=cv2.INTER_NEAREST
            )
            mask = confidence == 2  # high confidence depth only
            new_points, new_colors = get_point_cloud(
                frame["rgb"], depth, fx=fx, fy=fy, cx=cx, cy=cy, c2w=c2w, mask=mask
            )
            new_points, new_colors = downsample_point_cloud(
                new_points, new_colors, voxel_size=0.02
            )
            # append
            if points is None:
                points, colors = new_points, new_colors
            else:
                points = np.vstack((points, new_points))
                colors = np.vstack((colors, new_colors))
            # downsample to 2cm
            points, colors = downsample_point_cloud(
                points, colors, voxel_size=voxel_size
            )
            rr.log(
                "pcd",
                rr.Points3D(positions=points, colors=colors, radii=voxel_size / 2),
            )

        # Detect AprilTags
        gray = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2GRAY)
        detections = detector.detect(gray)
        annotated_frame = draw_apriltags(frame["rgb"].copy(), detections)
        rr.log("cam/april_tag", rr.Image(annotated_frame))
        if not detections:
            continue

        _log.info(
            f"Detected {len(detections)} AprilTags ({[d.tag_id for d in detections]}) in frame {idx}"
        )
        for detection in detections:
            tag_id = detection.tag_id

            # Extract pose from detection
            pose, init_error, final_error = detector.detection_pose(
                detection, [fx, fy, cx, cy], tag_size
            )
            _log.info(
                f"Tag {tag_id} init error: {init_error}, final error: {final_error}"
            )

            # Transform tag pose to world coordinates
            tag_pose = c2w @ pose

            flip_xz = np.eye(3)
            flip_xz[0, 0] = -1
            flip_xz[2, 2] = -1
            # Rotate the pose 90 degrees around its own z axis
            tag_pose[:3, :3] = (
                tag_pose[:3, :3]
                @ Rotation.from_euler("z", 90, degrees=True).as_matrix()
            )
            tag_pose[:3, :3] = tag_pose[:3, :3] @ flip_xz
            tag_id_to_poses[tag_id].append(tag_pose)

            rr.log(
                f"april_tag/{tag_id}",
                rr.Transform3D(translation=tag_pose[:3, 3], mat3x3=tag_pose[:3, :3]),
            )
            rr.log_components(
                f"april_tag/{tag_id}", [rr.components.AxisLength(0.5)], static=True
            )

    # Average the poses for each tag using a robust method for rotation
    tag_id_to_poses = {k: np.array(v) for k, v in tag_id_to_poses.items()}
    tag_id_to_pose = {}
    for tag_id, poses in tag_id_to_poses.items():
        # Extract rotation matrices and translations from poses
        rotations = poses[:, :3, :3]
        translations = poses[:, :3, 3]

        # Convert rotations to quaternions (scipy uses [x,y,z,w] ordering)
        quats = Rotation.from_matrix(rotations).as_quat()

        # Use the robust averaging function for quaternions
        mean_quat = average_quaternions(quats)

        # Mean translation (arithmetic mean is fine for positions)
        mean_tvec = np.mean(translations, axis=0)

        # Form the homogeneous transform for the averaged pose
        mat4x4 = np.eye(4)
        mat4x4[:3, :3] = Rotation.from_quat(mean_quat).as_matrix()
        mat4x4[:3, 3] = mean_tvec
        tag_id_to_pose[tag_id] = mat4x4

        tag_poses[str(tag_id)] = (mean_tvec.tolist(), mean_quat.tolist())
        pos_err = np.linalg.norm(translations - mean_tvec, axis=1)
        _log.info(f"Tag {tag_id} mean translation error: {pos_err}")

        rr.log(
            f"april_tag/{tag_id}",
            rr.Transform3D(translation=mean_tvec, mat3x3=mat4x4[:3, :3]),
        )

    _log.info("Check rerun for visualization")
    return tag_poses


def compute_graphnav_T_polycam(
    polycam_fid_poses: dict, graphnav_fid_poses: dict
) -> np.ndarray:
    # Convert polycam keys to strings to match graphnav keys and find the overlapping set.
    overlapping_tags = set(str(tag) for tag in polycam_fid_poses.keys()) & set(
        graphnav_fid_poses.keys()
    )

    if len(overlapping_tags) < 1:
        raise ValueError(
            "No overlapping tags found between Polycam and GraphNav fiducials."
        )

    polycam_points = []
    graphnav_points = []

    for tag in overlapping_tags:
        # polycam fiducial poses: (translation, quaternion)
        polycam_translation = np.array(polycam_fid_poses[tag][0])
        graphnav_translation = np.array(graphnav_fid_poses[tag][0])
        polycam_points.append(polycam_translation)
        graphnav_points.append(graphnav_translation)

    polycam_points = np.array(polycam_points)  # shape (N, 3)
    graphnav_points = np.array(graphnav_points)  # shape (N, 3)

    # Compute centroids
    centroid_polycam = polycam_points.mean(axis=0)
    centroid_graphnav = graphnav_points.mean(axis=0)

    # Center the points
    polycam_centered = polycam_points - centroid_polycam
    graphnav_centered = graphnav_points - centroid_graphnav

    # Compute cross-covariance matrix
    H = polycam_centered.T @ graphnav_centered

    # Compute SVD of H
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure a proper rotation matrix (determinant == 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = centroid_graphnav - R @ centroid_polycam

    # Build homogeneous transform matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


if __name__ == "__main__":
    dataset_path = os.path.join(
        PROJECT_ROOT, "environments/spot/world_scans/spot_room_graphnav"
    )

    polycam_fid_poses = demo_april_tag(dataset_path, max_images=20, viz_pcd=True)
    fid_poses = []
    graphnav_fid_poses = {}
    with open(os.path.join(dataset_path, "graph"), "rb") as graph_file:
        # Load the graph file and deserialize it. The graph file is a protobuf containing only the waypoints and the
        # edges between them.
        data = graph_file.read()
        current_graph = map_pb2.Graph()
        current_graph.ParseFromString(data)

        # Set up maps from waypoint ID to waypoints, edges, snapshots, etc.
        current_waypoints = {}
        current_waypoint_snapshots = {}
        current_edge_snapshots = {}
        current_anchors = {}
        current_anchored_world_objects = {}

        fiducials = []
        # Load the anchored world objects first so we can look in each waypoint snapshot as we load it.
        for anchored_world_object in current_graph.anchoring.objects:
            fname = str(anchored_world_object.id)
            fid_tf = anchored_world_object.seed_tform_object
            fid_pose = (
                (fid_tf.position.x, fid_tf.position.y, fid_tf.position.z),
                (
                    fid_tf.rotation.x,
                    fid_tf.rotation.y,
                    fid_tf.rotation.z,
                    fid_tf.rotation.w,
                ),
            )

            graphnav_fid_poses[str(fname)] = fid_pose

    print(polycam_fid_poses)
    print(graphnav_fid_poses)

    graphnav_T_polycam = compute_graphnav_T_polycam(
        polycam_fid_poses, graphnav_fid_poses
    )

    graphnav_T_polycam_pose = pbu.pose_from_tform(graphnav_T_polycam)
    for fname, gn_fid_pose in graphnav_fid_poses.items():
        fid_pose = pbu.multiply(pbu.invert(graphnav_T_polycam_pose), gn_fid_pose)
        rr.log(
            f"fiducial/{fname}",
            rr.Transform3D(
                translation=fid_pose[0],
                mat3x3=Rotation.from_quat(fid_pose[1]).as_matrix(),
            ),
        )
        rr.log_components(
            f"fiducial/{fname}", [rr.components.AxisLength(0.5)], static=True
        )

    print("GraphNav to Polycam transform:")
    print(pbu.pose_from_tform(graphnav_T_polycam))
