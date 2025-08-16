import apriltag
import cv2
import numpy as np

# --- Set Up Camera Parameters ---
# Replace these example values with your actual camera calibration data.
fx = 600  # focal length in pixels (x-direction)
fy = 600  # focal length in pixels (y-direction)
cx = 320  # principal point x-coordinate (in pixels)
cy = 240  # principal point y-coordinate (in pixels)
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

# Distortion coefficients: [k1, k2, p1, p2, k3]
dist_coeffs = np.zeros((5, 1), dtype=np.float32)  # assume no distortion for simplicity

# --- Define the AprilTag Size ---
# Set the physical side length of your AprilTag (in meters)
tag_size = 0.16  # for example, a 16 cm tag

# Define the 3D model points of the tag corners in the tag coordinate frame.
# We assume the tag is centered at (0,0,0) and lies on the xy-plane.
half_size = tag_size / 2.0
object_points = np.array(
    [
        [-half_size, half_size, 0],  # top-left
        [half_size, half_size, 0],  # top-right
        [half_size, -half_size, 0],  # bottom-right
        [-half_size, -half_size, 0],  # bottom-left
    ],
    dtype=np.float32,
)

# --- Load and Prepare the Image ---
# Replace 'path_to_your_image.jpg' with the actual path to your image file.
image = cv2.imread("path_to_your_image.jpg")
if image is None:
    raise IOError("Image not found. Please check the file path.")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- Set Up the AprilTag Detector ---
# Adjust the 'families' parameter to match your tag family, e.g., "tag36h11"
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)

# --- Detect AprilTags ---
detections = detector.detect(gray)
if len(detections) == 0:
    print("No AprilTags detected.")
    exit()

# For demonstration, we use the first detected tag.
detection = detections[0]

# The detected corners are in the order:
# [top-left, top-right, bottom-right, bottom-left]
image_points = np.array(detection.corners, dtype=np.float32)

# --- Pose Estimation Using solvePnP ---
# Compute the rotation and translation vectors using the PnP algorithm.
success, rvec, tvec = cv2.solvePnP(
    object_points, image_points, camera_matrix, dist_coeffs
)
if success:
    print("Pose estimation successful!")
    print("Rotation vector (rvec):\n", rvec)
    print("Translation vector (tvec):\n", tvec)
else:
    print("Pose estimation failed.")

# --- (Optional) Visualize the Results ---
# Project the 3D coordinate axes onto the image for visualization.
axis_length = tag_size * 0.5
axis_points = np.float32(
    [[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]]
)
imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)

corner = tuple(image_points[0].ravel().astype(int))
image = cv2.line(
    image, corner, tuple(imgpts[1].ravel().astype(int)), (0, 0, 255), 2
)  # X-axis in red
image = cv2.line(
    image, corner, tuple(imgpts[2].ravel().astype(int)), (0, 255, 0), 2
)  # Y-axis in green
image = cv2.line(
    image, corner, tuple(imgpts[3].ravel().astype(int)), (255, 0, 0), 2
)  # Z-axis in blue

cv2.imshow("Pose Estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
