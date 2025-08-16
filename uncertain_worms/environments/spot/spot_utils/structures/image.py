"""Data structures for images and pointclouds."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from bosdyn.client.math_helpers import SE3Pose  # type: ignore
from numpy.typing import NDArray


@dataclass
class Intrinsics:
    """Camera intrinsics."""

    rows: int
    cols: int
    fx: float
    fy: float
    cx: float
    cy: float

    def to_matrix(self) -> NDArray[np.float64]:
        """Converts intrinsics parameters into a 3x3 camera matrix."""
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])


@dataclass
class RGBImage:
    """Color image."""

    rgb: NDArray[np.uint8]
    frame: SE3Pose
    intrinsics: Intrinsics
    response: Any = None


@dataclass
class DepthImage:
    """Depth image."""

    depth: NDArray[np.uint16]
    frame: SE3Pose
    intrinsics: Intrinsics
    depth_scale: float = 1
    response: Any = None


@dataclass
class RGBDImage:
    """Color + depth image."""

    rgb: NDArray[np.uint8]
    depth: NDArray[np.uint16]
    frame: SE3Pose
    intrinsics: Intrinsics
    depth_scale: float = 1
    rgb_response: Any = None
    depth_response: Any = None


@dataclass
class PointCloud:
    """Cloud of points with positions and colors."""

    xyz: NDArray[np.float64]
    rgb: NDArray[np.uint8]
