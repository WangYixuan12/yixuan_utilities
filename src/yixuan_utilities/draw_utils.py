from typing import Optional, Tuple

import cv2
import numpy as np


def center_crop(img: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    th, tw = crop_size
    if h / w > th / tw:
        # image is taller than crop
        crop_w = w
        crop_h = int(round(w * th / tw))
    elif h / w < th / tw:
        # image is wider than crop
        crop_h = h
        crop_w = int(round(h * tw / th))
    else:
        return img
    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    return img[y1 : y1 + crop_h, x1 : x1 + crop_w]


def resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height), interpolation=cv2.INTER_LINEAR)


def project_coordinate_frame_to_image(
    pose_3d: np.ndarray,
    camera_extrinsics: np.ndarray,
    camera_intrinsics: np.ndarray,
    image: np.ndarray,
    axis_length: float = 0.1,
    line_thickness: int = 2,
) -> np.ndarray:
    """Project a 3D coordinate frame onto an image.

    Args:
        pose_3d (np.ndarray): 3D pose matrix of shape (4, 4) for the coordinate frame
        camera_extrinsics (np.ndarray): Camera extrinsics matrix of shape (4, 4)
        camera_intrinsics (np.ndarray): Camera intrinsics matrix of shape (3, 3)
        image (np.ndarray): Original image of shape (H, W, C)
        axis_length (float): Length of the coordinate frame axes in meters
        line_thickness (int): Thickness of the projected lines

    Returns:
        np.ndarray: Image with projected coordinate frame overlaid
    """
    # Define coordinate frame points (origin and axis endpoints)
    origin = np.array([0, 0, 0, 1])  # Origin point
    x_axis = np.array([axis_length, 0, 0, 1])  # X-axis endpoint (red)
    y_axis = np.array([0, axis_length, 0, 1])  # Y-axis endpoint (green)
    z_axis = np.array([0, 0, axis_length, 1])  # Z-axis endpoint (blue)

    # Transform points to world coordinates
    origin_world = pose_3d @ origin
    x_axis_world = pose_3d @ x_axis
    y_axis_world = pose_3d @ y_axis
    z_axis_world = pose_3d @ z_axis

    # Transform to camera coordinates
    origin_cam = camera_extrinsics @ origin_world
    x_axis_cam = camera_extrinsics @ x_axis_world
    y_axis_cam = camera_extrinsics @ y_axis_world
    z_axis_cam = camera_extrinsics @ z_axis_world

    # Project to image coordinates
    def project_point(point_3d: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        """Project a 3D point to 2D image coordinates"""
        # Normalize homogeneous coordinates
        point_3d = point_3d[:3] / point_3d[3]

        # Check if point is in front of camera
        if point_3d[2] <= 0:
            return None, None

        # Project using camera intrinsics
        point_2d = camera_intrinsics @ point_3d
        u, v = int(point_2d[0] / point_2d[2]), int(point_2d[1] / point_2d[2])

        return u, v

    # Project all points
    origin_2d = project_point(origin_cam)
    x_axis_2d = project_point(x_axis_cam)
    y_axis_2d = project_point(y_axis_cam)
    z_axis_2d = project_point(z_axis_cam)

    # Create a copy of the image for drawing
    result_image = image.copy()

    # Draw coordinate frame axes
    if origin_2d[0] is not None:
        origin_u, origin_v = origin_2d

        # X-axis (red)
        if x_axis_2d[0] is not None:
            cv2.line(
                result_image,
                (origin_u, origin_v),
                (x_axis_2d[0], x_axis_2d[1]),
                (0, 0, 255),  # BGR: red
                line_thickness,
            )

        # Y-axis (green)
        if y_axis_2d[0] is not None:
            cv2.line(
                result_image,
                (origin_u, origin_v),
                (y_axis_2d[0], y_axis_2d[1]),
                (0, 255, 0),  # BGR: green
                line_thickness,
            )

        # Z-axis (blue)
        if z_axis_2d[0] is not None:
            cv2.line(
                result_image,
                (origin_u, origin_v),
                (z_axis_2d[0], z_axis_2d[1]),
                (255, 0, 0),  # BGR: blue
                line_thickness,
            )

    return result_image
