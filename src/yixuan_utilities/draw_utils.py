import copy
import os
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import open3d as o3d
import torch


class ImgEncoding(Enum):
    """Image encoding format"""

    RGB_UINT8 = "rgb_uint8"
    BGR_UINT8 = "bgr_uint8"
    DEPTH_UINT16 = "depth_uint16"
    DEPTH_FLOAT = "depth_float"


class ExtriConvention(Enum):
    """Extrinsic convention for camera pose"""

    CAM_IN_WORLD = "cam_in_world"  # camera pose in world coord
    WORLD_IN_CAM = "world_in_cam"  # world pose in camera coord


def depth2fgpcd(
    depth: np.ndarray, mask: np.ndarray, cam_params: List, preserve_zero: bool = False
) -> np.ndarray:
    """Convert depth image to foreground point cloud"""
    # depth: (h, w)
    # fgpcd: (n, 3)
    # mask: (h, w)
    h, w = depth.shape
    if not preserve_zero:
        mask = np.logical_and(mask, depth > 0)
    fgpcd = np.zeros((mask.sum(), 3))
    fx, fy, cx, cy = cam_params
    pos_x, pos_y = np.meshgrid(np.arange(w), np.arange(h))
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    fgpcd[:, 0] = (pos_x - cx) * depth[mask] / fx
    fgpcd[:, 1] = (pos_y - cy) * depth[mask] / fy
    fgpcd[:, 2] = depth[mask]
    return fgpcd


def depth2fgpcd_tensor(depth: torch.Tensor, cam_params: torch.Tensor) -> torch.Tensor:
    """Convert depth image to foreground point cloud using PyTorch tensors.

    Args:
    depth (torch.Tensor): Depth image tensor of shape (b, h, w).
    cam_params (torch.Tensor): Camera parameters tensor of shape (b, 4).

    Returns:
    torch.Tensor: Foreground point cloud tensor of shape (b, h*w, 3).
    """
    b, h, w = depth.shape

    pos_x, pos_y = torch.meshgrid(
        torch.arange(w, device=depth.device),
        torch.arange(h, device=depth.device),
        indexing="xy",
    )
    pos_x = pos_x.unsqueeze(0).expand(b, -1, -1).reshape(b, h * w)
    pos_y = pos_y.unsqueeze(0).expand(b, -1, -1).reshape(b, h * w)

    cam_params = cam_params.unsqueeze(-1).unsqueeze(
        -1
    )  # Expand cam_params to match the shape of pos_x and pos_y

    fx = cam_params[:, 0]
    fy = cam_params[:, 1]
    cx = cam_params[:, 2]
    cy = cam_params[:, 3]

    fgpcd = torch.zeros((b, h * w, 3), device=depth.device, dtype=torch.float32)
    fgpcd[..., 0] = (pos_x - cx) * depth.reshape(b, h * w) / fx
    fgpcd[..., 1] = (pos_y - cy) * depth.reshape(b, h * w) / fy
    fgpcd[..., 2] = depth.reshape(b, h * w)

    return fgpcd


def aggr_point_cloud_from_data(
    colors: np.ndarray,
    depths: np.ndarray,
    Ks: np.ndarray,
    poses: np.ndarray,
    downsample: bool = True,
    downsample_r: float = 0.01,
    masks: np.ndarray = None,
    boundaries: Optional[Dict] = None,
    out_o3d: bool = True,
    #    excluded_pts=None,
    #    exclude_threshold=0.01,):
    color_fmt: ImgEncoding = ImgEncoding.RGB_UINT8,
    depth_fmt: ImgEncoding = ImgEncoding.DEPTH_FLOAT,
    pose_fmt: ExtriConvention = ExtriConvention.WORLD_IN_CAM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate point cloud from multi-view RGBD obs"""
    # colors: [N, H, W, 3] numpy array in uint8
    # depths: [N, H, W] numpy array in meters
    # Ks: [N, 3, 3] numpy array
    # poses: [N, 4, 4] numpy array
    # masks: [N, H, W] numpy array in bool
    N, H, W, _ = colors.shape
    if color_fmt == ImgEncoding.RGB_UINT8:
        colors = colors / 255.0
    elif color_fmt == ImgEncoding.BGR_UINT8:
        colors = colors[..., ::-1] / 255.0
    if depth_fmt == ImgEncoding.DEPTH_UINT16:
        depths = depths / 1000.0
    start = 0
    end = N
    step = 1
    pcds_ls = []
    pcd_colors = []
    # TODO: batch it
    for i in range(start, end, step):
        depth = depths[i]
        color = colors[i]
        K = Ks[i]
        cam_param = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]  # fx, fy, cx, cy
        if masks is None:
            mask = depth > 0
        else:
            mask = masks[i] & (depth > 0)
        # mask = np.ones_like(depth, dtype=bool)

        pcd = depth2fgpcd(depth, mask, cam_param)

        pose = poses[i]
        if pose_fmt == ExtriConvention.WORLD_IN_CAM:
            try:
                pose = np.linalg.inv(pose)
            except np.linalg.LinAlgError:
                print("singular matrix")
                pose = np.linalg.pinv(pose)

        trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
        trans_pcd = trans_pcd[:3, :].T

        color = color[mask]

        pcds_ls.append(trans_pcd)
        pcd_colors.append(color)

    pcds = np.concatenate(pcds_ls, axis=0)
    pcd_colors = np.concatenate(pcd_colors, axis=0)

    # post process 1: remove points outside of boundaries
    if boundaries is not None:
        x_lower = boundaries["x_lower"]
        x_upper = boundaries["x_upper"]
        y_lower = boundaries["y_lower"]
        y_upper = boundaries["y_upper"]
        z_lower = boundaries["z_lower"]
        z_upper = boundaries["z_upper"]

        pcd_mask = (
            (pcds[:, 0] > x_lower)
            & (pcds[:, 0] < x_upper)
            & (pcds[:, 1] > y_lower)
            & (pcds[:, 1] < y_upper)
            & (pcds[:, 2] > z_lower)
            & (pcds[:, 2] < z_upper)
        )

        pcds = pcds[pcd_mask]
        pcd_colors = pcd_colors[pcd_mask]

    # post process 2: downsample
    if downsample:
        pcds, pcd_colors = voxel_downsample_numpy(pcds, downsample_r, pcd_colors)

    # post process 3: return o3d point cloud if out_o3d is True
    if out_o3d:
        aggr_pcd = np2o3d(pcds, pcd_colors)
        return aggr_pcd
    else:
        return pcds, pcd_colors


def aggr_point_cloud_tensor(
    colors: np.ndarray,
    depths: np.ndarray,
    Ks: np.ndarray,
    poses: np.ndarray,
    masks: np.ndarray = None,
    downsample: bool = True,
    downsample_r: float = 0.01,
    boundaries: Optional[Dict] = None,
    color_fmt: ImgEncoding = ImgEncoding.RGB_UINT8,
    depth_fmt: ImgEncoding = ImgEncoding.DEPTH_FLOAT,
    pose_fmt: ExtriConvention = ExtriConvention.WORLD_IN_CAM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate point cloud from multi-view RGBD obs"""
    # colors: [N, H, W, 3] numpy array in uint8
    # depths: [N, H, W] numpy array in meters
    # Ks: [N, 3, 3] numpy array
    # poses: [N, 4, 4] numpy array
    N, H, W, _ = colors.shape
    if color_fmt == ImgEncoding.RGB_UINT8:
        colors = colors / 255.0
    elif color_fmt == ImgEncoding.BGR_UINT8:
        colors = colors[..., ::-1] / 255.0
    if depth_fmt == ImgEncoding.DEPTH_UINT16:
        depths = depths / 1000.0

    # convert to torch tensor
    device = "cuda"
    colors_tensor = torch.from_numpy(colors).to(dtype=torch.float32, device=device)
    depths_tensor = torch.from_numpy(depths).to(dtype=torch.float32, device=device)
    masks_tensor = torch.from_numpy(masks).to(dtype=torch.bool, device=device)
    Ks_tensor = torch.from_numpy(Ks).to(dtype=torch.float32, device=device)
    poses_tensor = torch.from_numpy(poses).to(dtype=torch.float32, device=device)

    # map to camera frame
    pcd_tensor = torch.zeros((N, H, W, 3), dtype=torch.float32, device=device)
    pos_y, pos_x = torch.meshgrid([torch.arange(H), torch.arange(W)])  # (H, W), (H, W)
    pos_y = pos_y.to(device)
    pos_x = pos_x.to(device)
    pos_x = torch.tile(pos_x[None], (N, 1, 1))  # [N, H, W]
    pos_y = torch.tile(pos_y[None], (N, 1, 1))  # [N, H, W]
    pcd_tensor[..., 0] = (
        (pos_x - Ks_tensor[:, 0:1, 2:3]) * depths_tensor / Ks_tensor[:, 0:1, 0:1]
    )
    pcd_tensor[..., 1] = (
        (pos_y - Ks_tensor[:, 1:2, 2:3]) * depths_tensor / Ks_tensor[:, 1:2, 1:2]
    )
    pcd_tensor[..., 2] = depths_tensor

    # map to world frame
    pcd_tensor = pcd_tensor.reshape(N, H * W, 3)
    pcd_tensor = torch.cat(
        [pcd_tensor, torch.ones((N, H * W, 1), dtype=torch.float32, device=device)],
        dim=-1,
    )  # (N, H*W, 4)
    if pose_fmt == ExtriConvention.WORLD_IN_CAM:
        try:
            poses_tensor = torch.linalg.inv(poses_tensor)
        except torch.linalg.LinAlgError:
            poses_tensor = torch.linalg.pinv(poses_tensor)
    pcd_tensor = torch.bmm(poses_tensor, pcd_tensor.permute(0, 2, 1))  # (N, 4, H*W)
    pcd_tensor = pcd_tensor[:, :3].permute(0, 2, 1).reshape(N, H, W, 3)

    # post process 1: mask out
    pcd_tensor = pcd_tensor[masks_tensor]  # [M, 3]
    pcd_colors = colors_tensor[masks_tensor]  # [M, 3]

    # post process 2: remove points outside of boundaries
    if boundaries is not None:
        x_lower = boundaries["x_lower"]
        x_upper = boundaries["x_upper"]
        y_lower = boundaries["y_lower"]
        y_upper = boundaries["y_upper"]
        z_lower = boundaries["z_lower"]
        z_upper = boundaries["z_upper"]

        pcd_mask = (
            (pcd_tensor[:, 0] > x_lower)
            & (pcd_tensor[:, 0] < x_upper)
            & (pcd_tensor[:, 1] > y_lower)
            & (pcd_tensor[:, 1] < y_upper)
            & (pcd_tensor[:, 2] > z_lower)
            & (pcd_tensor[:, 2] < z_upper)
        )

        pcd_tensor = pcd_tensor[pcd_mask]
        pcd_colors = pcd_colors[pcd_mask]

    # post process 3: downsample
    if downsample:
        pcd_tensor, pcd_colors = voxel_downsample_torch(
            pcd_tensor, downsample_r, pcd_colors
        )

    return pcd_tensor.cpu().numpy(), pcd_colors.cpu().numpy()


def voxel_downsample_torch(
    points: torch.Tensor, voxel_size: float, points_color: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Ensure the points tensor is in float format for division
    points = points.float()

    # Compute voxel indices
    voxel_indices = (points / voxel_size).floor()

    # Create a unique index for each voxel using its 3D index
    unique_voxels, indices = torch.unique(voxel_indices, return_inverse=True, dim=0)

    # Initialize voxel_points to store the sum of points in each voxel
    voxel_points = torch.zeros(
        unique_voxels.size(0), points.size(1), dtype=torch.float, device=points.device
    )

    # Sum points in each voxel
    voxel_points.index_add_(0, indices, points)

    # Count the number of points in each voxel
    counts = torch.zeros(
        unique_voxels.size(0), 1, dtype=torch.float, device=points.device
    )
    counts.index_add_(0, indices, torch.ones(points.size(0), 1, device=points.device))

    # Calculate the centroid
    centroids = voxel_points / counts
    if points_color is not None:
        points_color = points_color.float()
        voxel_colors = torch.zeros(
            unique_voxels.size(0),
            points_color.size(1),
            dtype=torch.float,
            device=points.device,
        )
        voxel_colors.index_add_(0, indices, points_color)
        centroids_color = voxel_colors / counts
    else:
        centroids_color = torch.Tensor()

    return centroids, centroids_color


def voxel_downsample_numpy(
    points: np.ndarray, voxel_size: float, points_color: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    # Ensure the points array is in float format for division
    points = points.astype(np.float32)

    # Compute voxel indices
    voxel_indices = np.floor(points / voxel_size)

    # Create a unique index for each voxel using its 3D index
    unique_voxels, indices = np.unique(voxel_indices, return_inverse=True, axis=0)

    # Initialize voxel_points to store the sum of points in each voxel
    voxel_points = np.zeros((unique_voxels.shape[0], points.shape[1]), dtype=np.float32)

    # Sum points in each voxel
    np.add.at(voxel_points, indices, points)

    # Count the number of points in each voxel
    counts = np.zeros((unique_voxels.shape[0], 1), dtype=np.float32)
    np.add.at(counts, indices, 1)

    # Calculate the centroid
    centroids = voxel_points / counts
    if points_color is not None:
        points_color = points_color.astype(np.float32)
        voxel_colors = np.zeros(
            (unique_voxels.shape[0], points_color.shape[1]), dtype=np.float32
        )
        np.add.at(voxel_colors, indices, points_color)
        centroids_color = voxel_colors / counts
    else:
        centroids_color = np.array([])

    return centroids, centroids_color


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


def np2o3d(
    pcd: np.ndarray, color: Union[None, np.ndarray] = None
) -> o3d.geometry.PointCloud:
    # pcd: (n, 3)
    # color: (n, 3)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None:
        assert pcd.shape[0] == color.shape[0]
        assert color.max() <= 1
        assert color.min() >= 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d


class o3dVisualizer:
    """open3d visualizer"""

    def __init__(
        self, view_ctrl_info: Optional[Dict] = None, save_path: Optional[str] = None
    ) -> None:
        """initialize o3d visualizer

        Args:
            view_ctrl_info (dict): view control info containing front, lookat, up, zoom
            save_path (_type_, optional): _description_. Defaults to None.
        """
        self.view_ctrl_info = view_ctrl_info
        self.save_path = save_path
        self.visualizer = o3d.visualization.Visualizer()
        self.vis_dict: Dict[str, o3d.geometry.PointCloud] = {}
        self.mesh_vertices: Dict[str, np.ndarray] = {}
        self.is_first = True
        self.internal_clock = 0
        if save_path is not None:
            os.system(f"mkdir -p {save_path}")

    def start(self) -> None:
        """start the visualizer"""
        self.visualizer.create_window()

    def update_pcd(self, mesh: o3d.geometry.PointCloud, mesh_name: str) -> None:
        """update point cloud"""
        if mesh_name not in self.vis_dict.keys():
            self.vis_dict[mesh_name] = o3d.geometry.PointCloud()
            self.vis_dict[mesh_name].points = mesh.points
            self.vis_dict[mesh_name].colors = mesh.colors
            self.visualizer.add_geometry(self.vis_dict[mesh_name])
        else:
            self.vis_dict[mesh_name].points = mesh.points
            self.vis_dict[mesh_name].colors = mesh.colors

    def add_triangle_mesh(
        self,
        type: str,
        mesh_name: str,
        color: Optional[np.ndarray] = None,
        radius: float = 0.1,
        width: float = 0.1,
        height: float = 0.1,
        depth: float = 0.1,
        size: float = 0.1,
    ) -> None:
        """add triangle mesh to the visualizer"""
        if type == "sphere":
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        elif type == "box":
            mesh = o3d.geometry.TriangleMesh.create_box(
                width=width, height=height, depth=depth
            )
        elif type == "origin":
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        else:
            raise NotImplementedError
        if color is not None:
            mesh.paint_uniform_color(color)
        self.vis_dict[mesh_name] = mesh
        self.mesh_vertices[mesh_name] = np.array(mesh.vertices).copy()
        self.visualizer.add_geometry(self.vis_dict[mesh_name])

    def update_triangle_mesh(self, mesh_name: str, tf: np.ndarray) -> None:
        """update triangle mesh"""
        tf_vertices = self.mesh_vertices[mesh_name] @ tf[:3, :3].T + tf[:3, 3]
        self.vis_dict[mesh_name].vertices = o3d.utility.Vector3dVector(tf_vertices)

    def update_custom_mesh(
        self, mesh: o3d.geometry.TriangleMesh, mesh_name: str
    ) -> None:
        """update custom mesh"""
        if mesh_name not in self.vis_dict.keys():
            self.vis_dict[mesh_name] = copy.deepcopy(mesh)
            self.visualizer.add_geometry(self.vis_dict[mesh_name])
        else:
            self.visualizer.remove_geometry(self.vis_dict[mesh_name], False)
            del self.vis_dict[mesh_name]
            self.vis_dict[mesh_name] = copy.deepcopy(mesh)
            self.visualizer.add_geometry(self.vis_dict[mesh_name])
        self.visualizer.update_geometry(self.vis_dict[mesh_name])

    def render(
        self,
        render_names: Optional[List[str]] = None,
        save_name: Optional[str] = None,
        curr_view_ctrl_info: Optional[Dict] = None,
    ) -> np.ndarray:
        """render the scene"""
        if self.view_ctrl_info is not None and curr_view_ctrl_info is None:
            view_control = self.visualizer.get_view_control()
            view_control.set_front(self.view_ctrl_info["front"])
            view_control.set_lookat(self.view_ctrl_info["lookat"])
            view_control.set_up(self.view_ctrl_info["up"])
            view_control.set_zoom(self.view_ctrl_info["zoom"])
        elif curr_view_ctrl_info is not None:
            view_control = self.visualizer.get_view_control()
            view_control.set_front(curr_view_ctrl_info["front"])
            view_control.set_lookat(curr_view_ctrl_info["lookat"])
            view_control.set_up(curr_view_ctrl_info["up"])
            view_control.set_zoom(curr_view_ctrl_info["zoom"])
        if render_names is None:
            for mesh_name in self.vis_dict.keys():
                self.visualizer.update_geometry(self.vis_dict[mesh_name])
        else:
            for mesh_name in self.vis_dict.keys():
                if mesh_name in render_names:
                    self.visualizer.update_geometry(self.vis_dict[mesh_name])
                else:
                    self.visualizer.remove_geometry(self.vis_dict[mesh_name], False)
        self.visualizer.poll_events()
        self.visualizer.update_renderer()
        # self.visualizer.run()

        img = None
        if self.save_path is not None:
            if save_name is None:
                save_fn = f"{self.save_path}/{self.internal_clock}.png"
            else:
                save_fn = f"{self.save_path}/{save_name}.png"
            self.visualizer.capture_screen_image(save_fn)
            img = cv2.imread(save_fn)
            self.internal_clock += 1

        # add back
        if render_names is not None:
            for mesh_name in self.vis_dict.keys():
                if mesh_name not in render_names:
                    self.visualizer.add_geometry(self.vis_dict[mesh_name])

        return img

    def close(self) -> None:
        """close the visualizer"""
        self.visualizer.destroy_window()


def test_o3d_vis() -> None:
    view_ctrl_info = {
        "front": [0.36137433126422974, 0.5811161319788094, 0.72918628200022917],
        "lookat": [0.45000000000000001, 0.45000000000000001, 0.45000000000000001],
        "up": [-0.17552920841503886, 0.81045157347999874, -0.55888974229000143],
        "zoom": 1.3400000000000005,
    }
    o3d_vis = o3dVisualizer(view_ctrl_info=view_ctrl_info, save_path="tmp")
    o3d_vis.start()
    for i in range(100):
        rand_pcd_np = np.random.rand(100, 3)
        rand_pcd_colors = np.random.rand(100, 3)
        rand_pcd_o3d = np2o3d(rand_pcd_np, rand_pcd_colors)
        o3d_vis.update_pcd(rand_pcd_o3d, "rand_pcd")
        if i == 0:
            o3d_vis.add_triangle_mesh(
                "sphere", "sphere", color=[1.0, 0.0, 0.0], radius=0.1
            )
            o3d_vis.add_triangle_mesh(
                "box", "box", color=[0.0, 1.0, 0.0], width=0.1, height=0.1, depth=0.1
            )
            o3d_vis.add_triangle_mesh("origin", "origin", size=1.0)
        else:
            sphere_tf = np.eye(4)
            sphere_tf[0, 3] = 0.01 * i
            o3d_vis.update_triangle_mesh("sphere", sphere_tf)

            box_tf = np.eye(4)
            box_tf[1, 3] = -0.01 * i
            o3d_vis.update_triangle_mesh("box", box_tf)
        o3d_vis.render(curr_view_ctrl_info=view_ctrl_info)
        time.sleep(0.1)
