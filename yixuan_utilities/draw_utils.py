import copy
import os
import time
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import open3d as o3d


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
