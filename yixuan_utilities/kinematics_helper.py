import copy
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import sapien.core as sapien
import transforms3d
import urdfpy

from yixuan_utilities.draw_utils import np2o3d

logger = logging.getLogger(__name__)


class KinHelper:
    """Helper class for kinematics-related functions"""

    def __init__(self, robot_name: str = "trossen_vx300s_tactile_thin"):
        # load robot
        current_dir = Path(__file__).parent
        package_dir = (current_dir / ".." / "assets").resolve()
        if "trossen" in robot_name:
            trossen_urdf_prefix = "_".join(robot_name.split("_")[1:])
            urdf_path = (
                f"{package_dir}/robot/trossen_description/{trossen_urdf_prefix}.urdf"
            )
            self.eef_name = "vx300s/ee_arm_link"
        elif robot_name == "panda":
            urdf_path = f"{package_dir}/robot/panda/panda.urdf"
            self.eef_name = "panda_hand"
        elif robot_name == "pyrep_panda":
            urdf_path = f"{package_dir}/robot/pyrep_panda/panda.urdf"
            self.eef_name = "Pandatip"
        self.robot_name = robot_name
        self.urdf_robot = urdfpy.URDF.load(urdf_path)

        # load sapien robot
        self.engine = sapien.Engine()
        self.scene = self.engine.create_scene()
        loader = self.scene.create_urdf_loader()
        self.sapien_robot = loader.load(urdf_path)
        self.robot_model = self.sapien_robot.create_pinocchio_model()
        self.link_name_to_idx: dict = {}
        for link_idx, link in enumerate(self.sapien_robot.get_links()):
            self.link_name_to_idx[link.name] = link_idx
        self.sapien_eef_idx = self.link_name_to_idx[self.eef_name]

        # load meshes and offsets from urdf_robot
        self.meshes = {}
        self.scales = {}
        self.offsets = {}
        for link in self.urdf_robot.links:
            if len(link.collisions) > 0:
                collision = link.collisions[0]
                if (
                    collision.geometry.mesh is not None
                    and len(collision.geometry.mesh.meshes) > 0
                ):
                    mesh = collision.geometry.mesh.meshes[0]
                    self.meshes[link.name] = mesh.as_open3d
                    self.meshes[link.name].compute_vertex_normals()
                    self.meshes[link.name].paint_uniform_color([0.2, 0.2, 0.2])
                    self.scales[link.name] = (
                        collision.geometry.mesh.scale[0]
                        if collision.geometry.mesh.scale is not None
                        else 1.0
                    )
                    self.offsets[link.name] = collision.origin
        self.pcd_dict: dict = {}
        self.tool_meshes: dict = {}

    def _mesh_poses_to_pc(
        self,
        poses: np.ndarray,
        meshes: list[o3d.geometry.TriangleMesh],
        offsets: list[np.ndarray],
        num_pts: list[int],
        scales: list[int],
        pcd_name: Optional[str] = None,
    ) -> np.ndarray:
        # poses: (N, 4, 4) numpy array
        # offsets: (N, ) list of offsets
        # meshes: (N, ) list of meshes
        # num_pts: (N, ) list of int
        # scales: (N, ) list of float
        try:
            assert poses.shape[0] == len(meshes)
            assert poses.shape[0] == len(offsets)
            assert poses.shape[0] == len(num_pts)
            assert poses.shape[0] == len(scales)
        except AssertionError:
            logger.critical("Input shapes do not match")
            exit(1)

        N = poses.shape[0]
        all_pc = []
        for index in range(N):
            mat = poses[index]
            if (
                pcd_name is None
                or pcd_name not in self.pcd_dict
                or len(self.pcd_dict[pcd_name]) <= index
            ):
                mesh = copy.deepcopy(meshes[index])  # .copy()
                mesh.scale(scales[index], center=np.array([0, 0, 0]))
                sampled_cloud = mesh.sample_points_poisson_disk(
                    number_of_points=num_pts[index]
                )
                cloud_points = np.asarray(sampled_cloud.points)
                if pcd_name not in self.pcd_dict:
                    self.pcd_dict[pcd_name] = []
                self.pcd_dict[pcd_name].append(cloud_points)
            else:
                cloud_points = self.pcd_dict[pcd_name][index]

            tf_obj_to_link = offsets[index]

            mat = mat @ tf_obj_to_link
            transformed_points = cloud_points @ mat[:3, :3].T + mat[:3, 3]
            all_pc.append(transformed_points)
        all_pc = np.concatenate(all_pc, axis=0)
        return all_pc

    def compute_robot_pcd(
        self,
        qpos: np.ndarray,
        link_names: Optional[list[str]] = None,
        num_pts: Optional[list[int]] = None,
        pcd_name: Optional[str] = None,
    ) -> np.ndarray:
        """Compute point cloud of robot links given joint positions"""
        self.robot_model.compute_forward_kinematics(qpos)
        if link_names is None:
            link_names = list(self.meshes.keys())
        if num_pts is None:
            num_pts = [500] * len(link_names)
        link_idx_ls = []
        for link_name in link_names:
            for link_idx, link in enumerate(self.sapien_robot.get_links()):
                if link.name == link_name:
                    link_idx_ls.append(link_idx)
                    break
        link_pose_ls = np.stack(
            [
                self.robot_model.get_link_pose(link_idx).to_transformation_matrix()
                for link_idx in link_idx_ls
            ]
        )
        meshes_ls = [self.meshes[link_name] for link_name in link_names]
        offsets_ls = [self.offsets[link_name] for link_name in link_names]
        scales_ls = [self.scales[link_name] for link_name in link_names]
        pcd = self._mesh_poses_to_pc(
            poses=link_pose_ls,
            meshes=meshes_ls,
            offsets=offsets_ls,
            num_pts=num_pts,
            scales=scales_ls,
            pcd_name=pcd_name,
        )
        return pcd

    def compute_robot_meshes(
        self,
        qpos: np.ndarray,
        link_names: Optional[list[str]] = None,
    ) -> list[o3d.geometry.TriangleMesh]:
        """Compute meshes of robot links given joint positions"""
        self.robot_model.compute_forward_kinematics(qpos)
        if link_names is None:
            link_names = list(self.meshes.keys())
        link_idx_ls = []
        for link_name in link_names:
            for link_idx, link in enumerate(self.sapien_robot.get_links()):
                if link.name == link_name:
                    link_idx_ls.append(link_idx)
                    break
        link_pose_ls = np.stack(
            [
                self.robot_model.get_link_pose(link_idx).to_transformation_matrix()
                for link_idx in link_idx_ls
            ]
        )
        offsets_ls = [self.offsets[link_name] for link_name in link_names]
        meshes_ls = []
        for link_idx, link_name in enumerate(link_names):
            import copy

            mesh = copy.deepcopy(self.meshes[link_name])
            mesh.scale(0.001, center=np.array([0, 0, 0]))
            tf = link_pose_ls[link_idx] @ offsets_ls[link_idx]
            mesh.transform(tf)
            meshes_ls.append(mesh)
        return meshes_ls

    def compute_fk_from_link_idx(
        self,
        qpos: np.ndarray,
        link_idx: list[int],
    ) -> list[np.ndarray]:
        """Compute forward kinematics of robot links given joint positions"""
        self.robot_model.compute_forward_kinematics(qpos)
        link_pose_ls = []
        for i in link_idx:
            pose = self.robot_model.get_link_pose(i)
            link_pose_ls.append(pose.to_transformation_matrix())
        return link_pose_ls

    def compute_fk_from_link_names(
        self,
        qpos: np.ndarray,
        link_names: list[str],
        in_obj_frame: bool = False,
    ) -> dict[str, np.ndarray]:
        """Compute forward kinematics of robot links given joint positions"""
        self.robot_model.compute_forward_kinematics(qpos)
        link_idx_ls = [self.link_name_to_idx[link_name] for link_name in link_names]
        poses_ls = self.compute_fk_from_link_idx(qpos, link_idx_ls)
        if in_obj_frame:
            for i in range(len(link_names)):
                if link_names[i] in self.offsets:
                    poses_ls[i] = poses_ls[i] @ self.offsets[link_names[i]]
        return {link_name: pose for link_name, pose in zip(link_names, poses_ls)}

    def compute_all_fk(
        self, qpos: np.ndarray, in_obj_frame: bool = False
    ) -> dict[str, np.ndarray]:
        """Compute forward kinematics of all robot links given joint positions"""
        all_link_names = [link.name for link in self.sapien_robot.get_links()]
        return self.compute_fk_from_link_names(qpos, all_link_names, in_obj_frame)

    def compute_ik(
        self,
        initial_qpos: np.ndarray,
        cartesian: np.ndarray,
    ) -> np.ndarray:
        """Compute inverse kinematics given initial joint pos and target pose"""
        tf_mat = np.eye(4)
        tf_mat[:3, :3] = transforms3d.euler.euler2mat(
            ai=cartesian[3], aj=cartesian[4], ak=cartesian[5], axes="sxyz"
        )
        tf_mat[:3, 3] = cartesian[0:3]
        pose = sapien.Pose.from_transformation_matrix(tf_mat)
        if "trossen" in self.robot_name:
            active_qmask = np.array([True, True, True, True, True, True, False, False])
        elif "panda" in self.robot_name:
            active_qmask = np.array(
                [True, True, True, True, True, True, True, True, True]
            )
        qpos = self.robot_model.compute_inverse_kinematics(
            link_index=self.sapien_eef_idx,
            pose=pose,
            initial_qpos=initial_qpos,
            active_qmask=active_qmask,
            eps=1e-3,
            damp=1e-1,
        )
        # verify ik
        fk_pose = self.compute_fk_from_link_idx(qpos[0], [self.sapien_eef_idx])[0]
        # print('target pose for IK:', tf_mat)
        # print('fk pose for IK:', fk_pose)
        pose_diff = np.linalg.norm(fk_pose[:3, 3] - tf_mat[:3, 3])
        rot_diff = np.linalg.norm(fk_pose[:3, :3] - tf_mat[:3, :3])
        if pose_diff > 0.01 or rot_diff > 0.01:
            print("ik pose diff:", pose_diff)
            print("ik rot diff:", rot_diff)
            logger.warning("ik pose diff or rot diff too large. Return initial qpos.")
            return initial_qpos
        return qpos[0]


def test_kin_helper_trossen() -> None:
    robot_name = "trossen_vx300s_v3"
    finger_names = None
    num_pts = None
    init_qpos = np.array(
        [
            0.851939865243963,
            -0.229601035617388,
            0.563932102437065,
            -0.098902024821519,
            1.148033168114365,
            1.016116677288259,
            0.0,
            -0.0,
        ]
    )
    end_qpos = np.array(
        [
            0.788165775078753,
            -0.243655597686374,
            0.573832680057706,
            -0.075632950397682,
            1.260574309772582,
            2.000622093036658,
            0.0,
            -0.0,
        ]
    )

    kin_helper = KinHelper(robot_name=robot_name)
    for i in range(100):
        curr_qpos = init_qpos + (end_qpos - init_qpos) * i / 100
        fk_pose = kin_helper.compute_fk_from_link_idx(
            curr_qpos, [kin_helper.sapien_eef_idx]
        )[0]
        print("fk pose:", fk_pose)
        start_time = time.time()
        pcd = kin_helper.compute_robot_pcd(
            curr_qpos, link_names=finger_names, num_pts=num_pts, pcd_name="finger"
        )
        print("compute_robot_pcd time:", time.time() - start_time)
        pcd_o3d = np2o3d(pcd)
        if i == 0:
            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window()
            curr_pcd = copy.deepcopy(pcd_o3d)
            visualizer.add_geometry(curr_pcd)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            visualizer.add_geometry(origin)
        curr_pcd.points = pcd_o3d.points
        curr_pcd.colors = pcd_o3d.colors
        visualizer.update_geometry(curr_pcd)
        visualizer.update_geometry(origin)
        visualizer.poll_events()
        visualizer.update_renderer()
        if i == 0:
            visualizer.run()


def test_kin_helper_panda() -> None:
    robot_name = "panda"
    total_steps = 100
    finger_names = None
    num_pts = None
    init_qpos = np.array(
        [
            -2.21402311,
            0.17274992,
            2.23800898,
            -2.27481246,
            -0.16332519,
            2.16096449,
            0.90828639,
            0.09,
            0.09,
        ]
    )
    end_qpos = np.array(
        [
            -2.18224038,
            0.26588862,
            2.40268749,
            -2.54840559,
            -0.2473307,
            2.33424677,
            1.19656971,
            0,
            0,
        ]
    )

    kin_helper = KinHelper(robot_name=robot_name)
    for i in range(total_steps):
        curr_qpos = init_qpos + (end_qpos - init_qpos) * i / total_steps
        fk_pose = kin_helper.compute_fk_from_link_idx(
            curr_qpos, [kin_helper.sapien_eef_idx]
        )[0]
        print("fk pose:", fk_pose)
        start_time = time.time()
        pcd = kin_helper.compute_robot_pcd(
            curr_qpos, link_names=finger_names, num_pts=num_pts, pcd_name="finger"
        )
        print("compute_robot_pcd time:", time.time() - start_time)
        pcd_o3d = np2o3d(pcd)
        if i == 0:
            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window()
            curr_pcd = copy.deepcopy(pcd_o3d)
            visualizer.add_geometry(curr_pcd)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            visualizer.add_geometry(origin)
        curr_pcd.points = pcd_o3d.points
        curr_pcd.colors = pcd_o3d.colors
        visualizer.update_geometry(curr_pcd)
        visualizer.update_geometry(origin)
        visualizer.poll_events()
        visualizer.update_renderer()
        if i == 0:
            visualizer.run()


def test_fk() -> None:
    robot_name = "trossen_vx300s_v3"
    init_qpos = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    end_qpos = np.array([0, -0.96, 1.16, 0, -0.3, 0, 0.09, -0.09])

    kin_helper = KinHelper(robot_name=robot_name)
    START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
    for i in range(100):
        curr_qpos = init_qpos + (end_qpos - init_qpos) * i / 100
        fk = kin_helper.compute_fk_from_link_idx(
            curr_qpos, [kin_helper.sapien_eef_idx]
        )[0]
        fk_euler = transforms3d.euler.mat2euler(fk[:3, :3], axes="sxyz")

        if i == 0:
            init_ik_qpos = np.array(START_ARM_POSE)
        ik_qpos = kin_helper.compute_ik(
            init_ik_qpos, np.array(list(fk[:3, 3]) + list(fk_euler)).astype(np.float32)
        )
        re_fk_pos_mat = kin_helper.compute_fk_from_link_idx(
            ik_qpos, [kin_helper.sapien_eef_idx]
        )[0]
        re_fk_euler = transforms3d.euler.mat2euler(re_fk_pos_mat[:3, :3], axes="sxyz")
        re_fk_pos = re_fk_pos_mat[:3, 3]
        print("re_fk_pos diff:", np.linalg.norm(re_fk_pos - fk[:3, 3]))
        print(
            "re_fk_euler diff:",
            np.linalg.norm(np.array(re_fk_euler) - np.array(fk_euler)),
        )

        init_ik_qpos = ik_qpos.copy()
        print("fk_euler:", fk_euler)
        print("gt qpos:", curr_qpos)
        print("ik qpos:", ik_qpos)
        print("qpos diff:", np.linalg.norm(ik_qpos[:6] - curr_qpos[:6]))
        qpos_diff = np.linalg.norm(ik_qpos[:6] - curr_qpos[:6])
        if qpos_diff > 0.01:
            logger.warning(
                "qpos diff too large",
            )

        print()

        time.sleep(0.1)


if __name__ == "__main__":
    # test_kin_helper()
    test_kin_helper_panda()
    # test_fk()
