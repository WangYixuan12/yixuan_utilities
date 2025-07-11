import copy
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import pinocchio
import sapien
import transforms3d
import urdfpy

from yixuan_utilities.draw_utils import np2o3d

logger = logging.getLogger(__name__)


class KinHelper:
    """Helper class for kinematics-related functions"""

    def __init__(self, robot_name: str = "trossen_vx300s"):
        # load robot
        current_dir = Path(__file__).parent
        package_dir = (current_dir / "assets").resolve()
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
        elif robot_name == "vega":
            urdf_path = f"{package_dir}/robot/vega-urdf/vega_no_effector.urdf"
            self.eef_name = "none"
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

        # # load pinocchio model
        # self.pinocchio_model = PinocchioModel.createPinocchioModel(urdf_path)

        # find end effector frame id
        if self.eef_name != "none":
            self.sapien_eef_idx = self.link_name_to_idx[self.eef_name]
            # self.sapien_eef_idx = self.pinocchio_model.model.getFrameId(self.eef_name)
        else:
            self.sapien_eef_idx = None
            # self.sapien_eef_idx = None

        # # create link name to frame id mapping
        # self.link_name_to_frame_id: dict = {}
        # for frame_id in range(self.pinocchio_model.model.nframes):
        #     frame_name = self.pinocchio_model.model.frames[frame_id].name
        #     self.link_name_to_frame_id[frame_name] = frame_id

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

    def convert_from_sapien_joint_order(
        self, arr: np.ndarray, joint_names: list[str]
    ) -> np.ndarray:
        """Convert array in sapien joint order to xml joint order"""
        original2sapien_joint = np.zeros(len(joint_names))
        for sapien_joint_idx, sapien_joint in enumerate(
            self.sapien_robot.get_active_joints()
        ):
            for original_joint_idx, original_joint in enumerate(joint_names):
                if sapien_joint.name == original_joint:
                    original2sapien_joint[original_joint_idx] = sapien_joint_idx
                    break
        original2sapien_joint = original2sapien_joint.astype(int)
        return arr[original2sapien_joint]

    def convert_to_sapien_joint_order(
        self, arr: np.ndarray, joint_names: list[str]
    ) -> np.ndarray:
        """Convert array in xml joint order to sapien joint order"""
        sapien2original_joint = np.zeros(len(joint_names))
        for sapien_joint_idx, sapien_joint in enumerate(
            self.sapien_robot.get_active_joints()
        ):
            for original_joint_idx, original_joint in enumerate(joint_names):
                if sapien_joint.name == original_joint:
                    sapien2original_joint[sapien_joint_idx] = original_joint_idx
                    break
        sapien2original_joint = sapien2original_joint.astype(int)
        return arr[sapien2original_joint]

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
        # self.pinocchio_model.computeForwardKinematics(qpos)
        if link_names is None:
            link_names = list(self.meshes.keys())
        if num_pts is None:
            num_pts = [500] * len(link_names)
        link_idx_ls = []
        # link_pose_ls = []
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
        # for link_idx, link in enumerate(self.sapien_robot.get_links()):
        #     if link_name in self.link_name_to_frame_id:
        #         frame_id = self.link_name_to_frame_id[link_name]
        #         pose = pinocchio.updateFramePlacements(self.pinocchio_model.model,
        # self.pinocchio_model.data)[frame_id]
        #         link_pose_ls.append(pose)
        #     else:
        #         # fallback to neutral pose if frame not found
        #         link_pose_ls.append(np.eye(4))
        # link_pose_ls = np.stack(link_pose_ls)
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
        # self.pinocchio_model.computeForwardKinematics(qpos)
        if link_names is None:
            link_names = list(self.meshes.keys())
        link_idx_ls = []
        # link_pose_ls = []
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
        # for link_name in link_names:
        #     if link_name in self.link_name_to_frame_id:
        #         frame_id = self.link_name_to_frame_id[link_name]
        #         pose = pinocchio.updateFramePlacements(self.pinocchio_model.model,
        # self.pinocchio_model.data)[frame_id]
        #         link_pose_ls.append(pose)
        #     else:
        #         link_pose_ls.append(np.eye(4))
        # link_pose_ls = np.stack(link_pose_ls)
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
        # self.pinocchio_model.computeForwardKinematics(qpos)
        link_pose_ls = []
        for i in link_idx:
            pose = self.robot_model.get_link_pose(i)
            link_pose_ls.append(pose.to_transformation_matrix())
            # if i < len(self.pinocchio_model.model.frames):
            #     pose = pinocchio.updateFramePlacements(self.pinocchio_model.model,
            # self.pinocchio_model.data)[i]
            #     link_pose_ls.append(pose)
            # else:
            #     link_pose_ls.append(np.eye(4))
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
        # self.pinocchio_model.computeForwardKinematics(qpos)
        # poses_ls = []
        # for link_name in link_names:
        #     if link_name in self.link_name_to_frame_id:
        #         frame_id = self.link_name_to_frame_id[link_name]
        #         pose = pinocchio.updateFramePlacements(self.pinocchio_model.model,
        # self.pinocchio_model.data)[frame_id]
        #         poses_ls.append(pose)
        #     else:
        #         poses_ls.append(np.eye(4))
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
        # all_link_names = [frame.name for frame in self.pinocchio_model.model.frames]
        return self.compute_fk_from_link_names(qpos, all_link_names, in_obj_frame)

    def compute_ik(
        self,
        initial_qpos: np.ndarray,
        cartesian: np.ndarray,
        damp: float = 1e-1,
        eef_idx: Optional[int] = None,
        active_qmask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute inverse kinematics given initial joint pos and target pose"""
        tf_mat = np.eye(4)
        tf_mat[:3, :3] = transforms3d.euler.euler2mat(
            ai=cartesian[3], aj=cartesian[4], ak=cartesian[5], axes="sxyz"
        )
        tf_mat[:3, 3] = cartesian[0:3]
        return self.compute_ik_from_mat(
            initial_qpos=initial_qpos,
            tf_mat=tf_mat,
            damp=damp,
            eef_idx=eef_idx,
            active_qmask=active_qmask,
        )

    def compute_ik_from_mat(
        self,
        initial_qpos: np.ndarray,
        tf_mat: np.ndarray,
        damp: float = 1e-1,
        eef_idx: Optional[int] = None,
        active_qmask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute IK given initial joint pos and target pose in matrix form"""
        pose = sapien.Pose(tf_mat)
        if "trossen" in self.robot_name:
            active_qmask = np.array([True, True, True, True, True, True, False, False])
        elif "panda" in self.robot_name:
            active_qmask = np.array(
                [True, True, True, True, True, True, True, True, True]
            )
        assert active_qmask is not None
        qpos = self.robot_model.compute_inverse_kinematics(
            link_index=eef_idx if eef_idx is not None else self.sapien_eef_idx,
            pose=pose,
            initial_qpos=initial_qpos,
            active_qmask=active_qmask,
            eps=1e-3,
            damp=damp,
        )
        return qpos[0]

        # """Compute IK given initial joint pos and target pose in matrix form"""
        # if eef_idx is None:
        #     eef_idx = self.sapien_eef_idx
        # if eef_idx is None:
        #     raise ValueError("No end effector frame specified")

        # # Convert target pose to position and quaternion
        # position = tf_mat[:3, 3]
        # rotation_matrix = tf_mat[:3, :3]
        # quaternion = transforms3d.quaternions.mat2quat(rotation_matrix)
        # quaternion = np.array([quaternion[1],
        # quaternion[2], quaternion[3], quaternion[0]])  # xyzw to wxyz

        # # Use the advanced IK solver
        # qpos, success, error = self.pinocchio_model.computeInverseKinematics(
        #     link_idx=eef_idx,
        #     pose=(position, quaternion),
        #     initial_qpos=initial_qpos,
        #     active_qmask=active_qmask,
        #     eps=1e-3,
        #     damp=damp,
        # )
        # if not success:
        #     logger.warning("IK did not converge to desired precision")

        # return qpos


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


class PinocchioModel:
    """Pinocchio model wrapper that imitates the C++ implementation"""

    def __init__(self, urdf_path: str, gravity: Optional[np.ndarray] = None):
        # Load model from URDF
        self.model = pinocchio.buildModelFromUrdf(urdf_path)
        if gravity is None:
            gravity = np.array([0, 0, -9.81])
        self.model.gravity = pinocchio.Motion(gravity, np.zeros(3))
        self.data = self.model.createData()

        # Initialize joint ordering
        self.indexS2P = np.eye(self.model.nv)
        self.QIDX = np.array([])
        self.NQ = np.array([])
        self.NV = np.array([])

        # Initialize link ordering
        self.linkIdx2FrameIdx: list[int] = []

        # Set default joint order (all joints in model order)
        default_joint_names = []
        for joint_id in range(1, len(self.model.joints)):  # Skip universe joint
            joint_name = self.model.names[joint_id]
            if joint_name != "universe":
                default_joint_names.append(joint_name)
        self.setJointOrder(default_joint_names)

    @classmethod
    def fromURDF(
        cls, urdf_path: str, gravity: Optional[np.ndarray] = None
    ) -> "PinocchioModel":
        """Create PinocchioModel from URDF file"""
        # Build model from URDF file
        model = pinocchio.buildModelFromUrdf(urdf_path)
        if gravity is None:
            gravity = np.array([0, 0, -9.81])
        model.gravity = pinocchio.Motion(gravity, np.zeros(3))

        # Create instance
        instance = cls.__new__(cls)
        instance.model = model
        instance.data = model.createData()

        # Initialize joint ordering
        instance.indexS2P = np.eye(instance.model.nv)
        instance.QIDX = np.array([])
        instance.NQ = np.array([])
        instance.NV = np.array([])

        # Initialize link ordering
        instance.linkIdx2FrameIdx = []

        # Set default joint order (all joints in model order)
        default_joint_names = []
        for joint_id in range(1, len(instance.model.joints)):  # Skip universe joint
            joint_name = instance.model.names[joint_id]
            if joint_name != "universe":
                default_joint_names.append(joint_name)
        instance.setJointOrder(default_joint_names)

        return instance

    def posS2P(self, qext: np.ndarray) -> np.ndarray:
        """Convert standard joint values to Pinocchio internal configuration format"""
        qint = np.zeros(self.model.nq)
        count = 0

        for N in range(len(self.QIDX)):
            start_idx = self.QIDX[N]
            nq = self.NQ[N]
            # nv = self.NV[N]

            if nq == 0:  # Fixed joint
                continue
            elif nq == 1:  # Revolute or prismatic joint
                qint[start_idx] = qext[count]
                count += 1
            elif nq == 2:  # Planar joint (cos, sin representation)
                qint[start_idx] = np.cos(qext[count])
                qint[start_idx + 1] = np.sin(qext[count])
                count += 1
            else:
                raise RuntimeError("Unsupported joint in computation.")

        assert count == qext.size, "posS2P failed"
        return qint

    def posP2S(self, qint: np.ndarray) -> np.ndarray:
        """Convert Pinocchio internal configuration format to standard joint values"""
        qext = np.zeros(self.model.nv)
        count = 0

        for N in range(len(self.QIDX)):
            start_idx = self.QIDX[N]
            nq = self.NQ[N]
            # nv = self.NV[N]

            if nq == 0:  # Fixed joint
                continue
            elif nq == 1:  # Revolute or prismatic joint
                qext[count] = qint[start_idx]
                count += 1
            elif nq == 2:  # Planar joint (cos, sin representation)
                qext[count] = np.arctan2(qint[start_idx + 1], qint[start_idx])
                count += 1
            else:
                raise RuntimeError("Unsupported joint in computation.")

        assert count == self.model.nv, "posP2S failed"
        return qext

    def setJointOrder(self, names: list[str]) -> None:
        """Set the joint order for conversion between standard and Pinocchio formats"""
        v = np.zeros(self.model.nv, dtype=int)
        count = 0

        for name in names:
            joint_id = self.model.getJointId(name)
            if joint_id == self.model.njoints:
                raise ValueError(f"Invalid joint name: {name}")

            size = self.model.nvs[joint_id]
            qi = self.model.idx_vs[joint_id]
            for s in range(size):
                v[count] = qi + s
                count += 1

        assert count == self.model.nv, "setJointOrder failed"
        self.indexS2P = np.eye(self.model.nv)[v]

        self.QIDX = np.zeros(len(names), dtype=int)
        self.NQ = np.zeros(len(names), dtype=int)
        self.NV = np.zeros(len(names), dtype=int)

        for N, name in enumerate(names):
            joint_id = self.model.getJointId(name)
            if joint_id == self.model.njoints:
                raise ValueError(f"Invalid joint name: {name}")

            self.NQ[N] = self.model.nqs[joint_id]
            self.NV[N] = self.model.nvs[joint_id]
            self.QIDX[N] = self.model.idx_qs[joint_id]

    def setLinkOrder(self, names: list[str]) -> None:
        """Set the link order for frame indexing"""
        self.linkIdx2FrameIdx = []

        for name in names:
            frame_id = self.model.getFrameId(name, pinocchio.BODY)
            if frame_id == self.model.nframes:
                raise ValueError(f"Invalid link name: {name}")
            self.linkIdx2FrameIdx.append(frame_id)

    def getRandomConfiguration(self) -> np.ndarray:
        """Get a random configuration in standard format"""
        return self.posP2S(pinocchio.randomConfiguration(self.model))

    def computeForwardKinematics(self, qpos: np.ndarray) -> None:
        """Compute forward kinematics"""
        pinocchio.forwardKinematics(self.model, self.data, self.posS2P(qpos))

    def getLinkPose(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get link pose (position and quaternion)"""
        if index >= len(self.linkIdx2FrameIdx):
            raise ValueError("Link index out of bounds")

        frame = self.linkIdx2FrameIdx[index]
        parent_joint = self.model.frames[frame].parent
        link2joint = self.model.frames[frame].placement
        joint2world = self.data.oMi[parent_joint]

        link2world = joint2world * link2joint
        position = link2world.translation
        quaternion = pinocchio.Quaternion(link2world.rotation)

        return position, np.array(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        )

    def computeFullJacobian(self, qpos: np.ndarray) -> None:
        """Compute full Jacobian matrices"""
        pinocchio.computeJointJacobians(self.model, self.data, self.posS2P(qpos))

    def getLinkJacobian(self, index: int, local: bool = False) -> np.ndarray:
        """Get link Jacobian matrix"""
        if index >= len(self.linkIdx2FrameIdx):
            raise ValueError("Link index out of bounds")

        frame_idx = self.linkIdx2FrameIdx[index]
        joint_idx = self.model.frames[frame_idx].parent

        link2joint = self.model.frames[frame_idx].placement
        joint2world = self.data.oMi[joint_idx]
        link2world = joint2world * link2joint

        J = pinocchio.getJointJacobian(
            self.model, self.data, joint_idx, pinocchio.ReferenceFrame.WORLD
        )

        if local:
            J = link2world.toActionMatrixInverse() @ J

        # Permute Jacobian to standard format
        return J @ self.indexS2P

    def computeSingleLinkLocalJacobian(
        self, qpos: np.ndarray, index: int
    ) -> np.ndarray:
        """Compute single link local Jacobian"""
        if index >= len(self.linkIdx2FrameIdx):
            raise ValueError("Link index out of bounds")

        frame_idx = self.linkIdx2FrameIdx[index]
        joint_idx = self.model.frames[frame_idx].parent
        link2joint = self.model.frames[frame_idx].placement

        J = pinocchio.computeJointJacobian(
            self.model, self.data, self.posS2P(qpos), joint_idx
        )

        return link2joint.toActionMatrixInverse() @ J @ self.indexS2P

    def computeGeneralizedMassMatrix(self, qpos: np.ndarray) -> np.ndarray:
        """Compute generalized mass matrix"""
        pinocchio.crba(self.model, self.data, self.posS2P(qpos))
        # Make symmetric
        M = self.data.M.copy()
        M = (M + M.T) / 2
        return self.indexS2P.T @ M @ self.indexS2P

    def computeCoriolisMatrix(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """Compute Coriolis matrix"""
        return (
            self.indexS2P.T
            @ pinocchio.computeCoriolisMatrix(
                self.model, self.data, self.posS2P(qpos), self.indexS2P @ qvel
            )
            @ self.indexS2P
        )

    def computeInverseDynamics(
        self, qpos: np.ndarray, qvel: np.ndarray, qacc: np.ndarray
    ) -> np.ndarray:
        """Compute inverse dynamics (RNEA)"""
        return self.indexS2P.T @ pinocchio.rnea(
            self.model,
            self.data,
            self.posS2P(qpos),
            self.indexS2P @ qvel,
            self.indexS2P @ qacc,
        )

    def computeForwardDynamics(
        self, qpos: np.ndarray, qvel: np.ndarray, qf: np.ndarray
    ) -> np.ndarray:
        """Compute forward dynamics (ABA)"""
        return self.indexS2P.T @ pinocchio.aba(
            self.model,
            self.data,
            self.posS2P(qpos),
            self.indexS2P @ qvel,
            self.indexS2P @ qf,
        )

    def computeInverseKinematics(
        self,
        link_idx: int,
        pose: Tuple[np.ndarray, np.ndarray],  # (position, quaternion)
        initial_qpos: Optional[np.ndarray] = None,
        active_qmask: Optional[np.ndarray] = None,
        eps: float = 1e-4,
        max_iter: int = 1000,
        dt: float = 1e-1,
        damp: float = 1e-6,
    ) -> Tuple[np.ndarray, bool, np.ndarray]:
        """Compute inverse kinematics with advanced error handling"""
        if link_idx >= len(self.linkIdx2FrameIdx):
            raise ValueError("Link index out of bounds")

        # Initialize configuration
        if initial_qpos is None or len(initial_qpos) == 0:
            q = pinocchio.neutral(self.model)
        else:
            q = self.posS2P(initial_qpos)

        # Set up active joint mask
        if active_qmask is not None and len(active_qmask) > 0:
            mask = self.indexS2P @ active_qmask.astype(float)
        else:
            mask = np.ones(self.model.nv)

        # Set up target pose
        frame_idx = self.linkIdx2FrameIdx[link_idx]
        joint_idx = self.model.frames[frame_idx].parent
        # link2joint = self.model.frames[frame_idx].placement

        position, quaternion = pose
        l2w = pinocchio.SE3(
            pinocchio.Quaternion(
                quaternion[3], quaternion[0], quaternion[1], quaternion[2]
            ).toRotationMatrix(),
            position,
        )
        # oMdes = l2w * link2joint.inverse()
        oMdes = l2w

        # IK iteration
        success = False
        min_error = 1e10
        best_q = q.copy()
        best_err = np.zeros(6)

        for _ in range(max_iter):
            pinocchio.forwardKinematics(self.model, self.data, q)
            # Compute error in joint frame
            iMd = self.data.oMi[joint_idx].actInv(oMdes)
            err = pinocchio.log6(iMd).vector  # in joint frame

            err_norm = np.linalg.norm(err)
            if err_norm < min_error:
                min_error = err_norm
                best_q = q.copy()
                best_err = err.copy()

            if err_norm < eps:
                success = True
                break

            J = pinocchio.computeJointJacobian(self.model, self.data, q, joint_idx)
            J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)

            # Apply active joint mask if provided
            if active_qmask is not None and len(active_qmask) > 0:
                mask = self.indexS2P @ active_qmask.astype(float)
                J = J @ np.diag(mask)

            # Damped pseudo-inverse
            JJt = J @ J.T
            JJt += damp * np.eye(6)
            v = -J.T @ np.linalg.solve(JJt, err)

            # Integrate configuration
            q = pinocchio.integrate(self.model, q, v * dt)

        return self.posP2S(best_q), success, best_err

    @classmethod
    def createPinocchioModel(
        cls, urdf_path: str, gravity: Optional[np.ndarray] = None
    ) -> "PinocchioModel":
        """Create PinocchioModel from URDF file"""
        # Create model from URDF file
        if gravity is None:
            gravity = np.array([0, 0, -9.81])
        pm = cls.fromURDF(urdf_path, gravity)

        # Extract joint and link names from URDF file
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # Get joint names (only joints with DOF > 0)
        joint_names = []
        for joint in root.findall(".//joint"):
            joint_type = joint.get("type")
            if joint_type in ["revolute", "prismatic", "continuous"]:
                joint_name = joint.get("name")
                if joint_name:
                    joint_names.append(joint_name)

        # Get link names
        link_names = []
        for link in root.findall(".//link"):
            link_name = link.get("name")
            if link_name:
                link_names.append(link_name)

        # Set joint and link order
        if joint_names:
            pm.setJointOrder(joint_names)
        if link_names:
            pm.setLinkOrder(link_names)

        return pm


if __name__ == "__main__":
    # test_kin_helper()
    test_kin_helper_panda()
    # test_fk()
