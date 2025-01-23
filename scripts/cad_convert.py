import os
from pathlib import Path

import trimesh
import urchin

current_dir = Path(__file__).parent
package_dir = (current_dir / ".." / "assets").resolve()
urdf_path = package_dir / "robot" / "panda" / "panda.urdf"
urdf_robot = urchin.URDF.load(f"{urdf_path}")

# load meshes and offsets from urdf_robot
meshes = {}
trimeshes = {}
scales = {}
offsets = {}
for link in urdf_robot.links:
    if len(link.collisions) > 0:
        visual = link.visuals[0]
        if len(visual.geometry.mesh.meshes) > 0:
            mesh = trimesh.util.concatenate(visual.geometry.mesh.meshes)
            os.system(
                f"mkdir -p {urdf_path}/../franka_description/meshes/visual/{link.name}"
            )
            link_path = (
                urdf_path
                / ".."
                / "franka_description"
                / "meshes"
                / "visual"
                / link.name
            ).resolve()
            link_path.mkdir(parents=True, exist_ok=True)
            mesh.export(f"{link_path}/{link.name}.obj")
            trimeshes[link.name] = mesh
            meshes[link.name] = mesh.as_open3d
            meshes[link.name].compute_vertex_normals()
            meshes[link.name].paint_uniform_color([0.2, 0.2, 0.2])
            scales[link.name] = (
                visual.geometry.mesh.scale[0]
                if visual.geometry.mesh.scale is not None
                else 1.0
            )
            offsets[link.name] = visual.origin
