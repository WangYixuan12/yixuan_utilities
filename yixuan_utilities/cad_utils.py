import trimesh


def glb2obj(glb_path: str, obj_path: str) -> None:
    mesh = trimesh.load(glb_path)
    all_meshes = [m for m in mesh.geometry.values()]
    mesh_combined = trimesh.util.concatenate(all_meshes)
    mesh_combined.export(obj_path)


if __name__ == "__main__":
    for i in range(2, 6):
        glb2obj(
            f"/home/yixuan/diffusion_final/diffusion_policy_code/sapien_env/sapien_env/assets/yx/battery_{i}/battery_{i}.glb",
            f"/home/yixuan/diffusion_final/diffusion_policy_code/sapien_env/sapien_env/assets/yx/battery_{i}/battery_{i}.obj",
        )
