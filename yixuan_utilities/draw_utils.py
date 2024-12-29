from typing import Union

import numpy as np
import open3d as o3d


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
