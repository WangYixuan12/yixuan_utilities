from enum import Enum


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
