"""transforms_tools —— 相机位姿(transforms.json)标定与统一坐标系工具集。

独立发布版:核心功能仅依赖 numpy / scipy;可视化(transforms_viz)另需可选依赖 [viz]。
"""

from .use_existed_trans import use_existed_trans
from .canonicalize import canonicalize
from .calibrate_rig import calibrate_rig
from .finalize import finalize
from .set_aabb import set_aabb
from .check_repeatability import check_repeatability
from .copy_first_images import copy_first_images

__version__ = "1.0.0"
__all__ = [
    "use_existed_trans",
    "canonicalize",
    "calibrate_rig",
    "finalize",
    "set_aabb",
    "check_repeatability",
    "copy_first_images",
]
