"""任务成功与距离指标计算。"""

import math

from .objects import TableObject, TargetZone


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """计算二维欧氏距离。"""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def find_target_object(
    objects: list[TableObject],
    color: str,
    shape: str,
) -> TableObject | None:
    """根据颜色和形状查找目标物体。"""
    for obj in objects:
        if obj.color == color and obj.shape == shape:
            return obj
    return None


def is_object_in_zone(obj: TableObject, zone: TargetZone) -> bool:
    """判断物体中心是否进入目标区域。"""
    return distance(obj.position, zone.center) <= zone.radius
