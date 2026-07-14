"""环境中的物体和目标区域定义。"""

from dataclasses import dataclass


@dataclass
class TableObject:
    """桌面上的一个可抓取物体。"""

    color: str
    shape: str
    position: tuple[float, float]
    size: str = "small"
    held: bool = False

    @property
    def radius(self) -> float:
        """根据尺寸返回物体近似半径。"""
        return 0.045 if self.size == "small" else 0.065

    @property
    def description(self) -> str:
        """返回颜色和形状组成的文本描述。"""
        return f"{self.color} {self.shape}"


@dataclass
class TargetZone:
    """放置区域的几何定义。"""

    center: tuple[float, float]
    radius: float
