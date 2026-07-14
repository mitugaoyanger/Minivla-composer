"""使用 Pillow 渲染俯视桌面 RGB 图像。"""

from collections.abc import Iterable

import numpy as np
from PIL import Image, ImageDraw

from .objects import TableObject, TargetZone

COLOR_MAP = {
    "red": (220, 70, 70),
    "blue": (65, 115, 220),
    "green": (80, 175, 90),
    "yellow": (230, 190, 55),
}


def to_pixel(pos: tuple[float, float], image_size: int) -> tuple[int, int]:
    """将 [0,1] 桌面坐标转换为图像像素坐标。"""
    x = int(np.clip(pos[0], 0.0, 1.0) * (image_size - 1))
    y = int((1.0 - np.clip(pos[1], 0.0, 1.0)) * (image_size - 1))
    return x, y


def draw_object(draw: ImageDraw.ImageDraw, obj: TableObject, image_size: int) -> None:
    """绘制一个带颜色和形状的物体。"""
    cx, cy = to_pixel(obj.position, image_size)
    r = max(4, int(obj.radius * image_size))
    color = COLOR_MAP.get(obj.color, (120, 120, 120))
    if obj.shape == "circle":
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=color,
            outline=(40, 40, 40),
            width=2,
        )
    elif obj.shape == "triangle":
        pts = [
            (cx, cy - r),
            (cx - int(0.9 * r), cy + r),
            (cx + int(0.9 * r), cy + r),
        ]
        draw.polygon(pts, fill=color, outline=(40, 40, 40))
    else:
        draw.rectangle(
            [cx - r, cy - r, cx + r, cy + r],
            fill=color,
            outline=(40, 40, 40),
            width=2,
        )


def render_tabletop(
    objects: Iterable[TableObject],
    target_zone: TargetZone,
    gripper_pos: tuple[float, float],
    gripper_closed: bool,
    image_size: int = 128,
) -> np.ndarray:
    """渲染当前环境状态，返回 H x W x 3 的 uint8 RGB 图像。"""
    img = Image.new("RGB", (image_size, image_size), (250, 250, 250))
    draw = ImageDraw.Draw(img, "RGBA")

    # 先画目标区，使用半透明绿色边框突出放置区域。
    tx, ty = to_pixel(target_zone.center, image_size)
    tr = int(target_zone.radius * image_size)
    draw.ellipse(
        [tx - tr, ty - tr, tx + tr, ty + tr],
        fill=(90, 210, 120, 45),
        outline=(40, 160, 70, 180),
        width=3,
    )

    for obj in objects:
        draw_object(draw, obj, image_size)

    # 画夹爪：张开为两条线，闭合为小十字。
    gx, gy = to_pixel(gripper_pos, image_size)
    g = int(0.045 * image_size)
    grip_color = (30, 30, 30, 255)
    if gripper_closed:
        draw.line([gx - g, gy, gx + g, gy], fill=grip_color, width=3)
        draw.line([gx, gy - g, gx, gy + g], fill=grip_color, width=3)
    else:
        draw.arc(
            [gx - g, gy - g, gx + g, gy + g],
            start=25,
            end=155,
            fill=grip_color,
            width=3,
        )
        draw.arc(
            [gx - g, gy - g, gx + g, gy + g],
            start=205,
            end=335,
            fill=grip_color,
            width=3,
        )

    # 轻微网格帮助观察运动，不参与状态计算。
    for v in np.linspace(0, image_size, 5)[1:-1]:
        draw.line(
            [0, int(v), image_size, int(v)],
            fill=(230, 230, 230, 100),
            width=1,
        )
        draw.line(
            [int(v), 0, int(v), image_size],
            fill=(230, 230, 230, 100),
            width=1,
        )
    return np.asarray(img, dtype=np.uint8)
