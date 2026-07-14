"""生成简单英文桌面操作指令。"""

import random

COLORS = ["red", "blue", "green", "yellow"]
SHAPES = ["square", "circle", "triangle"]
TEMPLATES = [
    "Move the {color} {shape} to the target zone.",
    "Put the {color} {shape} into the target area.",
    "Move the {color} {shape} into the target area.",
]


def generate_task(
    color: str | None = None,
    shape: str | None = None,
    rng: random.Random | None = None,
) -> dict[str, str]:
    """生成一条任务指令和结构化目标，可注入随机数生成器。"""
    generator = rng or random
    color = color or generator.choice(COLORS)
    shape = shape or generator.choice(SHAPES)
    if color not in COLORS or shape not in SHAPES:
        raise ValueError(f"不支持的目标：color={color!r}, shape={shape!r}")
    template = generator.choice(TEMPLATES)
    return {
        "instruction": template.format(color=color, shape=shape),
        "target_color": color,
        "target_shape": shape,
        "target_zone": "target zone",
    }
