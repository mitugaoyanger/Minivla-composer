"""用规则从指令中解析颜色和形状。"""

from .task_generator import COLORS, SHAPES


def parse_instruction(instruction: str) -> dict[str, str]:
    """从英文指令中抽取目标颜色和形状。"""
    text = instruction.lower()
    color = next((value for value in COLORS if value in text), None)
    shape = next((value for value in SHAPES if value in text), None)
    if color is None or shape is None:
        raise ValueError(f"指令中缺少可识别的颜色或形状：{instruction!r}")
    return {"target_color": color, "target_shape": shape, "target_zone": "target zone"}
