"""专家策略使用的基础运动原语。"""

import numpy as np


def move_towards(
    current: np.ndarray,
    target: tuple[float, float],
    max_step: float,
) -> np.ndarray:
    """生成朝目标点移动的一小步 dx、dy。"""
    target_arr = np.asarray(target, dtype=np.float32)
    delta = target_arr - current.astype(np.float32)
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        return np.zeros(2, dtype=np.float32)
    scale = min(max_step, norm) / norm
    return (delta * scale).astype(np.float32)
