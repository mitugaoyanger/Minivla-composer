"""评估指标计算函数。"""

import numpy as np


def action_smoothness(actions: list[np.ndarray]) -> float:
    """计算相邻动作差分的平均范数，越小越平滑。"""
    if len(actions) < 2:
        return 0.0
    arr = np.asarray(actions, dtype=np.float32)
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).mean())
