"""随机种子工具。"""

import random

import numpy as np


def set_seed(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # 没有安装 torch 时，数据采集脚本仍然可以运行到环境部分。
        pass
