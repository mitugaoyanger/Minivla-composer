"""小型 CNN 视觉编码器。"""

import torch
from torch import nn


class VisionEncoder(nn.Module):
    """提取图像特征，并保留物体位置信息。"""

    def __init__(self, output_dim: int = 128) -> None:
        """创建轻量卷积特征提取网络。"""
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, output_dim),
            nn.ReLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """输入 B x 3 x H x W 图像，输出视觉特征。"""
        return self.net(image)
