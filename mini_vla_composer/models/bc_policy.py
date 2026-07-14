"""图像-语言-状态行为克隆策略。"""

import torch
from torch import nn

from .language_encoder import LanguageEncoder
from .vision_encoder import VisionEncoder


class BCPolicy(nn.Module):
    """共享多模态特征，并分别预测连续位移与离散夹爪状态。"""

    def __init__(self, state_dim: int, vocab_size: int) -> None:
        """初始化三个编码分支和 MLP 动作头。"""
        super().__init__()
        self.vision = VisionEncoder(output_dim=128)
        self.language = LanguageEncoder(vocab_size=vocab_size, output_dim=64)
        if (state_dim - 6) % 10 != 0:
            raise ValueError(
                f"v2 状态维度应满足 (state_dim-6)%10==0，实际为 {state_dim}"
            )
        self.num_objects = (state_dim - 6) // 10
        self.object_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.language_query = nn.Linear(64, 64)
        self.state_proj = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64 + 64 + 6, 128),
            nn.ReLU(),
        )
        self.motion_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh(),
        )
        self.gripper_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """输出归一化位移 ``motion`` 与未过 sigmoid 的夹爪 logit。"""
        v = self.vision(image)
        l = self.language(tokens)
        s = self.state_proj(state)
        # 状态中的每个物体由位置、持有状态和颜色/形状 one-hot 编码。
        objects = state[:, 3 : 3 + self.num_objects * 10].reshape(
            -1,
            self.num_objects,
            10,
        )
        object_features = self.object_encoder(objects)
        object_logits = (
            (object_features * self.language_query(l).unsqueeze(1)).sum(dim=-1)
            / 8.0
        )
        # 语言到物体的注意力显式定位指令目标。
        attention = object_logits.softmax(dim=-1)
        target_position = (attention.unsqueeze(-1) * objects[:, :, :2]).sum(dim=1)
        target_held = (attention * objects[:, :, 2]).sum(dim=1, keepdim=True)
        gripper_position = state[:, :2]
        zone_position = state[:, -3:-1]
        any_held = objects[:, :, 2].amax(dim=1, keepdim=True)
        relational = torch.cat(
            [
                target_position - gripper_position,
                zone_position - gripper_position,
                any_held,
                target_held,
            ],
            dim=-1,
        )
        features = self.fusion(torch.cat([v, l, s, relational], dim=-1))
        return {
            "motion": self.motion_head(features),
            "gripper_logit": self.gripper_head(features).squeeze(-1),
            "object_logits": object_logits,
        }
