"""加载 BC 策略并执行单步推理。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mini_vla_composer.models.action_codec import decode_policy_output
from mini_vla_composer.models.bc_policy import BCPolicy
from mini_vla_composer.models.language_encoder import SimpleTokenizer


@dataclass(frozen=True)
class LoadedPolicy:
    """评估阶段所需的模型、分词器与检查点配置。"""

    model: BCPolicy
    tokenizer: SimpleTokenizer
    checkpoint: dict[str, Any]
    device: torch.device


def load_policy(checkpoint_path: str | Path, device: str = "cpu") -> LoadedPolicy:
    """从 v2 检查点恢复一份只用于推理的策略。"""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到模型：{path}")

    torch_device = torch.device(device)
    checkpoint = torch.load(path, map_location=torch_device)
    if int(checkpoint.get("format_version", 0)) != 2:
        raise ValueError("检查点不是 v2 双头策略，请先使用新版数据重新训练")

    model = BCPolicy(
        state_dim=int(checkpoint["state_dim"]),
        vocab_size=int(checkpoint["vocab_size"]),
    ).to(torch_device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return LoadedPolicy(
        model=model,
        tokenizer=SimpleTokenizer(),
        checkpoint=checkpoint,
        device=torch_device,
    )


def observation_to_tensors(
    observation: dict[str, Any],
    tokenizer: SimpleTokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把单条环境观测转换为带 batch 维度的模型输入。"""
    image = (
        torch.from_numpy(observation["image"].copy())
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    )
    state = torch.from_numpy(observation["state"].copy()).float().unsqueeze(0)
    tokens = torch.tensor(
        [tokenizer.encode(observation["instruction"])],
        dtype=torch.long,
    )
    return image.to(device), state.to(device), tokens.to(device)


def predict_action(policy: LoadedPolicy, observation: dict[str, Any]) -> np.ndarray:
    """预测一个环境动作，其中夹爪值由二分类结果解码为 0 或 1。"""
    image, state, tokens = observation_to_tensors(
        observation,
        policy.tokenizer,
        policy.device,
    )
    with torch.inference_mode():
        output = policy.model(image, state, tokens)
        action = decode_policy_output(
            output["motion"],
            output["gripper_logit"],
            float(policy.checkpoint["gripper_speed"]),
        )
    return action.cpu().numpy()[0]
