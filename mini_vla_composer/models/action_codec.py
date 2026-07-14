"""策略动作的归一化、监督目标与环境动作转换。"""

import torch


def encode_action_targets(
    actions: torch.Tensor,
    gripper_speed: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把环境动作转换成位移回归目标和夹爪二分类目标。"""
    if actions.ndim != 2 or actions.shape[-1] != 3:
        raise ValueError(f"actions 应为 B x 3，实际为 {tuple(actions.shape)}")
    motion = (actions[:, :2] / float(gripper_speed)).clamp(-1.0, 1.0)
    gripper = actions[:, 2].clamp(0.0, 1.0)
    return motion, gripper


def decode_policy_output(
    motion: torch.Tensor,
    gripper_logit: torch.Tensor,
    gripper_speed: float,
) -> torch.Tensor:
    """把模型输出转换成环境动作，其中夹爪严格输出 0 或 1。"""
    gripper = (gripper_logit >= 0.0).to(motion.dtype).unsqueeze(-1)
    return torch.cat(
        [motion.clamp(-1.0, 1.0) * float(gripper_speed), gripper],
        dim=-1,
    )
