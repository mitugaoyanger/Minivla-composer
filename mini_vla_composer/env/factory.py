"""从配置创建桌面环境，保证采集与评估使用相同接口。"""

from typing import Any, Mapping

from .tabletop_env import TabletopEnv


def env_kwargs_from_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """提取并规范化环境参数，避免脚本各自遗漏配置字段。"""
    defaults: dict[str, Any] = {
        "image_size": 128,
        "max_steps": 80,
        "num_objects": 4,
        "target_radius": 0.12,
        "gripper_speed": 0.04,
        "grasp_radius": 0.06,
        "randomize_layout": True,
        "seed": 7,
    }
    values = {key: cfg.get(key, default) for key, default in defaults.items()}
    for key in ("image_size", "max_steps", "num_objects", "seed"):
        values[key] = int(values[key])
    for key in ("target_radius", "gripper_speed", "grasp_radius"):
        values[key] = float(values[key])
    values["randomize_layout"] = bool(values["randomize_layout"])
    return values


def make_env(cfg: Mapping[str, Any], *, seed: int | None = None) -> TabletopEnv:
    """创建环境；可单独覆盖 seed 以运行可复现的多回合评估。"""
    kwargs = env_kwargs_from_config(cfg)
    if seed is not None:
        kwargs["seed"] = int(seed)
    return TabletopEnv(**kwargs)
