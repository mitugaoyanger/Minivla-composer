"""使用脚本专家采集行为克隆数据。"""

from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        """没有安装 tqdm 时退化为普通迭代器。"""
        return x

from mini_vla_composer.env.factory import env_kwargs_from_config, make_env
from mini_vla_composer.expert.scripted_expert import ScriptedExpert
from mini_vla_composer.language.instruction_parser import parse_instruction
from mini_vla_composer.language.subgoal_generator import generate_subgoals
from mini_vla_composer.utils.io import ensure_dir, save_json
from mini_vla_composer.utils.seed import set_seed


def collect_dataset(config: dict[str, Any]) -> None:
    """采集多条专家 episode 并保存为 npz/json。"""
    seed = int(config.get("seed", 7))
    set_seed(seed)
    save_dir = ensure_dir(config.get("save_dir", "results/datasets/demo_v2"))
    existing = list(save_dir.glob("episode_*.npz"))
    if existing:
        raise FileExistsError(
            f"数据目录已包含 {len(existing)} 条 episode：{save_dir}。"
            "请使用新的 save_dir，避免新旧数据静默混合。"
        )
    num_episodes = int(config.get("num_episodes", 120))
    success_only = bool(config.get("success_only", True))
    noise_std = float(config.get("action_noise_std", 0.0))
    env = make_env(config)
    expert = ScriptedExpert(env)
    rng = np.random.default_rng(seed)
    saved = 0
    for _ in tqdm(range(num_episodes), desc="采集专家数据"):
        obs = env.reset()
        expert.reset()
        images, states, actions = [], [], []
        info = {"success": False, "failure_reason": "not_started"}
        for _ in range(int(config.get("max_steps", 80))):
            action = expert.act(obs)
            # 仅扰动移动步骤；专家会从偏移后的新状态继续纠正。
            if noise_std > 0.0 and np.linalg.norm(action[:2]) > 1e-6:
                action[:2] += rng.normal(0.0, noise_std, size=2).astype(np.float32)
                action[:2] = np.clip(action[:2], -env.gripper_speed, env.gripper_speed)
            images.append(obs["image"])
            states.append(obs["state"])
            actions.append(action)
            obs, _, done, info = env.step(action)
            if done:
                break
        if success_only and not info.get("success", False):
            continue
        saved += 1
        stem = f"episode_{saved:06d}"
        np.savez_compressed(
            save_dir / f"{stem}.npz",
            images=np.asarray(images, dtype=np.uint8),
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
        )
        parsed = parse_instruction(env.instruction)
        metadata = {
            "instruction": env.instruction,
            "target_color": parsed["target_color"],
            "target_shape": parsed["target_shape"],
            "success": bool(info.get("success", False)),
            "num_steps": len(actions),
            "subgoals": generate_subgoals(env.instruction),
        }
        save_json(metadata, save_dir / f"{stem}.json")
    save_json(
        {
            "format_version": 2,
            "num_episodes": saved,
            "action": {"shape": [3], "motion": "dx_dy", "gripper": "0=open, 1=closed"},
            "environment": env_kwargs_from_config(config),
            "action_noise_std": noise_std,
        },
        save_dir / "dataset_info.json",
    )
    print(f"数据采集完成：保存 {saved} 条 episode 到 {Path(save_dir)}")
