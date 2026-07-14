"""检查训练数据是否符合 v2 动作与元数据约定。"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mini_vla_composer.utils.config import load_config
from mini_vla_composer.utils.console import configure_utf8_console
from mini_vla_composer.utils.io import load_json


def check_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """验证文件配对、数组形状、夹爪标签和位移方向覆盖。"""
    dataset_dir = Path(config.get("dataset_dir", "results/datasets/demo_v2"))
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"找不到数据集目录：{dataset_dir}")

    episode_files = sorted(dataset_dir.glob("episode_*.npz"))
    metadata_files = sorted(dataset_dir.glob("episode_*.json"))
    if not episode_files:
        raise FileNotFoundError(f"目录中没有 episode：{dataset_dir}")
    if {path.stem for path in episode_files} != {
        path.stem for path in metadata_files
    }:
        raise ValueError("npz 与 episode 元数据未一一对应")

    dataset_info = load_json(dataset_dir / "dataset_info.json")
    if int(dataset_info.get("format_version", 0)) != 2:
        raise ValueError("数据集不是 v2 格式")

    action_batches: list[np.ndarray] = []
    state_dims: set[int] = set()
    for path in episode_files:
        with np.load(path) as episode:
            images = episode["images"]
            states = episode["states"]
            actions = episode["actions"]
            if images.ndim != 4 or states.ndim != 2 or actions.ndim != 2:
                raise ValueError(f"{path.name} 的数组维度不合法")
            if not (len(images) == len(states) == len(actions)):
                raise ValueError(f"{path.name} 的帧数不一致")
            if actions.shape[1] != 3:
                raise ValueError(f"{path.name} 的动作维度不是 3")
            action_batches.append(actions.copy())
            state_dims.add(int(states.shape[1]))

    if len(state_dims) != 1:
        raise ValueError(f"数据集包含多个状态维度：{sorted(state_dims)}")
    actions = np.concatenate(action_batches, axis=0)
    if not np.isin(actions[:, 2], (0.0, 1.0)).all():
        raise ValueError("夹爪标签必须严格为 0 或 1")
    if not (np.any(actions[:, :2] < 0) and np.any(actions[:, :2] > 0)):
        raise ValueError("位移数据没有同时覆盖正、负方向")

    targets = Counter()
    for path in metadata_files:
        metadata = load_json(path)
        targets[(metadata["target_color"], metadata["target_shape"])] += 1

    return {
        "dataset_dir": str(dataset_dir),
        "episodes": len(episode_files),
        "samples": len(actions),
        "state_dim": state_dims.pop(),
        "motion_min": actions[:, :2].min(axis=0).tolist(),
        "motion_max": actions[:, :2].max(axis=0).tolist(),
        "gripper_counts": {
            "open": int(np.count_nonzero(actions[:, 2] == 0)),
            "closed": int(np.count_nonzero(actions[:, 2] == 1)),
        },
        "target_counts": {
            f"{color} {shape}": count
            for (color, shape), count in sorted(targets.items())
        },
    }


def parse_args() -> argparse.Namespace:
    """解析训练配置路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train_bc.yaml",
        help="用于定位 dataset_dir 的训练配置",
    )
    return parser.parse_args()


def main() -> None:
    """运行数据检查并输出简洁摘要。"""
    configure_utf8_console()
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    summary = check_dataset(load_config(config_path))
    print(f"数据集：{summary['dataset_dir']}")
    print(
        f"episodes={summary['episodes']}，samples={summary['samples']}，"
        f"state_dim={summary['state_dim']}"
    )
    print(
        f"motion_min={summary['motion_min']}，"
        f"motion_max={summary['motion_max']}，"
        f"gripper={summary['gripper_counts']}"
    )
    print(f"targets={summary['target_counts']}")
    print("ALL CHECKS PASSED：数据文件和 v2 动作语义有效。")


if __name__ == "__main__":
    main()
