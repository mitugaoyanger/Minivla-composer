"""命令行入口：训练行为克隆策略。"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mini_vla_composer.train.trainer_v2 import train_bc
from mini_vla_composer.utils.config import load_config
from mini_vla_composer.utils.console import configure_utf8_console


def parse_args() -> argparse.Namespace:
    """解析训练配置路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train_bc.yaml",
        help="训练配置文件路径",
    )
    return parser.parse_args()


def main() -> None:
    """读取配置并启动训练。"""
    configure_utf8_console()
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到训练配置文件：{config_path}")

    # 配置中的数据与输出路径均以项目根目录为基准。
    os.chdir(ROOT)
    print(f"训练配置：{config_path}")
    train_bc(load_config(config_path))


if __name__ == "__main__":
    main()
