"""命令行入口：采集脚本专家数据。"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mini_vla_composer.data.collect_dataset import collect_dataset
from mini_vla_composer.utils.config import load_config
from mini_vla_composer.utils.console import configure_utf8_console


def parse_args() -> argparse.Namespace:
    """解析数据采集配置路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "data.yaml",
        help="数据采集配置文件路径",
    )
    return parser.parse_args()


def main() -> None:
    """读取配置并启动数据采集。"""
    configure_utf8_console()
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config

    # 配置中的保存路径以项目根目录为基准。
    os.chdir(ROOT)
    collect_dataset(load_config(config_path))


if __name__ == "__main__":
    main()
