"""命令行入口：闭环评估行为克隆策略。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mini_vla_composer.eval.evaluator_v2 import evaluate_policy
from mini_vla_composer.utils.config import load_config
from mini_vla_composer.utils.console import configure_utf8_console


def parse_args() -> argparse.Namespace:
    """解析评估配置路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "eval.yaml",
        help="评估配置文件路径",
    )
    return parser.parse_args()


def main() -> None:
    """读取配置并启动闭环评估。"""
    configure_utf8_console()
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    evaluate_policy(load_config(config_path))


if __name__ == "__main__":
    main()
