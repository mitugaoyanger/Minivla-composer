"""命令行入口：动态播放一条数据集 episode。"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mini_vla_composer.data.visualize_episode import visualize_episode
from mini_vla_composer.utils.console import configure_utf8_console


def parse_args() -> argparse.Namespace:
    """解析 episode 路径与播放选项。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True, help="episode npz 路径")
    parser.add_argument("--fps", type=float, default=10.0, help="动画播放帧率")
    parser.add_argument("--repeat", action="store_true", help="循环播放")
    parser.add_argument("--save-gif", type=Path, default=None, help="可选的 GIF 保存路径")
    parser.add_argument("--no-window", action="store_true", help="不打开播放窗口")
    return parser.parse_args()


def main() -> None:
    """创建动画，并按命令行选项播放或保存。"""
    configure_utf8_console()
    args = parse_args()
    figure, animation = visualize_episode(args.episode, args.fps, args.repeat)
    if args.save_gif is not None:
        args.save_gif.parent.mkdir(parents=True, exist_ok=True)
        animation.save(args.save_gif, writer="pillow", fps=args.fps)
        print(f"动画已保存：{args.save_gif}")
    if args.no_window:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
