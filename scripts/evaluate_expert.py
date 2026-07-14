"""评估脚本专家在随机闭环任务上的成功率。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mini_vla_composer.env.factory import make_env
from mini_vla_composer.expert.scripted_expert import ScriptedExpert
from mini_vla_composer.utils.config import load_config
from mini_vla_composer.utils.console import configure_utf8_console


def parse_args() -> argparse.Namespace:
    """解析评估配置与可选覆盖参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "eval.yaml",
        help="环境配置文件路径",
    )
    parser.add_argument("--episodes", type=int, default=None, help="评估回合数")
    parser.add_argument("--seed", type=int, default=None, help="评估随机种子")
    return parser.parse_args()


def main() -> None:
    """运行专家闭环评估并打印逐回合结果。"""
    configure_utf8_console()
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    seed = args.seed if args.seed is not None else int(config.get("seed", 123))
    num_episodes = (
        args.episodes
        if args.episodes is not None
        else int(config.get("num_eval_episodes", 30))
    )

    env = make_env(config, seed=seed)
    expert = ScriptedExpert(env)
    successes = 0
    for episode in range(num_episodes):
        observation = env.reset()
        expert.reset()
        info = {"success": False, "step": 0, "final_distance": 1.0}
        for _ in range(env.max_steps):
            action = expert.act(observation)
            observation, _, done, info = env.step(action)
            if done:
                break

        successes += int(info["success"])
        print(
            f"Episode {episode + 1:03d}: success={info['success']}, "
            f"steps={info['step']}, distance={info['final_distance']:.4f}"
        )

    print(f"专家成功率：{successes / num_episodes:.2%}")


if __name__ == "__main__":
    main()
