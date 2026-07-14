"""随机运行一条闭环评估轨迹，并以动画方式播放。"""

import argparse
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mini_vla_composer.env.factory import make_env
from mini_vla_composer.eval.policy_runtime import load_policy, predict_action
from mini_vla_composer.utils.console import configure_utf8_console


@dataclass(frozen=True)
class Rollout:
    """一次闭环评估的画面、动作与最终状态。"""

    seed: int
    instruction: str
    frames: list[np.ndarray]
    actions: list[np.ndarray]
    gripper_closed: list[bool]
    held_objects: list[str]
    final_info: dict[str, Any]


def run_random_rollout(
    checkpoint_path: Path,
    seed: int,
    device: str = "cpu",
    max_steps: int | None = None,
) -> Rollout:
    """用给定随机种子生成任务，并记录一条完整闭环轨迹。"""
    policy = load_policy(checkpoint_path, device)
    environment_config = dict(policy.checkpoint.get("environment", {}))
    if max_steps is not None:
        environment_config["max_steps"] = max_steps
    env = make_env(environment_config, seed=seed)
    observation = env.reset()

    frames = [observation["image"].copy()]
    actions: list[np.ndarray] = []
    gripper_closed = [bool(env.gripper_closed)]
    held_objects = [_held_object_name(env)]
    final_info: dict[str, Any] = {
        "success": False,
        "failure_reason": "timeout",
        "final_distance": float("nan"),
    }

    for _ in range(env.max_steps):
        action = predict_action(policy, observation)
        observation, _, done, final_info = env.step(action)
        actions.append(action.copy())
        frames.append(observation["image"].copy())
        gripper_closed.append(bool(env.gripper_closed))
        held_objects.append(_held_object_name(env))
        if done:
            break

    return Rollout(
        seed=seed,
        instruction=observation["instruction"],
        frames=frames,
        actions=actions,
        gripper_closed=gripper_closed,
        held_objects=held_objects,
        final_info=final_info,
    )


def _held_object_name(env: Any) -> str:
    """返回当前被夹持物体的可读名称。"""
    if env.held_index is None:
        return "none"
    obj = env.objects[env.held_index]
    return f"{obj.color} {obj.shape}"


def create_animation(
    rollout: Rollout,
    fps: float = 8.0,
    repeat: bool = False,
) -> tuple[plt.Figure, FuncAnimation]:
    """创建环境画面与动作时间线同步更新的动画。"""
    if fps <= 0:
        raise ValueError("fps 必须大于 0")

    figure = plt.figure(figsize=(10, 5.6), layout="constrained")
    grid = figure.add_gridspec(2, 2, width_ratios=(1.35, 1.0))
    image_axis = figure.add_subplot(grid[:, 0])
    motion_axis = figure.add_subplot(grid[0, 1])
    gripper_axis = figure.add_subplot(grid[1, 1])

    image_artist = image_axis.imshow(rollout.frames[0], interpolation="nearest")
    image_axis.set_title(rollout.instruction)
    image_axis.axis("off")

    steps = np.arange(1, len(rollout.actions) + 1)
    action_array = (
        np.asarray(rollout.actions, dtype=np.float32)
        if rollout.actions
        else np.empty((0, 3), dtype=np.float32)
    )
    motion_axis.plot(steps, action_array[:, 0], label="dx", color="tab:blue")
    motion_axis.plot(steps, action_array[:, 1], label="dy", color="tab:orange")
    motion_axis.axhline(0.0, color="0.75", linewidth=0.8)
    motion_axis.set_ylabel("motion")
    motion_axis.legend(loc="upper right")
    motion_axis.grid(alpha=0.2)

    gripper_axis.step(
        steps,
        action_array[:, 2],
        where="post",
        color="tab:green",
    )
    gripper_axis.set_ylim(-0.1, 1.1)
    gripper_axis.set_yticks((0, 1), labels=("open", "closed"))
    gripper_axis.set_xlabel("step")
    gripper_axis.set_ylabel("gripper")
    gripper_axis.grid(alpha=0.2)

    motion_cursor = motion_axis.axvline(0, color="tab:red", linewidth=1.2)
    gripper_cursor = gripper_axis.axvline(0, color="tab:red", linewidth=1.2)
    status_artist = figure.text(0.5, 0.01, "", ha="center", va="bottom")

    def update(frame_index: int):
        """让画面、时间线游标和状态文本保持同步。"""
        image_artist.set_data(rollout.frames[frame_index])
        action_index = max(0, frame_index - 1)
        cursor = frame_index
        motion_cursor.set_xdata([cursor, cursor])
        gripper_cursor.set_xdata([cursor, cursor])

        if frame_index == 0:
            action_text = "action: waiting"
        else:
            action = rollout.actions[action_index]
            action_text = (
                f"action: dx={action[0]:+.4f}, dy={action[1]:+.4f}, "
                f"gripper={int(action[2])}"
            )
        result_text = ""
        if frame_index == len(rollout.frames) - 1:
            success = bool(rollout.final_info.get("success", False))
            result = "SUCCESS" if success else rollout.final_info.get(
                "failure_reason",
                "FAILED",
            )
            distance = float(rollout.final_info.get("final_distance", float("nan")))
            result_text = f" | result={result} | distance={distance:.4f}"
        status_artist.set_text(
            f"seed={rollout.seed} | step={frame_index}/{len(rollout.actions)} | "
            f"{action_text} | held={rollout.held_objects[frame_index]}{result_text}"
        )
        return image_artist, motion_cursor, gripper_cursor, status_artist

    animation = FuncAnimation(
        figure,
        update,
        frames=len(rollout.frames),
        interval=1000.0 / fps,
        repeat=repeat,
        blit=False,
    )
    return figure, animation


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "results" / "checkpoints" / "bc_policy_v2.pt",
        help="v2 策略检查点路径",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="任务随机种子；不指定时每次随机选择",
    )
    parser.add_argument("--device", default="cpu", help="推理设备，例如 cpu 或 cuda")
    parser.add_argument("--fps", type=float, default=8.0, help="动画播放帧率")
    parser.add_argument("--max-steps", type=int, default=None, help="可选的最大步数覆盖值")
    parser.add_argument(
        "--save-gif",
        type=Path,
        default=ROOT / "results" / "figures",
        help="GIF保存文件夹",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="不打开窗口，适合自动化验证",
    )
    return parser.parse_args()


def main() -> None:
    """随机执行一次评估，并按需播放或保存动画。"""
    configure_utf8_console()
    args = parse_args()
    seed = args.seed if args.seed is not None else secrets.randbelow(1_000_000)
    rollout = run_random_rollout(
        checkpoint_path=args.checkpoint,
        seed=seed,
        device=args.device,
        max_steps=args.max_steps,
    )

    success = bool(rollout.final_info.get("success", False))
    reason = "success" if success else rollout.final_info.get(
        "failure_reason",
        "unknown",
    )
    print(f"随机任务：{rollout.instruction}")
    print(f"seed={seed}，steps={len(rollout.actions)}，success={success}，reason={reason}")
    print(f"复现命令：python scripts/visualize_eval.py --seed {seed}")

    if args.no_window and args.save_gif is None:
        return

    figure, animation = create_animation(rollout, fps=args.fps)
    save_dir = args.save_gif
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"rollout_seed_{seed}.gif"

    animation.save(
        str(save_path),
        writer="pillow",
        fps=args.fps,
    )

    print(f"动画已保存：{save_path}")
    if args.no_window:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
