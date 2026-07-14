"""动态播放数据集中的一条 episode。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from mini_vla_composer.utils.io import load_json


def visualize_episode(
    episode_path: str | Path,
    fps: float = 10.0,
    repeat: bool = False,
) -> tuple[plt.Figure, FuncAnimation]:
    """同步展示环境画面、动作曲线与夹爪轨迹。"""
    if fps <= 0:
        raise ValueError("fps 必须大于 0")

    ep_path = Path(episode_path)
    if not ep_path.is_file():
        raise FileNotFoundError(f"找不到 episode：{ep_path}")

    with np.load(ep_path) as data:
        images = data["images"].copy()
        states = data["states"].copy()
        actions = data["actions"].copy()
    if not (len(images) == len(states) == len(actions)):
        raise ValueError("images、states 与 actions 的帧数必须一致")
    if len(images) == 0:
        raise ValueError("episode 不包含任何帧")

    metadata_path = ep_path.with_suffix(".json")
    metadata = load_json(metadata_path) if metadata_path.is_file() else {}
    instruction = metadata.get("instruction", ep_path.stem)

    figure = plt.figure(figsize=(10, 5.6), layout="constrained")
    grid = figure.add_gridspec(2, 2, width_ratios=(1.35, 1.0))
    image_axis = figure.add_subplot(grid[:, 0])
    motion_axis = figure.add_subplot(grid[0, 1])
    path_axis = figure.add_subplot(grid[1, 1])

    image_artist = image_axis.imshow(images[0], interpolation="nearest")
    image_axis.set_title(instruction)
    image_axis.axis("off")

    steps = np.arange(len(actions))
    motion_axis.plot(steps, actions[:, 0], label="dx", color="tab:blue")
    motion_axis.plot(steps, actions[:, 1], label="dy", color="tab:orange")
    motion_axis.step(
        steps,
        actions[:, 2],
        where="post",
        label="gripper",
        color="tab:green",
        alpha=0.75,
    )
    motion_axis.set_title("Recorded action")
    motion_axis.set_xlabel("step")
    motion_axis.grid(alpha=0.2)
    motion_axis.legend(loc="upper right")
    motion_cursor = motion_axis.axvline(0, color="tab:red", linewidth=1.2)

    # 状态向量前两维固定表示夹爪平面位置。
    path_axis.plot(states[:, 0], states[:, 1], color="0.7", linewidth=1.2)
    position_artist = path_axis.scatter(
        [states[0, 0]],
        [states[0, 1]],
        color="tab:red",
        zorder=3,
    )
    path_axis.set_title("Gripper path")
    path_axis.set_xlabel("x")
    path_axis.set_ylabel("y")
    path_axis.set_aspect("equal", adjustable="box")
    path_axis.grid(alpha=0.2)
    status_artist = figure.text(0.5, 0.01, "", ha="center", va="bottom")

    def update(frame_index: int):
        """同步更新当前图像、曲线游标和夹爪位置。"""
        image_artist.set_data(images[frame_index])
        motion_cursor.set_xdata([frame_index, frame_index])
        position_artist.set_offsets(states[frame_index, :2][None, :])
        action = actions[frame_index]
        status_artist.set_text(
            f"frame={frame_index + 1}/{len(images)} | "
            f"dx={action[0]:+.4f}, dy={action[1]:+.4f}, "
            f"gripper={int(action[2])}"
        )
        return image_artist, motion_cursor, position_artist, status_artist

    animation = FuncAnimation(
        figure,
        update,
        frames=len(images),
        interval=1000.0 / fps,
        repeat=repeat,
        blit=False,
    )
    return figure, animation
