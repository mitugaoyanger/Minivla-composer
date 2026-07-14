"""PyTorch 行为克隆数据集。"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mini_vla_composer.language.task_generator import COLORS, SHAPES
from mini_vla_composer.models.language_encoder import SimpleTokenizer
from mini_vla_composer.utils.io import load_json


class BCDataset(Dataset):
    """把 episode 展开为单步行为克隆样本。"""

    def __init__(
        self,
        dataset_dir: str,
        tokenizer: SimpleTokenizer | None = None,
        files: Sequence[Path] | None = None,
    ) -> None:
        """加载目录中所有 npz/json 文件。"""
        self.dataset_dir = Path(dataset_dir)
        self.files = (
            list(files)
            if files is not None
            else sorted(self.dataset_dir.glob("episode_*.npz"))
        )
        if not self.files:
            raise FileNotFoundError(
                f"没有找到数据：{self.dataset_dir}，请先运行 "
                "python scripts/collect_data.py --config configs/data.yaml"
            )
        self.tokenizer = tokenizer or SimpleTokenizer()
        self.index: list[tuple[int, int]] = []
        self.instructions: list[str] = []
        self.target_indices: list[int] = []
        self.cache: dict[int, dict[str, np.ndarray]] = {}
        for i, file in enumerate(self.files):
            meta = load_json(file.with_suffix(".json"))
            self.instructions.append(meta["instruction"])
            with np.load(file) as data:
                self.index.extend((i, t) for t in range(int(data["actions"].shape[0])))
                state = data["states"][0]
                # 状态布局为 gripper(3) + objects(N*10) + zone(3)。
                num_objects = (state.size - 6) // 10
                objects = state[3 : 3 + num_objects * 10].reshape(num_objects, 10)
                color_i = COLORS.index(meta["target_color"])
                shape_i = SHAPES.index(meta["target_shape"])
                matches = np.flatnonzero(
                    (objects[:, 3 + color_i] > 0.5)
                    & (objects[:, 7 + shape_i] > 0.5)
                )
                if matches.size != 1:
                    raise ValueError(f"{file} 无法唯一定位目标物体")
                self.target_indices.append(int(matches[0]))

    def __len__(self) -> int:
        """返回展开后的样本数量。"""
        return len(self.index)

    def _load_episode(self, episode_i: int) -> dict[str, np.ndarray]:
        """按需读取并缓存 episode。"""
        if episode_i not in self.cache:
            with np.load(self.files[episode_i]) as data:
                self.cache[episode_i] = {key: data[key] for key in data.files}
        return self.cache[episode_i]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        """返回单步训练样本。"""
        episode_i, t = self.index[idx]
        ep = self._load_episode(episode_i)
        image = torch.from_numpy(ep["images"][t]).float().permute(2, 0, 1) / 255.0
        state = torch.from_numpy(ep["states"][t]).float()
        action = torch.from_numpy(ep["actions"][t]).float()
        tokens = torch.tensor(
            self.tokenizer.encode(self.instructions[episode_i]),
            dtype=torch.long,
        )
        return {
            "image": image,
            "state": state,
            "tokens": tokens,
            "action": action,
            "target_index": self.target_indices[episode_i],
        }
