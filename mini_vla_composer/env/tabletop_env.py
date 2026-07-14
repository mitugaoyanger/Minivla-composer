"""二维桌面抓取环境。"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from mini_vla_composer.env.objects import TableObject, TargetZone
from mini_vla_composer.env.renderer import render_tabletop
from mini_vla_composer.env.task_checker import distance, find_target_object, is_object_in_zone
from mini_vla_composer.language.instruction_parser import parse_instruction
from mini_vla_composer.language.task_generator import COLORS, SHAPES, generate_task


class TabletopEnv:
    """最小二维桌面环境，支持 reset、step 和 RGB 观测。"""

    def __init__(
        self,
        image_size: int = 128,
        max_steps: int = 80,
        num_objects: int = 4,
        target_radius: float = 0.12,
        gripper_speed: float = 0.04,
        grasp_radius: float = 0.06,
        randomize_layout: bool = True,
        seed: int | None = None,
    ) -> None:
        """初始化环境参数。"""
        self.image_size = image_size
        self.max_steps = max_steps
        self.num_objects = num_objects
        self.target_radius = target_radius
        self.gripper_speed = gripper_speed
        self.grasp_radius = grasp_radius
        self.randomize_layout = randomize_layout
        self.rng = random.Random(seed)
        self.objects: list[TableObject] = []
        self.target_zone = TargetZone(center=(0.82, 0.78), radius=target_radius)
        self.gripper_pos = np.array([0.12, 0.12], dtype=np.float32)
        self.gripper_closed = False
        self.held_index: int | None = None
        self.instruction = ""
        self.target_color = ""
        self.target_shape = ""
        self.steps = 0

    def reset(self, instruction: str | None = None) -> dict[str, Any]:
        """重置环境并返回初始观测。"""
        self.steps = 0
        self.gripper_pos = np.array([0.12, 0.12], dtype=np.float32)
        self.gripper_closed = False
        self.held_index = None
        self._sample_layout()
        if instruction is None:
            target_obj = self.rng.choice(self.objects)
            task = generate_task(target_obj.color, target_obj.shape, rng=self.rng)
            self.instruction = task["instruction"]
            self.target_color = task["target_color"]
            self.target_shape = task["target_shape"]
        else:
            self.instruction = instruction
            parsed = parse_instruction(instruction)
            self.target_color = parsed["target_color"]
            self.target_shape = parsed["target_shape"]
        return self._get_obs()

    def step(
        self,
        action: np.ndarray | list[float],
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """执行动作 [dx, dy, gripper]，返回 observation、reward、done、info。"""
        self.steps += 1
        act = np.asarray(action, dtype=np.float32)
        if act.shape != (3,):
            raise ValueError(f"动作必须是形状 (3,) 的向量，实际为 {act.shape}")
        delta = np.clip(act[:2], -self.gripper_speed, self.gripper_speed)
        self.gripper_pos = np.clip(self.gripper_pos + delta, 0.0, 1.0)

        # 对外动作语义与数据集保持一致：0=打开，1=闭合。
        want_closed = bool(act[2] >= 0.5)
        # 未抓住物体时持续尝试，可容忍策略提前闭合夹爪。
        if want_closed and self.held_index is None:
            self._try_grasp()
        if not want_closed and self.gripper_closed:
            self._release()
        self.gripper_closed = want_closed

        # 被抓住的物体跟随夹爪中心移动。
        if self.held_index is not None:
            self.objects[self.held_index].position = tuple(float(x) for x in self.gripper_pos)
            self.objects[self.held_index].held = True

        target_obj = find_target_object(self.objects, self.target_color, self.target_shape)
        # 物体进入目标区但仍被夹着不算完成；必须松手后才判定成功。
        success = (
            target_obj is not None
            and self.held_index is None
            and is_object_in_zone(target_obj, self.target_zone)
        )
        timeout = self.steps >= self.max_steps
        done = success or timeout
        reward = 1.0 if success else -0.01
        final_distance = (
            distance(target_obj.position, self.target_zone.center)
            if target_obj
            else 1.0
        )
        info = {
            "success": bool(success),
            "failure_reason": "" if success else ("timeout" if timeout else "not_done"),
            "target_object": f"{self.target_color} {self.target_shape}",
            "final_distance": float(final_distance),
            "step": self.steps,
        }
        return self._get_obs(), reward, done, info

    def _sample_layout(self) -> None:
        """采样夹爪、目标区和物体，覆盖桌面的各个运动方向。"""
        if not self.randomize_layout:
            self.gripper_pos = np.array([0.12, 0.12], dtype=np.float32)
            self.target_zone = TargetZone(center=(0.84, 0.80), radius=self.target_radius)
            self.objects = self._sample_objects(avoid=[self.target_zone.center])
            return

        margin = max(0.12, self.target_radius + 0.03)
        self.gripper_pos = np.array(self._sample_position(margin), dtype=np.float32)
        zone_center = self._sample_position(
            margin,
            avoid=[tuple(self.gripper_pos)],
            min_distance=0.30,
        )
        self.target_zone = TargetZone(center=zone_center, radius=self.target_radius)
        self.objects = self._sample_objects(avoid=[zone_center, tuple(self.gripper_pos)])

    def _sample_position(
        self,
        margin: float,
        avoid: list[tuple[float, float]] | None = None,
        min_distance: float = 0.16,
    ) -> tuple[float, float]:
        """在桌面边界内采样一个与已有点保持间距的位置。"""
        avoid = avoid or []
        for _ in range(200):
            candidate = (
                self.rng.uniform(margin, 1.0 - margin),
                self.rng.uniform(margin, 1.0 - margin),
            )
            if all(distance(candidate, other) >= min_distance for other in avoid):
                return candidate
        raise RuntimeError("无法采样互不重叠的桌面布局，请减小物体数量或最小间距")

    def _sample_objects(self, avoid: list[tuple[float, float]]) -> list[TableObject]:
        """随机生成多个不同颜色、形状且互不重叠的物体。"""
        pairs = [(c, s) for c in COLORS for s in SHAPES]
        self.rng.shuffle(pairs)
        objects: list[TableObject] = []
        for index in range(self.num_objects):
            color, shape = pairs[index]
            pos = self._sample_position(
                0.10,
                avoid=avoid + [obj.position for obj in objects],
            )
            size = "large" if self.rng.random() < 0.25 else "small"
            objects.append(TableObject(color=color, shape=shape, position=pos, size=size))
        return objects

    def _try_grasp(self) -> None:
        """夹爪闭合时尝试抓住最近的物体。"""
        best_i = None
        best_d = 999.0
        for i, obj in enumerate(self.objects):
            d = distance(tuple(self.gripper_pos), obj.position)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i is not None and best_d <= self.grasp_radius:
            self.held_index = best_i
            self.objects[best_i].held = True

    def _release(self) -> None:
        """打开夹爪并释放当前物体。"""
        if self.held_index is not None:
            self.objects[self.held_index].held = False
        self.held_index = None

    def _state_vector(self) -> np.ndarray:
        """编码夹爪、物体语义和目标区，形成固定长度状态向量。"""
        values = [
            float(self.gripper_pos[0]),
            float(self.gripper_pos[1]),
            float(self.gripper_closed),
        ]
        for obj in self.objects:
            color_one_hot = [float(obj.color == color) for color in COLORS]
            shape_one_hot = [float(obj.shape == shape) for shape in SHAPES]
            values.extend(
                [
                    obj.position[0],
                    obj.position[1],
                    float(obj.held),
                    *color_one_hot,
                    *shape_one_hot,
                ]
            )
        while len(values) < 3 + self.num_objects * 10:
            values.extend([0.0] * 10)
        values.extend(
            [
                self.target_zone.center[0],
                self.target_zone.center[1],
                self.target_zone.radius,
            ]
        )
        return np.asarray(values, dtype=np.float32)

    def _get_obs(self) -> dict[str, Any]:
        """返回包含图像、状态和语言指令的观测。"""
        image = render_tabletop(
            self.objects,
            self.target_zone,
            tuple(self.gripper_pos),
            self.gripper_closed,
            self.image_size,
        )
        return {"image": image, "state": self._state_vector(), "instruction": self.instruction}
