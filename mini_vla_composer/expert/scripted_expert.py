"""用于自动生成演示数据的脚本专家。"""

import numpy as np

from mini_vla_composer.env.tabletop_env import TabletopEnv
from mini_vla_composer.env.task_checker import distance, find_target_object
from mini_vla_composer.expert.motion_primitives import move_towards
from mini_vla_composer.language.instruction_parser import parse_instruction


class ScriptedExpert:
    """按 approach-grasp-move-release 阶段输出动作的简单专家。"""

    def __init__(self, env: TabletopEnv, reach_threshold: float = 0.025) -> None:
        """保存环境引用并初始化专家阶段。"""
        self.env = env
        self.reach_threshold = reach_threshold
        self.phase = "approach_object"

    def reset(self) -> None:
        """每条 episode 开始时重置专家内部阶段。"""
        self.phase = "approach_object"

    def act(self, obs: dict) -> np.ndarray:
        """根据当前观测和任务输出动作 [dx, dy, gripper]。"""
        parsed = parse_instruction(obs["instruction"])
        target_obj = find_target_object(
            self.env.objects,
            parsed["target_color"],
            parsed["target_shape"],
        )
        if target_obj is None:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if self.phase == "approach_object":
            if (
                distance(tuple(self.env.gripper_pos), target_obj.position)
                <= self.reach_threshold
            ):
                self.phase = "verify_grasp"
                return np.array([0.0, 0.0, 1.0], dtype=np.float32)
            delta = move_towards(
                self.env.gripper_pos,
                target_obj.position,
                self.env.gripper_speed,
            )
            return np.array([delta[0], delta[1], 0.0], dtype=np.float32)
        if self.phase == "verify_grasp":
            if self.env.held_index is None:
                self.phase = "approach_object"
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.phase = "move_to_target"
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if self.phase == "move_to_target":
            if (
                distance(tuple(self.env.gripper_pos), self.env.target_zone.center)
                <= self.reach_threshold
            ):
                self.phase = "release_object"
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)
            delta = move_towards(
                self.env.gripper_pos,
                self.env.target_zone.center,
                self.env.gripper_speed,
            )
            return np.array([delta[0], delta[1], 1.0], dtype=np.float32)
        if self.phase == "release_object":
            self.phase = "done"
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
