"""环境动作语义和策略输出接口的回归测试。"""

import unittest

import numpy as np
import torch

from mini_vla_composer.env.tabletop_env import TabletopEnv
from mini_vla_composer.env.task_checker import find_target_object
from mini_vla_composer.expert.scripted_expert import ScriptedExpert
from mini_vla_composer.models.action_codec import decode_policy_output, encode_action_targets


class ActionCodecTest(unittest.TestCase):
    """验证模型动作与环境动作之间的转换约定。"""

    def test_round_trip_uses_binary_gripper(self) -> None:
        actions = torch.tensor([[0.04, -0.02, 1.0], [-0.04, 0.01, 0.0]])
        motion, gripper = encode_action_targets(actions, gripper_speed=0.04)
        logits = torch.where(gripper > 0.5, torch.tensor(10.0), torch.tensor(-10.0))
        decoded = decode_policy_output(motion, logits, gripper_speed=0.04)
        torch.testing.assert_close(decoded, actions)


class EnvironmentTest(unittest.TestCase):
    """验证环境终止条件和专家覆盖范围。"""

    def test_success_requires_release(self) -> None:
        env = TabletopEnv(seed=1)
        env.reset()
        target = find_target_object(env.objects, env.target_color, env.target_shape)
        self.assertIsNotNone(target)
        index = env.objects.index(target)
        env.gripper_pos = np.asarray(env.target_zone.center, dtype=np.float32)
        target.position = env.target_zone.center
        target.held = True
        env.held_index = index
        env.gripper_closed = True

        _, _, _, held_info = env.step([0.0, 0.0, 1.0])
        self.assertFalse(held_info["success"])
        _, _, done, released_info = env.step([0.0, 0.0, 0.0])
        self.assertTrue(done)
        self.assertTrue(released_info["success"])

    def test_expert_handles_all_directions(self) -> None:
        env = TabletopEnv(seed=123)
        expert = ScriptedExpert(env)
        actions = []
        successes = 0
        for _ in range(30):
            obs = env.reset()
            expert.reset()
            for _ in range(env.max_steps):
                action = expert.act(obs)
                actions.append(action)
                obs, _, done, info = env.step(action)
                if done:
                    break
            successes += int(info["success"])
        array = np.asarray(actions)
        self.assertEqual(successes, 30)
        self.assertTrue(np.any(array[:, 0] < 0) and np.any(array[:, 0] > 0))
        self.assertTrue(np.any(array[:, 1] < 0) and np.any(array[:, 1] > 0))

    def test_seed_reproduces_task_and_layout(self) -> None:
        first = TabletopEnv(seed=17)
        second = TabletopEnv(seed=17)
        first_observation = first.reset()
        second_observation = second.reset()

        self.assertEqual(first.instruction, second.instruction)
        np.testing.assert_array_equal(
            first_observation["state"],
            second_observation["state"],
        )


if __name__ == "__main__":
    unittest.main()
