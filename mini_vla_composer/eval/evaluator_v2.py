"""双头 BC 策略的闭环批量评估。"""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mini_vla_composer.env.factory import make_env
from mini_vla_composer.eval.metrics import action_smoothness
from mini_vla_composer.eval.policy_runtime import load_policy, predict_action
from mini_vla_composer.utils.io import ensure_dir, save_csv, save_json
from mini_vla_composer.utils.seed import set_seed

ROOT = Path(__file__).resolve().parents[2]


def evaluate_policy(config: dict[str, Any]) -> dict[str, float]:
    """在未见随机种子上运行闭环评估，并保存逐回合结果。"""
    checkpoint_path = Path(
        config.get("checkpoint_path", "results/checkpoints/bc_policy_v2.pt")
    )
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path

    seed = int(config.get("seed", 123))
    set_seed(seed)
    policy = load_policy(checkpoint_path, str(config.get("device", "cpu")))
    env = make_env(
        {**policy.checkpoint.get("environment", {}), **config},
        seed=seed,
    )

    num_episodes = int(config.get("num_eval_episodes", 30))
    save_examples = int(config.get("save_examples", 5))
    example_dir = ensure_dir(ROOT / "results" / "figures" / "eval_examples_v2")
    table_dir = ensure_dir(ROOT / "results" / "tables")
    rows: list[dict[str, Any]] = []

    for episode in range(num_episodes):
        observation = env.reset()
        actions: list[np.ndarray] = []
        final_frame = observation["image"].copy()
        info: dict[str, Any] = {
            "success": False,
            "failure_reason": "not_started",
            "final_distance": 1.0,
        }

        for _ in range(env.max_steps):
            action = predict_action(policy, observation)
            actions.append(action.copy())
            observation, _, done, info = env.step(action)
            final_frame = observation["image"].copy()
            if done:
                break

        switches = (
            int(np.count_nonzero(np.diff(np.asarray(actions)[:, 2])))
            if len(actions) > 1
            else 0
        )
        row = {
            "episode": episode,
            "success": bool(info["success"]),
            "steps": len(actions),
            "failure_reason": info["failure_reason"],
            "final_distance": float(info["final_distance"]),
            "action_smoothness": action_smoothness(actions),
            "gripper_switches": switches,
        }
        rows.append(row)
        print(
            f"Episode {episode + 1:03d}: success={row['success']}, "
            f"steps={row['steps']}, distance={row['final_distance']:.4f}"
        )
        if episode < save_examples:
            Image.fromarray(final_frame).save(
                example_dir / f"eval_{episode:03d}_final.png"
            )

    summary = _summarize(rows)
    json_path = table_dir / "eval_results_v2.json"
    csv_path = table_dir / "eval_results_v2.csv"
    save_json({"summary": summary, "episodes": rows}, json_path)
    save_csv(rows, csv_path)
    print(f"评估完成：{summary}")
    print(f"结果：{json_path}")
    return summary


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    """汇总逐回合指标；空输入返回全零结果。"""
    if not rows:
        return {
            "success_rate": 0.0,
            "average_steps": 0.0,
            "timeout_rate": 0.0,
            "average_final_distance": 0.0,
            "action_smoothness": 0.0,
            "average_gripper_switches": 0.0,
        }
    return {
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "average_steps": float(np.mean([row["steps"] for row in rows])),
        "timeout_rate": float(
            np.mean([row["failure_reason"] == "timeout" for row in rows])
        ),
        "average_final_distance": float(
            np.mean([row["final_distance"] for row in rows])
        ),
        "action_smoothness": float(
            np.mean([row["action_smoothness"] for row in rows])
        ),
        "average_gripper_switches": float(
            np.mean([row["gripper_switches"] for row in rows])
        ),
    }
