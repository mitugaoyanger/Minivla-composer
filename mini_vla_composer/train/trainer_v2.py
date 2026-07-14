"""训练带连续位移头和夹爪分类头的行为克隆策略。"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        """没有安装 tqdm 时退化为普通迭代器。"""
        return x

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from mini_vla_composer.data.dataset import BCDataset
from mini_vla_composer.models.action_codec import encode_action_targets
from mini_vla_composer.models.bc_policy import BCPolicy
from mini_vla_composer.models.language_encoder import SimpleTokenizer
from mini_vla_composer.utils.io import ensure_dir, load_json, save_json
from mini_vla_composer.utils.seed import set_seed


def _split_episode_files(
    dataset_dir: Path,
    val_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """按 episode 划分训练集和验证集，避免相邻帧泄漏。"""
    files = sorted(dataset_dir.glob("episode_*.npz"))
    if len(files) < 2:
        raise ValueError("至少需要 2 条 episode 才能划分训练集和验证集")
    random.Random(seed).shuffle(files)
    val_count = max(1, min(len(files) - 1, round(len(files) * val_fraction)))
    return files[val_count:], files[:val_count]


def _run_epoch(
    model: BCPolicy,
    loader: DataLoader,
    device: torch.device,
    gripper_speed: float,
    motion_weight: float,
    gripper_weight: float,
    grounding_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """运行一轮训练或验证并返回分项指标。"""
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "motion_loss": 0.0,
        "gripper_loss": 0.0,
        "grounding_loss": 0.0,
        "motion_mae": 0.0,
        "gripper_correct": 0.0,
        "grounding_correct": 0.0,
    }
    samples = 0
    for batch in tqdm(loader, desc="训练" if training else "验证", leave=False):
        image = batch["image"].to(device)
        state = batch["state"].to(device)
        tokens = batch["tokens"].to(device)
        action = batch["action"].to(device)
        target_index = batch["target_index"].to(device)
        motion_target, gripper_target = encode_action_targets(action, gripper_speed)
        with torch.set_grad_enabled(training):
            output = model(image, state, tokens)
            motion_loss = F.smooth_l1_loss(output["motion"], motion_target)
            gripper_loss = F.binary_cross_entropy_with_logits(
                output["gripper_logit"],
                gripper_target,
            )
            grounding_loss = F.cross_entropy(output["object_logits"], target_index)
            loss = (
                motion_weight * motion_loss
                + gripper_weight * gripper_loss
                + grounding_weight * grounding_loss
            )
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        batch_size = image.shape[0]
        samples += batch_size
        totals["loss"] += float(loss.item()) * batch_size
        totals["motion_loss"] += float(motion_loss.item()) * batch_size
        totals["gripper_loss"] += float(gripper_loss.item()) * batch_size
        totals["grounding_loss"] += float(grounding_loss.item()) * batch_size
        totals["motion_mae"] += (
            float((output["motion"] - motion_target).abs().mean().item())
            * gripper_speed
            * batch_size
        )
        totals["gripper_correct"] += float(
            ((output["gripper_logit"] >= 0) == (gripper_target >= 0.5))
            .sum()
            .item()
        )
        totals["grounding_correct"] += float(
            (output["object_logits"].argmax(dim=-1) == target_index).sum().item()
        )
    return {
        "loss": totals["loss"] / samples,
        "motion_loss": totals["motion_loss"] / samples,
        "gripper_loss": totals["gripper_loss"] / samples,
        "grounding_loss": totals["grounding_loss"] / samples,
        "motion_mae": totals["motion_mae"] / samples,
        "gripper_accuracy": totals["gripper_correct"] / samples,
        "grounding_accuracy": totals["grounding_correct"] / samples,
    }


def train_bc(config: dict[str, Any]) -> None:
    """训练模型，以验证损失选择最佳检查点。"""
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = torch.device(config.get("device", "cpu"))
    dataset_dir = Path(config.get("dataset_dir", "results/datasets/demo_v2"))
    info_path = dataset_dir / "dataset_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"缺少数据集元信息：{info_path}。请用新版采集脚本重新采集数据。"
        )
    dataset_info = load_json(info_path)
    if int(dataset_info.get("format_version", 0)) != 2:
        raise ValueError("数据集格式不是 v2，夹爪标签可能仍为 -1/1")
    gripper_speed = float(dataset_info["environment"]["gripper_speed"])
    train_files, val_files = _split_episode_files(
        dataset_dir,
        float(config.get("val_fraction", 0.1)),
        seed,
    )
    tokenizer = SimpleTokenizer()
    train_set = BCDataset(str(dataset_dir), tokenizer=tokenizer, files=train_files)
    val_set = BCDataset(str(dataset_dir), tokenizer=tokenizer, files=val_files)
    batch_size = int(config.get("batch_size", 32))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    sample = train_set[0]
    model = BCPolicy(
        state_dim=sample["state"].numel(),
        vocab_size=tokenizer.vocab_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )
    motion_weight = float(config.get("motion_loss_weight", 1.0))
    gripper_weight = float(config.get("gripper_loss_weight", 1.0))
    grounding_weight = float(config.get("grounding_loss_weight", 0.5))
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_state = None

    print(
        f"训练/验证 episode：{len(train_files)}/{len(val_files)}，"
        f"样本：{len(train_set)}/{len(val_set)}"
    )
    for epoch in range(int(config.get("epochs", 30))):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            gripper_speed,
            motion_weight,
            gripper_weight,
            grounding_weight,
            optimizer,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            device,
            gripper_speed,
            motion_weight,
            gripper_weight,
            grounding_weight,
        )
        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch {epoch + 1:02d}: train={train_metrics['loss']:.4f}, "
            f"val={val_metrics['loss']:.4f}, "
            f"motion_mae={val_metrics['motion_mae']:.5f}, "
            f"grip_acc={val_metrics['gripper_accuracy']:.2%}, "
            f"ground_acc={val_metrics['grounding_accuracy']:.2%}"
        )

    if best_state is None:
        raise RuntimeError("训练未产生有效检查点")
    model.load_state_dict(best_state)
    ckpt_path = Path(
        config.get("checkpoint_path", "results/checkpoints/bc_policy_v2.pt")
    )
    ensure_dir(ckpt_path.parent)
    torch.save(
        {
            "format_version": 2,
            "model": model.state_dict(),
            "state_dim": sample["state"].numel(),
            "vocab_size": tokenizer.vocab_size,
            "gripper_speed": gripper_speed,
            "environment": dataset_info["environment"],
            "best_val_loss": best_loss,
        },
        ckpt_path,
    )
    history_path = Path(
        config.get("history_path", "results/tables/train_history_v2.json")
    )
    save_json({"config": config, "history": history}, history_path)
    fig_path = Path(
        config.get("loss_curve_path", "results/figures/train_loss_v2.png")
    )
    ensure_dir(fig_path.parent)
    if plt is not None:
        plt.figure(figsize=(6, 4))
        plt.plot([row["train"]["loss"] for row in history], label="train")
        plt.plot([row["val"]["loss"] for row in history], label="validation")
        plt.xlabel("epoch")
        plt.ylabel("combined loss")
        plt.title("BC training and validation loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
    print(f"最佳模型：{ckpt_path}（val_loss={best_loss:.6f}）")
