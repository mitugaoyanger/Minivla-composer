# MiniVLA-Composer

MiniVLA-Composer 是一个适合 Windows + PyCharm 本地运行的轻量级 VLA 学习项目。它不依赖真实机械臂、相机、MuJoCo、LIBERO 或大模型，而是在二维桌面仿真环境中学习“图像 + 语言 + 状态 -> 动作”的行为克隆策略。

## 为什么做这个项目

真实 VLA 系统通常涉及机器人硬件、仿真器、视觉编码器、语言模型和策略学习，初学者很难一次性搭起来。本项目把问题缩小到可观察、可调试、可训练的二维桌面任务，帮助你先理解数据采集、语言条件、状态表示、动作预测和闭环评估的完整链路。

## 和真实 VLA 的关系

真实 VLA 会根据图像、语言指令和机器人状态输出动作。MiniVLA-Composer 保留了这个核心接口：环境返回 RGB 图像、数值状态和英文任务指令，模型输出 `[dx, dy, gripper]`。区别是这里的世界是二维简化版，动作也是夹爪在平面上的小位移。

## 为什么第一版不使用 MuJoCo / LIBERO / 大模型

第一版目标是最小可运行 MVP。MuJoCo 和 LIBERO 会带来安装、资产、任务接口和调试复杂度；大模型会带来显存、下载和推理成本。本项目先用 Pillow 渲染、规则语言解析、小型 CNN 和 Embedding 编码器，保证 Windows + PyCharm 上容易运行。

## 项目结构

```text
MiniVLA-Composer/
  configs/                 # 环境、数据、训练、评估配置
  mini_vla_composer/
    env/                   # 二维桌面环境和渲染器
    language/              # 指令生成、解析和子目标
    expert/                # 脚本专家和运动原语
    data/                  # 数据采集、Dataset、可视化
    models/                # 视觉编码器、语言编码器、BC 策略
    train/                 # 训练逻辑
    eval/                  # 闭环评估和指标
    utils/                 # 配置、IO、随机种子
  scripts/                 # 命令行入口
  results/                 # 数据、模型、图像和评估输出
```

## 环境安装

建议使用 Python 3.10 或 3.11。在 PyCharm 中打开 `C:\VLA\MiniVLA-Composer`，创建虚拟环境后运行：

```bash
pip install -r requirements.txt
```

## 快速开始

第一步：采集专家数据

```bash
python scripts/collect_data.py --config configs/data.yaml
```

第二步：可视化一条数据

```bash
python scripts/visualize_episode.py --episode results/datasets/demo_v2/episode_000001.npz
```

第三步：训练 BC 模型

```bash
python scripts/train_bc.py --config configs/train_bc.yaml
```

第四步：评估模型

```bash
python scripts/evaluate_bc.py --config configs/eval.yaml
```

随机选择一条闭环任务并动态播放：

```bash
python scripts/visualize_eval.py
```

脚本会打印本次任务的随机 `seed`。使用 `--seed 123` 可以复现同一任务，使用
`--save-gif results/figures/eval.gif` 可以额外保存动画。

运行回归测试：

```bash
python -m unittest discover -s tests -v
```

## 数据格式

每条 episode 保存为一个 `.npz` 和一个 `.json`。

`.npz` 包含：

- `images`: `T x H x W x 3` 的 RGB 图像。
- `states`: `T x state_dim` 的状态向量。
- `actions`: `T x 3` 的动作，分别是 `dx, dy, gripper`。

`.json` 包含：

- `instruction`: 英文任务指令。
- `target_color`: 目标颜色。
- `target_shape`: 目标形状。
- `success`: 专家是否成功。
- `num_steps`: episode 步数。
- `subgoals`: 固定子目标列表。

## 环境状态说明

桌面坐标范围是 `[0, 1] x [0, 1]`。状态包含夹爪、目标区，以及每个物体的位置、抓取状态、颜色 one-hot 和形状 one-hot。

## 动作空间说明

动作由二维连续位移和一个离散夹爪状态组成：

```text
[dx, dy, gripper]
```

`dx, dy` 表示夹爪的小位移；`gripper=0` 表示打开，`gripper=1` 表示闭合。位移使用 Smooth L1 回归，夹爪使用二分类交叉熵。

## 模型结构说明

模型由三部分组成：

- `VisionEncoder`: 小型 CNN，把 RGB 图像编码为视觉特征。
- `LanguageEncoder`: 简单词表 + Embedding + 平均池化，把指令编码为语言特征。
- `BCPolicy`: 用语言条件注意力定位目标物体，再由独立的位移回归头和夹爪分类头输出动作。

## 评估指标说明

评估脚本会输出：

- `success_rate`: 成功率。
- `average_steps`: 平均步数。
- `timeout_rate`: 超时比例。
- `average_final_distance`: 目标物体到目标区的最终平均距离。
- `action_smoothness`: 相邻动作变化的平均范数。
- `average_gripper_switches`: 每回合夹爪开合切换次数。

当前配置在固定测试种子 `123` 的 30 条随机布局任务上达到 `100%` 成功率，平均约 `24.87` 步。该数字是本地轻量基准结果，不代表真实机器人性能。

## 当前版本局限

当前版本只是可运行 MVP：语言解析是规则匹配，专家策略比较简单，渲染是二维俯视图，BC 模型只预测单步动作，没有 action chunk、历史帧或扩散策略。

## 后续扩展方向

- 加入 Action Chunk 模型。
- 加入语言改写测试。
- 加入组合泛化测试。
- 加入子目标增强模型。
- 后续迁移到 MuJoCo 机械臂环境。
- 后续接入简化 Diffusion Policy。

## 工程参考

项目保持独立的二维轻量实现，未引入其 MuJoCo、LeRobot 或模型依赖。
