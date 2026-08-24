# HolisticMotion

[English](README.md) | 简体中文

HolisticMotion 是一个专注的 C++17 机器人运动库，提供 Python 绑定，涵盖
URDF 机器人模型、运动学、流形数学、求解器相关算法、轨迹生成、可选的
Pinocchio/Coal 碰撞查询以及姿态重定向。

机器人资源不随库分发。所有需要模型的 API 都要求调用方传入明确的 URDF
路径，项目不会隐式下载机器人模型。

## 主要功能

- C++17 核心库和 pybind11 Python 绑定。
- URDF 解析以及数值、OPW、UR、SRS、FEP 运动学。
- 满足约束的 Double-S、梯形速度轨迹和 TOPPRA 路径时间参数化。
- 默认启用、可按需关闭的 CUDA 批处理后端。
- 默认启用、由 Conan 管理 Pinocchio 和 Coal 的碰撞查询。
- 支持单臂、双臂和全身模式的 Pinocchio/Pink 风格 retargeting 工具箱。
- 英文和简体中文文档。

## 快速开始

```bash
./scripts/build.sh
source scripts/activate.sh
```

默认构建 Python 绑定、CUDA 后端以及由 Conan 管理的 Pinocchio/Coal 碰撞组件。
不需要对应功能时使用 `--no-cuda` 或 `--no-collision`；加入 `--tests` 可运行测试。
激活脚本会同时设置本地 Python 包和 Conan 动态库路径。只运行单条命令时也可
使用 `./scripts/run.sh python3 ...`。

传入自己的 URDF 和关节构型运行碰撞检测 demo：

```bash
python3 examples/python/collision/basic_query.py \
  --urdf /绝对路径/robot.urdf -q 0 0 0 0 0 0
```

## Python 示例

```python
import holistic_motion as hm

robot = hm.Robot("/absolute/path/to/robot.urdf")
q = [0.0] * robot.dof
pose = robot.kinematics.forward(q)
solution = robot.kinematics.inverse(pose, q)
```

无需安装外部 TOPPRA 即可对路点路径进行重定时：

```python
from holistic_motion.trajectory import ToppraTrajectory

trajectory = ToppraTrajectory(
    [[0.0, 0.0], [0.4, -0.2], [1.0, 0.5]],
    max_velocity=[1.0, 0.8],
    max_acceleration=[2.0, 1.5],
)
times, positions, velocities, accelerations = trajectory.sample_uniform(200)
```

Retargeting API 位于 `holistic_motion.kit.retargeting`：

```python
from holistic_motion.kit.retargeting import PinkRetargetingSolver

solver = PinkRetargetingSolver(
    "/absolute/path/to/robot.urdf",
    frames={"left_hand": "left_ee", "right_hand": "right_ee", "head": "head_ee"},
    joint_groups={
        "left_arm": ["left_j1", "left_j2"],
        "right_arm": ["right_j1", "right_j2"],
    },
)
solver.set_mode("dual_arm")
result = solver.solve({"left_hand": left_pose, "right_hand": right_pose})
```

使用 `python -m pip install '.[retargeting]'` 安装运行时依赖。

## 文档

```bash
python -m pip install '.[docs]'
./scripts/docs.sh
```

- English 输出：`docs/_build/html/en/index.html`
- 中文输出：`docs/_build/html/zh_CN/index.html`

安装、核心概念、教程、API、架构和贡献说明请阅读[中文文档](docs/zh_CN/index.md)。

## 验证

```bash
./scripts/build.sh --tests
conan create . --no-remote -o '&:with_python=False'
```
