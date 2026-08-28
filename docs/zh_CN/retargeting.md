# 姿态重定向

<div class="language-switcher"><a href="../en/retargeting.html">English</a> · 简体中文</div>

安装 Pinocchio Python 运行时并导入工具箱：

```bash
python -m pip install '.[retargeting]'
```

```python
from holistic_motion.kit.retargeting import (
    FrameTask,
    PinkRetargetingSolver,
    PostureTask,
)

solver = PinkRetargetingSolver(
    "/path/to/robot.urdf",
    frames={
        "left_hand": "left_ee",
        "right_hand": "right_ee",
        "head": "head_ee",
    },
    joint_groups={
        "left_arm": ["left_j1", "left_j2", "left_j3"],
        "right_arm": ["right_j1", "right_j2", "right_j3"],
    },
    frame_tasks={
        "left_hand": FrameTask(position_cost=1.0, orientation_cost=0.2),
        "right_hand": FrameTask(position_cost=1.0, orientation_cost=0.2),
    },
    posture_task=PostureTask(cost=1e-3),
)
```

## 模式

| 模式 | 必需目标 | 激活速度变量 |
|---|---|---|
| `left_arm` | `left_hand` | 左臂关节组 |
| `right_arm` | `right_hand` | 右臂关节组 |
| `dual_arm` | 左右手 | 两个手臂关节组 |
| `whole_body` | 左右手和头部 | 整个模型 |

```python
solver.set_mode("dual_arm")
result = solver.solve({
    "left_hand": left_pose,
    "right_hand": right_pose,
})

if result.success:
    send_joint_command(result.configuration)
```

目标、任务权重、模式描述和求解结果都会保存调用方数组与序列的不可变快照。
应用需要调整结果时，应先复制 `result.configuration`，从而避免 UI 或控制器代码
通过返回值意外修改求解器历史状态。

求解器会把上一次结果作为热启动。传入 `seed=` 可以覆盖热启动，调用
`solver.reset()` 会恢复到 Pinocchio 中性构型。

## 求解器选择

- `PinocchioRetargetingSolver` 提供紧凑的阻尼最小二乘 IK。
- `PinkRetargetingSolver` 构建 Pink 风格的加权任务目标，并提供各向异性末端
  权重、姿态正则、LM 阻尼以及速度受限积分。

Pink 风格实现由 HolisticMotion 自身维护，运行时不导入上游 `pink` 或
`qpsolvers`。完整命令行示例位于
`examples/python/retargeting/pink_dual_arm.py`。

如果碰撞组件使用 Conan 的 Pinocchio，而 Python Pinocchio 来自 `pin` wheel，
不要让两套原生动态库进入同一个进程。使用专用启动器只加载纯 Python 工具箱：

```bash
./scripts/run-python-toolkit.sh python3 \
  examples/python/retargeting/pink_dual_arm.py --help
```

通过左右手和头部 gizmo 交互执行 URDF retargeting：

```bash
python3 examples/python/visualization/pink_robot_viser.py \
  --urdf /path/to/robot.urdf
```

Viser 面板支持左臂、右臂、双臂和全身模式，以及连续求解、目标复位、残差、
迭代次数和求解耗时显示。
求解器使用独立的位置与姿态容差、自适应阻尼、回溯步长接受、速度裁剪和停滞
检测。面板会显示每个目标的误差和终止原因；不可达目标每次 gizmo 事件只求解
一次，不会在每个渲染帧重复占用计算资源。

实时控制循环可以在每个周期只执行一次带约束 QP：

```python
result = solver.step(targets, seed=current_configuration)
```

`step()` 会联合关节位置和速度限制约束构型增量；配置 `acceleration_limits` 后，
还会限制相对上一周期指令速度的变化。Viser Demo 中对应 `Single QP step` 策略，
离线或事件驱动收敛仍可选择 `Iterative solve`。

```{warning}
通用 retargeting 求解器本身不执行碰撞检查。如果应用需要拒绝碰撞指令，请在
接受结果前使用可选的 C++/Python `CollisionModel` 组件。
```
