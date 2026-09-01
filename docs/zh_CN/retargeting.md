# 姿态重定向

<div class="language-switcher"><a href="../en/retargeting.html">English</a> · 简体中文</div>

安装 Pinocchio Python 运行时并导入工具箱：

```bash
python -m pip install '.[retargeting]'
```

```python
from holistic_motion.kit.retargeting import (
    CenterOfMassTask,
    FrameTask,
    PinkRetargetingSolver,
    PostureTask,
    SupportPolygonTask,
    ZmpTask,
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
    center_of_mass_task=CenterOfMassTask(cost=[1.0, 1.0, 0.2]),
    support_polygon_task=SupportPolygonTask(
        [[-0.12, -0.08], [0.12, -0.08], [0.12, 0.08], [-0.12, 0.08]],
        cost=10.0,
        margin=0.01,
    ),
    zmp_task=ZmpTask(cost=[1.0, 1.0]),
)
solver.set_center_of_mass_target([0.0, 0.0, 0.85])
solver.set_zmp_target([0.0, 0.0])
solver.set_center_of_mass_acceleration([0.0, 0.0, 0.0])
```

## 模式

| 模式 | 必需目标 | 激活速度变量 |
|---|---|---|
| `left_arm` | `left_hand` | 左臂关节组 |
| `right_arm` | `right_hand` | 右臂关节组 |
| `dual_arm` | 左右手 | 两个手臂关节组 |
| `left_leg` | `left_foot` | 左腿关节组 |
| `right_leg` | `right_foot` | 右腿关节组 |
| `dual_leg` | 左右脚 | 两个腿部关节组 |
| `whole_body` | 左右手和头部 | 整个模型 |
| `full_body` | 左右手、左右脚、头部和骨盆 | 整个模型 |

frame 映射与关节组会在选择对应模式时惰性校验，因此纯双臂应用无需提供虚假的
脚部或骨盆映射。腿部模式需要 `left_leg`、`right_leg` 关节组以及
`left_foot`、`right_foot` frame 映射。`whole_body` 保留原有手部/头部接口；
需要显式约束脚部和骨盆时使用 `full_body`。

`CenterOfMassTask` 会把世界坐标系质心残差和 Pinocchio 解析质心 Jacobian 加入
同一个带约束 QP。可通过 `center_of_mass_tolerance` 单独设置收敛容差，最终误差
由 `result.center_of_mass_residual` 返回。该功能提供加权质心跟踪，不能仅根据
较小的质心残差推断支撑稳定性。

`SupportPolygonTask` 为质心超出严格凸 XY 支撑多边形的情况增加可微惩罚。
顺时针顶点会自动规范化；重复、共线、凹形或非有限多边形会被拒绝。`margin`
用于收缩可用区域，`result.support_polygon_violation` 返回最大的边界违反量。
CoM 与支撑任务对每个状态共用一次质心和 Jacobian 计算。当前仍属于软 QP 目标。
设置 `reference="zmp"` 可改为约束运动学 ZMP，并要求同时配置 `ZmpTask`。若只
需要多边形不等式而不跟踪单点 ZMP，可将 ZMP 跟踪权重设为零，此时无需设置
单点 ZMP 目标。

`ZmpTask` 跟踪运动学近似
`zmp_xy = com_xy - (com_z - plane_height) * com_acceleration_xy / gravity`。
默认零加速度时，
它退化为准静态 CoM 投影。使用上面的 setter 设置 XY 目标，以及可用时的世界
坐标系 CoM 加速度；最终误差由 `result.zmp_residual` 返回。它是加权求解目标，
不是接触力、摩擦锥或完整刚体动力学约束。收敛容差只检查任务权重大于零的轴；
零权重轴仍会包含在原始残差遥测中，但不会阻止求解成功。

每次非线性求解期间，输入的 CoM 加速度会被视为常量。因此解析 ZMP Jacobian
只对 CoM 位置和高度求导，不包含与构型耦合的加速度模型。支撑平面不在世界
坐标 `z=0` 时，应设置 `plane_height`。

```python
solver.prepare("dual_arm")
result = solver.solve({
    "left_hand": left_pose,
    "right_hand": right_pose,
})

if result.success:
    send_joint_command(result.configuration)
```

应在应用初始化阶段或非实时的模式切换阶段调用 `prepare(mode)`。它会预先验证
frame 与关节组映射，并缓存 active indices、速度限制和 Pink 数值工作区。
`set_mode(mode)` 保留原有的延迟准备语义，适合逐步组装配置，但随后的第一次
求解可能承担模式准备开销。

目标、任务权重、模式描述和求解结果都会保存调用方数组与序列的不可变快照。
应用需要调整结果时，应先复制 `result.configuration`，从而避免 UI 或控制器代码
通过返回值意外修改求解器历史状态。
目标矩阵必须是有效 SE(3) 变换，包含正确的齐次末行和右手正交旋转矩阵。

求解器会把上一次结果作为热启动。传入 `seed=` 可以覆盖热启动，调用
`solver.reset()` 会恢复到 Pinocchio 中性构型。

## 求解器选择

- `PinocchioRetargetingSolver` 提供紧凑的阻尼最小二乘 IK。
- `PinkRetargetingSolver` 构建 Pink 风格的加权任务目标，并提供各向异性末端
  权重、姿态正则、LM 阻尼以及速度受限积分。
- `CuroboRetargetingSolver` 增加确定性多 seed 优化、流形上的 seed 扰动、
  最优结果选择，以及首个 seed 收敛后的可选提前退出。

cuRobo 风格求解器与 `PinkRetargetingSolver` 使用相同的 frame、关节组、任务、
模式和目标接口：

```python
from holistic_motion.kit.retargeting import CuroboRetargetingSolver

solver = CuroboRetargetingSolver(
    "/path/to/robot.urdf",
    frames=frames,
    joint_groups=joint_groups,
    num_seeds=8,
    seed_spread=0.35,
    sampler_seed=451,
)
result = solver.solve(targets, seed=current_configuration)
```

`last_seed_index` 和 `last_num_seeds_evaluated` 提供 seed 选择统计。实时场景可设置
`stop_on_success=True` 降低延迟；需要比较所有不同 seed 的解质量时保持关闭。
`num_seeds` 是候选数上限：零扰动或限位投影产生的重复 seed 会在优化前去重。
排序会先选择已收敛的 seed，随后使用优化过程中相同的加权 `objective`，而不是
用未加权的碰撞距离覆盖姿态误差权重。
`step()` 始终只使用物理 primary seed：随机替代构型无法满足相对于机器人当前
状态的单周期加速度约束。未启用加速度强制时，`solve()` 仍提供多 seed 优化。

## 碰撞代价与梯度

`PinkRetargetingSolver` 和 `CuroboRetargetingSolver` 支持可选的
`collision_cost`、`collision_gradient` 与组合 `collision_cost_gradient` 回调。
非负碰撞代价参与回溯接受和收敛
判断，其切空间梯度直接改变 IK 更新方向。未提供解析梯度时，求解器会沿激活的
速度坐标执行关节限位感知的有限差分。

球碰撞模型可以构造二次 clearance 代价；解析梯度位于 `nv` 维切空间中，也支持
流形模型的 `nq != nv`：

```python
clearance = 0.05

def collision_cost(q):
    distance = sphere_model.minimum_distance(q).distance
    return max(0.0, clearance - distance) ** 2

def collision_cost_gradient(q):
    result = sphere_model.minimum_distance_with_gradient(q)
    deficit = max(0.0, clearance - result.distance_result.distance)
    return deficit**2, -2.0 * deficit * result.gradient

solver = CuroboRetargetingSolver(
    urdf_path,
    frames=frames,
    joint_groups=joint_groups,
    collision_cost=collision_cost,
    collision_cost_gradient=collision_cost_gradient,
    collision_cost_weight=10.0,
    collision_tolerance=1e-8,
)
```

标量回调让只比较 objective 的回溯候选保持低开销；构建 QP 时，组合回调通过一次
查询同时获得距离与梯度。若代价和梯度来自不同系统，仍可使用
`collision_gradient`。

`RetargetingResult` 会报告最终加权 `objective`、`collision_cost`、
`collision_evaluations` 和 `collision_gradient_evaluations`。该功能属于可微
软约束，并不保证连续运动严格无碰撞；安全要求较高时仍需验证最终指令以及
相邻指令之间的运动。
碰撞代价和梯度会按精确构型在一次 `solve()` 的所有 seed 间共享缓存；求解结束
后立即丢弃，因此下一次调用仍能观察到动态场景变化。

Pink 与 cuRobo 风格实现均由 HolisticMotion 自身维护，运行时不导入上游
`pink`、`qpsolvers`、cuRobo、Torch 或 Warp。完整命令行示例位于
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
碰撞回调是可选项；未配置时 retargeting 不执行碰撞检测。即使启用了软碰撞
代价，如果应用需要硬性拒绝碰撞指令，仍应在接受结果前使用可选的 C++/Python
`CollisionModel` 组件。
```
