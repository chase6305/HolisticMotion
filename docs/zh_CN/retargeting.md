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

## 可选 dex-retargeting 灵巧手后端

`DexHandRetargetingSolver` 将真实的 dex-retargeting 0.5.0 `VectorOptimizer`
接入重映射模块，独立求解手指关节。使用 Python 3.9–3.12 安装可选依赖；
上游包暂不接受 Python 3.13 及以上版本：

```bash
python -m pip install '.[hand-retargeting]'
```

如需使用本地源码，先执行 `python -m pip install /path/to/dex-retargeting`；
适配层面向 0.5.0 版本。可选依赖显式补上上游已使用但未声明的 PyTorch，
CPU 即可运行，不要求 CUDA。普通工具箱导入不会加载 dex-retargeting、
NLopt 或 PyTorch。

```python
import numpy as np
from holistic_motion.kit.retargeting import DexHandRetargetingSolver

hand = DexHandRetargetingSolver(
    "/path/to/hand.urdf",
    joint_names=["thumb_joint", "index_joint"],  # 参与优化的独立关节
    origin_links=["palm", "palm"],
    task_links=["thumb_tip", "index_tip"],
    keypoint_pairs=[[0, 1], [0, 2]],  # 每行对应一组机器人 link 向量
    scaling=1.0,
    temporal_weight=4e-3,
    vector_tolerance=0.02,
    max_evaluations=100,
)
points = np.load("/path/to/keypoints.npy", allow_pickle=False)  # (N, 3)
result = hand.solve(points)
if result.success:
    hand_positions = dict(result.joint_positions)
else:
    print(result.termination_reason, result.message)
```

关节名、link 名和关键点索引须替换为实际模型的对应关系。输入关键点使用
手模型基座的坐标轴方向和长度单位（通常为米），先完成腕部方向对齐。
目标比较关键点之差，因此整体平移会抵消；`scaling` 乘在人手向量上。
时序正则项惩罚相对本次 seed 的关节位移，适配层保证其标量值与梯度一致。
`objective_tolerance` 默认 `1e-9`，控制目标函数绝对变化量，相比上游 `1e-6`
减少提前停止。它与检查最大几何误差的 `vector_tolerance` 分开设置，优化器
评估次数仍受 `max_evaluations` 限制。

可在两次求解之间修改 `scaling`、`temporal_weight`、`huber_delta`、
`vector_tolerance` 和 `objective_tolerance`。下一次 `solve()` 先校验全部
五项设置，再同步优化器的损失、梯度和停止容差。非法设置会在修改后端参数
或热启动历史前报错，修正对应属性后即可重试；正在求解时应保持参数不变。

`solve()` 可在 `torch.no_grad()` 或 `torch.inference_mode()` 中调用，便于
衔接关键点模型推理。求解器也可在这些上下文中创建，缓存索引使用普通张量，
确保后续反向传播可用。适配器仅在 NLopt 请求导数时局部开启自动微分，
仅计算标量值时不构建梯度图；正常返回或异常退出都会恢复调用方的梯度与
推理模式。该梯度仅用于手部优化，返回的 NumPy 结果不提供回传到关键点
模型的端到端梯度。

传入独立的标量关节手部 URDF，支持有限位的转动/移动关节，不支持浮动、
球关节和连续转动关节。直接 mimic 从动关节由参与优化的主关节推导，
不应写入 `joint_names`；从动限位同时约束主关节，不使用上游额外限位扩张。
嵌套 mimic 和由未参与优化关节驱动的 mimic 会被拒绝。
主关节区间按 float64 先乘后加的实际映射与从动限位求交，覆盖从动锁定和
极小非零耦合系数，避免逆算边界的舍入或溢出误拒绝可行姿态。没有可表示
可行主关节位置的区间仍会被拒绝。
显式 `seed` 用名称映射提供全部独立手部关节位置，未参与优化的关节保持
该值。省略 seed 时沿用上次成功结果，首次使用经限位夹取的中性姿态。
`reset()` 恢复中性姿态，`reset(configuration)` 接受校验后的命名配置。

`HandRetargetingResult` 返回按名称组织的完整手部位置（含 mimic 和未优化
关节）、最大向量残差、目标函数值、耗时、评估次数、原生状态与终止原因。
输入 seed 已满足向量容差时直接保持原姿态，返回 `converged`、零评估次数
和原生状态零（未调用优化器）。全部活动关节上下界相同，包括由 mimic
限位导致的锁定时，未达标目标返回 `no_feasible_motion`，同样不调用优化器。
这两条路径跳过 Jacobian，保留输入姿态；成功保持会成为下一次热启动。
后续目标需要运动时恢复正常优化，诊断不会沿用之前求解的状态或评估次数。

需要优化时，只有优化器收敛且残差达到容差才成功。预算耗尽、非法输出和原生求解失败
均明确报告失败，此时返回位置和残差描述输入 seed，保留此前热启动；
回调异常向上传播。
每次原生优化在完成、失败或中断时解除当前帧回调，避免原生优化器通过该
回调持续持有手部求解器、旧目标或异常堆栈。后续运动帧重新注册目标函数，
返回结果仍保留原生状态码和评估次数。这解决对象生命周期问题，不保证
Python 或原生内存分配器立即将已释放空间归还操作系统。

运行 `python examples/python/retargeting/dex_hand.py --help` 查看命令行示例，
显式传入 URDF、向量对应关系、关键点 `.npy`，可选传入 seed JSON 对象。
求解失败时退出状态为 1。

腰臂手协同时，现有求解器处理腰部与手臂姿态，每只手分别使用一个手部
求解器，再按名称合并成功的手指结果；不要用手部结果中的固定腕关节条目
覆盖腰臂结果。手部与整机 URDF 的关节名应一致，或由调用方显式提供映射。
该后端只提供位置重映射，组合后仍需轨迹限速及整机碰撞校验，不承担 WBC、
速度/加速度约束或碰撞检测。

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
| `torso` | `torso` | 躯干关节组 |
| `torso_left_arm` | `left_hand` | 躯干和左臂关节组 |
| `torso_right_arm` | `right_hand` | 躯干和右臂关节组 |
| `torso_dual_arm` | 左右手 | 躯干和两个手臂关节组 |

frame 映射与关节组会在选择对应模式时惰性校验，因此纯双臂应用无需提供虚假的
脚部或骨盆映射。腿部模式需要 `left_leg`、`right_leg` 关节组以及
`left_foot`、`right_foot` frame 映射。`whole_body` 保留原有手部/头部接口；
需要显式约束脚部和骨盆时使用 `full_body`。

Pink 和 cuRobo 风格求解器的 `FrameTask` 分别计算笛卡尔位置误差与旋转向量
误差。目标位姿使用世界坐标系；分轴代价沿当前被跟踪 frame 的坐标轴施加，
与局部 Jacobian 一致。设置 `orientation_cost=0` 后，位置跟踪不受目标朝向
影响。零代价轴不阻止收敛，但仍包含在原始遥测中。`position_residual` 及
各目标的位置残差表示欧氏距离，姿态残差表示旋转角度，单位为弧度。
位置项不再使用 SE(3) 对数的平移部分，因此同时存在位置与姿态误差时，
目标函数数值和候选排序可能改变。

frame 各轴代价不同时，任务 Jacobian 会计入当前位置误差坐标轴的转动以及
旋转向量的导数，使更新梯度与加权目标一致。位置或姿态各轴等权重的部分
保留原有速度任务更新方式。局部单轴目标可能通过转动坐标轴满足，而完整
末端距离没有减小；原始位置遥测仍报告完整距离。

任务 `gain` 缩放 QP 的期望修正量。回溯与候选排序使用的 `objective` 对每个
跟踪或姿态偏好项只乘一次增益，即 `0.5 * gain * ||weighted_error||²`，再加
独立的碰撞代价。因此不同任务的增益也会影响冲突目标之间的折中。QP 的
Hessian 与原有按增益缩放的 LM 阻尼保持原有行为。`residual` 表示按代价
加权的跟踪误差范数，不含增益、姿态偏好或碰撞项；收敛仍使用实际误差容差。
相较以前的增益处理，非单位增益可能改变目标函数、残差、离线更新接受结果
和候选排序；所有增益均为默认值 1 时保持不变。

将质心、ZMP 跟踪或支撑多边形任务的全部代价设为零，会关闭该任务的求解与
诊断计算，不再要求其跟踪目标。若没有其他启用任务依赖质心，也不会计算
质心或其 Jacobian，因此不需要为纯运动学模型补充质量信息。
关闭后的 CoM/ZMP 残差为 `NaN`，支撑违反量为 `0`，这些默认值不代表测量
结果或稳定性保证。部分轴代价为零、但任务仍启用时，保留完整残差遥测。
启用的 ZMP 支撑约束仍需要 `ZmpTask` 提供支撑平面和重力参数，即使该对象的
点跟踪代价为零；完全关闭的 ZMP 支撑任务不要求这个依赖。

`CenterOfMassTask` 会把世界坐标系质心残差和 Pinocchio 解析质心 Jacobian 加入
同一个带约束 QP。可通过 `center_of_mass_tolerance` 单独设置收敛容差，最终误差
由 `result.center_of_mass_residual` 返回。该功能提供加权质心跟踪，不能仅根据
较小的质心残差推断支撑稳定性。

`SupportPolygonTask` 为质心超出严格凸 XY 支撑多边形的情况增加可微惩罚。
顺时针顶点会自动规范化；重复、共线、凹形或非有限多边形会被拒绝。`margin`
用于收缩可用区域，`result.support_polygon_violation` 返回最大的边界违反量。
CoM 与支撑任务对每个状态共用一次质心和 Jacobian 计算。当前仍属于软 QP 目标。
凸性检查使用单位边方向与归一化的相对坐标，其容差不依赖长度单位；跨度、
边长或平面偏移无法表示为有限浮点数时会拒绝输入。边界距离先减去对应顶点
再投影，减少远离世界原点时的相消误差，但不能恢复输入坐标已经丢失的精度。
设置 `reference="zmp"` 可改为约束运动学 ZMP，并要求同时配置 `ZmpTask`。若只
需要多边形不等式而不跟踪单点 ZMP，可将 ZMP 跟踪权重设为零，此时无需设置
单点 ZMP 目标。

`ZmpTask` 跟踪运动学近似
`zmp_xy = com_xy - (com_z - plane_height) * com_acceleration_xy / gravity`。
默认零加速度时，
它退化为准静态 CoM 投影。使用上面的 setter 设置 XY 目标，以及可用时的世界
坐标系 CoM 加速度；最终误差由 `result.zmp_residual` 返回。它是加权求解目标，
不是接触力、摩擦锥或完整刚体动力学约束。收敛容差只检查任务权重大于零的轴；
跟踪任务仍启用时，零权重轴会包含在原始残差遥测中，但不会阻止求解成功。

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
基础 `PinocchioRetargetingSolver` 在结果校验通过后才更新历史；非有限诊断
被拒绝或结果构造中止时，即使显式传入了 seed，也保留之前的热启动配置。
正常返回但已耗尽迭代预算的结果仍会作为下一次 seed。
残差范数使用避免溢出的累积方式，因此很大的有限位姿误差仍可作为失败诊断
返回。若残差可表示而它的平方目标值不可表示，`result.objective` 为 `NaN`；
残差和配置仍有效，正常返回的配置会成为下一次热启动。

配置会先按模型流形归一化，再施加坐标限位，避免裁剪放大的连续关节正余弦
或浮动关节四元数时改变朝向；标量关节仍按位置限位夹取。全零朝向坐标、
以及数值上无法归一化的配置会抛出 `ValueError`。非法 seed、重置配置或姿态
偏好不会改变先前配置、速度历史或偏好目标，也不会修改调用方的输入数组。

## 腰臂协同与先转腰策略

腰臂联合模式把 `torso` 与参与操作的手臂关节放进同一次 IK，共享腰部变量
只出现一次。调用方显式指定 `torso` 关节组，可包含一个或多个躯干关节。
这些模式只要求对应的手部目标，不激活未列入关节组的关节；单独对齐躯干时，
才需要额外的 `torso` frame 映射。

```python
from holistic_motion.kit.retargeting import (
    FrameTask, PinkRetargetingSolver, PostureTask, solve_torso_first,
)

solver = PinkRetargetingSolver(
    "/path/to/robot.urdf",
    frames={"torso": "torso_link", "left_hand": "left_ee", "right_hand": "right_ee"},
    joint_groups={
        "torso": ["waist_yaw"],
        "left_arm": left_arm_joint_names,
        "right_arm": right_arm_joint_names,
    },
    frame_tasks={"torso": FrameTask(position_cost=0.0)},
    posture_task=PostureTask(cost=1e-3, joint_costs={"waist_yaw": 0.01}),
    joint_motion_costs={"waist_yaw": 0.1},
)
solver.set_posture_target(preferred_configuration)
solver.prepare("torso_dual_arm")
result = solver.solve(
    {"left_hand": left_pose, "right_hand": right_pose}, seed=current_configuration,
)
```

示例关节名和 link 名须替换为调用方 URDF 中的名称。姿态偏好使用完整模型
配置，末端目标使用世界坐标系 SE(3)。权重仅用于示意，需要按任务调节；
较强的腰部姿态偏好可能与手部目标冲突。

`PostureTask.joint_costs` 按关节名覆盖默认 `cost`，作用于该关节的所有切空间
自由度；默认 `cost=0` 时仍可单独启用关节偏好。`joint_motion_costs` 为每次
局部 QP 增加 `0.5 * sum((cost * delta_q) ** 2)` 正则项，未指定关节默认
为零。两类映射都保存快照，按 `nv` 解析，支持 `nq != nv` 的模型。
运动惩罚影响局部更新，不计入返回的 `objective` 或多初值排序，不约束全程
位移，也不代替速度、加速度限制。姿态偏好是软目标，不单独阻止收敛；手部等
任务已满足容差时，求解器可以直接返回，而不继续优化腰部偏好。

严格先后顺序使用两阶段入口：

```python
stages = solve_torso_first(
    solver,
    torso_target=desired_torso_pose,
    hand_targets={"left_hand": left_pose, "right_hand": right_pose},
    seed=current_configuration,
    arm_mode="dual_arm",  # 也可选 left_arm 或 right_arm
)
if stages.success:
    aligned_configuration = stages.torso.configuration
    manipulation_configuration = stages.arms.configuration
```

第一阶段固定手臂关节，双手随躯干移动；第二阶段固定躯干，仅求所选手臂。
两阶段活动关节必须互不重叠。转腰失败时 `stages.arms` 为 `None`；手臂失败
仍保留两阶段诊断。函数始终恢复原模式，正常返回后热启动为最后一次尝试的
配置，异常时恢复配置、速度历史及多初值候选索引与数量。这也包括
`KeyboardInterrupt` 和 `SystemExit`；回滚后原中断继续向上传播。
它支持 cuRobo 风格求解器，其中性和随机
候选均只改变当前模式的活动关节。

`seed` 必须显式提供有效配置；传入 `None` 会报错，不会退回中性姿态。
`TorsoFirstResult` 要求第一阶段为 `torso`，第二阶段为单臂或双臂模式，且两阶段
配置维度一致。普通不收敛仍返回最后尝试的诊断；抛异常时才回滚到调用前状态。

返回值是离线 IK 路点。调用方仍需规划并检查两段路径，执行时确认躯干到位
后才进入手臂阶段。此策略不提供转腰期间的世界坐标手部保持、刚性双手抓握
约束、接触力或动力学平衡保证。需要同时补偿手部运动时，使用腰臂联合模式
并持续提供手部目标，路径约束仍需另行检查。

经典命令行示例可在调用方显式提供的 URDF 上运行同一流程。一个关节组包含
多个关节时，重复对应的关节选项：

```bash
./scripts/run-python-toolkit.sh python3 \
  examples/python/retargeting/classic_torso_first.py \
  --urdf /absolute/path/to/robot.urdf \
  --torso-frame torso_link --torso-joint waist_yaw \
  --left-frame left_tool --left-joint left_shoulder --left-joint left_elbow \
  --right-frame right_tool --right-joint right_shoulder --right-joint right_elbow \
  --joint-delta waist_yaw=-0.2 --joint-delta left_elbow=0.4
```

默认模式为 `dual_arm`，也可改为 `left_arm` 或 `right_arm`。示例从中性构型
叠加标量关节增量生成可达目标，并拒绝超出 URDF 限位的增量。
`--torso-delta` 和 `--arm-delta` 提供组内默认值；重复传入
`--joint-delta JOINT=DELTA` 可覆盖指定的已配置关节。输出仍是离线
IK 路点，并非经过碰撞检查、可直接执行的路径。每个阶段会输出收敛诊断、
求解耗时、带名称的活动关节值和完整构型，便于进入规划前检查结果。

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
候选按主初值、中性初值、确定性随机初值的顺序按需生成。提前成功时既不
求解、也不生成剩余候选；完整搜索保留原有顺序和排序规则。后续候选生成
或求解发生异常时，配置、速度历史及候选统计恢复为本次调用前的状态。
Pink 和 cuRobo 风格的 `solve()`/`step()` 遇到 `KeyboardInterrupt` 或
`SystemExit` 时也会回滚，然后继续向上传播原中断。单步求解会恢复此前的
速度历史，后续调用不会继承尚未返回的更新。
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

即使同一模式中还有可动关节，数值差分也会跳过零速度限位或标量位置上下界
相同的自由度。这些固定分量在内部数值梯度中置零，不做采样积分和代价查询；
可动分量保留原有采样与缓存。解析及组合回调仍需返回完整 `nv` 维梯度。
`limit_hits` 比较约束增量与无约束增量，因此固定分量梯度改变后该诊断计数
可能不同。

限位裁剪使两侧实际步长不同时，差分使用当前点与两侧采样的三点公式，在
当前配置处估计导数，避免将采样区间中点的斜率误用于当前配置。两侧等距时
保留中心差分，恰好位于限位时保留单边差分并复用当前代价；不增加回调次数。

采样有效性按相对尺度判断，不使用固定的关节位移绝对阈值。若一侧正步长
小于另一侧的 `sqrt(machine epsilon)` 倍，则使用较长一侧的单边差分，
避免放大代价舍入误差；否则保留可表示的非零位移，包括小范围关节的采样。
因舍入而回到当前配置的采样继续复用当前代价。

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

`solve(..., enforce_acceleration=True)` 同样按单周期处理：即使 `max_iterations`
或构造时的迭代预算大于一，也最多执行一次 QP 更新。预算仍须是合法的正整数。
这样返回的总位移和保存的指令速度对应同一个 `integration_dt`，未显式配置
加速度限制时也适用。下一周期应再次调用 `step()` 并传入新的实测配置。
离线迭代收敛继续使用默认的 `enforce_acceleration=False`；离线结果不能直接
当作单周期运动指令。

模型含连续转动或浮动关节、使 `nq != nv` 时，标量转动与移动关节的位置限位
仍按各自的配置索引映射到 QP 速度变量。手臂接近限位时，联合求解可将剩余
动作分配给腰部。此映射不把连续关节的正余弦坐标或姿态四元数当成标量关节
限位。关节组和命名加速度限制都拒绝 `universe` 占位关节。

`step()` 会联合关节位置和速度限制约束构型增量；配置 `acceleration_limits` 后，
还会限制相对上一周期指令速度的变化。Viser Demo 中对应 `Single QP step` 策略，
离线或事件驱动收敛仍可选择 `Iterative solve`。

对于 Pink 和 cuRobo 风格求解器，零速度限位会把对应自由度固定在输入 seed
经归一化及限位夹取后的位置，不会解释为无限速。多初值的中性和随机候选也
不改变零限速自由度。活动速度限位为负值或 NaN 时，模式准备或求解会报错；
正无穷仍表示不设速度上限。准备失败或中断后恢复原模式，重复调用会重新校验，
不会使用尚未完整建立的限速缓存。

若关节边界只允许零增量，求解器只评估一次当前残差和标量碰撞代价，跳过
Jacobian、碰撞梯度与 QP 更新。目标未满足时返回 `no_feasible_motion`；
没有活动自由度的模式仍返回 `no_active_dofs`；任务已满足时返回 `converged`。
这些情况均保留当前配置，接受步数为零。`step()` 的保持指令会清零速度历史，
离线 `solve()` 则保留原历史。被边界强制要求的非零制动增量仍走正常约束更新。

盒约束 QP 使用保持可行的活动集更新：向约束下的局部最优解移动时，只走到
首先触及的关节限位，再重新求解其余自由度，避免对耦合关节分别夹取造成
活动集反复切换。上下界相同的自由度在整个求解过程中保持固定。

离线 `solve()` 允许 `step_size > 1`，但回溯的初始倍率最多放大到首先触及的
关节增量边界。所有关节使用同一倍率保持 QP 搜索方向，随后从这个可行倍率
逐次减半；整个增量仍在边界内时保留放大能力。因此即使配置较大步长，每次
局部迭代仍遵守位置和速度增量边界。离线多次迭代的总结果仍不是单周期指令；
`step()` 保持原有不经倍率放大的约束更新。

即使输入姿态已满足任务容差，`step()` 也会检查零增量是否满足当前活动关节的
速度与加速度边界。无法在一个周期内停住时，仍通过 QP 求受约束增量；可以
保持不动时，返回原配置并将指令速度历史清零，避免下一次启动沿用旧速度。
`result.success` 仍表示返回姿态满足任务容差，不表示机器人已经停止。
模式切换会固定未激活关节，其减速过渡需要调用方安排；离线 `solve()` 不承担
逐周期速度历史更新，重新开始连续调用前可用 `reset(configuration)` 清零历史。

```{warning}
碰撞回调是可选项；未配置时 retargeting 不执行碰撞检测。即使启用了软碰撞
代价，如果应用需要硬性拒绝碰撞指令，仍应在接受结果前使用可选的 C++/Python
`CollisionModel` 组件。
```
