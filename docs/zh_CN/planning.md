# 采样式路径规划

<div class="language-switcher"><a href="../en/planning.html">English</a> · 简体中文</div>

HolisticMotion 自主实现了一套紧凑的 C++17 关节空间规划算法，不链接、不内置，
也不导入 OMPL。目前有意只提供 `RRT_CONNECT`、`RRT_STAR` 和
`INFORMED_RRT_STAR` 三种算法。

## 职责划分

采样规划器生成无碰撞几何路径，shortcut 删除不必要的路径点，随后应由 TOPPRA
施加速度和加速度约束。控制器不属于本库范围。

单次机器人规划默认推荐 `RRT_CONNECT`。`RRT_STAR` 和
`INFORMED_RRT_STAR` 会持续使用完整时间预算优化路径，因此即使提前找到初始解，
求解耗时通常仍接近 `timeout_seconds`。

三种算法在扩展随机树之前都会完整检查起点到终点的直连边。如果整条边有效，
规划器会立即返回仅含两个点的最短路径；只有障碍物阻断直连时，优化型算法才会
使用完整时间预算。HolisticMotion 当前只报告精确解，超时和失败结果不会夹带
语义不明确的部分路径。

## Python 示例

```python
collision = hm.CollisionModel(urdf_path, [str(urdf_path.parent)])
joint_names = left_arm_joints + right_arm_joints
context = collision.configuration_from_joint_positions(start_positions)

planner = hm.SamplingPlanner.from_collision_joints(
    collision, joint_names, context, security_margin=0.005
)
options = hm.PlanningOptions()
options.algorithm = hm.SamplingAlgorithm.RRT_CONNECT
options.timeout_seconds = 2.0
options.extension_range = 0.2
options.edge_resolution = 0.03
options.random_seed = 42

result = planner.plan(start, goal, options)
if not result.success:
    raise RuntimeError(result.message)
```

`from_collision_joints` 只搜索指定的标量关节；每次 Coal 查询都会从 `context`
恢复未激活关节和流形关节。这是双臂规划推荐接口。左右臂应当放在同一个 active
space 中，才能在每条搜索边上正确检查双臂之间的碰撞。

通用构造函数也接受 validator 回调：

```python
planner = hm.SamplingPlanner(lower, upper, lambda q: is_valid(q))
```

该方式适合测试和非 Coal 环境。机器人规划应优先使用原生碰撞工厂，以免每个采样
构型都跨越 Python 边界。

可选的 `security_margin` 不仅拒绝已经穿透的构型，也会拒绝 collision mesh
间距小于指定距离的构型。Marvin Viewer 默认同时检查双臂互碰、各臂自碰以及
手臂与躯干，并会复检每一个实际显示的路径采样点。

## 保持可行性的路径优化

`PathOptimizer` 为已有可行几何路径提供轻量后处理。它固定起点和终点，迭代降低
关节加权的路径长度与二阶差分平滑度组合目标。只有目标下降且相邻两条边都通过
碰撞 validator 时，候选路点才会被接受。

```python
optimizer = hm.PathOptimizer.from_collision_joints(
    collision,
    joint_names,
    context,
    security_margin=0.005,
    clearance=0.02,
)
optimization = hm.PathOptimizationOptions()
optimization.timeout_seconds = 0.5
optimization.smoothness_weight = 0.5
optimization.state_cost_weight = 2.0
optimized = optimizer.optimize(result.path, optimization)
if not optimized.success:
    raise RuntimeError(optimized.message)
path = optimized.path
```

`security_margin` 是不可越过的硬可行边界；`clearance` 是软目标。最小碰撞距离
低于它时，原生适配器会加入平方 hinge 代价。优化器使用中心有限差分
（`finite_difference_step`）估计代价梯度，再与解析几何梯度组合；
`state_cost_step_size` 在组合方向归一化前缩放 clearance 梯度。保持
`clearance=0` 和 `state_cost_weight=0` 会完全关闭该阶段。
相邻迭代会交替采用正向和反向路点扫描，以减小原地更新产生的方向偏差。

通用接口可通过 `optimizer.set_state_cost(callable)` 使用相同机制。代价必须有限、
非负，并适合被重复调用。统计中的 `state_cost_evaluations` 与碰撞可行性检查分开
计数，`line_search_evaluations` 则记录实际评估的候选步长数量。
`iterations` 统计实际进入的每轮优化，包括最后一次收敛或超时的 sweep。

如果应用已有解析梯度、自动微分梯度或基于运动学 Jacobian 的梯度，还可以调用
`optimizer.set_state_cost_gradient(callable)`。回调应为每个 active joint 返回一个
有限梯度值。它会替代默认中心有限差分，把每个候选路点的梯度工作从 `2 * DoF`
次代价查询降为一次梯度调用；`state_cost_gradient_evaluations` 会单独统计该路径。
不设置回调时仍自动使用有限差分。
在有限关节的边界上，差分采样如果被截回当前状态，会复用该路点已缓存的代价。

超时时仍返回当前最佳可行路径。非法输入路径会被拒绝而不是尝试修复，因此采样
规划仍负责找到初始可行同伦路径。该设计参考 cuRobo 中 seed 生成、可行性优化和
最佳解跟踪的分层，但不导入其 Torch、Warp 或 CUDA 优化器栈。之后仍由 TOPPRA
完成速度和加速度时间参数化。
如果在全部初始 state cost 计算完成前超时，优化器会返回已经验证的输入路径；由于
完整目标从未得到计算，两个 objective 统计值均为 `NaN`。

路点更新采用增量目标计算：移动一个内部路点时，只重算相邻的两项路径长度和最多
三个受影响的二阶差分项。每轮外层迭代再完整计算一次目标，以限制浮点累计误差。
该次扫描会复用相邻差分并同时累计长度和平滑度。因此每轮几何目标维护随路点数量
线性增长，而不是二次增长；碰撞和 state-cost
回调仍是主要计算开销。

每个路点更新都会执行有界回溯搜索：`line_search_steps` 控制最多尝试的步长
数量，`line_search_decay` 控制每次重试的缩放比例。初始步长发生碰撞或越过
下降区间时，优化器会尝试更小的可行步长，并且不会接受不可行的中间路径。
由于 state cost 必须非负，如果候选即使取得零代价仍无法满足
`minimum_improvement`，优化器会在调用代价和碰撞回调前直接剪枝。
梯度为零或被关节限位截回当前状态时，会立即结束该路点搜索，不再执行重复回调。
验证通过的两点路径没有内部路点或 state cost 需要优化，因此会立即返回。

更新方向使用已启用的路径长度和二阶差分项的解析梯度，并与可选 state-cost
梯度组合。梯度先使用关节权重的逆进行预条件，再按最大分量归一化，因此各项目标
权重同时影响搜索方向和候选接受判定。
默认的关节范围平方倒数权重必须有限且为正；数值上无法表示的关节范围会在构造时
被拒绝。

## 分辨率与安全性

每条搜索边都会按 active joint 中不超过 `edge_resolution` 的间隔离散验证。
数值越小，碰撞覆盖越密，但 Coal 查询次数也越多。起终点和所有 shortcut 都会
验证。输出仍是几何路径；如果规划后环境发生变化，应用层必须再次验证。
每条边只计算一次 wrapped 关节差分并供所有采样复用，连续关节始终沿最短圆弧。
极小的正分辨率采用饱和式分段计数，确保由 deadline 安全终止且不会发生浮点到
整数的溢出。
未配置 state validator 时会完全跳过边采样；有限关节的合法端点通过凸插值即可
保证限位可行性。
此时 `collision_checks` 保持为零；配置 validator 后，该统计值等于实际回调次数。

## Demo

不使用机器人资源对比三种算法：

```bash
python3 examples/python/planning/rrt_variants.py
```

规划并播放 Marvin 双臂碰撞路径：

```bash
./scripts/run.sh python3 \
  examples/python/visualization/rrt_robot_viser.py \
  --urdf /absolute/path/to/robot_with_ee.urdf
```

增加 `--validate-only` 可执行无界面的规划验证。机器人资源继续由调用方显式提供，
不会进入测试或安装包。

Viewer 使用橙色和紫色绘制左右末端的 RRT 空间轨迹；绿色标记表示起点，末端颜色
标记表示目标，青色标记随播放位置移动。细红线表示会发生碰撞的关节直线路径，
两类轨迹都可以在 Viser 面板中独立显示或隐藏。

Marvin profile 还在左右臂工作区分别定义了镜像 box。它们会加入
`CollisionModel`，参与每一次 Coal 状态和边检查，并以半透明方块绘制。使用
`--no-obstacles` 可与仅检查
自碰、双臂互碰和躯干的结果对比。应用可以通过
`add_box_obstacle(name, size, pose)` 和 `remove_obstacle(name)` 管理 box。

Demo 默认使用 `--planning-space symmetric`：RRT 在镜像的 7 自由度手臂子空间
中搜索，并把每个待检查状态展开为完整的 14 自由度机器人。因此双臂互碰、手臂与
躯干、夹具及环境碰撞仍全部生效，同时左右轨迹保持均衡。独立或非对称双臂任务可
使用 `--planning-space coupled`。规划阶段额外增加 1 mm 净空，保护对外要求的
5 mm 安全裕量不受边离散影响；最终显示的 180 点路径还会独立复检。

## 当前限制

- 碰撞适配器目前支持固定基座的标量 active joints。
- 最近邻目前使用线性搜索，面向常见机械臂搜索树规模。
- 碰撞验证为单线程。
- 路径优化器是局部 CPU 平滑器，不是全局轨迹优化器。
- 暂不包含动态障碍、动力学规划、浮动基座规划和控制器执行。
