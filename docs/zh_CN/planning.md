# 采样式路径规划

<div class="language-switcher"><a href="../en/planning.html">English</a> · 简体中文</div>

HolisticMotion 自主实现了一套紧凑的 C++17 关节空间规划算法，不链接、不内置，
也不导入 OMPL。目前有意只提供 `RRT_CONNECT`、`RRT_STAR` 和
`INFORMED_RRT_STAR` 三种算法。

## 职责划分

采样规划器生成无碰撞几何路径，shortcut 删除不必要的路径点，随后应由 TOPPRA
施加速度和加速度约束。控制器不属于本库范围。

单次机器人规划默认推荐 `RRT_CONNECT`。`RRT_STAR` 和
`INFORMED_RRT_STAR` 会持续优化路径直到时间或迭代预算耗尽。迭代预算较大时，
即使提前找到初始解，求解耗时通常仍接近 `timeout_seconds`。

三种算法在扩展随机树之前都会完整检查起点到终点的直连边。如果整条边有效，
规划器会立即返回仅含两个点的最短路径；只有障碍物阻断直连时，优化型算法才会
使用完整时间预算。HolisticMotion 当前只报告精确解，超时和失败结果不会夹带
语义不明确的部分路径。

未配置状态校验回调时会跳过边采样：有限关节的插值保持在限位内，连续关节正常
周期化。此时 `edge_resolution` 不产生校验工作，`collision_checks` 为零。
需要检查障碍物或其他约束时，应显式配置校验回调。

对于不超过八个坐标、同时含有限关节和周期关节的空间，搜索树达到 128 个状态后
使用内部平衡空间索引。索引仅沿有限关节划分，候选仍使用原有加权周期距离计算；
新增节点在下次重建前通过线性扫描查询，等距结果和邻域列表保持插入顺序。
小树、全部有限或全部周期关节以及更高维空间仍使用线性搜索；无需额外依赖或
公开配置选项。

连续关节的限位范围必须覆盖 `2*pi`，且一个圆周内须存在不同的浮点坐标。
若任一限位处的浮点间距不小于 `2*pi`，`set_continuous_joints` 会拒绝该配置，
设置失败时保留原有拓扑。
间距过大时，插值校验采样可能塌缩，即使原始起终点仍不同。
已在规范区间内的角度保持原值，避免插值或优化中的可表示小步长被取模舍入吞掉。

关节权重控制搜索距离，不充当目标容差。即使权重很小，直连解也会保留不同的起点
和终点；RRT-Connect 只有在经过验证的扩展实际到达目标状态后才连接两棵树，
不会因为加权距离很小就跳过连接边的运动校验。

RRT* 与 Informed RRT* 再次采样到已有最近邻状态时，会复用该节点，同时继续考虑
更好的父节点和邻域重连，避免重复存储目标样本。邻域半径仍根据已接纳的样本数
计算；`tree_nodes` 统计实际存储的树节点数。

找到初始解后，Informed RRT* 每次迭代最多尝试 128 个均匀样本进入知情子集。
若全部被拒绝，则重新访问目标节点以尝试改善其父节点，避免子集过窄时拒绝采样
耗尽时间预算。这会改变受影响场景的固定种子采样序列；每条新连接仍须通过运动校验。

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

输入校验使用原始路点数值，包括每条边的精确终点。插值重建终点可能产生舍入误差，
将其推到可行边界另一侧，因此只有边内部采样使用插值。这保留既有回调顺序，候选
更新时也不会重复检查已经验证的相邻端点。

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
创建具有正 `clearance` 的网格或球体碰撞优化器时，模型必须含有活动碰撞对。
两次调用之间清空碰撞对时，已有优化器暂时关闭碰撞校验和间距代价；恢复碰撞对后
重新启用。碰撞对修改不能与优化调用重叠。
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
两侧实际偏移不等时使用当前路点处的三点插值导数，避免限位投影使梯度偏向
采样区间中点；一侧几乎坍缩时按两侧偏移的相对比例省略该侧查询。
周期关节的差分步长最大为 `pi/2`，以保证采样落在当前路点的两侧；
无法用有限浮点数表示的数值梯度会抛出 `ValueError`。
预条件计算会保留各关节下降方向的比例，即使合法正权重的倒数超出浮点范围也不会
静默跳过更新。几何梯度的中间运算出现非有限值时会使用扩展精度回退；如果组合后的
目标梯度仍无法表示，则在执行候选回调前抛出 `ValueError`。

超时时仍返回当前最佳可行路径。非法输入路径会被拒绝而不是尝试修复，因此采样
规划仍负责找到初始可行同伦路径。该设计参考 cuRobo 中 seed 生成、可行性优化和
最佳解跟踪的分层，但不导入其 Torch、Warp 或 CUDA 优化器栈。之后仍由 TOPPRA
完成速度和加速度时间参数化。
每次校验回调返回后都会检查截止时间，包括一条边的最后一个采样点。输入路径
校验期间超时会返回空路径和 `success=False`；候选路径校验超时则丢弃候选，
保留之前已经验证的路径。回调同步执行且无法中断，因此单次慢回调仍可能超过
请求的墙钟时间预算。
如果在全部初始 state cost 计算完成前超时，优化器会返回已经验证的输入路径；由于
完整目标从未得到计算，两个 objective 统计值均为 `NaN`。

路点更新采用增量目标计算：移动一个内部路点时，只重算相邻的两项路径长度和最多
三个受影响的二阶差分项。初始与最终报告的目标值分别完整计算，候选接受条件直接
比较受影响的局部项，避免其他路点不变的大代价掩盖局部改进或放行代价上升的候选。
最终 state-cost 统计从已接受值的缓存重新求和，不增加回调，也避免反复相减造成漂移。
如果原始代价总和溢出，则先缩放缓存中的各项再求和，保留仍可表示的加权总值。
因此在固定回溯预算下，每轮几何目标计算随路点数量线性增长；碰撞和 state-cost
回调另有计算成本。

每次 `optimize()` 使用独立工作区，复用边差分、二阶差分、梯度、方向和候选缓冲区。
单个路点的回溯搜索缓存两侧不变的外邻边，移动到下一路点时重新读取，反向扫描也
如此。梯度与代价计算之间复用相邻边长，切换路点或候选后重新计算。
工作区及输出初始化后，局部几何目标与梯度计算不再申请 Eigen 堆内存；
完整优化器仍会为路径、校验采样和可选 state-cost 缓冲区分配内存。

目标权重为零的项不会参与代价和梯度计算，既省去无效的几何计算，也避免已关闭项
的溢出污染本来有限的目标值。

`step_size` 必须处于 `(0, 1]`。即使坐标变化小于绝对值 `1e-15`，只要仍可表示就会
继续评估；当舍入后构型完全不变时停止该次试探。
每个路点更新都会执行有界回溯搜索：`line_search_steps` 控制最多尝试的步长
数量，`line_search_decay` 控制每次重试的缩放比例。初始步长发生碰撞或越过
下降区间时，优化器会尝试更小的可行步长，并且不会接受不可行的中间路径。
由于 state cost 必须非负，如果候选即使取得零代价仍无法满足
`minimum_improvement`，优化器会在调用代价和碰撞回调前直接剪枝。
梯度为零或被关节限位截回当前状态时，会立即结束该路点搜索，不再执行重复回调。
验证通过的两点路径没有内部路点或 state cost 需要优化，因此会立即返回。

更新方向使用已启用的路径长度和二阶差分项的解析梯度，并与可选 state-cost
梯度组合。梯度先使用关节权重的逆进行预条件，再按最大分量归一化；指数分解可恢复因溢出、下溢或
次正规数舍入而丢失的方向比例，因此各项目标
权重同时影响搜索方向和候选接受判定。可表示的正边长不会因绝对值很小而被跳过；
`minimum_improvement` 仍是绝对目标改进阈值，改变目标尺度时应相应调整。
几何梯度也会恢复提前下溢的加权中间乘积，避免在随后除以小边长或乘以大差值前丢失可表示的结果。
单关节相邻边同向时，长度导数会在缩放前精确抵消，避免局部恒定的长度产生放大的舍入残差。
即使未加权的平方会溢出或下溢，加权路径长度和几何代价仍保留可表示的结果；
平方代价超出常规范围时，平滑度系数会在累积前参与缩放，保留有限非零目标值。
普通尺度查询保持原有计算。当距离平方无法区分可表示的运动时，
最近邻和半径查询会改用实际范数比较。默认的关节范围平方倒数权重必须有限且为正，
无法表示默认权重的范围会在构造时被拒绝。

Python 规划输入必须为实数。复数限位、状态、权重、选项（含计数、开关和关节索引）
以及代价回调结果
会抛出 `TypeError`，避免转换时丢弃虚部；实数整数数组和非连续切片仍受支持。

## 分辨率与安全性

RRT* 和 Informed RRT* 在重连后刷新最佳目标代价，包括祖先节点改进所传播的
路径缩短。固定随机种子并完成给定迭代预算时，继续搜索会保留或改进 shortcut
之前的最佳路径；墙钟时间限制会影响实际完成的搜索和后处理工作量。

每条搜索边都会按 active joint 中不超过 `edge_resolution` 的间隔离散验证。
数值越小，碰撞覆盖越密，但 Coal 查询次数也越多。起终点和所有 shortcut 都会
验证。输出仍是几何路径；如果规划后环境发生变化，应用层必须再次验证。
shortcut 还会重新检查两侧保留的边片段，因为截短边会改变碰撞采样位置。
shortcut 的改进判断采用相对舍入容差，全局缩放关节权重不会因固定绝对阈值而跳过可表示的正路径长度。
可选路径插值保留全部原始拐点，并优先细分最长线段，使采样尽量均匀。
`interpolation_points` 是增加路点时的目标数量，不会删除已有路点。
插值后的子段验证通过才会被接受；验证失败或超时时保留原路径，因此返回数量
可能与目标不同。离散验证不能排除采样点之间的碰撞，分辨率应结合应用选择。
每条边只计算一次 wrapped 关节差分并供所有采样复用，连续关节始终沿最短圆弧。
极小的正分辨率采用饱和式分段计数，确保由 deadline 安全终止且不会发生浮点到
整数的溢出。
未配置 state validator 时会完全跳过边采样；有限关节的合法端点通过凸插值即可
保证限位可行性。
此时 `collision_checks` 保持为零；配置 validator 后，该统计值等于实际回调次数。

## 规划与轨迹之间的关节空间契约

`JointSpace` 是限位、度量权重、周期关节和可选关节名称顺序的不可变快照。
规划器和路径优化器均可接收它，避免后处理阶段丢失搜索时的度量与周期拓扑：

```python
space = hm.JointSpace(lower, upper, joint_names)
space = space.with_joint_weights(weights).with_continuous_joints([0])
planner = hm.SamplingPlanner(space, state_validator=is_valid)
optimizer = hm.PathOptimizer(planner.joint_space, state_validator=is_valid)
```

原有限位构造与 setter 保持兼容；setter 只替换该消费者的空间，不改变已有快照或
其他消费者。命名碰撞工厂在 `joint_space.joint_names` 中保留选定关节顺序。
快照不包含 validator、碰撞模型、代价回调或共享工作区，这些状态需分别提供。
复制只保留共享的只读坐标数据，不重新分配限位、权重或名称数组；builder 会创建
新快照，因此已有副本和结果保持原值。C++ 快照移动后，源对象仍保留可用坐标数据。
规划和优化结果也保存本次请求的 `joint_space`，之后的 setter 不会改变既有路径
对应的坐标契约。

每次请求会在调用校验或目标函数前独立持有端点或路径坐标、选项和回调包装。
重入修改调用方输入、选项对象或规划器回调只影响后续请求，当前请求始终使用
同一份已验证的采样分辨率、度量和目标。
回调捕获的模型、列表或共享工作区保持原有归属，不会深拷贝；C++ 按值复制的可变
函数对象则使用请求局部状态，需要共享时应显式捕获共享状态。并发调用 setter
仍需调用方同步。

C++ `NullSpacePlanner::Plan()` 可以将旧输出路径中的行用作新的起点或偏好向量，
SRS/FEP `Solve()` 也可以将旧输出候选用作种子。这些调用会在替换输出容器前保留
所需输入值。

Python 的 `SRSKinematics.null_space_velocity()` 与 `NullSpacePlanner.plan()`
复制输入后在原生计算期间释放 GIL。只读查询可共享求解器或规划器，每次查询持有
独立输出；模型、坐标系和限位更新仍须外部同步。

Python 的 `SamplingPlanner.plan()` 和 `PathOptimizer.optimize()` 接受仅限关键字的 `release_gil=True`，用于并行
执行原生计算。绑定先持有 GIL 复制输入、选项和规划器或优化器，再为本次请求释放
GIL；Python 回调会自动重新获取它。默认 `release_gil=False` 保留既有原生回调的
调用契约；启用后，访问 Python API 的 C++ 回调须自行获取 GIL。回调捕获的碰撞模型
和其他可变工作区仍需由各 worker 分别持有。可在启动 worker 前创建副本：

```python
worker_models = [collision.clone() for _ in range(4)]
worker_planners = [
    hm.SamplingPlanner.from_collision_joints(model, joint_names, context)
    for model in worker_models
]
```

每个副本保留源对象当前的障碍物和碰撞对选择，并持有新的查询工作区。复制期间
源对象应无活动查询或更新。每个并发请求使用不同的规划器与模型，再通过
`plan(..., release_gil=True)` 并行执行原生计算。

CPython 的校验、代价和梯度回调参与循环引用回收，包括捕获自身规划器或优化器的
回调，以及反向引用工厂规划器或优化器的碰撞模型。工厂回调的每个副本独立持有
其 Python 碰撞对象引用；借用模型仍由外部 C++ 所有者管理原生生命周期。替换或清除全部相关回调后，活动请求与外部 C++ 副本结束时即可释放
该资源。执行中的请求会独立持有回调引用，直到请求结束。通过借用引用暴露给 Python
的 C++ 规划对象仍遵循其 C++ 所有者的生命周期；Python 回收不会清除该外部对象
中的回调。
包含 Python 回调或资源的 C++ 副本应在对应解释器关闭前释放。
这些回调在复制或销毁时会获取 GIL；嵌入端等待执行这些操作的 worker 结束时，
应释放 GIL。
在 PyPy 上应避免回调捕获自身所有者，或在使用结束后显式调用
`set_state_validator(None)`、`set_state_cost(None)` 和
`set_state_cost_gradient(None)`；已测试的 PyPy 3.10 回收器不会回收原生回调循环。

将包含周期关节的规划结果交给标量轨迹插值器前，应调用
`result.joint_space.unwrap_path(result.path)`。它按规划使用的最短圆弧展开各边，
有界坐标保持原值，展开后的连续坐标可以超出其规划代表区间。
例如 `pi - 0.1` 到 `-pi + 0.1` 会展开为正向 `0.2` 弧度，而不是穿过零点的近一整圈。
空路径、非法源状态和无法表示的展开步进会被拒绝；周回平移须使各源角度的正弦与
余弦分量误差均不超过 1e-10。该操作保留折线的周期拓扑；
样条或非零 blending 仍可能改变几何，需另行验证。离散碰撞检查不能证明采样点之间
的连续路径无碰撞。

## Demo

建议先运行自包含的经典“规划并重定时”流程：

```bash
./scripts/run.sh python3 \
  examples/python/planning/classic_plan_and_retime.py
./scripts/run.sh python3 \
  examples/python/planning/classic_plan_and_retime.py --plot
```

示例使用固定随机种子的 RRT-Connect 让二维点机器人绕过墙体，再用原生
Double-S 进行时间参数化，并复查采样位置和导数限位。它显式使用单位度量权重，
因此统计长度就是图中的欧氏长度；规划裕量大于离散边分辨率，并保持零圆滑容差，
避免时间参数化切过已检查的拐角。默认无需机器人资源或可选求解器，只有
`--plot` 需要 Matplotlib；传入 `--profile trapezoidal` 可对比另一种原生剖面。
在时间参数化前，示例还会确定性地保留每个路点处最远的可直达后续点，并用
相同裕量重新检查每条候选边，从而删除冗余 RRT 点、减少不必要的完全停车，
同时保持已检查折线的语义。

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
