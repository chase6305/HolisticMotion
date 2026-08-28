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

## 分辨率与安全性

每条搜索边都会按 active joint 中不超过 `edge_resolution` 的间隔离散验证。
数值越小，碰撞覆盖越密，但 Coal 查询次数也越多。起终点和所有 shortcut 都会
验证。输出仍是几何路径；如果规划后环境发生变化，应用层必须再次验证。

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
- 暂不包含动态障碍、动力学规划、浮动基座规划和控制器执行。
