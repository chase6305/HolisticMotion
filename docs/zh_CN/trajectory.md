# 轨迹时间参数化

<div class="language-switcher"><a href="../en/trajectory.html">English</a> · 简体中文</div>

HolisticMotion 将几何路径与时间参数化分开处理。`ToppraTrajectory` 在不改变
关节路径几何形状的前提下，对已有路点路径进行满足速度、加速度限制的重定时。

```python
from holistic_motion.trajectory import ToppraTrajectory

trajectory = ToppraTrajectory(
    [[0.0, 0.0], [0.4, -0.2], [1.0, 0.5]],
    max_velocity=[1.0, 0.8],
    max_acceleration=[2.0, 1.5],
)
times, position, velocity, acceleration = trajectory.sample_uniform(200)
```

路点、约束和时间参数化结果数组都是自包含的不可变快照。`sample()` 与
`sample_uniform()` 返回新的数组，应用可以自由修改而不会影响轨迹对象。

算法以路径速度平方为状态，先反向传播可控区间的上下界，再正向选择可达的最大速度。
运行时不依赖上游 TOPPRA 包，也不需要额外 LP/QP 求解器。

`start_path_velocity` 和 `end_path_velocity` 指归一化弦长参数的起止速度，
默认均为零。可行时结果保留指定值；在当前网格的约束下不可行时抛出 `ValueError`。
求解完成后不会通过整体时间缩放修改这些边界值。

运行示例：

```bash
./scripts/run.sh python3 examples/python/trajectory/toppra_retiming.py
./scripts/run.sh python3 examples/python/trajectory/toppra_retiming.py --plot
```

使用 Viser 交互查看关节路径、播放游标、速度/加速度曲线和实时约束利用率：

```bash
./scripts/run-python-toolkit.sh python3 \
  examples/python/visualization/toppra_viser.py --autoplay --loop
```

在完整 Marvin URDF、双臂末端轨迹和全部机器人网格上播放 TOPPRA 结果：

```bash
./scripts/run.sh python3 \
  examples/python/visualization/toppra_robot_viser.py \
  --urdf /path/to/robot.urdf --autoplay --loop
```

轨迹图表按左臂、右臂标签页分组，图例使用统一颜色的 `J1`–`J7` 短名称，并提供
完整 URDF 关节名映射表。点击图例条目可以单独查看一条曲线。

应传入 `--asset-root` 与 `--profile`，或者明确的 `--urdf` 路径；同时兼容
`--urdf-path` 和 `--urdf_path`。在已经编译的源码仓库中直接执行时，Demo 会
自动进入 `scripts/run.sh` 环境。

路径插值器使用关节空间自然三次样条。每段通过路径一阶导数的解析极值限制速度，
通过二次加速度多项式的 Bernstein 系数限制加速度，覆盖整个区间及节点两侧。
这些区间约束偏保守，不能保证连续问题的全局时间最优；增加 `grid_size` 可以减少
保守程度。`ToppraResult` 同时校验路径速度、加速度和时间数组的一致性。


C++ 的 `PathSegBezierCurve5th` 对曲线长度的各项系数统一使用完整切空间度量，
SE3 路径包含旋转分量。笛卡尔标志不会将切向量截取为三维，因此一维和二维 Rn
曲线也可安全使用该标志；沿固定轴的直线纯旋转曲线与线段采用相同的路径长度定义。
