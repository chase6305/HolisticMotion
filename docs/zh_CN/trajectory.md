# 轨迹时间参数化

<div class="language-switcher"><a href="../en/trajectory.html">English</a> · 简体中文</div>

HolisticMotion 将几何路径与时间参数化分开处理。`ToppraTrajectory` 在不改变
关节路径几何形状的前提下，对已有路点路径进行时间最优重定时。

```python
from holistic_motion.trajectory import ToppraTrajectory

trajectory = ToppraTrajectory(
    [[0.0, 0.0], [0.4, -0.2], [1.0, 0.5]],
    max_velocity=[1.0, 0.8],
    max_acceleration=[2.0, 1.5],
)
times, position, velocity, acceleration = trajectory.sample_uniform(200)
```

算法以路径速度平方为状态，先反向传播可控集，再执行时间最优正向通过。
运行时不依赖上游 TOPPRA 包，也不需要额外 LP/QP 求解器。

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
  examples/python/visualization/toppra_robot_viser.py --autoplay --loop
```

轨迹图表按左臂、右臂标签页分组，图例使用统一颜色的 `J1`–`J7` 短名称，并提供
完整 URDF 关节名映射表。点击图例条目可以单独查看一条曲线。

可通过 `--profile`、`--asset-root` 或明确的 `--urdf` 路径覆盖默认资产配置；
同时兼容 `--urdf-path` 和 `--urdf_path`。在已经编译的源码仓库中直接执行时，
Demo 会自动进入 `scripts/run.sh` 环境。

路径插值器使用关节空间自然三次样条。约束作用于可达性网格点上的解析导数；
对于曲率较大的路径，应增加 `grid_size`，并在下发硬件前检查采样结果。
