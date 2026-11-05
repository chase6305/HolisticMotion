# NVIDIA cuRobo v2 规划

<div class="language-switcher"><a href="../en/curobo_v2_fast.html">English</a> · 简体中文</div>

`holistic_motion.kit.curobo_planner` 实际接入新版 NVIDIA cuRobo v2
`MotionPlanner`，启用 RRT、B 样条优化与 TimeOpt。支持静止边界关节规划、具名工具
位姿规划、FK 和场景替换。后端可选，独立于原生碰撞扩展。所有规划阶段显式启用
动力学，即使调用方的上游示例 YAML 写着 `load_dynamics: false`。

适配器使用上游多种子 IK，并关闭 LM 种子生成。验证的提交中，LM CUDA graph 会在
连续位姿请求时持有历史梯度输入；此配置用于保持长期规划的显存占用有界。

## 安装可选运行时

需要 Python 3.10+、NVIDIA GPU 以及匹配的 CUDA／PyTorch 环境。cuRobo 使用单独
安装的 CUDA 运行时；HolisticMotion 的原生 `--cuda` 构建选项控制 FEP 批处理组件。按
[上游安装说明](https://github.com/NVlabs/curobo/blob/b29a8c2ee6869585bb05b056b9ab53b9c0753106/docs/getting-started/installation.rst)
选择 CUDA 版本。例如，在单独的 CUDA 13 环境中：

```bash
python -m pip install 'nvidia-curobo[cu13-torch] @ git+https://github.com/NVlabs/curobo.git@b29a8c2ee6869585bb05b056b9ab53b9c0753106'
```

适配器检查 v2 的公开 planner 与 TimeOpt API，不支持 v1 `MotionGen`。
普通 HolisticMotion 导入不加载 cuRobo、Torch、Warp 或 CUDA，不会自动安装依赖或
下载机器人。既有 `CuroboRetargetingSolver` 仍是独立的本地多种子 IK 实现，不调用
这个 NVIDIA 后端。

## 使用调用方提供的机器人

显式提供本地 cuRobo 机器人 YAML，其中包含 URDF／mesh 路径、基座／工具坐标系、
碰撞球、锁定关节、惯量及动力学限值。外部资源在配置中使用绝对路径。适配器不自动
转换 HolisticMotion 碰撞模型，不推测几何，也不在项目中包含机器人资源。

```python
from holistic_motion.kit.curobo_planner import CuroboPlanner

with CuroboPlanner(
    "/absolute/path/robot.yml",
    scene="/absolute/path/scene.yml",  # 无障碍物时省略。
    collision_cache={"cuboid": 8},
) as planner:
    planner.warmup()
    result = planner.plan(q_start, q_goal)
    if not result.success:
        raise RuntimeError(result.message)
    trajectory = result.trajectory
    joint_names = trajectory.joint_names
    times = trajectory.time
    positions = trajectory.position
```

输入为 `planner.joint_names` 顺序的标量位置，不是完整 Pinocchio `nq`。
可选 `joint_names=` 可重排全部主动关节；锁定关节保持配置值。
`lower_limits`／`upper_limits` 返回同一顺序的独立数组。输入必须有限、满足限位，
转为 float32 的误差不超过 `1e-6`。适配器不推测周期关节的绕圈方向。

位姿目标使用基座坐标系中的 4×4 齐次矩阵，平移单位为米。必须提供每个已配置工具
的目标，不需要猜测四元数顺序：

在同一个尚未关闭的 planner 上下文中：

```python
targets = planner.forward_kinematics(q_goal)  # 也可由调用方提供矩阵。
result = planner.plan_pose(q_start, targets)
```

构造时配置 `position_tolerance`（默认 0.005 m）和 `orientation_tolerance`
（默认 0.05 rad），对返回终点重新做 FK 检查。`plan()` 则使用独立的
`endpoint_tolerance` 检查关节端点，默认 `1e-3`。

两种规划方法均支持 `enable_graph_attempt=0`，从第一次尝试就使用 RRT；默认值 `1`
先直接优化，再于第二次尝试启用 RRT。`max_attempts` 限制重试次数。

## 场景、结果与生命周期

`update_link_inertial(link_name, mass=..., com=..., inertia=...)` 完整替换单个
link 的惯性参数。COM 为 link 坐标系中的三维米制向量；质心处惯量按 link 轴表达，
使用 `[ixx, iyy, izz, ixy, ixz, iyz]`，单位 kg*m²。适配器检查 float32 有限值、
非负质量及物理一致的主惯量，更新上游动力学并清空 RRT 状态。这不会附着碰撞几何。
底层惯量更新失败后阻止后续规划，必须重建实例。

`update_link_collision_spheres(link_name, spheres)` 用调用方提供的
`[x, y, z, radius]` 行完整替换单个 link 的碰撞球，坐标相对该 link，单位为米，
半径必须为正。机器人配置须预留足够球槽，例如附着物 link 上的
`extra_collision_spheres`；多余槽位会禁用。传入 `None` 恢复该 link 的原始配置球。
更新同时作用于 IK、RRT 和轨迹碰撞检查，并清空 RRT 状态；质量和环境障碍物需
另外显式更新。底层球更新部分失败后须重建规划器。

```python
planner.update_link_collision_spheres("attached_object", [[0, 0, 0.1, 0.05]])
# 使用调用方定义的载荷几何规划，随后恢复 YAML 中的模型。
planner.update_link_collision_spheres("attached_object", None)
```

`scene=` 和 `update_world(scene)` 接受本地场景 YAML、上游场景字典或
`curobo.scene.Scene`。障碍物位姿相对已配置基座，遵循上游 `xyz, wxyz` 顺序。
`None` 表示空环境，自碰撞仍开启。更新会替换整个场景并清空 RRT 搜索状态；构造时
预留碰撞容量。更新失败后阻止规划，直到完整替换成功。同一实例的调用和更新串行化；
同进程不同实例的后端调用也串行化：上游延迟 capture 使用 CUDA 全局模式，会禁止
另一规划线程同时分配设备数据。并行 GPU 请求使用独立进程（Python multiprocessing
使用 `spawn`）。更新返回前完成设备操作，后续调用可切换 CUDA stream。

验证的上游／Warp 组合存在跨实例 capture 限制：有环境碰撞缓存的存活实例中，只要
任意实例开启 CUDA graphs，就须使用相同缓存布局。适配器在构造时检查并拒绝不兼容
组合，避免新增 Warp 泛型特化使另一实例的旧 capture 失效。相同布局的场景、惯量和
生命周期仍各自独立。需要不同布局时使用独立进程，或将**所有**实例构造为
`use_cuda_graph=False`。没有分配环境碰撞缓存的实例不施加此限制。

场景字典支持 `cuboid`、`sphere`、`capsule`、`cylinder`、`mesh`、`voxel`。
替换环境前拒绝未知类型、重名障碍物、无效尺寸及非单位位姿四元数。所有障碍物须有
显式位姿，缺少 mesh 位姿时在加载器清除现有场景前即拒绝输入。
公开 `Scene` 的 `objects` 必须与各类型障碍物列表一致。验证的上游加载器会遗漏
环境球，因此适配器将球转换为端点重合的解析胶囊体，不引入 mesh 近似。
后续需要替换球或胶囊体时，构造中预留 `collision_cache={"capsule": N}`。

完整替换环境时也会清除旧 Warp 网格几何，避免上游仅按名称缓存而复用同名旧形状，
或持续积累已移除的网格。释放网格前等待设备工作完成，保留碰撞存储容量和 capture。
只移动已有障碍物时，使用 `update_obstacle_pose(name, transform)`，传入基座系
4×4 矩阵；此路径保留几何并清空 RRT 状态。位姿更新失败后须成功完整替换环境，
才能继续规划。

成功的 `CuroboPlanningResult` 包含 `CuroboTrajectory`：只读且自行持有数据的
NumPy `time`、`position`、`velocity`、`acceleration` 以及可选 `jerk` 数组。
按有效插值长度、实际返回步长和请求关节顺序转换，移除 padding 与锁定关节。
下一次求解或关闭规划器不会改变先前结果。失败保留可用的上游验收诊断且没有轨迹；无效调用参数在
求解前抛错。结果是采样数据，不是插值轨迹，不能经 TOPPRA 重拟合后沿用原约束结论。

上游采样碰撞／动力学验收不等于连续时间可行性证明。现有 HolisticMotion TOPPRA
约束速度与加速度，不具备等价的力矩／jerk 契约。实际使用前检查目标负载下的碰撞球
覆盖和动力学数据；控制器及硬件执行不在适配器范围内。

NumPy 接口自行管理后端所需的 autograd 上下文，可在调用方的 `torch.no_grad()` 或
`torch.inference_mode()` 中构造和调用；每次调用后恢复调用方的状态。

## 示例与测试

```bash
HOLISTICMOTION_PURE_PYTHON=1 PYTHONPATH=python python examples/python/planning/curobo_plan.py \
  --robot-config /absolute/path/robot.yml \
  --start 0 0 0 --goal 0.1 0.2 0.3 --output /tmp/trajectory.npz
```

按实际机器人的关节数填写参数。示例预热后规划、报告完整调用耗时，成功时保存具名
采样数组，无需可视化依赖。

CPU 接口测试为 `tests/python/test_curobo_planner.py`。真实 GPU 测试由调用方
显式提供配置和 JSON 关节数组：

```bash
export HOLISTICMOTION_TEST_CUROBO_ROBOT_CONFIG=/absolute/path/robot.yml
export HOLISTICMOTION_TEST_CUROBO_START='[0, 0, 0]'
export HOLISTICMOTION_TEST_CUROBO_GOAL='[0.1, 0.2, 0.3]'
HOLISTICMOTION_PURE_PYTHON=1 PYTHONPATH=python python -m pytest tests/python/test_curobo_planner_gpu.py -q
```

验证对应上游提交 `b29a8c2ee6869585bb05b056b9ab53b9c0753106`，覆盖关节／位姿
规划、结果所有权、障碍物拒绝与场景替换后的恢复。未提供配置时跳过 GPU 测试；
配置后必须实际构造并运行后端。
