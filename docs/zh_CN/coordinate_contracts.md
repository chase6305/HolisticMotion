# 运动模块之间的坐标契约

<div class="language-switcher"><a href="../en/coordinate_contracts.html">English</a> · 简体中文</div>

组合 IK、规划和重定时时，先确定关节名称和完整参考配置。向量长度不能单独识别
配置：两个模型即使维度相同，也可能采用不同的关节顺序或编码。

## 各接口接受哪一种向量？

| 接口 | 位置坐标及顺序 | 位姿坐标系 |
|---|---|---|
| `robot.kinematics`、`robot.dof` | 自动选择的默认运动链标量位置 | 运动链基座 |
| `robot.create_kinematics(base, tip)` | 按 `robot.chain_joint_names(base, tip)` 排列的标量位置 | 显式指定的运动链基座 |
| `RobotModel`、`robot.model_dof`、`robot.nq/nv` | 全部独立标量关节，连续关节保存一个角度 | URDF 模型根节点 |
| 原生碰撞模型 | 完整 `nq` 配置，通过 `joint_coordinates` 查询块索引 | URDF 模型根节点 |
| Pinocchio/Pink/多初值重定向 | Python Pinocchio 模型顺序的完整 `nq` 配置 | URDF 模型根节点 |
| `SamplingPlanner`、`PathOptimizer` | `JointSpace` 顺序的标量位置 | 由 validator 定义 |
| `CuroboPlanner` | 按 `planner.joint_names` 排列的全部活动标量关节，不包含配置中锁定的关节 | cuRobo 配置中的基座 |
| 原生关节轨迹与 TOPPRA | 保留传入路点的列顺序 | 不转换位姿坐标系 |

原生 `Robot` 默认选择活动关节最多的受支持根到叶运动链。分支机器人应显式指定
`base` 和 `tip`。`JointSpace.joint_names` 只标记坐标，不重排输入；可选的
`CuroboPlanner(joint_names=...)` 则接受全部活动关节的排列，并转换后端输入输出顺序。

标量转动/连续关节使用弧度，平移关节使用米。导数限制使用对应关节单位每秒及每秒
平方。标量轨迹的位置采样不是完整流形配置，其导数也不是完整模型的切空间向量。

## 配置索引和切空间索引

| 关节表示 | 位置块（`q`，长度 `nq`） | 切空间块（`v`，长度 `nv`） |
|---|---|---|
| 标量转动或平移关节 | 一个角度或位移 | 一个速度 |
| 原生标量连续关节 | 一个角度 | 一个角速度 |
| Pinocchio 连续关节 | `[cos(angle), sin(angle)]` | 一个角速度 |
| Pinocchio 浮动基座 | 平移和四元数 | 六个切空间坐标 |

使用 `idx_q/nq` 填充配置，使用 `idx_v/nv` 提取速度、位移增量或切空间梯度。
一个连续关节就可能使其后所有位置索引和速度索引不同。
`JointPositionMap.joint_velocities()` 只提取**已经处于切空间**的梯度；它不负责
对原始 `nq` 配置代价求导，也不旋转笛卡尔 Jacobian。

## 一次准备命名映射

```python
names = robot.chain_joint_names("torso", "left_tool")
chain = robot.create_kinematics("torso", "left_tool")
if chain is None:
    raise ValueError("unsupported chain")

# 应用提供的完整上下文，包含另一条手臂和基座姿态。
collision.validate_configuration(reference)
mapping = collision.create_joint_position_map(names)
chain_q = mapping.joint_positions(reference)
base_T_tool = chain.forward(chain_q)
full_q = mapping.configuration_with_joint_positions(reference, chain_q)
```

映射持有拓扑快照，保持传入的关节名称顺序。填充返回独立的完整向量，并保留所有
未选择的坐标块；连续关节提取返回主值角，不保留绕圈历史。映射不传播 mimic
约束，也不检查限位或碰撞。提取时只验证被选中的连续关节单位对，未选择的基座
四元数仍需由所属模型验证。模型拓扑改变后应重新建立映射。

Python Pinocchio 模型无需加载原生扩展，也可使用相同的映射契约：

```python
from holistic_motion.configuration import JointPositionMap

mapping = JointPositionMap.from_model(solver.model, names)
full_q = mapping.configuration_with_joint_positions(reference, chain_q)
```

每个集成边界使用对应模型的参考配置和映射。Python 映射不会让独立打包的原生碰撞
与 Python Pinocchio 运行时变得兼容，仍须遵守 {doc}`collision` 中的进程边界。
运动链 FK 返回 `base_T_tool`，得到 `world_T_tool` 时需要左乘完整模型中的当前
`world_T_base`。

## 从规划到重定时保留坐标拓扑

1. 用相同名称和完整参考配置创建命名碰撞规划/优化工厂。上下文在构造时保存；
   更改另一条手臂或浮动基座时，应重新创建工厂。
2. 使用路径前检查规划和优化是否成功。
3. 对最后成功的结果调用 `result.joint_space.unwrap_path(result.path)`。
   即使之后更改了规划器，也继续使用本次请求的限位与周期拓扑。
4. 对展开后的标量路径重定时，再通过预备映射把位置采样填入完整配置并复检运动。

跨越 `+pi/-pi` 时，展开保留规划得到的最短圆弧。展开后的周期角度可能超出代表
区间 `[-pi, pi]`，这是预期行为。展开后的终点不一定等于目标的代表角度，应比较
正余弦或填充后的配置，例如 `+pi` 与 `-pi` 等价。有界关节仍保留原坐标。硬件需要多圈运动或绝对
绕圈限制时，应使用应用定义的有界标量空间，不能从主值角推断绕圈历史。

TOPPRA 通过路点构造自然三次样条。多路点样条或原生轨迹的 blend 可能改变已检查的
几何路径。`named_plan_and_retime.py` 示例把每条折线边分别作为两点 TOPPRA 路径，
在折点处停稳，以额外停顿换取原折线几何的保留；连接处加速度可以跳变，jerk 没有
约束。离散规划与输出复检仍受采样分辨率限制。

## 运行完整的命名关节示例

显式提供机器人名称、资源、起终点及导数限制：

```bash
./scripts/run.sh python3 examples/python/planning/named_plan_and_retime.py \
  --urdf /absolute/path/robot.urdf \
  --joint-names left_shoulder left_elbow \
  --start 0.1 -0.2 --goal 0.3 -0.4 \
  --max-velocity 0.8 0.8 --max-acceleration 1.5 1.5 \
  --output /tmp/named-motion.npz
```

关节名和限制需替换为调用方模型数据。`--reference` 接受碰撞模型的完整 `nq`
配置，省略时使用中性上下文。起点、终点、导数限制和导出的标量列均遵循相同名称
顺序；NPZ 同时保存完整配置与参考配置。只有被选关节运动，需要双臂协同时应将
两臂关节放入同一个搜索空间。

JSON 分别报告规划、优化、重定时与复检耗时，并显示活动碰撞对数量；没有活动对
的模型不会执行碰撞检查。规划或优化失败时不导出轨迹。示例顺序使用一个模型，
并行 worker 需要独立碰撞模型。

算法和后端选择见 {doc}`architecture`，可重复的端到端测量见 {doc}`development`。
