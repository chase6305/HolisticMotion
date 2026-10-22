# 核心概念

<div class="language-switcher"><a href="../en/concepts.html">English</a> · 简体中文</div>

## 显式模型所有权

HolisticMotion 不捆绑机器人资源。请向 `Robot`、`CollisionModel` 和 retargeting
求解器传入绝对路径或由应用解析后的 URDF 路径。

默认运动链选择从根到叶节点、活动关节最多的完整受支持路径；数量相同时保留先访问
的叶节点，末端固定变换仍计入工具位姿。包含 mimic 或不支持关节的路径不参与选择。
显式创建基座到末端的求解器时，大模型中的长路径使用本次调用内的增量名称索引，
短路径保留直接查询。C++ 对公开名称与父子关系的修改会在下次调用生效，重复名称
保持首次匹配行为。父链遍历遇到环时会结束并返回 `nullptr`（Python 为 `None`）。

## 核心组件与可选组件

核心 `holistic_motion` 库包含机器人模型、运动学、流形、零空间路径生成和轨迹。
碰撞查询位于可选的 `holistic_motion_collision` 配套库中，因此普通运动学用户
不需要引入 Pinocchio 和 Coal。Python 也使用独立的核心与碰撞扩展：普通包导入
只加载核心，首次使用碰撞类型或碰撞规划工厂时才加载碰撞。尚未加载 Conan 碰撞
扩展时，核心算法可与 Python Pinocchio 在同一进程运行。
离线几何预处理在首次访问 `hm.geometry` 时加载；普通包导入不会初始化 NumPy，
数值数组操作会在需要时加载它。

## 坐标和顺序约定

- 目标位姿是有限的 4×4 齐次变换矩阵。
- Retargeting 目标使用 URDF 世界坐标系。
- 通用 retargeting 结果采用 Pinocchio 模型关节顺序。
- 控制器特定的关节重排应在应用集成边界完成。

对于内置浮点标量，C++ `SO3` 的 `EulerAnglesType::rota_vec` 构造器会保留
范数可表示的旋转，即使范数平方发生溢出或下溢；自动微分标量保持原有计算路径。
内置浮点类型的 SO3 切向量指数映射和 Jacobian 在大角度下使用单位轴公式，
避免中间幂次溢出和小横向导数被相消。逆 Jacobian 的条目仍须可表示，整圈旋转处仍有
奇异性。Eigen 自动微分标量的指数映射与 Jacobian 通过有限差分核对。

## 关节名称与坐标映射

完整配置 `q` 的长度为 `nq`，速度、位移增量和距离梯度的长度为 `nv`。
原生 `Robot` 的完整标量模型采用 `nq = nv = model_dof`，连续关节保存角度；
Pinocchio 碰撞和重定向模型的连续关节采用 `[cos(theta), sin(theta)]`，
因此单个连续关节为 `nq=2, nv=1`。不能把配置索引直接用于梯度。
`robot.dof` 和 `robot.kinematics` 描述默认选择的运动链；`robot.model_dof`、
`robot.nq`、`robot.nv` 描述所有独立标量关节，包括其他分支。向链求解器传入链维度
的位置，再按关节名称嵌入完整碰撞或重定向配置。
`Robot` 的标量模型不包含浮动、平面或 mimic 从动坐标；它不是完整 Pinocchio
配置的替代品。这些坐标需要由碰撞或重定向模型的参考配置提供。

`joint_coordinates` 返回各关节的 `name, idx_q, nq, idx_v, nv` 快照。
`create_joint_position_map(names)` 按传入名称顺序建立可复用映射：

```python
names = robot.chain_joint_names("base", "left_tool")
mapping = collision_model.create_joint_position_map(names)
context = collision_model.neutral_configuration()
full_q = mapping.configuration_with_joint_positions(context, chain_q)
chain_q = mapping.joint_positions(full_q)
chain_gradient = mapping.joint_velocities(full_distance_gradient)
```

映射复制参考配置，仅替换所选关节，保留另一条手臂或浮动基座。重复名称、未知名称
和被选中的多自由度关节会被拒绝。连续角提取返回主值角，不保存绕圈历史；链模型与
Pinocchio 模型的轴和单位仍须一致。映射是拓扑快照，模型拓扑变化后需要重新创建。
Python Pinocchio 模型可通过
`holistic_motion.configuration.JointPositionMap.from_model(model, names)` 使用同一语义。
重定向求解器也提供 `create_joint_position_map()`，不需要加载原生扩展。

`Robot`、两个原生碰撞模型，以及 Python 重定向/手部求解器均通过稳定缩放，将
URDF 中的转动、连续和平移关节轴归一化为单位方向。
导入后的单位轴在原生范数计算下也精确为 1，避免轴长度的舍入误差改变极大关节角
相对于 Pinocchio 的旋转相位。
轴向量的长度不缩放关节角度或位移，负方向仍保留符号；零轴和非有限
轴会被拒绝。这修正了原生 URDF 路径此前按轴长度缩放运动的行为，也避免碰撞后端
接受零轴后产生非刚体变换；调用方 URDF 文件保持原样。直接在 C++ 中构造
`JointNode` 时仍保留显式轴缩放，因此手动构造的运动链仍须与碰撞/重定向模型使用
一致的轴约定。`LoadURDF()` 仅在新模型完整构造并通过校验后替换旧模型；解析或
校验失败时保留原有模型及求解器。
延迟调用 `LoadVisuals()` 时发布与运动学同一次 URDF 解析的视觉元数据；切换工作
目录、替换或删除源文件均不会混入另一份视觉模型。延迟期间只保留视觉元数据，不
读取 mesh 资产，发布时也不再解析 URDF。`GetURDFPath()` / `robot.urdf_path`
仍保留调用者传入的路径。
原生模型导入会拒绝所有有界标量关节中颠倒的位置上下限，包括默认运动链以外的
分支。上下限相等仍表示合法的锁定位置；连续关节没有位置限位。

映射负责坐标转换，不传播 mimic 关节约束，也不检查硬件限位或碰撞。
原生碰撞模型的 `validate_configuration(q)` 可单独检查全配置维度、有限性以及
连续关节/四元数的归一化，不更新碰撞工作区。命名关节规划工厂在创建时校验需要保留
的上下文流形坐标，避免把非法浮动基座留到搜索阶段才发现。

碰撞模型的 `joint_lower_limits` / `joint_upper_limits` 为连续角提供 `[-pi, pi]`
的规划区间，而非 cosine/sine 的分量界。`from_collision_joints()` 规划器和优化器
自动将这些关节设为周期坐标。需要多圈约束时，应使用调用方定义的标量规划空间。

## 初始化阶段与运行阶段

模型构建和配置应放在实时循环之外。`Robot`、原生碰撞模型、连续 IK 跟踪器和
TOPPRA 轨迹会在构造阶段解析拓扑并分配持久状态。Retargeting 求解器应在构造后
或非实时模式切换期间调用 `prepare(mode)`；之后的 `solve()` 与 `step()` 会复用
mode indices 和数值工作区。采样规划器与路径优化器在对象中保存关节拓扑，而每次
`plan()` 或 `optimize()` 只持有本次请求的搜索状态。球体拟合属于离线初始化操作，
生成的球体模型才是运行时数据。

有状态重定向求解器与碰撞模型会复用可变工作区，因此同一实例不支持并发调用。需要并行
查询时，应为每个运行 worker 创建并 prepare 独立实例。

| 类或操作 | 同一实例的并发契约 |
|---|---|
| `Robot` 树查询、名称映射构造 | 模型、公开字段及 visual 数据不变时可并发读取；加载或编辑需外部同步 |
| `JointPositionMap` | 不可变拓扑快照，每次调用拥有独立输出，可共享 |
| `NumericalKinematics`、OPW、UR、FEP 的只读求解 | 无配置更新且调用方各自持有种子和输出时可并发；不与 TCP、限位或模型更新并发 |
| `SRSKinematics` | 构造和模型更新时准备几何；配置不变且种子/输出独立时，可共享实例执行只读查询 |
| `NullSpacePlanner` | 路径与工作区属于单次调用；持有的 SRS 求解器在配置不变时可共享 |
| `SRSContinuousTracker` / `FEPContinuousTracker` | 历史可变，每个 worker 独立构造跟踪器；原生求解器可用于并发只读查询 |
| `PinocchioRetargetingSolver`、`PinkRetargetingSolver`、`CuroboRetargetingSolver` | Pinocchio Data 和求解历史可变，每个 worker 独立 prepare，回调状态也须遵守相同归属约定 |
| `DexHandRetargetingSolver` | 手部工作区、优化器及上次结果可变，每个 worker 独立初始化 |
| `RetargetingModeManager` | 模式选择可变；每个求解 worker 独立持有管理器，仅共享不可变的 `mode_specs` |
| `CollisionModel`、`SphereCollisionModel` | 查询会更新 FK/碰撞工作区，每个 worker 独立实例 |
| `SamplingPlanner`、`PathOptimizer` | 搜索状态属于单次调用；只有所有回调也可并发时才能共享。绑定原生碰撞或球体距离缓存时使用独立实例 |
| 内置 C++ 轨迹、NumPy TOPPRA 采样 | 构造完成且几何、限制及结果数据不变时可并发只读采样；自定义路径段负责自身契约 |
| PyTorch TOPPRA | 只读前向查询需遵守 PyTorch 的设备/流与 autograd 规则；并行反向使用独立计算图 |
| `SimulationSession` | 单一状态推进者；并发或重入的 reset/step/close 会被拒绝，适配器自行管理缓冲区和设备同步 |

Python GIL 不作为线程安全保证：NumPy、Pinocchio 和设备操作可能释放 GIL，
应用回调也可能重入。`prepare()`、模式切换和配置更新应发生在该 worker 的查询之外。

内置求解器的 `solve_ik()`、SRS/FEP 专用 `solve()` 和 `solve_configuration()`、
SRS 详细报告及用户坐标系求解接口，以及 FEP `forward_batch()`，会先复制 Python 输入
（包括分支配置），再在原生计算期间释放 GIL，并在输出转换时重新持有它。因此 Python worker 可以共享配置不变
的求解器并分别持有输出；并发调用 setter 或共享 tracker 历史仍不在此契约内。
自定义求解器（包括 numerical/OPW/UR 的派生类）默认保留 GIL；只有显式重写能力
查询并声明支持并发时才释放它。此声明也包含不持有 Python GIL 的执行条件。
规划与路径优化可显式传入 Python 关键字 `release_gil=True`；绑定在释放前复制
输入和请求配置，Python 回调自行重新获取 GIL。默认调用方式兼容需要持有 GIL 的
既有原生回调。
内置数值、SRS 和 FEP 的标量 FK 只计算末端位姿，`GetAllFK()` 仍返回完整帧序列。
自定义数值求解器派生类默认保留 `GetAllFK()` 虚调用；只有通过受保护的
`UsesDefaultForwardKinematics()` 钩子声明使用默认链语义时，才采用末端专用路径。

每个 worker 的碰撞适配器应绑定独立构造的模型；复制规划器或闭包不会复制其捕获的
模型和共享距离缓存。
复制 C++ `Robot` 也会共享 `Joint`/`Link` 对象及默认求解器。需要编辑模型或修改
求解器配置时，各 worker 应独立构造模型；重定向 worker 可共享不可变模式描述，
但应分别构造模式管理器。

## 七轴运动学

SRS 在构造和模型更新时准备标准坐标系与偏移模型，不缓存查询目标或工作缓冲。
TCP 修改在下一次查询生效；复制求解器只共享不可变的准备数据。
详细报告只计算所需的奇异值，不生成未使用的奇异向量；限位余量取硬件与用户限位
的交集。闭式分支选择与普通 IK 共用有界周期坐标逻辑。手工 C++ 关节轴的尺度保留
在标准坐标和肘部种子中，导入 URDF 的轴仍会归一化。偏置链的局部精化保留 TCP
和有效限位交集。
标准基坐标系按上臂所在一侧定向，不随首轴正方向翻转。标准肩、肘、腕分支及臂角
描述物理相位；给关节坐标增加完整周期不会改变它们。
SRS 与 FEP 按各关节的坐标周期 `2*pi/axis.norm()` 比较候选；去重比较物理角相位，
分支排序保持关节坐标距离。FEP 的 CPU/CUDA 评分使用相同周期，冗余采样范围按物理
半圈计算。

选择可选操作前，可读取 Python 的 `solver.capabilities` 或通过 C++
`KinematicsBase::GetCapabilities()` 查询。结果快照包含求解器家族、默认 IK 算法、
自由度、Jacobian、冗余求解、闭式 IK、批量 FK、已编译 CUDA 和并发查询能力。
查询不会初始化 CUDA，也不保证某个目标可达；自定义 C++ 求解器默认报告保守能力，
可按实际实现覆盖。Python 的 `KinematicsBase` 是原 `AnalyticKinematics` 基类的
别名，原名称继续可用。

SRS/FEP 继承的 `inverse()` / `GetNearestIK()` 使用数值 IK，专用 `solve()`
需要显式调用。七转动关节链具有 SRS 类型，并不表示满足闭式几何条件；
`supports_closed_form_ik` 按默认几何容差报告这一差别。
`analyze_geometry()` 的自定义容差只影响本次报告，不改变能力快照或求解配置。
FEP 当前使用数值修正，因此不声明闭式 IK 能力。

所有内置求解器提供统一的
`solve_ik(target, seed, selection=IKSelection.NEAREST)`，返回独立持有数据的报告。
`status` 区分 `SUCCESS`、`INVALID_INPUT` 和 `NO_SOLUTION`，`algorithm` 报告
通用入口实际对应的数值或解析算法。调用不改变种子，失败时 `solutions` 和
`distances` 均为空。`IKSelection.ALL` 请求从当前种子可得到的候选，不保证完备枚举。
Python 参数类型或形状错误仍在转换时抛出异常。
C++ 对应 `KinematicsBase::SolveIK()` 和 `IKSolveReport`。
原有 `inverse()`、`solve_all()` 及专用求解 API 的行为保持兼容。

```python
import holistic_motion as hm

report = robot.kinematics.solve_ik(target_pose, seed)
if report.success:
    joints = report.solutions[0]
elif report.status == hm.IKSolveStatus.INVALID_INPUT:
    raise ValueError("invalid IK target or seed")
```

兼容的七旋转关节球形肩腕结构使用 `SRSKinematics`。其零空间投影会拒绝非有限
输入，并采用相对奇异值阈值，避免近奇异位形放大数值噪声。投影直接使用保留的
右奇异向量，不再构造伪逆矩阵。`NullSpacePlanner` 要求起点同时满足硬件和用户
关节限位。非零偏好方向的幅值不改变规划步进，极小和极大的有限输入也适用。
若种子步长无法改变当前浮点关节坐标，或 IK 将候选纠正回当前构型，规划失败并
清空输出路径。
理想 SRS 模型保留
纯闭式结果；存在小量 URDF 连杆偏置时，以闭式结果作为保持分支的初值进行严格
局部修正。等价旋转角直接根据声明的关节限位区间选择，不再依赖固定绕圈次数。

连续目标流应使用有状态 Python 跟踪器，而不是逐帧独立选择：

```python
from holistic_motion.kinematics import SRSContinuousTracker

tracker = SRSContinuousTracker(solver, initial_joints=q0)
result = tracker.solve(target_pose, dt=0.01)
q_command = result.joints
```

跟踪器会在预测状态附近展开周期角，执行构型迟滞，按位姿误差和动态限制过滤候选，
并报告奇异状态、构型切换、速度、加速度和残差。原生 `SRSKinematics` 不保存连续目标跟踪历史，可继续直接使用；
几何量在构造与成功更新模型时计算，`analyze_geometry(tolerance)` 只对只读快照
应用容差。不同容差的并发查询不会写共享状态；配置更新仍需外部同步。
调用方需要非异常状态以及逐候选构型、Jacobian 最小奇异值、逐关节限位余量、
近奇异标记和限位触达标记时，可调用 `solve_detailed(target, seed, method)`；现有
`solve()` 继续作为紧凑的异常式接口。
TCP 与用户坐标系属于显式运行配置。`forward()` 和 `solve()` 继续使用机器人基座
坐标；`forward_user()`、`inverse_user()`、`solve_user()` 与
`solve_detailed_user()` 按兼容 SDK 的 `user_T_base * base_T_tcp` 约定应用用户
变换。可通过 `tcp`、`user_frame` 属性读取，并用 `clear_tcp()`、
`clear_user_frame()` 恢复单位变换。

偏置七轴结构和批量 FK 使用 `FEPKinematics`。小批量使用 CPU，不初始化 CUDA；只有运行时存在
CUDA 设备且批量足以摊薄传输成本时，`AUTO` 才选择 CUDA。FEP IK 在必要的高精度
修正后还会独立检查最终结果，要求位置与角度误差分别不超过 10 微米和 10 微弧度。
显式请求 `CUDA` 时若后端不可用会直接失败，不会静默回退。
高精度修正会继承当前 TCP，使初次求解、修正目标和最终验证使用相同工具坐标系。
CPU 批量 FK 每次调用只准备一次关节轴和固定末端/TCP 变换，各行累积位姿时
不再分配临时关节向量和中间位姿列表。模型与 TCP 修改在下次调用生效。
标量 FK 使用轴角旋转正确处理斜轴旋转关节，C++ 连续关节节点也参与 FK 和
几何 Jacobian。空批量返回 `(0, 4, 4)`。即使关节输入与 TCP 有限，CPU 和 CUDA
批量 FK 也会拒绝非有限输出位姿；C++ 失败时清空全部结果，包括已经完成的行。
连续偏置七轴目标可使用 `FEPContinuousTracker`，其选项和诊断与 SRS 跟踪器一致。
两类跟踪器每帧只求当前构型，周期性刷新全部构型；当前构型失败或接近奇异点时会
立即枚举全部候选。非法候选（含嵌套复数和循环对象容器）会在查询分支信息前被过滤；种子结果不可用时，即使
尚未到周期刷新帧，也会重新枚举，包括因位姿或动态限制被淘汰的种子解。
每帧最多完整枚举一次；刷新后仍失败时保留跟踪历史。
可通过 `candidate_refresh_interval` 调整刷新频率。
两种跟踪器用稳定的位置范数和 `atan2` 旋转角保留小位姿误差；非有限的动态量、
FK 位姿和得分会在排序前被过滤。无法表示的预测会被拒绝，保留已有历史。
初始关节、动态限制、权重和结果快照必须包含实数；复数输入会在转换前抛出
`TypeError`。
跟踪器的标量权重和阈值保存为独立浮点值，修改调用方的零维数组不会改变已验证的
选项；这些标量字段拒绝非标量数组。

数值 IK 在单次求解内复用薄 SVD、Jacobian 和关节步进缓冲区；已收敛种子会跳过
这些缓冲区的分配。工作区属于当前调用，不引入求解器共享可变缓存。C++ 的误差
阈值、步长和阻尼参数要求有限正数，非法值不改变已有设置。`GetIK()` 在校验前
清空解和距离输出，失败调用不会残留上次求解的距离。
若 C++ 种子引用上次 IK 结果中的元素，数值、UR 和 OPW 求解器会先复制种子再
清空输出；独立种子无需额外复制。
C++ IK 目标与 TCP、用户坐标系采用相同校验：系数有限，四元数平方范数与一的
误差不超过 `1e-9`。非法目标返回失败并清空解输出，保留有效的独立种子。
原生 Python FK/IK、SRS/FEP 查询及数值求解器坐标系设置，会在 Eigen 转换前拒绝
复数数组，包括包含复数标量的对象数组。目标矩阵在构造四元数前校验，非法旋转块
或齐次末行会被拒绝。实数列表和对象数组仍受支持；非法配置更新保留原有状态。
SRS/FEP 配置标志和冗余值同样要求实数标量，非法赋值保留原配置。
UR/OPW 参数赋值和构造限位采用相同的实数要求。基座与工具变换必须为刚体变换；
非法赋值保留原参数值。
数值 `GetNearestIK()` 和 Python `inverse()` 在收敛分支的周期表示中，选择相对
调用方有限种子加权关节位移最小的解。改用零种子重试时，仍按原始种子选择多圈
表示并报告距离。
当极端尺度导致奇异值、误差投影或正则化平方项溢出，或损失次正规数精度时，
工作区改用归一化分解与有界的尾数、指数运算。
平移范数在平方项下溢时仍保留非零误差；旋转误差使用 `atan2`，避免旋转矩阵迹
舍入为 3 时丢失小角度。FK 也会保留因范数平方下溢而被忽略的极小转动。
大旋转向量的平方溢出时，FK 使用稳定范数保留仍可表示的旋转幅值；角度或平移本身
无法表示时，仍会拒绝查询。
标量 FK、CPU/CUDA 批量 FK 与周期坐标转换共用主机准备的轴尺度，避免对
`q * axis` 求范数或在设备端重新计算尺度时，因舍入改变大转角相位。六自由度的
原有肘部朝上过滤同样使用轴角运动；前缀几何或候选坐标非法时会清空输出，避免越界。

C++ 求解器在分配与模型维度相关的状态前，统一校验关节字段和坐标布局。数值模型
最后一个节点必须是固定工具变换（兼容接受 `UNKNOWN`）；OPW 和 UR 至少需要六个
坐标槽位，后续节点只能为固定或 unknown 类型。非法模型抛出 `std::invalid_argument`。
`NumericalKinematics::SetDOF()` 只接受当前模型推导出的维度，改变维度需要新建求解器。
仅含一个固定节点的数值模型为零自由度，Jacobian 为 `6 x 0`；只有目标在容差内与
其位姿相符时，IK 才返回一个空关节向量。

`SetJointNode()` 拒绝节点数量变化、不支持的关节类型、非法字段，以及超出坐标槽位的
活动节点。硬件限位和用户限位更新均须保持非空交集；拒绝更新时保留原模型或限位。
C++ TCP 和用户坐标系要求有限系数与单位四元数。FK 还检查有限关节输入引起的算术
溢出：失败时清空位姿列表，并通过 Python 的 `forward()`、`forward_all()` 和
`jacobian()` 抛出 `ValueError`。完整模型校验仅在构造或更新时执行，FK 保留轻量的
维度与运算检查。

`IkRtn::GetLimitsIK()` 在硬件与用户限位的闭区间内展开转动/连续关节等价解。
可行圈数区间直接计算，耗时不再随越界种子的圈数线性增长；普通取余损失精度时，
极大种子改用其正弦/余弦相位，返回角度须以 `1e-10` 容差保留该相位。
非周期坐标只检查范围。原值已在范围内且没有其他周期表示时，直接保留该配置，
跳过圈数枚举及其工作区。
直接通过 C++ `JointNode` 提供的非单位转轴，其坐标周期为 `2*pi/axis.norm()`；
URDF 导入的轴为单位方向。回绕会按 FK 使用的实际旋转角
检查相位，包括斜向转轴。

默认最多生成 65536 个配置，计数发生在后续去重之前。C++ 重载
`GetLimitsIK(nodes, max_solutions)` 可显式传入 1 至 `INT_MAX` 的上限。展开超限
或圈数范围无法表示时返回 `false` 并清空结果，不返回部分解。数值 `solve_all()`
沿用默认上限，因此很宽的多圈限位可能在单关节均存在有效解时仍因组合过多而失败。
解析求解器的最近分支筛选继续使用独立的 `WrapToLimitsNear()` 路径。
数值 `GetNearestIK()` 和 Python `inverse()` 也直接为每个关节选择一个周期表示，
因此可在所有圈数组合超出全解预算时成功。最近解回绕只检查非周期坐标，不改变
其数值；NaN 限位和无法表示的所选周期相位会被拒绝。

`IkRtn` 结果整理以 `ik_joints` 为准，同步缓存计数 `ik_number`。去重按字典序
保留代表解；仅当每个坐标差都严格小于有限、非负容差时合并，零容差只合并完全
相同的解。该确定性贪心规则不把近似相等当作传递关系。距离排序使用调用方提供的
有限距离，距离相同时保留输入顺序。去重、排序和关节过滤均清空 `joint_states`；
输入格式非法时清空解输出，排序失败还会清空距离。每份可变结果由单一调用方或
worker 持有。

## 功能边界

可选的碰撞采样规划器提供由 HolisticMotion 自主实现的一组关节空间 RRT 算法。
控制器、标定、隐式模型下载以及强制可视化依赖仍不属于核心库范围。
