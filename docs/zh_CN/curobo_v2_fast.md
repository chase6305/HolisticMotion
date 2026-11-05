# cuRoboV2-Fast 集成提案

<div class="language-switcher"><a href="../en/curobo_v2_fast.html">English</a> · 简体中文</div>

状态：设计提案。2026-11-05 核对 HolisticMotion `da453ca` 与上游 cuRobo
`b29a8c2ee6869585bb05b056b9ab53b9c0753106`。本页没有引入运行时集成。
下文区分已查证的上游 API 和拟议的 HolisticMotion 实现。

## 建议与首期范围

先将 NVIDIA cuRobo 接为调用方显式选择的可选 Python 规划后端，验证机器人、
碰撞、动力学及轨迹所有权后，再评估独立的 C++ 部署组件。保留现有 CPU 采样
规划器和 TOPPRA 接口。

[论文](https://arxiv.org/pdf/2611.06913v1) 将 GPU RRT、三次 B 样条轨迹优化、
可微逆动力学与固定路径 TimeOpt 组合起来。其 RTX 5090、3 kg 负载下平均
4.4 ms 是上游基准，不能视为本项目集成后的预期延迟。TimeOpt 在固定样条上调整
速度，约束采样点的速度、加速度、jerk 和力矩；这值得接入，但不意味着现有轨迹
接口可以直接互换。

首期限定为固定基座、具名标量旋转／移动关节、固定锁定关节集合、单场景、串行
请求，以及起终点静止的关节目标。浮动基座、主动耦合／mimic 坐标、非静止边界和
批量规划需要另行验证。GPU 失败必须保留失败语义；不能静默替换成约束不同的 CPU
路径。调用方仍可通过现有 API 使用 CPU 规划。

## 现有能力与缺口

| 现有能力 | 集成决策 |
| --- | --- |
| `JointSpace` 保存名称、限位、权重和周期拓扑快照 | 用具名标量顺序作为边界，验证与 cuRobo 主动坐标兼容。 |
| `SamplingPlanner`、`PathOptimizer` 输出几何路点 | 保持其几何语义；拟合样条可能改变路径，需要重新验证碰撞。 |
| NumPy／Torch TOPPRA | 当前约束速度和加速度，没有力矩及 jerk 约束；保留其契约，作为独立基线。 |
| 可选 `CollisionModel`、`SphereCollisionModel` | 在可表达时复用调用方几何和碰撞对配置；不支持的转换应报错。 |
| `kit.retargeting.CuroboRetargetingSolver` | 是基于 Pinocchio 重定向实现的本地多种子求解器，没有导入 NVIDIA cuRobo；保留原 API，并与新后端明确区分。 |
| FEP 原生 CUDA | 不代表已经安装 cuRobo、设备兼容或共享模型内存；需要独立探测。 |

## 已核对的上游入口

v2 已改写 v1 API。新集成不应继续基于旧的
`curobo.wrap.reacher.motion_gen.MotionGen`。参见
[上游 README](https://github.com/NVlabs/curobo/blob/b29a8c2ee6869585bb05b056b9ab53b9c0753106/README.md)。
以本页核对的提交作为初始复现参考，在基准中记录实际安装版本／提交。

| 用途 | 核对提交中的公开 API |
| --- | --- |
| 配置 | `curobo.motion_planner.MotionPlannerCfg.create(...)`；显式设置 `graph_planner_algorithm="rrt"`、`enable_timeopt=True`。图规划算法默认是 `"prm"`。 |
| 构造与预热 | `MotionPlanner(config)`、`planner.warmup(enable_graph=True, ...)`，构造一次并复用。 |
| 关节目标 | `planner.plan_cspace(goal_state, current_state, ...)`，输入带名称的 `JointState`；目标在前、起点在后。 |
| 后续位姿目标 | `planner.plan_pose(goal_tool_poses, current_state, ...)`，用 `GoalToolPose` 指定工具名称。 |
| 单条成功结果 | 检查非 `None` 和成功掩码后，调用 `result.get_interpolated_plan()`，保留返回的关节顺序及 `dt`。 |
| 独立重定时 | `curobo.trajectory_timeopt` 导出的 `TrajectoryTimeOpt`、`TrajectoryTimeOptCfg`、`TrajectoryTimeOptResult`。 |
| 场景／负载 | `update_world(...)`、`reset_seed()`、`update_link_inertial(...)`、`update_links_inertial(...)`；碰撞物体附着是另一项操作。 |

依据：[规划配置](https://github.com/NVlabs/curobo/blob/b29a8c2ee6869585bb05b056b9ab53b9c0753106/curobo/_src/motion/motion_planner_cfg.py)、
[规划实现](https://github.com/NVlabs/curobo/blob/b29a8c2ee6869585bb05b056b9ab53b9c0753106/curobo/_src/motion/motion_planner.py)、
[TimeOpt 公开导出](https://github.com/NVlabs/curobo/blob/b29a8c2ee6869585bb05b056b9ab53b9c0753106/curobo/trajectory_timeopt.py)、
[TimeOpt 实现](https://github.com/NVlabs/curobo/blob/b29a8c2ee6869585bb05b056b9ab53b9c0753106/curobo/_src/motion/motion_trajectory_timeopt.py)。
这里引用私有实现用于核查；适配器应从公开模块导入。

## 适配器边界

建议后续实现先放在一个职责完整的模块
`python/holistic_motion/kit/curobo_planner.py`。仅在显式构造该后端时导入
cuRobo／Torch／Warp，普通包导入与 CPU 安装继续独立运行。调用方负责准备经验证的
CUDA 专用上游环境；本 PR 不修改依赖、不安装 cuRobo、不下载模型。
上游当前要求 Python 3.10+，而本项目支持 3.9+，版本限制只作用于可选后端。
还需评估上游包含 Viser 在内的传递依赖，不能将其变为本项目核心依赖。

首个适配器接受调用方提供的绝对 URDF、mesh 根目录、cuRobo 机器人配置路径、具名
`JointSpace` 和显式场景。必须验证配置描述的是同一 URDF／模型及坐标系；仅路径
存在不足以证明模型一致。先使用经验证的已有配置，不急于加入自动 URDF 转换器。

| 边界 | 必须满足的行为 |
| --- | --- |
| 关节顺序 | 建立双向名称排列，拒绝重名、缺失和不支持的主动关节，保留锁定关节上下文；不能将完整 Pinocchio `nq` 直接当标量位置。 |
| 周期关节 | 导出 CPU 路径时使用请求拓扑及 `JointSpace.UnwrapPath`／Python `unwrap_path`；核对 cuRobo 的代表区间与绕圈行为，不支持时拒绝，不能跨 seam 裁剪。 |
| 坐标与单位 | 使用米、弧度、秒，将目标和障碍物变换至配置基座坐标系，显式转为上游 `wxyz` 四元数；用 FK 比对基座和工具。 |
| 数值转换 | 转换前验证有限实数；转为 float32 CUDA 后再次检查有限值、限位及约定的转换误差。 |
| 碰撞几何 | 显式转换支持的 link 局部球、世界 cuboid 和排除对，区别 link／joint 坐标系，匹配 margin 及机器人／环境启用对；无法等价表达的场景应报错。 |
| 动力学 | 要求有效惯量、质心坐标系、重力、速度／加速度／jerk 限制及非对称 effort 上下限；缺失数据时阻止力矩受限模式，不能补零惯量或无限 effort。 |
| 负载 | 分别更新惯性与附着碰撞几何，预留附着容量，附着时关闭世界副本，释放时在实际释放位姿恢复。 |

构造阶段先比对中性及非平凡配置的 FK、关节限位、代表性自碰撞／环境碰撞结果。
球近似需要检查覆盖，URDF 相同不代表碰撞几何相同。已安装可选原生碰撞组件时，
可用 Coal mesh 查询独立验收。CPU Python validator 不能直接插入已捕获的 GPU
碰撞循环，应作为最终验证阶段，并将成本计入端到端延迟。

## 轨迹与失败契约

为后端返回独立的带时间结果，不向几何 `PlanningResult` 添加无关字段。
首个实现只需要成功／失败诊断，以及成功时自行持有的关节名称、显式时间、位置、
速度、加速度数组。仅在实际评估后提供 jerk／力矩，保留上游诊断，不虚构与 CPU
等价的迭代数或碰撞检查数。

对单条规划结果使用公开 helper 去掉 padding，从返回的 `dt` 构造时间，并一致
重排所有导数。独立 TimeOpt 使用其返回的 `time` 数组，不能由路径采样网格推定
均匀时间。下一次求解或负载更新可能覆盖复用缓冲区，需先复制结果，并测试所有权。

不要再经过 `ToppraTrajectory` 或通用插值器重新拟合；这可能改变几何、导数、时间
及力矩可行性。未来若封装统一轨迹接口，必须评估原始合格表示并保留其 owner，
否则明确输出只是采样数据。

结果为空、成功掩码为假、非有限输出、无效 horizon、端点错误、转换不一致或最终
验证失败时，不得返回可使用的轨迹。保留上游失败细节，不能把所有失败猜成 timeout。
未来可显式提供 CPU fallback，但需独立结果及约束契约。

## TimeOpt 与部署路线

1. **先接完整 GPU 流程。** 由 cuRobo 管理 RRT 初始化、样条优化和重定时，先验证
   上游行为，避免在求解器边界同时引入第二种路径表示。
2. **再评估 CPU 规划加 GPU 重定时。** 独立 TimeOpt 接受固定三次 B 样条控制布局、
   起终点及可选原始 `knot_dt`，不接受任意路点表或本项目自然三次样条。能精确转换
   时验证等价，否则明确拟合并重新检查改变后的几何，再重定时。当前实现要求 CUDA
   float32、1–64 自由度、32 个时间控制量，能力检查应绑定验证过的上游版本。
3. **最后评估可选原生 runtime。** 上游已经提供
   [C++ 规划与 TimeOpt](https://github.com/NVlabs/curobo/blob/b29a8c2ee6869585bb05b056b9ab53b9c0753106/cpp_runtime/README.md)。
   先验证 Python，再导出匹配模型／布局／GPU 的 artifacts，评估单独启用的 C++
   组件及其制品生命周期、打包和 ABI；这不能仅靠链接现有 FEP CUDA target 完成。
   控制器、标定及可视化继续留在核心库之外。

求解器、场景容量及附着容量一次构造并复用；计时前预热所有实际使用分支，包括
RRT fallback。场景、负载及锁定状态修改与求解串行化；场景变化使图搜索状态失效，
不兼容布局或容量变化后重新初始化并预热。并发请求使用独立工作区。若使用独立
TimeOpt 实例，也必须同步同一份更新后的动力学。

上游明确将 TimeOpt 成功定义为采样可行。三次路径在 knot 处可能发生 jerk 跳变，
检查时覆盖两侧。加密后的独立碰撞／动力学检查能改善验收覆盖，但不能证明连续
时间可行。固定路径重定时也不能保证对时变障碍物安全，后者需要另行建模。

## 后续 PR 与验收依据

| 阶段 | 交付物与通过条件 |
| --- | --- |
| 1：隔离适配器 | 显式可选导入、合格模型／配置、固定基座静止边界 `plan_cspace`、持有数据的带时间输出、清晰的不支持输入错误；CPU 导入和既有 API 回归通过。 |
| 2：场景及动力学 | 覆盖阻塞路径、自碰撞、坐标系／margin、负载附着及释放；使用相同惯量与约定，独立对比 Pinocchio FK／逆动力学。 |
| 3：重定时与部署 | 验证独立路径转换及 TimeOpt，再按实际部署需求评估 C++ 制品。 |

坐标、坐标系及异常测试使用临时 URDF。GPU 集成测试由调用方显式传入机器人配置、
URDF、mesh 路径并记录；缺少 GPU 或跳过不能算集成验收通过。覆盖关节名置换、
周期 seam、float32 精度损失、无效惯量／effort、起终点碰撞、静止路径、重力／负载
力矩不可行、陈旧场景及多次调用结果所有权。使用独立网格验证轨迹导数和力矩，
包含 knot 两侧及端点。

在同一请求、几何、限位、margin 和预算下，比较 CPU RRT-Connect 加现有 TOPPRA、
完整 cuRobo，之后再加入混合路线。报告成功率与拒绝输出数、端到端延迟中位数及
p95／p99、轨迹时长、最小间隙和导数／effort 利用率。分别报告冷启动构造／编译／
捕获、热 GPU 求解、数据传输、最终检查及适配器总耗时，计时同步 CUDA。
记录 GPU／CPU、driver／CUDA、Python／Torch／Warp／cuRobo 版本、种子及配置。
比较轨迹时长时必须说明约束差别：当前 TOPPRA 基线不约束力矩和 jerk。
性能目标待完成同条件测量后制定。

本提案仅以源码核查和文档构建验证，没有完成 cuRobo 安装、GPU 运行、动力学数值
对照，也没有宣称本项目获得加速。
