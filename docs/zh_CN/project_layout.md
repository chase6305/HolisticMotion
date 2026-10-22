# 项目目录

仓库根目录就是项目边界，因此不再额外增加一层包裹 `include`、`src`、`bindings`
和 `python` 的 `holistic_motion/`。C++ 安装命名空间已经位于
`include/holistic_motion/`，Python import 命名空间位于
`python/holistic_motion/`。

```text
include/holistic_motion/   按机器人领域划分的公共 C++ API
src/                       与公共模块对应的 C++ 实现
bindings/python/           编译型 Python binding 单元
python/holistic_motion/    纯 Python 轨迹、retargeting 与可视化 API
examples/python/           按领域整理的 Python 示例
tests/cpp/                 C++ 单元与集成测试
tests/python/              Python 单元与集成测试
cmake/modules/             target 源文件清单和构建模块
docs/en, docs/zh_CN/       中英文文档
scripts/                   受支持的开发入口
```

迁移期间 `python/examples/` 保留兼容启动器。新的可复用 Viser 组件或碰撞策略
代码应进入 `python/holistic_motion/`，而不是继续堆积在 Demo 脚本中。

算法集成按功能领域组织，而不是照搬上游项目名：TOPPRA 时间参数化放在
`python/holistic_motion/trajectory/`，Pink 风格 IK 放在
`python/holistic_motion/kit/retargeting/`。上游仓库不会嵌套进本目录，也不会
注册为 Git submodule。

`bindings/python/module.cpp` 初始化核心扩展，`collision_module.cpp` 在首次使用时
初始化可选碰撞扩展。领域注册分别放在
`RobotBindings.cpp`、`KinematicsBindings.cpp`、`SevenAxisBindings.cpp`、
`TrajectoryBindings.cpp` 以及规划/碰撞文件中。可选碰撞适配和类型链接到独立的
`_holistic_motion_collision` 模块，核心 `_holistic_motion` 只注册延迟加载的碰撞
工厂，从而保留公开规划 API，并避免普通导入链接 Pinocchio/Coal。按名称转换坐标的共享入口是 `robot/JointPositionMap.h` 和纯 Python 的
`configuration.py`，后者还可适配 Python Pinocchio 模型，无需加载原生扩展。

轨迹注册将笛卡尔操作交给 `CartesianTrajectoryBindings.cpp`，关节模板分到四个
编译单元，分别覆盖 1–8、9–16、17–24 和 25–32 维。私有头文件提供共享数组转换
和关节包装实现，入口文件不再让单个编译器进程实例化所有维度。

原生轨迹模板负责路径几何与段落组装。较重的 `PathBezierCurve`、`PathSegment`、
`TrajectoryBase` 和 `TrajectoryReport` 将算法定义保存在私有 `*Impl.h` 中；`.cpp`
分别实例化 1–8、9–16、17–24 和 25–32 维，SE(3) 放在首批。
`TrajectoryInstantiationBatches.h` 集中定义类型列表，报告文件只实例化报告成员。
这限制了单个编译器进程的内存，同时保留一份算法定义和现有安装头文件；静态
消费者可按实际引用的维度选取归档成员。
`src/trajectory/DoubleSProfile.cpp` 和
`TrapeziumProfile.cpp` 各自实现一份与 Lie 群维度无关的标量时钟；原有 protected
剖面方法仍作为包装入口，保持子类兼容。

`src/kinematics/JointMotion.h` 统一标量 FK、批量 FK 与周期坐标转换的主机轴尺度，
CUDA 接收主机已准备的尺度。`JointPeriod.h` 让普通 IK 与 SRS 闭式分支共用最近
可行周期坐标选择。SRS 标准几何和轴尺度作为 `SRSKinematicsInternal.h` 中的不可变
准备数据，请求工作区仍由每次调用独立持有。`RobotConfigFilter.cpp` 独立维护原有机器人配置几何
策略；`utility.cpp` 负责关节限位转换，`IKResults.cpp` 负责结果校验、
确定性去重、距离排序与关节过滤。

TOPPRA 和重定向公共模块保留现有入口。私有同级模块分别承担样条构造、数值约束、
速度剖面校验与修复、
任务/碰撞目标及有界 QP 计算；算法内部实现应放在这些模块中，避免增加公共导入或
让核心 toolkit 依赖可选机器人运行时。安装元数据与导入边界检查由 `runtime.py`
管理。
