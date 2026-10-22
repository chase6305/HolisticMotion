# 开发与文档维护

<div class="language-switcher"><a href="../en/development.html">English</a> · 简体中文</div>

## 仓库结构

- 公共头文件：`include/holistic_motion/`
- C++ 实现：`src/`
- Python 绑定：`bindings/python/`
- Python 包：`python/holistic_motion/`
- C++ 测试：`tests/cpp/`
- Python 测试：`tests/python/`
- Conan 消费端测试：`test_package/`

## C++ 诊断与性能基准

类模板的显式实例化应放在成员定义之后。提前实例化时，GCC 仍可能生成相关符号，
而 Clang 会缺少构造函数和方法。可选 Clang 14 核心任务使用独立的 Conan 依赖组合，
并通过 Python 绑定链接全部轨迹维度。其他可选任务覆盖 PyPy 3.10 和两种 CPython
3.13 ABI 布局。

可选运行时在独立进程中验证。启用 CUDA、测试和 Python 绑定并完成构建后，执行：

```bash
bash scripts/test-cuda.sh --build-dir build/cmake --python /path/to/python -q
# 同时验证可选的 Torch CUDA 轨迹后端：
bash scripts/test-cuda.sh --build-dir build/cmake --python /path/to/python --torch -q
```

该检查要求可用 GPU，CUDA 后端静默回退到 CPU 会使检查失败。它覆盖两个 FEP
内核、批量数组布局和跨线程块的候选评分。CI 另在 CPU runner 编译 CUDA 并验证
无设备回退，同时验证 Conan 导出包和真实 FEP 消费端链接；无需 GPU 也能发现
CUDA 运行时依赖没有正确传递的问题。Python Pinocchio 重定向可使用
`PYTHON_EXECUTABLE=/path/to/python bash scripts/test-retargeting.sh -q`；入口会
选择 toolkit 模式，并在运行前确认 Pinocchio 已安装。可选工作流还构建通用
toolkit wheel，验证受支持的 Python Pinocchio 版本和手部后端。
原生核心任务覆盖 CPython 3.9 和 3.12，并导出启用 Python 的 Conan 包，通过安装后
消费端验证记录的解释器 ABI。3.9 任务检查 NumPy 1.23，3.12 任务检查 sanitizer
及受支持的原生核心加 Python Pinocchio 组合；主碰撞任务使用 CPython 3.11。
主任务还会安装 Python Pinocchio，在新进程中验证默认原生包的两种导入顺序：碰撞
尚未加载时，核心 FK 与重定向可以组合运行；显式访问碰撞时则检查运行时边界。

`concurrent_queries_smoke` 在四个 worker 间共享不可变映射、只读 FEP 求解器和使用
无状态回调的规划器，各 worker 独立持有种子及输出；有设备时运行 CUDA 批处理。
有历史的求解器与可变碰撞工作区仍遵守
[概念](concepts.md) 中的每 worker 独立实例契约。

`benchmarks/joint_mapping_benchmark.py` 比较每次按名称映射、复用原生映射和
预计算 NumPy 映射的开销。例如：

```bash
PYTHONPATH=python:build/cmake python benchmarks/joint_mapping_benchmark.py \
  --joints 64 --selected 7 --continuous --iterations 10000 --repeats 7
```

合成 URDF 使用临时文件；输入解析、映射构造及输出等价性检查在计时外。CSV 报告
交替测量后的耗时中位数和输出校验和。各接口的输入校验仍计入单次调用，模型便捷
接口还会检查完整流形约束。该基准测量完整坐标转换，不代表碰撞或 FK 吞吐量。

`benchmarks/runtime_import_benchmark.py` 在新解释器进程中比较包导入及首次访问碰撞
类型的耗时，交替测量各安装目录，输出中位数和原始样本。Linux 还记录核心导入后
的常驻内存和已映射的 Pinocchio/Coal 动态库：

```bash
python benchmarks/runtime_import_benchmark.py --repeats 11 \
  --package baseline=/path/to/baseline/install \
  --package current=/path/to/current/install --json /tmp/imports.json
```

解释器须匹配两个扩展的 ABI。解释器启动和测量进程的标准库导入不计时；测量保留
操作系统现有文件缓存，不代表冷磁盘读取。该结果用于观察，不设置 CI 性能阈值。

`benchmarks/native_query_concurrency.py` 使用配置不变的共享求解器，比较单线程与
常驻 Python 线程池完成相同工作量的耗时，覆盖 14 关节不可达 IK、4096 行 CPU FK、
SRS 详细报告与 FEP 分支枚举。`--optimization-calls 16` 还会在共享的七坐标、
256 路点优化器上比较 `release_gil=False` 与 `True` 的纯几何路径优化。
模型构造、线程池创建和输出等价性检查在计时外，完整调用仍包含输入转换和独立输出：

```bash
PYTHONPATH=python:build/cmake python benchmarks/native_query_concurrency.py \
  --workers 4 --repeats 7 --optimization-calls 16 --json /tmp/native-threads.json
```

`test_native_concurrency.py` 检查报告/批量输出独立持有，以及日志异常后的恢复；
还在独立进程中确认较长的原生 IK 调用期间其他 Python 线程能前进，不设置加速比阈值。
规划回归覆盖两种 GIL 模式、回调异常和引用回收；外部 SDK 插件直接检查原生回调
所见的 GIL 状态，同时确认默认调用方式保持兼容。

独立的 CPU C++ 开发构建可启用 ASan、UBSan 和 Eigen 断言。选项默认关闭，
要求 GCC 或 Clang。检查覆盖项目源码，Conan 预编译依赖不带插桩；此配置用于
测试，不用于发布安装包。核心 CI 在普通 Python 构建后运行该配置。

```bash
conan install . --output-folder=build-core --build=missing \
  -o '&:with_python=False' -o '&:with_tests=True' \
  -o '&:with_cuda=False' -o '&:with_collision=False'
cmake -S . -B build-sanitizers \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/build-core/build/Release/generators/conan_toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DHOLISTICMOTION_BUILD_PYTHON=OFF -DHOLISTICMOTION_BUILD_TESTS=ON \
  -DHOLISTICMOTION_ENABLE_CUDA=OFF -DHOLISTICMOTION_ENABLE_COLLISION=OFF \
  -DHOLISTICMOTION_ENABLE_SANITIZERS=ON
cmake --build build-sanitizers --parallel 2
ctest --test-dir build-sanitizers --output-on-failure --no-tests=error
```

该辅助开关支持 CPU 核心构建。实测中，本地 ASan 改变了 Eigen 分配/释放路径，
与默认预编译 Pinocchio 混用时，在销毁碰撞工作区时产生分配 ABI 不一致；CMake
现在会在查找依赖前拒绝这一组合。碰撞 sanitizer 验证需要单独构建一致的依赖
instrumentation 组合，不能直接使用该辅助开关。

性能测量使用关闭 sanitizer 的普通 Release 构建，开启
`HOLISTICMOTION_BUILD_BENCHMARKS=ON`，运行
`<构建目录>/sampling_planner_benchmark`。程序输出加权 RRT* 的 CSV 数据，覆盖
2、7、14 自由度及周期关节。各用例固定随机种子、4000 次迭代，预热一次后测量
五次取中位数。`success` 区分找到路径与搜索预算耗尽；未求解用例的零路径长度
不代表路径质量提升。不需要外部机器人资产或基准库，时间不作为 CI 通过门槛。

可传入 `[goal_bias [dof continuous]]` 单独测量一个场景，例如
`sampling_planner_benchmark 0.05 7 1`。显式维度支持 2–32；`continuous=1` 将
第一个关节设为周期关节。CSV 包含路径校验和，应先比较固定种子的结果，再比较
耗时。索引查询缓存非周期坐标的包围盒，并保留保守的舍入余量以正确处理边界并列。
距离查询为每个关节缓存一个字节的拓扑标记，避免扫描候选状态时反复解包位标记，
并保持加权求和顺序。

同一选项还构建 `<构建目录>/fep_batch_benchmark`。默认 CPU 基准使用合成七轴链、
非单位关节原点和 TCP、种子 7，覆盖批量 1/32/128/1024。每个用例预热三次，
随后测量七组，每组调用 `max(1, 16384 / batch_size)` 次，CSV 报告单次调用的
微秒中位数、与标量 FK 的最大误差和位姿校验和。输出容器复用已有容量；测量
不包括 Python 数据转换。传入 `CUDA` 或 `AUTO` 可选择其他后端，每组调用 100 次。
显式 CUDA 要求可用设备，计时包含主机打包、设备内存分配、传输、核函数及输出解包。

`fep_selection_benchmark [CPU|CUDA]` 测量七坐标候选评分的完整调用，与 CPU 参考
对照；数量为 1/32/129/1024，随机种子为 82。每项预热三次，测量七组；CUDA 每组
100 次，CPU 每组 `max(1, 16384 / count)` 次。CSV 包含最大评分误差与校验和。
两个 CUDA 操作均在每次请求中分配一块独立设备缓冲区，批量 FK 的模型常量合并为
一次上传。Linux 的 `cuda_workspace_smoke` 注入分配和传输失败，包括核函数启动
后的输出下载失败，检查资源回收和后续重试。无 GPU 时仍检查尺寸拒绝和分配失败。
候选评分还通过独立三角函数参考检查非单位坐标周期，覆盖 CPU 与 CUDA。种子和七个
周期共用一次上传，每次评分仍为三次传输和一次分配。

`<构建目录>/forward_kinematics_benchmark` 测量合成 0/1/3/7/14/64 坐标混合链及
七轴 SRS/FEP 的标量末端 FK。关节原点和 TCP 非单位变换，轴包含非单位尺度。
先将各状态与完整帧列表对照，再测量七组、每组 65536 次调用，报告纳秒中位数、
每次查询的普通 C++ `operator new` 次数及平移校验和。分配计数不包含对齐分配或
直接 `malloc`。内置数值、SRS 和 FEP 末端查询遍历关节链时不构造帧列表；未知
数值求解器派生类仍通过虚函数 `GetAllFK()` 调度。`forward_kinematics_smoke` 另与
独立矩阵链对照，并检查非法输入和自定义虚调用。

`joint_space_snapshot_benchmark` 测量 2/7/64/256 坐标、含较长关节名称的不可变空间
复制、直接规划及两路点优化。计时为七组、每组 32768 次调用；另用 128 次调用单独
统计分配。CSV 包含纳秒耗时、分配数、计数种类和校验和。Linux GCC/Clang 构建包装
链接的 `malloc`，包含 Eigen 分配；其他平台只统计普通 C++ `new`。快照复制不分配
新坐标数据。`joint_space_snapshot_smoke` 检查复制/移动、builder 隔离、结果持有及
并发快照构造。

`<构建目录>/numerical_ik_benchmark` 测量带非单位 TCP 的合成 1/3/6/7/14 轴链，
覆盖精确种子、附近可达目标和耗尽 60 次迭代的不可达目标。预热一组后测量七组，
三种场景每组分别调用 16384、1024、128 次；CSV 包含单次耗时中位数、成功状态、FK 次数、位姿残差和关节
校验和。比较耗时前应核对成功状态及 FK 次数，失败行的零残差只是占位值。
`damped_ik_workspace_smoke` 独立比较矩形、秩亏矩阵的求解结果与正则化法方程，
并用 Eigen 分配守卫禁止工作区计算步骤中的堆分配，首次计算也包含在检查中。

`kinematic_model_smoke` 覆盖非法构造和模型更新、零位/权重/限位状态保留、非法
坐标系、纯固定节点的 FK/IK 和 FK 溢出。测试通过子类注入空模型、错误自由度、活动
末端节点和空 FK 输出，检查运行时保护。Python 回归通过三个数值 FK/Jacobian 接口
验证有限输入造成的 FK 运算溢出，并确认后续普通查询仍能正常工作。

`concurrent_queries_smoke` 同时启动四个 C++ worker，共享只读 FEP 模型、配置与切向量
布局不同的命名映射，以及通用采样规划器/路径优化器；种子和输出由各自持有。
FK 与独立矩阵链对照，检查 IK 往返，并要求并发规划结果与顺序执行的固定种子参考
一致。启用 CUDA 且设备可用时还检查并发批量 FK，否则检查 CPU 批量路径；该回归也
包含在 CPU sanitizer 构建中。

`<构建目录>/robot_model_benchmark` 测量 URDF 加载和指定基座到末端的求解器创建。
临时生成 7/64/256/1024 个移动关节的链，以及带 1024 个固定叶节点的 256 关节主干。
预热一组后记录七组，每组加载模型 8 次或创建运动链 128 次；CSV 报告单次调用微秒
中位数、自由度和零位末端高度。URDF 生成与 FK 校验在计时外，加载计时包含文件
读取、解析、模型构建与析构。大型用例用于观察扩展成本，不代表常见七轴机械臂延迟。

`robot_model_smoke` 验证导入轴的单位方向语义、重载失败时模型/求解器状态的保留、
C++ 模型重命名、重复名称的首次匹配行为，以及修改父子
关系引入环后查询能够结束。Python 回归覆盖支持分支、mimic 与 planar 分支之间的
默认链选择。`compiler_check_smoke` 使用模拟 CMake 值检查 GCC、Clang、Apple
Clang 和 MSVC 的类型与版本比较，不替代各平台的真实构建。profile 配置见
[安装文档](installation.md)。

`<构建目录>/path_optimizer_benchmark` 比较 2/7/14 关节、32/256 路点的合成路径，
分别使用有限关节或跨越角度接缝的首个连续关节，并比较有、无轻量 C++ 校验回调。
各用例必须完成 12 轮扫描且接受更新；预热一组后记录七组，短路径每组调用 32 次，
长路径每组 4 次。CSV 包含耗时微秒中位数、迭代/更新/校验计数、目标值、最终长度
及路径校验和。比较耗时前应核对其余结果列；本基准不含 Python 回调开销或原生
网格碰撞计算成本。
可用四个位置参数单独选择用例，例如 `path_optimizer_benchmark 7 256 1 0`，
便于逐用例交替测量前后版本。
追加 `--finite-cost` 或 `--analytic-cost` 可启用二次状态代价，分别使用数值或
解析梯度。CSV 同时记录代价模式与回调次数，便于在工作量一致时比较缓冲区复用。

`benchmarks/continuous_tracker_benchmark.py` 使用固定变换和 Jacobian 夹具，测量
1 或 8 个候选的完整 Python 跟踪步骤。计时包含 NumPy SVD 和连续性选择，不包含
原生 FK/IK。使用 `HOLISTICMOTION_PURE_PYTHON=1 PYTHONPATH=python` 运行；
`--baseline` 可比较旧版 `srs_tracker.py`，并核对候选数量与求解结果。
`benchmarks/sphere_fit_benchmark.py --selection-workloads` 覆盖小型、大型及大量
重复候选的两种拟合模式；已有的 `--baseline` 会核对输出，并单独测量分配开销。

`benchmarks/urdf_geometry_benchmark.py` 测量含 1/32/128 个 link 的完整 URDF 加载，
覆盖唯一及重复的 box、cylinder 和 sphere 几何。使用相同纯 Python 环境，传入
`--baseline /path/to/old/urdf_sphere_fit.py` 进行配对测量，并精确核对顶点和面索引；
计时不包含输入文件生成。

`benchmarks/collision_reconfiguration_benchmark.py` 测量合成 7/32/128 关节模型的
公开碰撞对重置、分组修改与障碍物增删操作，每个模型只有两个球体几何。各操作先
预热三次，再测量七组、每组 100 次完整 Python/原生调用，并在计时后校验碰撞对数量
和物理距离。该基准测量几何工作区更新，不代表完整规划吞吐量；这些修改可以复用
不变运动学模型的 `pinocchio::Data`。

`path_geometry_workspace_smoke` 将局部目标变化与完整路径目标比较，并用中心
有限差分校验解析梯度，覆盖周期接缝、反向扫描、多次试探和重合路点。Eigen 分配
守卫覆盖工作区的首次及后续计算。

`<构建目录>/ik_limit_benchmark` 测量 1/7/14 个转动关节的限位展开。窄范围
只保留一个解；宽范围分别生成 2、128 和 16384 个配置。预热一组后测量七组，
窄范围每组调用 8192 次，宽范围的 1/7 关节每组 1024 次，14 关节每组 8 次。
CSV 包含单次微秒中位数、结果数量和校验和；计时包含输入/结果准备，独立相位
及限位校验在计时外。`ik_limit_smoke` 的 CTest 超时为十秒，覆盖极大种子、
闭区间端点、独立圈数枚举、组合输出顺序、预算和失败后的空结果。

`<build-directory>/path_construction_benchmark` 测量原生五次 Bezier 路径构造，
使用 2/7 维、32/512/4096 个合成路点，过渡容差为零，日志级别为 Warning。
CSV 输出一次预热后 30 次调用的中位数和路径长度校验和。计时包含路径构造
及关闭 Debug 后的日志开销，不包含时间参数化、Python 绑定或机器人资产。
应在相同构建和机器条件下比较版本；耗时不作为 CI 阈值。

`<build-directory>/trajectory_construction_benchmark` 覆盖 2/7/20/32 维、4/64 路点、
两种时间剖面，以及有无圆滑过渡的路径。`trajectory_query_benchmark` 在相同维度
及路点数量下比较各阶独立查询与组合状态查询。两者均输出耗时、时长及校验和 CSV；
请在相同机器及构建条件下进行配对测量。

`trajectory_cap_benchmark` 单独计时曲线速度上界采样、完整梯形轨迹构造，以及构造后
再次执行的组合限位检查；重复检查必须保持构造所得时长。
覆盖 2/7/20/32 维、64 个普通尺度路点及 4 个放大 `1e4` 倍的路点；几何、限位和
圆滑容差按同一比例缩放。另有两个 SE3 工况，旋转幅值保持固定，平移、圆滑容差和
六个限位一起缩放，这是指定的工作负载，不是位姿的单位换算。预热一次后测量 21 次，
CSV 标识所在群，并输出 `cap_us`、`limit_us`、`construction_us` 的中位数、
速度上界校验和及时长。各项操作分别计时，比例仅用于定位瓶颈，不等同于构造函数
内部的实测占比；不计几何路径创建时间。

`trajectory_report_benchmark` 测量 Double-S 原生约束报告，覆盖 2/7/20/32 维、4/64 路点，
以及 2/2001/20001 个请求的均匀采样点；报告还会检查阶段端点。共 24 种工况，
CSV 输出一次预热后 21 次调用的微秒中位数和校验和。计时不包含轨迹构造或
Python 绑定开销，不作为 CI 阈值。

`benchmarks/python_trajectory_benchmark.py` 测量完整 Python 调用，包括结果赋值及
上一结果的释放。默认输出 110 个状态查询和均匀采样配置；`--operations report`
选择 88 个约束报告配置，也可组合 `state uniform report`。`--dofs 1 4 32`
选择关节维度，仍包含笛卡尔工况；`--repeats` 控制每个配置的重复次数。
报告校验和包含全部诊断字段及数组形状、dtype、步长、所有权和可写性。例如：

```bash
PYTHONPATH=build/install python benchmarks/python_trajectory_benchmark.py \
  --operations report --dofs 1 4 32 --repeats 9
```

`benchmarks/compare_python_trajectories.py` 用两个常驻且独立的进程比较安装目录，
逐工况交替测量、随机排列顺序，并核对全部返回值校验和。默认覆盖 198 个状态、
均匀采样和报告配置；JSON 保留墙钟/线程 CPU 时间、每次重复，以及实际加载的
扩展路径和哈希。将同一安装目录同时作为基线和候选，可以估计环境噪声。
常驻进程能缩短配对间隔，但不能消除共享机器上的竞争；配对变化为负表示候选
耗时更少。

```bash
python benchmarks/compare_python_trajectories.py \
  --baseline /tmp/baseline-install --candidate build/install \
  --python /path/to/compatible/python --rounds 8 > comparison.json
```

两个安装目录须兼容所选解释器。支持 CPU 亲和性的平台可以用 `--cpu` 将两个
进程固定到同一个允许使用的逻辑 CPU；`--dofs`、`--operations`、`--repeats`
选择与单安装基准相同的工况。计时包括结果赋值、上一结果释放及计时器开销，
校验和计算在计时之外。

`benchmarks/trajectory_audit.py` 按显式种子生成原生轨迹，在均匀时间点和
全部分段边界检查有限性、端点和导数限位。例如执行
`PYTHONPATH=build/install python benchmarks/trajectory_audit.py --cases 3000`。
JSON 将构建拒绝与结果不变量失败分别计数，并记录前十个失败；用相同的
`--seed` 和 `--case-index` 重放。存在任意失败时返回状态码 1，不把仅构建
成功视为通过。这是诊断工具，不证明采样点之间无碰撞或始终满足导数约束，
其中的总耗时也不是独立性能基准。`--distribution stress` 增加不等长路径段、
近共线、近反向以及更宽的各关节限位，最高覆盖 32 维，每种形状均覆盖两种剖面。
重放时也须使用相同的分布选项。构造拒绝同样使命令返回状态码 1。

## 生成式数值回归测试

可选的 `test` extra 包含 pytest 和 Hypothesis。验证已有 Conan 安装时，可以只在
对应 Python 环境中安装测试依赖，避免再次触发原生包构建：

```bash
python -m pip install 'pytest>=7' 'hypothesis>=6.112,<7'
PYTHONPATH=build/install python -m pytest tests/python -q
HYPOTHESIS_PROFILE=deep PYTHONPATH=build/install \
  python -m pytest tests/python -m property --hypothesis-show-statistics
```

默认 `ci` 配置为每组性质生成 40 个确定性样例；`deep` 配置生成 300 个，并使用
Hypothesis 位于 `.hypothesis/` 的本地样例数据库。该目录已被 Git 忽略，重要失败
应转为显式样例或普通回归测试后随修复提交。未安装 Hypothesis 时只跳过生成测试
模块，CI 会安装该依赖。请使用维护中的 Python 补丁版本：Python 3.10.0 的
[带 slots 的 dataclass 初始化缺陷](https://github.com/python/cpython/issues/88815)
会导致当前 Hypothesis 无法运行。

性质覆盖加权关节空间的整圈等价性、混合关节 FK/IK 和几何 Jacobian 的基座变换、
TOPPRA 空间单位与时间缩放，以及增加 RRT* 迭代预算时最佳路径不会变差。
命名坐标性质测试生成含分支的混合关节 URDF、打乱关节选择顺序、保留浮动基座
参考配置，并将原生运动链的点 Jacobian 与完整模型的球体距离切空间梯度比较。
路径平滑性质测试独立重算完整加权目标，检查目标下降、端点精确保留和关节范围，
覆盖混合周期/有限关节及可选的二次状态代价。
规划测试关闭 shortcut，并检查固定迭代预算确实完成；相同时间预算不能保证搜索
相同数量的样例。轨迹采样性质补充已有的多项式解析极值测试，不构成采样点之间
无碰撞或满足动力学约束的证明。

## Viser 场景环境

仓库内的 Viser 应用统一调用
`holistic_motion.visualization.viser.configure_scene(server.scene)`。该入口设置
Z 轴朝上、模糊 Studio 背景、环境光/补光/主光、阴影和带实体底色的公制地板网格。
可通过 `ground_width`、`ground_height`、`ground_z` 和 `ground_center_xy`
匹配场景尺度与偏移；网格或碰撞球检查工具可调用
`configure_scene_from_bounds()`，根据有限包围盒居中地板并放在模型下方。
网格间距按十进制 1-2-5 序列随地板范围自动调整；只有需要固定间距时才传入
`grid_cell_size`。较新 Viser 使用完整效果，过渡版本会逐级减少不支持的环境和
灯光参数，并在网格 API 允许时保留地板位置。无法以有限坐标表示跨度或带边距
地板的包围盒会在创建场景前被拒绝。应用自身的几何体应避开预留的
`/environment` 场景子树。

## 构建文档

```bash
python -m pip install '.[docs]'
./scripts/docs.sh
```

构建会把 Sphinx 警告和无效 API 引用视为错误。生成文件统一位于
`docs/_build/`，不得提交。

修改公共行为时：

1. 更新相关公共头文件注释或 Python docstring。
2. 添加或更新聚焦测试。
3. 同步修改英文和中文指南。
4. 运行 `./scripts/docs.sh` 及受影响的测试套件。
