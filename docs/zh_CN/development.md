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

性能测量使用关闭 sanitizer 的普通 Release 构建，开启
`HOLISTICMOTION_BUILD_BENCHMARKS=ON`，运行
`<构建目录>/sampling_planner_benchmark`。程序输出加权 RRT* 的 CSV 数据，覆盖
2、7、14 自由度及周期关节。各用例固定随机种子、4000 次迭代，预热一次后测量
五次取中位数。`success` 区分找到路径与搜索预算耗尽；未求解用例的零路径长度
不代表路径质量提升。不需要外部机器人资产或基准库，时间不作为 CI 通过门槛。

同一选项还构建 `<构建目录>/fep_batch_benchmark`。该 CPU 基准使用合成七轴链、
非单位关节原点和 TCP、种子 7，覆盖批量 1/32/128/1024。每个用例预热三次，
随后测量七组，每组调用 `max(1, 16384 / batch_size)` 次，CSV 报告单次调用的
微秒中位数、与标量 FK 的最大误差和位姿校验和。输出容器复用已有容量；测量
不包括 Python 数据转换和 GPU 传输。

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

`<构建目录>/robot_model_benchmark` 测量 URDF 加载和指定基座到末端的求解器创建。
临时生成 7/64/256/1024 个移动关节的链，以及带 1024 个固定叶节点的 256 关节主干。
预热一组后记录七组，每组加载模型 8 次或创建运动链 128 次；CSV 报告单次调用微秒
中位数、自由度和零位末端高度。URDF 生成与 FK 校验在计时外，加载计时包含文件
读取、解析、模型构建与析构。大型用例用于观察扩展成本，不代表常见七轴机械臂延迟。

`robot_model_smoke` 验证 C++ 模型重命名、重复名称的首次匹配行为，以及修改父子
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

`benchmarks/trajectory_audit.py` 按显式种子生成原生轨迹，在均匀时间点和
全部分段边界检查有限性、端点和导数限位。例如执行
`PYTHONPATH=build/install python benchmarks/trajectory_audit.py --cases 3000`。
JSON 将构建拒绝与结果不变量失败分别计数，并记录前十个失败；用相同的
`--seed` 和 `--case-index` 重放。存在任意失败时返回状态码 1，不把仅构建
成功视为通过。这是诊断工具，不证明采样点之间无碰撞或始终满足导数约束，
其中的总耗时也不是独立性能基准。

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
路径平滑性质测试独立重算完整加权目标，检查目标下降、端点精确保留和关节范围，
覆盖混合周期/有限关节及可选的二次状态代价。
规划测试关闭 shortcut，并检查固定迭代预算确实完成；相同时间预算不能保证搜索
相同数量的样例。轨迹采样性质补充已有的多项式解析极值测试，不构成采样点之间
无碰撞或满足动力学约束的证明。

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
