# 安装

<div class="language-switcher"><a href="../en/installation.html">English</a> · 简体中文</div>

## 环境要求

- 支持 C++17 的编译器和 CMake 3.22 或更高版本。
- Conan 2.28 或更高版本。
- 构建 Python 绑定时需要 Python 3.9 或更高版本。
- 运行 Python 包的解释器中需要 NumPy 1.23 或更高版本。
- 涉及机器人模型的操作需要调用方提供 URDF。

## 标准构建

```bash
./scripts/build.sh
```

这一条命令默认构建 Python 和碰撞组件。使用 `--cuda` 可加入可选 CUDA 后端，
使用 `--no-collision` 可进一步精简构建。

Linux 本地构建会将 Conan 依赖目录写入各库的运行时搜索路径。
`source scripts/activate.sh` 和 `./scripts/run.sh python3 ...` 会启用本地安装，
同时保留调用方的动态库路径，供原生碰撞与 Python 重定向各自的独立进程使用。
旧安装和其他平台继续使用 Conan 运行环境。导出的 Conan 包保留标准配置；
本地搜索路径设置由 `scripts/build.sh` 启用。

启用 Python 的 Conan 包会通过运行环境的 `PYTHONPATH` 暴露安装目录。消费端测试
选择包中记录的 Python 次版本，验证安装后的绑定、模型加载、关节名称映射和轨迹
求值。运行 `conan create` 前，需在该解释器中安装 NumPy；Conan 管理原生依赖，
不负责安装 Python 运行时包。

Conan 的全局运行环境可能覆盖 Python Pinocchio 所需的动态库。Linux 下仅供本机
缓存使用的 Python 导出包，可启用各库独立的搜索路径：

```bash
conan create . --no-remote -o '&:with_python=True' -o '&:shared=True' \
  -c tools.build:jobs=2 \
  -c 'tools.cmake.cmaketoolchain:extra_variables={"CMAKE_INSTALL_RPATH_USE_LINK_PATH": True}'
```

此模式会记录 Conan 缓存的绝对路径，移动缓存后需重新构建。
需要可迁移的 Conan 包时使用默认配置。

手工配置仅供本机缓存使用的 CMake 安装时，应与构建脚本一样传入
`-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON`。默认的可迁移安装则需要在链接和运行
消费端时使用 Conan 运行环境。CMake 找到依赖包并不等于已经配置操作系统的动态库
加载器；各库独立的搜索路径也不证明两套 Pinocchio 运行时兼容，仍须遵守下文的
进程组合约定。

应针对安装目录运行测试，避免从源码目录导入旧文件：

```bash
PYTHONPATH=$PWD/build/install pytest tests/python
```

## 编译器与 Conan profile

新生成的 Conan 工具链会记录所选编译器类型与版本。CMake 在配置依赖之前，将其
与实际 C++ 编译器比较；不一致时中止配置，并提示如何选择匹配的编译器。
这项检查只覆盖编译器类型和版本，不验证预编译依赖的完整 ABI。独立 CMake 构建
及缺少此元数据的旧工具链会跳过检查；已有项目需要重新生成 Conan 工具链才能启用。

`scripts/build.sh` 通过 `CONAN_PROFILE_HOST` 和 `CONAN_PROFILE_BUILD` 接收
profile 名称或路径，默认均为 `default`。例如，系统已在以下路径安装 GCC 12 时，
可将以下内容保存为 `/tmp/hmotion-gcc12.profile`：

```ini
include(default)

[settings]
compiler=gcc
compiler.version=12
compiler.libcxx=libstdc++11
compiler.cppstd=gnu17

[conf]
tools.build:compiler_executables={"c": "/usr/bin/gcc-12", "cpp": "/usr/bin/g++-12"}
```

切换编译器后使用新的构建目录：

```bash
CONAN_PROFILE_HOST=/tmp/hmotion-gcc12.profile \
CONAN_PROFILE_BUILD=/tmp/hmotion-gcc12.profile \
BUILD_DIR="$PWD/build-gcc12" ./scripts/build.sh --tests --no-collision
```

profile 的 settings 描述目标编译器，`tools.build:compiler_executables` 选择实际
可执行文件，详见 [Conan CMakeToolchain 官方文档](https://docs.conan.io/2/reference/tools/cmake/cmaketoolchain.html)。
该示例不修改全局默认 profile。交叉编译时，应分别选择适用于目标平台与构建机器的
host、build profile。

## 可选功能

| Conan 选项 | 作用 | 默认值 |
|---|---|---|
| `with_python` | 构建 pybind11 扩展 | `True` |
| `python_version` | 纳入原生包身份的 Python 主/次版本 | `auto` |
| `python_abi` | 纳入原生包身份的解释器实现及对象布局 | `auto` |
| `with_tests` | 构建 C++ 冒烟测试 | `False` |
| `with_cuda` | 构建 CUDA FEP 批处理后端 | `False` |
| `cuda_architectures` | 纳入 CUDA 二进制包身份的 GPU 目标 | `auto` |
| `cuda_version` | 纳入二进制包身份的 CUDA 工具包主次版本 | `auto` |
| `with_collision` | 构建 Pinocchio/Coal 碰撞库 | `True` |

关闭 CUDA 和碰撞组件的最小 CPU 构建：

```bash
conan install . --output-folder=build --build=missing \
  -o '&:with_cuda=False' -o '&:with_collision=False'
```

默认配置下 Pinocchio 和 Coal 都是直接 Conan 依赖，由 `CMakeDeps` 提供 targets，
CMake 不会查找未受管理的系统副本。

`cuda_architectures=auto` 记录 `tools.cmake.cmaketoolchain:extra_variables` 中的
`HOLISTICMOTION_CUDA_ARCHITECTURES`，缺省为 `75;80;86;89;120`。显式选项也会配置
CMake。不同目标集合及 `-real` / `-virtual` 代码选择使用不同二进制包身份；
顺序与重复项会被规范化。显式选项与 CMake 配置冲突时会报错。关闭 CUDA 时，
该选项不参与包身份。

```bash
conan install . --output-folder=build-cuda --build=missing \
  -o '&:with_cuda=True' -o '&:cuda_architectures=120'
```

Conan CUDA 包要求显式数值目标列表。`all`、`native` 等依赖编译器或设备解析的
关键字仍可用于独立 CMake 构建；在 Conan 选择二进制包之前，它们不能确定可复现的
架构集合。CUDA 编译器及工具包通过构建环境选择。

`cuda_version=auto` 依次从 CMake extra variables 的 `CMAKE_CUDA_COMPILER`、
`tools.build:compiler_executables["cuda"]`、`CUDACXX` 和 PATH 选择 `nvcc`，
探测版本后将编译器固定给 CMake。不同工具包主次版本使用不同包 ID。
在 Conan 解析图时没有编译器、需要使用预编译包的环境中，可显式设置版本，
例如 `-o '&:cuda_version=12.8'`。CMake 在构建前校验实际工具包与包身份一致；
安装后的 CMake 和 Conan 消费端也要求相同的工具包主次版本，允许补丁更新。
这是保守的受支持组合，避免隐式替换 CUDA 运行时。关闭 CUDA 时，两个 CUDA
选项都不参与包身份。

`python_version=auto` 记录运行 Conan 的解释器版本；通过
`tools.cmake.cmaketoolchain:extra_variables` 指定 `Python3_EXECUTABLE` 时，则探测
该解释器。不同 Python 小版本使用不同原生包 ID；`with_python=False` 会从纯 C++
包身份中移除此选项。CMake 校验实际解释器与记录的小版本一致。
`scripts/build.sh` 会自动传递 `PYTHON_EXECUTABLE` 的小版本；也可显式指定：

```bash
conan create . --no-remote -o '&:python_version=3.12' \
  -c 'tools.cmake.cmaketoolchain:extra_variables={"Python3_EXECUTABLE": "/path/to/python3.12"}'
```

各 Python 小版本应使用独立构建/安装目录，使用已准备 Conan toolchain 的原生
pip 构建也一样。新元数据提供 `runtime_info().native_python_version`，包在加载
扩展前拒绝不匹配的小版本；同一小版本的补丁更新仍可使用。可移植 toolkit wheel
不带原生 Python ABI 要求。

`python_abi=auto` 同时探测选定解释器的 ABI。普通 CPython 使用 `cpython`，调试与
自由线程布局分别增加 `d` / `t`；PyPy 则包含其扩展 ABI 标记。同一 Python 小版本
的这些变体使用不同包 ID。只指定另一个 `python_version`、未指定解释器时，默认
目标为普通 CPython；其他布局应指定 `Python3_EXECUTABLE` 或显式 `python_abi`。
CMake 校验实际解释器，`runtime_info().native_python_abi` 让导入过程在加载扩展前
拒绝不匹配 ABI。纯 C++ 包移除两个 Python 选项；`scripts/build.sh` 同时传递所选
解释器的版本和 ABI。

ABI 隔离不改变各实例的并发契约。本扩展未声明无需 GIL，自由线程 CPython 使用其
[默认扩展 GIL 行为](https://docs.python.org/3.13/howto/free-threading-extensions.html)。

Linux 核心构建已在 CPython 3.9、3.10、3.12、3.13、自由线程 CPython 3.13 及
PyPy 3.10 上验证。可选原生 ABI CI 任务覆盖 PyPy 和两种 3.13 布局，关闭碰撞与
CUDA。这些检查不扩展下文单独验证的 Python Pinocchio wheel 组合范围。

## Python 原生运行时组合

仅安装 Python 工具层（重定向、TOPPRA、几何预处理和仿真适配接口）：

```bash
python -m pip install . \
  -Ccmake.define.HOLISTICMOTION_PYTHON_TOOLKIT_ONLY=ON \
  -Cwheel.platlib=false
```

该构建生成 `py3-none-any` wheel，不查找 C++ 编译器、Eigen、pybind11 CMake 包或 Conan
依赖，也不会加载原生扩展。可将 `.` 替换为 `'.[retargeting]'` 或其他可选 Python
依赖组。纯工具包自动启用工具层模式，无需设置环境变量；它不提供原生 `Robot`
和碰撞类型。默认原生构建继续使用上文的 Conan/CMake 工作流；从 pip 构建原生包
也需要显式传入已准备好的 Conan 工具链。

| 使用方式 | 同一进程中支持的组件 |
|---|---|
| 默认原生构建 | C++ 核心，可配合 Python Pinocchio 重定向或按需加载的 Conan 碰撞库 |
| `--no-collision` 原生构建 | C++ 核心，可同时使用 Python Pinocchio 重定向 |
| `HOLISTICMOTION_PURE_PYTHON=1` | Python Pinocchio 重定向、NumPy/PyTorch 轨迹及其他纯 Python 工具 |
| 纯 Python 工具包 wheel | 与上一行相同，安装元数据自动启用该模式 |

从原有单体 Python 扩展更新时，应一起重建并安装原生包、Python 文件与生成的
元数据。启用碰撞的安装现在包含核心与碰撞两个扩展文件，公开的 `holistic_motion`
导入名称继续可用。C++ 消费者也应根据更新后的公开头文件重新编译。
扩展与其链接的核心、碰撞库会核对由原生源码、编译设置和构建配置生成的身份。
公开碰撞加载器在注册 Python 类型前查询候选二进制；不匹配或无法核验的碰撞扩展
会返回明确的重建提示。替换原生二进制后，应启动新的 Python 进程。

独立安装的 Python Pinocchio 与 Conan 碰撞库应运行在不同进程。
Pinocchio 版本号相同也不能证明 Coal、Boost 和编译配置的 ABI 一致。
普通 `import holistic_motion` 只加载原生核心。首次访问 `hm.CollisionModel` 等碰撞
类型或调用碰撞规划工厂时，才加载独立的碰撞扩展。公开加载器会在加载碰撞前检查
Python Pinocchio 及独立的 `coal`、`hppfcl`、`eigenpy` 模块；重定向后端也会在
导入 Python Pinocchio 前检查原生碰撞组件，并发首次导入由同一把锁串行处理。
`from holistic_motion import *` 请求所有公开类型，因此在构建启用碰撞时会加载碰撞扩展。直接通过第三方代码导入依赖不经过这些检查，
因此仍需遵守进程边界。
Python Pinocchio worker 应从未启用全局 Conan 运行环境的环境启动：子进程会继承
动态库搜索路径，进程分离不会重置 `LD_LIBRARY_PATH`。本地构建脚本使用各库独立的
运行时路径，使原生应用与独立 Python worker 可以保留各自的依赖查找方式。

已通过可选检查验证的 Linux 运行时组合如下：

| 运行时 | Pinocchio | Boost | URDFDOM / TinyXML2 |
|---|---|---|---|
| Conan 碰撞组件 | 3.8.0 | 1.88.0 | Conan 管理的依赖 |
| Python 最低版本 wheel 组合 | 3.8.0 | 1.89.0 | 4.0.1 / 10.0.0 |
| Python 已验证 4.x wheel 组合 | 4.1.0 | 1.90.0 | 6.0.0 / 11.0.0 |

同一 Pinocchio 版本也可能使用不同的 Boost 库。使用 3.8.0 Python wheel 时，应显式
安装其配套解析依赖：

```bash
python -m pip install --only-binary=:all: 'pin==3.8.0' 'coal==3.0.2' \
  'cmeel-urdfdom==4.0.1' 'cmeel-tinyxml2==10.0.0'
```

单独安装 `pin==3.8.0` 可能解析到 URDFDOM 6，并在导入时报告缺少
`liburdfdom_sensor.so.4.0`。最低版本 CI 在 Python 3.10 上固定上述已验证组合；
当前版本任务在 Python 3.11 上跟随 `pin>=3.8,<5` 及其解析出的依赖。
纯 toolkit wheel 独立支持 Python 3.9 及更新版本。Python 3.9 不在已验证的重定向
wheel 组合中：Linux 安装探测能找到 Pinocchio 3.8 wheel，但无法解析其所需的
`eigenpy>=3.12` 二进制依赖。`--only-binary` 可避免安装过程自动转为未经验证的
原生源码构建。

```bash
# 默认构建的原生碰撞应用
./scripts/run.sh python3 examples/python/collision/basic_query.py --help

# 在独立进程中使用 Python Pinocchio
./scripts/run-python-toolkit.sh python3 examples/python/retargeting/pink_dual_arm.py --help
```

环境变量必须在导入包之前设置，不能通过运行中修改变量卸载已经加载的动态库。
使用 `holistic_motion.runtime_info()` 可查看实际加载的组件、构建开关和 Pinocchio
版本；此查询不会加载 Pinocchio、Torch 或碰撞扩展。`collision_compiled` 表示构建
支持，`collision_loaded` 表示当前进程是否已实际加载碰撞扩展。未找到构建元数据时，未知的
构建能力为 `None`。安装后会包含 CMake 生成的 `_build_config.py` 元数据。

## C++ 目标边界

使用方只链接实际需要的组件。运动学、轨迹和规划 API 使用
`HolisticMotion::holistic_motion`；碰撞查询使用独立的
`HolisticMotion::collision`，后者依赖 Pinocchio 和 Coal：

```cmake
find_package(HolisticMotion REQUIRED CONFIG)
target_link_libraries(my_motion_app PRIVATE HolisticMotion::holistic_motion)
target_link_libraries(my_collision_app PRIVATE HolisticMotion::collision)
```

CMake 安装目录和 Conan `CMakeDeps` 使用完全相同的目标名。开启 CUDA 时，只有
核心目标会传递 `CUDA::cudart`；碰撞组件仍然独立于 CUDA 和核心库。

已安装的 CMake SDK 支持显式选择组件；不指定组件时保持加载全部已构建组件的行为。
选择 `core` 不查找 Pinocchio/Coal，选择 `collision` 不加载核心库的 fmt/spdlog 依赖：

```cmake
find_package(HolisticMotion REQUIRED CONFIG COMPONENTS core)
target_link_libraries(my_motion_app PRIVATE HolisticMotion::holistic_motion)

# 需要两个组件时：COMPONENTS core collision
# 仅使用碰撞时：COMPONENTS collision
# 可选探测：COMPONENTS core OPTIONAL_COMPONENTS collision
```

两个组件均支持可选探测，通过 `HolisticMotion_core_FOUND` 或
`HolisticMotion_collision_FOUND` 检查是否可用。缺少可选核心依赖（包括 CUDA）
不影响仅使用碰撞组件；请求未构建的必需
组件或未知组件时会给出明确诊断。Conan 的完整依赖图仍由包选项决定，精简 Conan
安装应使用 `with_collision=False`。
