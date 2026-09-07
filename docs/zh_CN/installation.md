# 安装

<div class="language-switcher"><a href="../en/installation.html">English</a> · 简体中文</div>

## 环境要求

- 支持 C++17 的编译器和 CMake 3.22 或更高版本。
- Conan 2.28 或更高版本。
- 构建 Python 绑定时需要 Python 3.9 或更高版本。
- 涉及机器人模型的操作需要调用方提供 URDF。

## 标准构建

```bash
./scripts/build.sh
```

这一条命令默认构建 Python 和碰撞组件。使用 `--cuda` 可加入可选 CUDA 后端，
使用 `--no-collision` 可进一步精简构建。

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
| `with_tests` | 构建 C++ 冒烟测试 | `False` |
| `with_cuda` | 构建 CUDA FEP 批处理后端 | `False` |
| `with_collision` | 构建 Pinocchio/Coal 碰撞库 | `True` |

关闭 CUDA 和碰撞组件的最小 CPU 构建：

```bash
conan install . --output-folder=build --build=missing \
  -o '&:with_cuda=False' -o '&:with_collision=False'
```

默认配置下 Pinocchio 和 Coal 都是直接 Conan 依赖，由 `CMakeDeps` 提供 targets，
CMake 不会查找未受管理的系统副本。

## C++ 目标边界

使用方只链接实际需要的组件。运动学、轨迹和规划 API 使用
`HolisticMotion::holistic_motion`；碰撞查询使用独立的
`HolisticMotion::collision`，后者会传递 Pinocchio 和 Coal：

```cmake
find_package(HolisticMotion REQUIRED CONFIG)
target_link_libraries(my_motion_app PRIVATE HolisticMotion::holistic_motion)
target_link_libraries(my_collision_app PRIVATE HolisticMotion::collision)
```

CMake 安装目录和 Conan `CMakeDeps` 使用完全相同的目标名。开启 CUDA 时，只有
核心目标会传递 `CUDA::cudart`；碰撞组件仍然独立于 CUDA 和核心库。
