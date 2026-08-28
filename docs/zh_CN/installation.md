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
