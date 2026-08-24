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
