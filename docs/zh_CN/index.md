# HolisticMotion 文档

<div class="language-switcher"><a href="../en/index.html">English</a> · 简体中文</div>

HolisticMotion 提供可复用的 C++17 机器人基础能力和 Python 绑定，涵盖模型加载、
运动学、轨迹、碰撞查询和姿态重定向。机器人资源由调用方提供，项目不会隐式下载。

```{toctree}
:maxdepth: 2
:caption: 用户指南

installation
concepts
collision
trajectory
retargeting
```

```{toctree}
:maxdepth: 2
:caption: 参考资料

cpp_api
python_api
development
project_layout
```

## 开始使用

1. 按照 {doc}`installation` 使用 Conan 构建。
2. 阅读 {doc}`concepts`，了解组件边界和数据约定。
3. 通过 {doc}`retargeting` 使用单臂、双臂和全身 IK。
4. 在 {doc}`cpp_api` 和 {doc}`python_api` 中查询自动生成的 API。
