# C++ API 参考

<div class="language-switcher"><a href="../en/cpp_api.html">English</a> · 简体中文</div>

以下声明由 Doxygen 和 Breathe 根据公共头文件自动生成。

## 碰撞模型

```{doxygenclass} holistic_motion::robotics::collision::CollisionModel
:members:
```

```{doxygenclass} holistic_motion::robotics::collision::SphereCollisionModel
:members:
```

## 机器人模型

```{doxygenclass} holistic_motion::robotics::Robot
:members:
```

## 运动学能力与求解报告

所有内置求解器通过 `KinematicsBase::GetCapabilities()` 暴露能力快照。
默认 IK 算法对应通用 `GetIK()` / `GetNearestIK()`；SRS/FEP 专用求解方法需显式调用。

```{doxygenstruct} holistic_motion::robotics::KinematicsCapabilities
:members:
```

```{doxygenstruct} holistic_motion::robotics::IKSolveReport
:members:
```

`KinematicsBase::SolveIK()` 不改变调用方种子，返回独立持有数据的报告，
区分输入非法和没有可用解。

## 关节名称与坐标映射

```{doxygenstruct} holistic_motion::robotics::JointCoordinate
:members:
```

```{doxygenclass} holistic_motion::robotics::JointPositionMap
:members:
```

## 采样规划器

```{doxygenclass} holistic_motion::robotics::planning::JointSpace
:members:
```

```{doxygenclass} holistic_motion::robotics::planning::SamplingPlanner
:members:
```

```{doxygenclass} holistic_motion::robotics::planning::PathOptimizer
:members:
```

其他公共类会在头文件注释通过文档告警规则后加入自动生成的参考页面。
