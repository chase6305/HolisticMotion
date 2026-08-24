# 核心概念

<div class="language-switcher"><a href="../en/concepts.html">English</a> · 简体中文</div>

## 显式模型所有权

HolisticMotion 不捆绑机器人资源。请向 `Robot`、`CollisionModel` 和 retargeting
求解器传入绝对路径或由应用解析后的 URDF 路径。

## 核心组件与可选组件

核心 `holistic_motion` 库包含机器人模型、运动学、流形、零空间路径生成和轨迹。
碰撞查询位于可选的 `holistic_motion_collision` 配套库中，因此普通运动学用户
不需要引入 Pinocchio 和 Coal。

## 坐标和顺序约定

- 目标位姿是有限的 4×4 齐次变换矩阵。
- Retargeting 目标使用 URDF 世界坐标系。
- 通用 retargeting 结果采用 Pinocchio 模型关节顺序。
- 控制器特定的关节重排应在应用集成边界完成。

## 功能边界

零空间路径生成和姿态重定向属于求解器相关算法。通用运动规划、控制器、标定、
隐式模型下载以及强制可视化依赖不属于核心库范围。
