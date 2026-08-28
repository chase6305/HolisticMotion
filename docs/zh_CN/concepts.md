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

## 七轴运动学

兼容的七旋转关节球形肩腕结构使用 `SRSKinematics`。其零空间投影会拒绝非有限
输入，并采用相对奇异值阈值，避免近奇异位形放大数值噪声。理想 SRS 模型保留
纯闭式结果；存在小量 URDF 连杆偏置时，以闭式结果作为保持分支的初值进行严格
局部修正。等价旋转角直接根据声明的关节限位区间选择，不再依赖固定绕圈次数。

连续目标流应使用有状态 Python 跟踪器，而不是逐帧独立选择：

```python
from holistic_motion.kinematics import SRSContinuousTracker

tracker = SRSContinuousTracker(solver, initial_joints=q0)
result = tracker.solve(target_pose, dt=0.01)
q_command = result.joints
```

跟踪器会在预测状态附近展开周期角，执行构型迟滞，按位姿误差和动态限制过滤候选，
并报告奇异状态、构型切换、速度、加速度和残差。原生 `SRSKinematics` 仍保持
无状态，可继续直接使用。

偏置七轴结构和批量 FK 使用 `FEPKinematics`。小批量使用 CPU；只有运行时存在
CUDA 设备且批量足以摊薄传输成本时，`AUTO` 才选择 CUDA。FEP IK 在必要的高精度
修正后还会独立检查最终结果，要求位置与角度误差分别不超过 10 微米和 10 微弧度。
显式请求 `CUDA` 时若后端不可用会直接失败，不会静默回退。
连续偏置七轴目标可使用 `FEPContinuousTracker`，其选项和诊断与 SRS 跟踪器一致。
两类跟踪器每帧只求当前构型，周期性刷新全部构型；当前构型失败或接近奇异点时会
立即枚举全部候选。可通过 `candidate_refresh_interval` 调整刷新频率。

## 功能边界

可选的碰撞采样规划器提供由 HolisticMotion 自主实现的一组关节空间 RRT 算法。
控制器、标定、隐式模型下载以及强制可视化依赖仍不属于核心库范围。
