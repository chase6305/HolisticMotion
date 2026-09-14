# 仿真、相机与可视化后端

<div class="language-switcher"><a href="../en/simulation_backends.html">English</a> · 简体中文</div>

## 范围与实施状态

`holistic_motion.kit.simulation` 提供三类后端协议、`BackendRegistry` 和
`SimulationSession`，用于组合调用方提供的适配器。内置 Viewer 仅有无依赖
的 `NullViewer`；本模块不指定或安装物理引擎、相机渲染器及平台 Viewer。

HolisticMotion 提供运动计算，应用层负责引擎选型、动作语义、模型资产和训练
循环。具体集成通过这些接口实现，并自行管理可选依赖。

## 三类独立接口

| 接口 | 当前契约 | 实现来源 |
| --- | --- | --- |
| `PhysicsBackend` | 重置全部环境、按明确动作语义步进、返回原生状态、关闭资源 | 调用方提供 |
| `ObservationBackend` | 采集指定状态并返回原生 payload、关闭资源 | 调用方提供 |
| `VisualizationBackend` | 按独立频率发布状态、关闭资源 | 内置 NullViewer 或调用方提供 |

## 已实现的组合 API

显式注册工厂函数：物理工厂接收配置中的关键字参数，相机/Viewer 工厂额外
接收 `physics=engine`。可选依赖放在工厂内部导入。注册器在调用任何工厂前
检查后端名称、声明的场景格式和频率；消费者格式 `None` 表示不依赖具体
场景表示，格式不兼容时需要显式注册桥接适配器。`options` 按 `physics`、
`observation` 和 `viewer` 三个类别提供参数映射。

```python
from holistic_motion.kit.simulation import BackendRegistry

def compose(make_physics, make_camera):
    registry = BackendRegistry()
    registry.register("physics", "engine", make_physics, scene_format="engine-v1")
    registry.register("observation", "camera", make_camera, scene_format="engine-v1")
    return registry.create(
        physics="engine", observation="camera", viewer="null",
        capture_every_n_steps=4, publish_every_n_steps=20,
    )
```

将返回的 session 用作上下文管理器，先 `reset()`，再 `step(action, dt)`。
物理协议实现 `reset`、`step`、`close`，相机协议实现 `capture(StateFrame)`
和 `close`，可视化协议实现 `publish(StateFrame)` 和 `close`。已经创建的
适配器也可直接传给 `SimulationSession`。

每步返回 `StepResult.state`、可选的 `ObservationFrame` 和 `viewer_error`。
状态包含会话级 episode ID、step ID、仿真时间和原生 payload；相机结果引用
同一个状态封装，payload 保持原对象，不复制或转换数组。未到采集周期时
观测为 `None`，不会返回上一帧。payload 仍可能是可变或借用的缓冲区，适配器
须约定有效期与设备同步；组合层不校验相机元数据，也不保证 GPU 任务已完成。

当前重置作用于全部环境，并重新计数步数和采集周期。局部重置、逐环境
episode ID 和异步执行仍待实现。物理或相机异常向上传播，下一次 step 前
必须 reset，避免自动重试已经推进物理状态的动作。Viewer 异常会被报告并
在本会话中永久停用展示，reset 也不会恢复它；物理和采集继续推进。
`publish` 本身必须不阻塞，有界 Web 队列由适配器实现。

session 按逆序释放其拥有的适配器；工厂失败时关闭此前成功创建的适配器，
保留原始构造异常。显式 `close()` 尝试释放全部资源，将失败记录在
`close_errors` 并报错，重复 close 不重复释放。上下文退出时保留已有的应用
异常。工厂自身抛出异常前，须自行释放尚未交付的部分资源。

清理同样处理 `KeyboardInterrupt` 和 `SystemExit`：先尝试关闭剩余的
全部适配器，再由显式 `close()` 重新抛出第一个此类中断。`close_errors`
按清理顺序保留普通异常和中断。已有的构建异常或 with 块业务异常优先于
清理错误，包括清理期间的中断。此机制无法强制阻塞的 `close()` 返回，
也不能处理进程被直接终止的情况。

`reset`、`step`、`close` 共享会话级非阻塞互斥保护。并发调用或适配器回调
中的重入调用，在触及适配器或帧计数前抛出包含 `already running` 的
`RuntimeError`，不会自动排队。关闭前须等待当前调用结束，`close` 不负责
取消正在执行的 step。成功、异常和中断都会释放保护；物理或采集被中断后
仍须 reset。Viewer 中未捕获的重入错误按原有展示失败策略处理。

该保护只约束会话状态迁移，不保护绕过会话访问的适配器或借用缓冲区。
具体后端仍可能要求固定线程或显式 GPU 同步，不保证跨线程依次调用就安全。
UI 请求应交给应用循环处理，不应在回调内递归调用会话操作。

运行 `python examples/python/simulation/backend_composition.py` 可验证合成
示例在关闭 Viewer 时仍于第 2、4、6 步生成观测；该示例不进行机器人动力学
仿真或相机渲染。

## 兼容性与数据边界

注册器在创建适配器前比较声明的场景格式。接口独立不代表引擎和相机可以
任意组合：消费者须理解引擎的原生表示，或通过显式桥接适配器转换。
组合层不提供隐式引擎或渲染器回退。

应用按名称和自由度布局映射关节，明确单位及坐标系。配置与速度维度可能
不同（`nq != nv`）；位置、速度和力矩动作在不同适配器间须保持明确语义。

图像适配器应声明环境/相机 ID、时间戳、内外参、形状、dtype、颜色空间、
深度含义、有效掩码、设备、缓冲区所有权与完成事件。训练数据保留环境和
相机两个批量维度，显示拼图在 Viewer 内处理。通用层只传递不透明 payload，
不校验这些图像专属字段。

## 采集与显示独立

`observation=None` 关闭采集，`viewer="null"` 关闭显示。使用空 Viewer 不影响
已启用的观测适配器。物理、采集和显示各自计数，训练终止由应用预算或
取消请求决定，不由窗口状态决定。

Web 适配器应使用有界展示队列，将展示丢帧策略与训练观测交付分开。
异步消费者完成前不能复用设备缓冲区，具体同步策略由适配器负责。

## 验证范围

合成适配器测试覆盖工厂选择、兼容性拒绝、独立频率、重置和失败恢复、
资源清理及会话互斥保护，可运行 `pytest tests/python/test_simulation_backends.py`
验证接口契约。合成示例不代表真实引擎集成、相机精度、GPU 同步或训练吞吐
已验证；这些需要使用应用最终选择的适配器和实际负载另行验证。
