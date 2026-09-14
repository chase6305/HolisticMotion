# 日志

<div class="language-switcher"><a href="../en/logging.html">English</a> · 简体中文</div>

## 包级配置

HolisticMotion 使用 Python 标准 `logging`，命名空间为 `holistic_motion`。
默认导入仅安装 `NullHandler`，允许日志传播到应用配置的 root handler；
除非通过环境变量显式启用调试，否则不会创建控制台或文件 handler，也不会
修改 root 配置。原生扩展将 C++ 日志转发到 `holistic_motion.native`，保留
级别、文件、行号和函数信息，不依赖解析控制台文本。

```python
from holistic_motion import get_logger, setup_logging

setup_logging(level="INFO")
logger = get_logger("retargeting")  # holistic_motion.retargeting
logger.info("处理完第 %s 帧", 42)
```

`setup_logging()` 默认级别为 WARNING，只替换包级 handler，并关闭向 root
传播，避免重复输出。重复调用替换上一次配置。`handlers=None` 创建 stdout
handler；显式序列只使用传入的 handler；`handlers=[]` 关闭输出。传入列表
不会被修改，同一个 handler 对象只挂载一次。

自定义 handler 的级别和 formatter 保持原样，替换或重置时不会关闭这些
调用方持有的资源。模块自己创建的默认 handler 则会在替换时关闭。
`add_logger_handler(handler)` 可追加一个自定义 handler，重复添加不重复挂载。
`reset_logging()` 恢复 NullHandler 和向 root 传播，不修改应用 root 设置。

setup 和 reset 在替换 Python handler 前同步待生效的原生级别。如果同步
失败或被中断，原有 Python handler、级别和传播设置保持不变，本次新建的
handler 会被关闭，异常继续向调用方抛出。复用模块创建的默认 handler 时，
其所有权仍属于模块，包括 reset 创建的 NullHandler；后续配置移除它时
会负责关闭。

## JSON 与自定义输出

```python
from logging.handlers import RotatingFileHandler
from holistic_motion import setup_logging
from holistic_motion.logging import JsonFormatter

# 应用显式决定路径，并负责关闭这个 handler。
handler = RotatingFileHandler("run.jsonl", maxBytes=10 * 1024 * 1024,
                              backupCount=1, encoding="utf-8")
handler.setFormatter(JsonFormatter())
setup_logging("DEBUG", handlers=[handler])
```

stdout JSON 使用 `setup_logging("INFO", with_json_format=True)`，无需额外
依赖。字段包含时间戳、级别、logger 名称、消息、文件、函数和行号；异常、
堆栈及 `extra` 字段也会保留，中文直接输出。非有限浮点 extra 转换为字符串
`"NaN"`、`"Infinity"` 或 `"-Infinity"`，嵌套字典、列表、元组和非有限浮点
字典键也适用。容器循环引用标记为 `"<circular reference>"`，不同分支重复
引用同一个容器仍保留其内容。转换不修改原始 record 或 extra，有限数值
保留数值类型。其他不可直接序列化的对象转换为字符串。自定义 handler 自行选择
formatter，因此 `with_json_format` 只作用于默认 stdout handler。

应用通过 `LogRecordFactory` 添加的 `trace_id` 等上下文字段会保留在
JSON 中，不受 factory 在包导入前后注册的影响；导入不会调用应用 factory。
传输后的记录若已移除 `exc_info`、仍保留 `exc_text`，JSON 会使用缓存的
异常文本。有可用的 `exc_info` 时优先重新格式化，不修改记录中的缓存文本。

## 使用线程队列后台写入

标准库 `QueueHandler` 和 `QueueListener` 可以将输出目标的 I/O 放到
后台线程。将 JSON formatter 安装在队列 handler 上，输出端直接写入消息：

```python
import logging
from logging.handlers import QueueHandler, QueueListener
from queue import Queue
from holistic_motion import get_logger, reset_logging, setup_logging
from holistic_motion.logging import JsonFormatter

queued = QueueHandler(Queue())
queued.setFormatter(JsonFormatter())
destination = logging.StreamHandler()
destination.setFormatter(logging.Formatter("%(message)s"))
listener = QueueListener(queued.queue, destination)
listener.start()
try:
    setup_logging("INFO", [queued])
    get_logger("training").info("处理完成", extra={"frame": 42})
finally:
    # 排空队列前，先停止并等待应用的日志生产线程退出。
    reset_logging()
    listener.stop()
    queued.close()
    destination.close()
```

JSON 转换仍在产生日志的线程执行，入队前固定消息、extra 和异常内容。
输出端使用普通消息 formatter，避免将 JSON 再次编码。默认 `QueueHandler`
会将异常文本合并到消息，并清空 `exc_info` 和 `exc_text`；仅在 listener
输出端安装 `JsonFormatter` 无法从这种记录恢复独立的异常字段。
队列与 listener 的生命周期由应用管理，setup/reset 不会启动、停止或
排空借用的队列。此示例使用进程内线程队列。

## 环境变量

`HOLISTICMOTION_DEBUG=1` 在导入时启用 DEBUG 日志；同时设置
`HOLISTICMOTION_JSON_FORMAT_LOG=1` 可选择 JSON。仅设置 JSON 开关不会开启
输出。也支持 true/false、yes/no、on/off，不区分大小写；实际使用的开关值
非法时，在改动配置前抛出 `ValueError`。后续可显式调用
`auto_configure_debug_logging()` 重新读取环境变量。

## 原生日志过滤与异常

导入、`setup_logging()` 和 `reset_logging()` 会将
`holistic_motion.native` 的有效级别同步到 C++。低于已同步原生阈值的消息在
格式化或获取 GIL 前被过滤。应用应在导入前配置 root/子 logger 的级别；
后续调整应用管理的层级后，可调用 `reset_logging()` 重新同步原生阈值。

`LogError` 继续抛出 `std::runtime_error`，在 Python 中转换为 `RuntimeError`，
即使错误日志被过滤也仍抛异常。异常文本不含注入的 ANSI 颜色；日志 sink
失败不会替换原算法错误。其他回调失败继续传播。日志配置不改变求解返回值、
收敛诊断或执行控制。

原生日志转发到 Python 时，如果记录创建、过滤、格式化或 handler 回调
再次触发原生日志，会抑制同一线程的嵌套转发，避免递归调用。其他线程仍
独立转发日志；即使回调抛异常或被中断，也会清除防重入状态。嵌套的
`LogError` 仍会抛异常，只抑制其递归进入 Python 的日志记录。该机制不
处理纯 Python 日志递归、跨线程回调循环或独立 C++ sink；显式启用的
原生文件输出保持独立。

C++ 单例不会持有 Python 对象，桥接在 `atexit` 时断开；应用仍需在解释器
关闭前停止工作线程。handler 使用 Python logging 的正常同步机制，配置
适合在初始化或受控重配置阶段调整，不适合在关键运动周期中并发修改。

## 独立 C++ 使用

保留原有 `utility::LogInfo`、`LogWarning`、`LogDebug`、`LogError`、级别、
文本回调及文件接口。独立 C++ 默认级别调整为 WARNING（此前为 INFO），
控制台和异常文本不再注入 ANSI 颜色。`SetRecordFunction()` 接收拥有独立
字符串数据的 `LogRecord`，替代控制台分发，显式启用的文件输出保持独立。
`ResetRecordFunction()` 恢复此前文本 sink；`SetPrintFunction()` 选择文本
sink 并清除结构化回调。

配置受同步保护，回调在配置锁外调用；自定义 C++ 回调需支持并发调用。
文件使用独立的 spdlog 轮转 logger
（10 MiB、一个备份），不注册或修改应用的 spdlog logger。重复启用不增加
sink；启用期间修改路径立即切换文件，打开新文件失败时保留原路径与 sink。
文件记录包含源码位置且不注入 ANSI 颜色。

```cpp
#include <holistic_motion/utility/Logging.h>
using namespace holistic_motion;
utility::SetVerbosityLevel(utility::VerbosityLevel::Info);
utility::SetLoggerFilePath("run.log");
utility::EnableSaveToFile(true);
utility::LogInfo("加载 {} 个关节", 7);
utility::EnableSaveToFile(false);
```

设置 `HOLISTICMOTION_PURE_PYTHON=1` 的纯 Python 工具也可使用这些日志配置，
不需要原生扩展或可视化依赖。
