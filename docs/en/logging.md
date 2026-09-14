# Logging

<div class="language-switcher">English · <a href="../zh_CN/logging.html">简体中文</a></div>

## Package configuration

HolisticMotion uses Python's standard `logging` hierarchy under `holistic_motion`.
Import installs a `NullHandler`, permits propagation to application root handlers,
and creates no console or file handler unless debug logging is explicitly enabled
by environment. Root configuration is never changed. The native extension routes
C++ records to `holistic_motion.native` with their original severity, file, line,
and function, rather than parsing formatted console strings.

```python
from holistic_motion import get_logger, setup_logging

setup_logging(level="INFO")
logger = get_logger("retargeting")  # holistic_motion.retargeting
logger.info("Completed frame %s", 42)
```

`setup_logging()` defaults to WARNING and replaces handlers only on the package
logger, with propagation disabled to avoid duplicate root output. Repeated setup
replaces the previous configuration. `handlers=None` creates a stdout handler;
an explicit sequence selects only those handlers, and `handlers=[]` silences
output. Supplied lists are copied and duplicate handler objects are attached once.

Custom handlers are borrowed: their levels and formatters are preserved, and they
are not closed on replacement or reset. Default handlers created by this module
are closed when replaced. `add_logger_handler(handler)` attaches an additional
borrowed handler once. `reset_logging()` returns to a NullHandler and root
propagation; it does not change application root settings.

Setup and reset synchronize the pending native level before replacing Python
handlers. If synchronization fails or is interrupted, the existing Python
handlers, level, and propagation remain in place; handlers created for the
failed attempt are closed and the exception is re-raised. A retained default
handler remains module-owned, including NullHandlers created by reset, and is
closed when a later configuration removes it.

## JSON and custom destinations

```python
import logging
from logging.handlers import RotatingFileHandler
from holistic_motion import setup_logging
from holistic_motion.logging import JsonFormatter

# The application explicitly chooses the path and owns this handler.
handler = RotatingFileHandler("run.jsonl", maxBytes=10 * 1024 * 1024,
                              backupCount=1, encoding="utf-8")
handler.setFormatter(JsonFormatter())
setup_logging("DEBUG", handlers=[handler])
```

For stdout JSON, use `setup_logging("INFO", with_json_format=True)`. JSON needs
no additional dependency and includes timestamp, level, logger name, message,
filename, function and line. Exception/stack text and `extra` fields are included;
Unicode remains readable. Non-finite float extras become the strings `"NaN"`,
`"Infinity"`, or `"-Infinity"`, including inside dictionaries, lists and tuples;
non-finite float dictionary keys use the same strings. Circular container
references become `"<circular reference>"`. Shared containers in separate
branches retain their contents. Conversion does not mutate the original record
or extras, and finite numbers retain their numeric types. Other non-JSON objects
are represented as strings. Custom handlers select their own
formatter, so `with_json_format` affects only the default stdout handler.

Context fields added by an application `LogRecordFactory`, such as `trace_id`,
are included whether the factory is registered before or after package import.
Import does not call that factory. When a transported record has no `exc_info`
but retains `exc_text`, JSON uses that cached exception text. A live `exc_info`
takes precedence and is formatted without changing the record's cached text.

## Background output with a thread queue

Standard `QueueHandler` and `QueueListener` can move destination I/O to a worker
thread. Format JSON on the queue handler and write the resulting message directly
at the destination:

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
    get_logger("training").info("Completed frame", extra={"frame": 42})
finally:
    # Stop/join application producers before draining the listener.
    reset_logging()
    listener.stop()
    queued.close()
    destination.close()
```

JSON conversion still runs on the emitting thread and snapshots the message,
extras and exception before enqueueing. The destination uses a plain message
formatter so it does not encode JSON a second time. A default `QueueHandler`
merges exception text into the message and clears `exc_info` and `exc_text`;
placing `JsonFormatter` only on the listener cannot recover a separate exception
field from that record. Queue/listener lifecycle belongs to the application;
setup/reset do not start, stop or drain borrowed queues. This example uses an
in-process thread queue.

## Environment configuration

`HOLISTICMOTION_DEBUG=1` enables DEBUG logging at package import.
`HOLISTICMOTION_JSON_FORMAT_LOG=1` selects JSON for that debug handler. The JSON
flag alone does not enable output. Flags also accept true/false, yes/no and
on/off, case-insensitively. Invalid active flags raise `ValueError` before
configuration changes. `auto_configure_debug_logging()` rechecks the environment
when explicitly called later.

## Native filtering and error behavior

The C++ threshold is synchronized with the effective `holistic_motion.native`
level at import, `setup_logging()` and `reset_logging()`. Messages below that
native threshold are filtered before formatting or acquiring the GIL. Configure application
root/child levels before import, or call `reset_logging()` after changing an
application-managed hierarchy to resynchronize the native threshold.

`LogError` still throws `std::runtime_error`, translated to `RuntimeError` in
Python, even when error output is filtered. Its exception text is plain, without
ANSI escapes. A failing logging sink cannot replace that primary algorithm error.
Other callback failures propagate. Logging configuration does not alter solver
return values, convergence diagnostics, or execution control.

Native-to-Python dispatch suppresses nested native records on the same thread
while a record is being created, filtered, formatted, or handled. This prevents
recursive logging when one of those callbacks calls native code. Other threads
continue to forward records independently, and the guard is cleared even when
a callback raises or is interrupted. Nested `LogError` calls still throw; only
their recursive Python log forwarding is suppressed. This guard does not cover
pure Python logging recursion, cross-thread callback cycles, or standalone C++
sinks; explicitly enabled native file output remains independent.

The bridge holds no Python object in the C++ singleton and is disconnected by an
`atexit` hook. Applications remain responsible for stopping worker threads before
interpreter shutdown. Normal handler synchronization is provided by Python
logging; changing configuration is intended for initialization or controlled
reconfiguration, not concurrent changes during a critical motion cycle.

## Standalone C++

Existing `utility::LogInfo`, `LogWarning`, `LogDebug`, `LogError`, verbosity,
print-callback and file APIs remain available. The standalone default threshold
is now WARNING (previously INFO). Console and exception output no longer inject
ANSI colors. `SetRecordFunction()` accepts structured `LogRecord` values and
replaces console dispatch while leaving explicitly enabled file output independent.
`ResetRecordFunction()` restores the previous text sink; `SetPrintFunction()`
selects a text sink and clears structured dispatch.

Configuration is synchronized; callbacks run outside the configuration lock.
Custom C++ callbacks must support concurrent calls. File output uses a private rotating spdlog logger (10 MiB,
one backup) and does not register or modify application spdlog loggers. Enabling
it twice does not add a second sink. Changing the path while enabled switches
files immediately; failure to open a replacement preserves the current path and
sink. File records include source metadata and have no injected ANSI escapes.

```cpp
#include <holistic_motion/utility/Logging.h>
using namespace holistic_motion;
utility::SetVerbosityLevel(utility::VerbosityLevel::Info);
utility::SetLoggerFilePath("run.log");
utility::EnableSaveToFile(true);
utility::LogInfo("Loaded {} joints", 7);
utility::EnableSaveToFile(false);
```

Python-only toolkits retain logging support with
`HOLISTICMOTION_PURE_PYTHON=1`; no compiled extension or visualization dependency
is required for package logging.
