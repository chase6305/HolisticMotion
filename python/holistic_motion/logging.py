"""Package-scoped logging with optional native records and JSON formatting."""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
import sys
from threading import RLock, local

PACKAGE_NAME = "holistic_motion"
DEFAULT_FORMAT = (
    "%(asctime)s|%(levelname)s|%(name)s|"
    "%(filename)s-%(funcName)s-%(lineno)04d: %(message)s"
)
_CONFIG_LOCK = RLock()
_OWNED_HANDLERS = []
_NATIVE_DISABLE = None
_NATIVE_DISPATCH = local()
# Inspect the standard record, not the application's configurable factory:
# factory-injected context must remain visible as extras and import must be quiet.
_RECORD_FIELDS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
}


def _json_safe(value, active):
    """Copy JSON containers while preserving otherwise unencodable diagnostics."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    if not isinstance(value, (dict, list, tuple)):
        return value
    identity = id(value)
    if identity in active:
        return "<circular reference>"
    active.add(identity)
    try:
        if isinstance(value, dict):
            return {
                _json_safe(key, active) if isinstance(key, float) else key: _json_safe(
                    item, active
                )
                for key, item in value.items()
            }
        return [_json_safe(item, active) for item in value]
    finally:
        # Track only the current ancestor chain. A shared object encountered
        # again in a separate branch is not a cycle and must retain its content.
        active.remove(identity)


class JsonFormatter(logging.Formatter):
    """Format source metadata, exception text, and LogRecord extras as JSON."""

    def format(self, record):
        data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "filename": record.filename,
            "function": record.funcName,
            "line": record.lineno,
        }
        for name, value in record.__dict__.items():
            if name not in _RECORD_FIELDS and name not in data:
                data[name] = value
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        elif record.exc_text:
            # Transports can remove traceback objects but retain rendered text.
            data["exception"] = record.exc_text
        if record.stack_info:
            data["stack"] = self.formatStack(record.stack_info)
        return json.dumps(
            _json_safe(data, set()), ensure_ascii=False, default=str, allow_nan=False
        )


def get_logger(name=None):
    """Return a logger under holistic_motion, accepting short or full names."""
    if name is None or name == PACKAGE_NAME:
        return logging.getLogger(PACKAGE_NAME)
    if (
        not isinstance(name, str)
        or not name
        or any(not part for part in name.split("."))
    ):
        raise ValueError("logger name must contain non-empty dotted components")
    if not name.startswith(PACKAGE_NAME + "."):
        name = PACKAGE_NAME + "." + name
    return logging.getLogger(name)


def _level(value):
    if isinstance(value, str):
        value = logging.getLevelName(value.upper())
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("level must be a non-negative logging level or level name")
    return value


def setup_logging(level=logging.WARNING, handlers=None, with_json_format=False):
    """Replace package handlers, leaving root and caller-owned handlers unchanged.

    None creates a stdout handler; an explicit sequence uses only those handlers.
    An empty sequence silences output. Custom handlers retain their own level and
    formatter. JSON formatting applies only to the default handler.
    """
    level = _level(level)
    if not isinstance(with_json_format, bool):
        raise TypeError("with_json_format must be boolean")
    selected = [] if handlers is None else list(handlers)
    if any(not isinstance(handler, logging.Handler) for handler in selected):
        raise TypeError("handlers must contain logging.Handler instances")
    owned = []
    if handlers is None:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            JsonFormatter() if with_json_format else logging.Formatter(DEFAULT_FORMAT)
        )
        selected = [handler]
        owned = [handler]
    if not selected:
        selected = [logging.NullHandler()]
        owned = selected.copy()
    unique = []
    for handler in selected:
        if not any(handler is previous for previous in unique):
            unique.append(handler)
    _replace_configuration(unique, owned, level, propagate=False)


def _replace_configuration(selected, owned, level, propagate):
    """Synchronize the native threshold before publishing Python configuration."""
    global _OWNED_HANDLERS
    with _CONFIG_LOCK:
        logger = get_logger()
        selected_ids = {id(handler) for handler in selected}
        retired = [
            handler for handler in _OWNED_HANDLERS if id(handler) not in selected_ids
        ]
        retained = [
            handler for handler in _OWNED_HANDLERS if id(handler) in selected_ids
        ]
        try:
            _sync_native_level(package_level=level)
        except BaseException:
            # Only newly created handlers belong to the failed attempt. Keep
            # previous ownership and borrowed handlers intact, even on Ctrl-C.
            for handler in owned:
                handler.close()
            raise
        logger.handlers = selected
        logger.setLevel(level)
        logger.propagate = propagate
        _OWNED_HANDLERS = owned + retained
    for handler in retired:
        handler.close()


def add_logger_handler(handler):
    """Attach a borrowed handler once, preserving its level and formatter."""
    if not isinstance(handler, logging.Handler):
        raise TypeError("handler must be a logging.Handler")
    with _CONFIG_LOCK:
        get_logger().addHandler(handler)


def reset_logging():
    """Return to application-managed propagation without closing borrowed sinks."""
    handler = logging.NullHandler()
    _replace_configuration([handler], [handler], logging.NOTSET, propagate=True)


def _env_flag(name):
    value = os.environ.get(name, "0").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"{name} must be a boolean flag")


def auto_configure_debug_logging():
    """Enable package DEBUG output only when HOLISTICMOTION_DEBUG is true."""
    if _env_flag("HOLISTICMOTION_DEBUG"):
        setup_logging(
            logging.DEBUG, with_json_format=_env_flag("HOLISTICMOTION_JSON_FORMAT_LOG")
        )


def _emit_native(level, file, line, function, message):
    # User factories, filters and handlers can call native code that logs again.
    # Suppress that thread's nested forwarding before invoking any logging hook.
    # C++ LogError still throws after the bridge returns.
    if getattr(_NATIVE_DISPATCH, "active", False):
        return
    _NATIVE_DISPATCH.active = True
    try:
        logger = get_logger("native")
        severity = (logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG)[level]
        if logger.isEnabledFor(severity):
            record = logger.makeRecord(
                logger.name, severity, file, line, message, (), None, function
            )
            # Native paths can originate on another operating system.
            record.filename = file.replace("\\", "/").rsplit("/", 1)[-1]
            logger.handle(record)
    finally:
        _NATIVE_DISPATCH.active = False


def _disable_native_bridge():
    global _NATIVE_DISABLE
    if _NATIVE_DISABLE is not None:
        _NATIVE_DISABLE(False)
        _NATIVE_DISABLE = None


def _install_native_bridge():
    global _NATIVE_DISABLE
    native = sys.modules.get("holistic_motion._holistic_motion") or sys.modules.get(
        "_holistic_motion"
    )
    if native is not None and hasattr(native, "_set_python_logging"):
        native._set_python_logging(True)
        if _NATIVE_DISABLE is None:
            atexit.register(_disable_native_bridge)
        _NATIVE_DISABLE = native._set_python_logging
        _sync_native_level()


def _sync_native_level(package_level=None):
    if _NATIVE_DISABLE is None:
        return
    native = sys.modules.get("holistic_motion._holistic_motion") or sys.modules.get(
        "_holistic_motion"
    )
    logger = get_logger("native")
    if package_level is None or logger.level != logging.NOTSET:
        level = logger.getEffectiveLevel()
    elif package_level != logging.NOTSET:
        level = package_level
    else:
        # reset_logging restores inheritance from the application hierarchy.
        parent = get_logger().parent
        level = parent.getEffectiveLevel() if parent is not None else logging.NOTSET
    if level <= logging.DEBUG:
        native_level = 3
    elif level <= logging.INFO:
        native_level = 2
    elif level <= logging.WARNING:
        native_level = 1
    else:
        native_level = 0
    native._set_native_log_level(native_level)


# A NullHandler suppresses lastResort output while permitting application root
# handlers to collect library records. Import does not call basicConfig().
if not get_logger().handlers:
    get_logger().addHandler(logging.NullHandler())
