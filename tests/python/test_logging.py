"""Package logging contracts, including actual C++ to Python forwarding."""

import io
import json
import logging
import math
import os
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from queue import Queue
from threading import Barrier

import holistic_motion.logging as package_logging
import pytest
from holistic_motion import (
    add_logger_handler,
    auto_configure_debug_logging,
    get_logger,
    reset_logging,
    setup_logging,
)
from holistic_motion.logging import JsonFormatter


class Collector(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
        self.closed_by_setup = False

    def emit(self, record):
        self.records.append(record)

    def close(self):
        self.closed_by_setup = True
        super().close()


@pytest.fixture(autouse=True)
def restore_logging():
    logger = get_logger()
    state = logger.handlers[:], logger.level, logger.propagate
    native = get_logger("native")
    old_level = native.level
    reset_logging()
    native.setLevel(logging.NOTSET)
    yield
    reset_logging()
    logger.handlers, logger.level, logger.propagate = state
    native.setLevel(old_level)


def native_module():
    native = sys.modules.get("holistic_motion._holistic_motion") or sys.modules.get(
        "_holistic_motion"
    )
    if native is None:
        pytest.skip("native extension not loaded in pure Python mode")
    assert hasattr(native, "_log_native"), "rebuild the extension to test logging"
    return native


def test_import_is_quiet_and_preserves_application_root(tmp_path):
    env = dict(os.environ, HOLISTICMOTION_PURE_PYTHON="1", HOLISTICMOTION_DEBUG="0")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "python")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import logging
root = logging.getLogger()
h = logging.StreamHandler()
root.addHandler(h)
root.setLevel(logging.ERROR)
import holistic_motion
assert root.handlers == [h] and root.level == logging.ERROR
holistic_motion.get_logger('test').warning('hidden')
""",
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == result.stderr == ""
    assert not list(tmp_path.iterdir())


def test_names_share_package_hierarchy():
    assert get_logger("retargeting") is get_logger("holistic_motion.retargeting")
    assert get_logger().name == "holistic_motion"


@pytest.mark.parametrize("name", ["", ".bad", "bad..name", 5])
def test_invalid_logger_names(name):
    with pytest.raises(ValueError):
        get_logger(name)


def test_repeated_setup_deduplicates_and_preserves_borrowed_handlers():
    root = logging.getLogger()
    before = root.level, root.handlers[:]
    handler = Collector()
    formatter = logging.Formatter("custom %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    supplied = [handler, handler]
    for _ in range(3):
        setup_logging("DEBUG", supplied, with_json_format=True)
    add_logger_handler(handler)
    get_logger("test").debug("filtered")
    get_logger("test").info("once")
    assert [record.getMessage() for record in handler.records] == ["once"]
    assert supplied == [handler, handler]
    assert handler.formatter is formatter and handler.level == logging.INFO
    assert (root.level, root.handlers) == before
    setup_logging(handlers=[])
    assert not handler.closed_by_setup


def test_default_handler_is_replaced_and_closed(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    setup_logging("INFO")
    old = get_logger().handlers[0]
    closed = []
    monkeypatch.setattr(old, "close", lambda: closed.append(True))
    setup_logging("INFO")
    get_logger("test").info("one message")
    assert closed == [True]
    assert output.getvalue().count("one message") == 1


@pytest.mark.parametrize("reset", [False, True])
def test_reset_handler_is_owned_until_replaced(reset, monkeypatch):
    reset_logging()
    previous = get_logger().handlers[0]
    closed = []
    original_close = previous.close

    def close():
        closed.append(previous)
        original_close()

    monkeypatch.setattr(previous, "close", close)
    if reset:
        reset_logging()
    else:
        # Reusing an owned handler must preserve ownership until it is removed.
        setup_logging("INFO", [previous, previous])
        assert get_logger().handlers == [previous]
        assert closed == []
        setup_logging(handlers=[])
    assert closed == [previous]


def test_empty_handlers_silence_and_reset_restores_root_propagation(caplog):
    setup_logging("DEBUG", handlers=[])
    with caplog.at_level(logging.DEBUG):
        get_logger("test").error("hidden")
        reset_logging()
        get_logger("test").warning("visible")
    assert [record.message for record in caplog.records] == ["visible"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"level": "BAD"},
        {"level": True},
        {"handlers": [object()]},
        {"with_json_format": "yes"},
    ],
)
def test_invalid_setup_preserves_current_configuration(kwargs):
    handler = Collector()
    setup_logging("INFO", [handler])
    before = get_logger().handlers[:], get_logger().level, get_logger().propagate
    with pytest.raises((ValueError, TypeError)):
        setup_logging(**kwargs)
    assert (get_logger().handlers, get_logger().level, get_logger().propagate) == before
    assert not handler.closed_by_setup


@pytest.mark.parametrize("operation", ["default", "borrowed", "silent", "reset"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_failed_configuration_preserves_output_and_releases_candidates(
    operation, error_type, monkeypatch
):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    setup_logging("INFO")
    logger = get_logger()
    previous = logger.handlers
    old = previous[0]
    closed = []
    original_close = old.close

    def close_old():
        closed.append(old)
        original_close()

    monkeypatch.setattr(old, "close", close_old)
    # Populate the effective-level cache before trying a different level.
    child = get_logger("configuration_test")
    assert child.isEnabledFor(logging.INFO)
    borrowed = Collector()
    candidates = []
    failure = error_type("native configuration failed")

    class PendingStreamHandler(logging.StreamHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            candidates.append(self)

        def close(self):
            closed.append(self)
            super().close()

    class PendingNullHandler(logging.NullHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            candidates.append(self)

        def close(self):
            closed.append(self)
            super().close()

    def fail_sync(*args, **kwargs):
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(logging, "StreamHandler", PendingStreamHandler)
        patch.setattr(logging, "NullHandler", PendingNullHandler)
        patch.setattr(package_logging, "_sync_native_level", fail_sync)
        with pytest.raises(error_type) as caught:
            if operation == "reset":
                reset_logging()
            elif operation == "default":
                setup_logging("ERROR")
            else:
                setup_logging("ERROR", [borrowed] if operation == "borrowed" else [])

    assert caught.value is failure
    assert logger.handlers is previous
    assert logger.level == logging.INFO and not logger.propagate
    assert old not in closed and not borrowed.closed_by_setup
    assert closed == candidates
    child.info("still using previous configuration")
    assert output.getvalue().count("still using previous configuration") == 1
    # A failed replacement must not lose ownership of the original handler.
    setup_logging("WARNING", [borrowed])
    assert closed == candidates + [old]
    child.warning("replacement works")
    assert [record.getMessage() for record in borrowed.records] == ["replacement works"]


def test_json_contains_source_unicode_exceptions_and_extras():
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(JsonFormatter())
    setup_logging("INFO", [handler])
    try:
        raise ValueError("故障")
    except ValueError:
        get_logger("retargeting").exception("求解 %s", "失败", extra={"episode": 7})
    record = json.loads(output.getvalue())
    assert record["name"] == "holistic_motion.retargeting"
    assert record["message"] == "求解 失败"
    assert record["episode"] == 7
    assert record["filename"] == "test_logging.py" and record["line"] > 0
    assert (
        record["function"] == "test_json_contains_source_unicode_exceptions_and_extras"
    )
    assert "ValueError: 故障" in record["exception"]


@pytest.mark.parametrize("transported", [False, True])
def test_json_preserves_cached_exception_after_record_transport(transported):
    try:
        raise ValueError("远端故障")
    except ValueError:
        record = logging.LogRecord(
            "holistic_motion.remote",
            logging.ERROR,
            "remote.py",
            27,
            "frame %s",
            (8,),
            sys.exc_info(),
        )
    record.trace_id = "trace-8"
    logging.Formatter().format(record)
    cached = record.exc_text
    if transported:
        record.exc_info = None
        record = pickle.loads(pickle.dumps(record))
    state = record.__dict__.copy()
    payload = json.loads(JsonFormatter().format(record))
    assert payload["exception"] == cached
    assert payload["message"] == "frame 8"
    assert payload["trace_id"] == "trace-8"
    assert record.__dict__ == state


@pytest.mark.parametrize("install_before_import", [False, True])
def test_json_preserves_application_record_factory_fields(
    install_before_import, tmp_path
):
    env = dict(os.environ, HOLISTICMOTION_PURE_PYTHON="1", HOLISTICMOTION_DEBUG="0")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "python")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import logging
import sys
calls = []
original = logging.getLogRecordFactory()
def factory(*args, **kwargs):
    calls.append(True)
    record = original(*args, **kwargs)
    record.trace_id = 'trace-机器人'
    record.frame_id = 42
    return record
before = sys.argv[1] == 'True'
if before:
    logging.setLogRecordFactory(factory)
import holistic_motion
assert not calls, 'package import invoked the application record factory'
if not before:
    logging.setLogRecordFactory(factory)
holistic_motion.setup_logging('INFO', with_json_format=True)
holistic_motion.get_logger('pipeline').info('frame ready')
""",
            str(install_before_import),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["trace_id"] == "trace-机器人"
    assert payload["frame_id"] == 42
    assert result.stderr == ""


@pytest.mark.parametrize("exception", [False, True])
def test_json_queue_preserves_diagnostics_after_synchronous_handler(exception):
    class PreparedMessageFormatter(logging.Formatter):
        def format(self, record):
            # Older QueueHandler implementations retain stack_info after JSON
            # encoding. The destination writes the prepared message verbatim.
            return record.getMessage()

    synchronous = io.StringIO()
    asynchronous = io.StringIO()
    text_handler = logging.StreamHandler(synchronous)
    queue = Queue()
    queued = QueueHandler(queue)
    queued.setFormatter(JsonFormatter())
    destination = logging.StreamHandler(asynchronous)
    destination.setFormatter(PreparedMessageFormatter())
    listener = QueueListener(queue, destination)
    setup_logging("INFO", [text_handler, queued])
    metrics = {"residual": float("inf"), "frame": 8}
    logger = get_logger("queued")
    if exception:
        try:
            raise ValueError("异步故障")
        except ValueError:
            logger.exception("frame %s", 8, extra={"metrics": metrics}, stack_info=True)
    else:
        logger.info("frame %s", 8, extra={"metrics": metrics}, stack_info=True)
    # JSON is frozen before enqueueing; later application mutation is harmless.
    metrics["frame"] = 9
    listener.start()
    listener.stop()  # Drains records queued before the stop sentinel.
    queued.close()
    destination.close()
    text_handler.close()
    payload = strict_json(asynchronous.getvalue())
    assert payload["message"] == "frame 8"
    assert payload["metrics"] == {"residual": "Infinity", "frame": 8}
    assert "Stack (most recent call last)" in payload["stack"]
    if exception:
        assert payload["exception"].count("ValueError: 异步故障") == 1
        assert synchronous.getvalue().count("ValueError: 异步故障") == 1
    else:
        assert "exception" not in payload


def strict_json(text):
    def reject_constant(value):
        pytest.fail(f"non-standard JSON number: {value}")

    return json.loads(text, parse_constant=reject_constant)


@pytest.mark.parametrize(
    "number,expected",
    [(float("nan"), "NaN"), (float("inf"), "Infinity"), (-float("inf"), "-Infinity")],
)
@pytest.mark.parametrize("exception", [False, True])
def test_json_preserves_nonfinite_diagnostics(number, expected, exception, capsys):
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(JsonFormatter())
    setup_logging("WARNING", [handler])
    metrics = {"residual": number, "nested": [1.25, (number, {"cost": number})]}
    logger = get_logger("solver")
    if exception:
        try:
            raise ValueError("数值求解失败")
        except ValueError:
            logger.exception("求解失败", extra={"metrics": metrics})
    else:
        logger.warning("求解失败", extra={"metrics": metrics})
    lines = output.getvalue().splitlines()
    assert len(lines) == 1
    record = strict_json(lines[0])
    assert record["message"] == "求解失败"
    assert record["metrics"] == {
        "residual": expected,
        "nested": [1.25, [expected, {"cost": expected}]],
    }
    if exception:
        assert "ValueError: 数值求解失败" in record["exception"]
    assert metrics["residual"] is number
    assert metrics["nested"][1][0] is number
    assert isinstance(metrics["nested"][1], tuple)
    assert not math.isfinite(metrics["nested"][1][1]["cost"])
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "number,expected",
    [(float("nan"), "NaN"), (float("inf"), "Infinity"), (-float("inf"), "-Infinity")],
)
def test_json_handles_nonfinite_mapping_keys(number, expected):
    record = logging.makeLogRecord({"msg": "metrics", "metrics": {number: 2.0}})
    assert strict_json(JsonFormatter().format(record))["metrics"] == {expected: 2.0}
    assert next(iter(record.metrics)) is number


@pytest.mark.parametrize("container_type", [list, dict])
def test_json_marks_cycles_without_changing_original(container_type):
    metrics = container_type()
    if container_type is list:
        metrics.append(metrics)
        expected = ["<circular reference>"]
    else:
        metrics["self"] = metrics
        expected = {"self": "<circular reference>"}
    record = logging.makeLogRecord({"msg": "cycle", "metrics": metrics})
    assert strict_json(JsonFormatter().format(record))["metrics"] == expected
    assert (metrics[0] if container_type is list else metrics["self"]) is metrics


def test_json_repeated_references_are_not_cycles():
    shared = {"finite": 0.125, "nested": [1, True, None, "中文"]}
    record = logging.makeLogRecord({"msg": "shared", "metrics": [shared, shared]})
    assert strict_json(JsonFormatter().format(record))["metrics"] == [shared, shared]
    assert record.metrics[0] is record.metrics[1] is shared


def test_environment_debug_json_is_opt_in(monkeypatch, capsys):
    monkeypatch.setenv("HOLISTICMOTION_DEBUG", "0")
    before = get_logger().handlers[:]
    auto_configure_debug_logging()
    assert get_logger().handlers == before
    monkeypatch.setenv("HOLISTICMOTION_DEBUG", "true")
    monkeypatch.setenv("HOLISTICMOTION_JSON_FORMAT_LOG", "1")
    auto_configure_debug_logging()
    get_logger("env").debug("enabled")
    assert json.loads(capsys.readouterr().out)["message"] == "enabled"


def test_bad_environment_is_rejected_before_configuration(monkeypatch):
    monkeypatch.setenv("HOLISTICMOTION_DEBUG", "1")
    monkeypatch.setenv("HOLISTICMOTION_JSON_FORMAT_LOG", "invalid")
    before = get_logger().handlers[:]
    with pytest.raises(ValueError, match="HOLISTICMOTION_JSON_FORMAT_LOG"):
        auto_configure_debug_logging()
    assert get_logger().handlers == before


@pytest.mark.parametrize(
    "level,severity", [(1, logging.WARNING), (2, logging.INFO), (3, logging.DEBUG)]
)
def test_native_records_preserve_metadata_without_duplicate_console(
    level, severity, capsys
):
    native = native_module()
    handler = Collector()
    setup_logging("DEBUG", [handler])
    native._log_native(level, "值 {literal}", "C:\\code\\solver.cpp", 42, "Solve")
    assert len(handler.records) == 1
    record = handler.records[0]
    assert (
        record.name,
        record.levelno,
        record.filename,
        record.lineno,
        record.funcName,
    ) == ("holistic_motion.native", severity, "solver.cpp", 42, "Solve")
    assert record.getMessage() == "值 {literal}"
    assert capsys.readouterr() == ("", "")


def test_native_filtering_and_error_semantics():
    native = native_module()
    handler = Collector()
    setup_logging("WARNING", [handler])
    native._log_native(3, "filtered")
    with pytest.raises(RuntimeError, match="primary error"):
        native._log_native(0, "primary error", "solver.cpp", 9, "Solve")
    assert len(handler.records) == 1
    assert handler.records[0].levelno == logging.ERROR
    setup_logging("CRITICAL", handlers=[])
    with pytest.raises(RuntimeError, match="still throws"):
        native._log_native(0, "still throws")


@pytest.mark.parametrize("operation", ["setup", "inherit", "reset"])
@pytest.mark.parametrize("child_level", [logging.NOTSET, logging.DEBUG])
def test_native_reconfiguration_uses_pending_level_and_child_override(
    operation, child_level, caplog, monkeypatch
):
    native = native_module()
    setup_logging("ERROR", handlers=[])
    get_logger("native").setLevel(child_level)
    updates = []
    original = native._set_native_log_level

    def record_update(level):
        updates.append(level)
        original(level)

    monkeypatch.setattr(native, "_set_native_log_level", record_update)
    collector = Collector()
    with caplog.at_level(logging.WARNING):
        if operation == "reset":
            reset_logging()
        else:
            setup_logging("INFO" if operation == "setup" else "NOTSET", [collector])
        expected = (
            logging.DEBUG
            if child_level == logging.DEBUG
            else logging.INFO
            if operation == "setup"
            else logging.WARNING
        )
        assert updates == [
            {logging.DEBUG: 3, logging.INFO: 2, logging.WARNING: 1}[expected]
        ]
        if operation == "reset":
            # Capture DEBUG child records too without altering root inheritance.
            caplog.handler.setLevel(logging.NOTSET)
        for level, severity in [
            (3, logging.DEBUG),
            (2, logging.INFO),
            (1, logging.WARNING),
        ]:
            native._log_native(level, f"native severity {severity}")
        records = caplog.records if operation == "reset" else collector.records
        assert [record.levelno for record in records] == [
            severity
            for severity in (logging.DEBUG, logging.INFO, logging.WARNING)
            if severity >= expected
        ]


def test_native_threads_can_emit_with_gil_released():
    native = native_module()
    handler = Collector()
    setup_logging("INFO", [handler])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda value: native._log_native(2, str(value)), range(100)))
    assert sorted(int(record.getMessage()) for record in handler.records) == list(
        range(100)
    )


@pytest.mark.parametrize("site", ["factory", "filter", "handler", "formatter"])
def test_native_reentry_is_suppressed_throughout_dispatch(site, monkeypatch):
    native = native_module()
    output = io.StringIO()
    calls = []

    def reenter():
        calls.append(True)
        # Bound recursion so the regression fails safely without the guard.
        if len(calls) < 3:
            native._log_native(1, "nested")

    class ReentrantHandler(logging.StreamHandler):
        def emit(self, record):
            if site == "handler":
                reenter()
            super().emit(record)

    class ReentrantFormatter(JsonFormatter):
        def format(self, record):
            if site == "formatter":
                reenter()
            return super().format(record)

    factory = logging.getLogRecordFactory()

    def make_record(*args, **kwargs):
        if site == "factory":
            reenter()
        return factory(*args, **kwargs)

    def filter_record(record):
        if site == "filter":
            reenter()
        return True

    monkeypatch.setattr(logging, "_logRecordFactory", make_record)
    handler = ReentrantHandler(output)
    handler.setFormatter(ReentrantFormatter())
    handler.addFilter(filter_record)
    setup_logging("INFO", [handler])
    native._log_native(2, "outer")
    native._log_native(2, "next")
    assert [json.loads(line)["message"] for line in output.getvalue().splitlines()] == [
        "outer",
        "next",
    ]
    assert len(calls) == 2


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_native_dispatch_recovers_after_factory_failure(error_type, monkeypatch):
    native = native_module()
    collector = Collector()
    setup_logging("INFO", [collector])
    failure = error_type("record creation failed")

    def fail(*args, **kwargs):
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(logging, "_logRecordFactory", fail)
        with pytest.raises(error_type) as caught:
            native._log_native(2, "failed")
    assert caught.value is failure
    native._log_native(2, "recovered")
    assert [record.getMessage() for record in collector.records] == ["recovered"]


@pytest.mark.parametrize("outer_level", [0, 1])
def test_native_reentrant_error_still_throws_and_dispatch_recovers(outer_level):
    native = native_module()

    class ReentrantErrorHandler(Collector):
        def emit(self, record):
            super().emit(record)
            if len(self.records) < 3:
                native._log_native(0, "nested error")

    handler = ReentrantErrorHandler()
    setup_logging("INFO", [handler])
    expected = "primary error" if outer_level == 0 else "nested error"
    with pytest.raises(RuntimeError, match=expected):
        native._log_native(outer_level, "primary error")
    assert [record.getMessage() for record in handler.records] == ["primary error"]
    collector = Collector()
    setup_logging("INFO", [collector])
    native._log_native(2, "recovered")
    assert [record.getMessage() for record in collector.records] == ["recovered"]


def test_native_dispatch_guard_is_local_to_each_thread(monkeypatch):
    native = native_module()
    collector = Collector()
    setup_logging("INFO", [collector])
    barrier = Barrier(2)
    factory = logging.getLogRecordFactory()

    def rendezvous(*args, **kwargs):
        record = factory(*args, **kwargs)
        if record.name == "holistic_motion.native":
            # Meet before the handler lock; both threads must be in dispatch.
            barrier.wait(timeout=5)
        return record

    monkeypatch.setattr(logging, "_logRecordFactory", rendezvous)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(
            pool.map(
                lambda message: native._log_native(2, message), ["first", "second"]
            )
        )
    assert sorted(record.getMessage() for record in collector.records) == [
        "first",
        "second",
    ]


@pytest.mark.parametrize(
    "level,expected", [(0, "primary failure"), (1, "handler failure")]
)
def test_native_python_handler_failure_preserves_primary_error(level, expected):
    native = native_module()

    class FailingHandler(logging.Handler):
        def emit(self, record):
            raise RuntimeError("handler failure")

    setup_logging("DEBUG", [FailingHandler()])
    with pytest.raises(RuntimeError, match=expected):
        native._log_native(level, "primary failure")
