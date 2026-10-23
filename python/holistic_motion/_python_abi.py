"""Interpreter ABI identity; usable standalone before importing the package."""

import json
import sys
import sysconfig


def python_abi():
    implementation = sys.implementation.name
    if implementation == "cpython":
        flags = "d" if sysconfig.get_config_var("Py_DEBUG") in (1, "1") else ""
        if sysconfig.get_config_var("Py_GIL_DISABLED") in (1, "1"):
            flags += "t"
        tag = implementation + ("-" + flags if flags else "")
        if sysconfig.get_config_var("Py_TRACE_REFS") in (1, "1"):
            tag += "-tracerefs"
        return tag
    soabi = sysconfig.get_config_var("SOABI") or sys.implementation.cache_tag
    return implementation + "-" + soabi


if __name__ == "__main__":
    if sys.argv[1:] == ["--abi"]:
        print(python_abi())
    else:
        print(
            json.dumps({"version": "%d.%d" % sys.version_info[:2], "abi": python_abi()})
        )
