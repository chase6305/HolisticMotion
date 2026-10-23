"""Compare package imports and first collision access in fresh interpreters.

Each --package is LABEL=PYTHONPATH (an install prefix or source/build path list).
Alternating order limits drift. Timings include Python/native initialization but
exclude interpreter startup and this worker's standard-library imports. Linux
also reports current resident memory and actually mapped collision libraries.
No URDF, optional Python robotics package or model download is required.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

WORKER = r"""
import json
import sys
from pathlib import Path
from time import perf_counter

def memory_kib():
    status = Path('/proc/self/status')
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    return None

def collision_libraries():
    maps = Path('/proc/self/maps')
    if not maps.exists():
        return None
    names = {Path(line.split()[-1]).name for line in maps.read_text().splitlines()}
    return sorted(name for name in names if name.startswith(('libpinocchio', 'libcoal', 'libhpp-fcl')))

started = perf_counter()
import holistic_motion as hm
import_ms = (perf_counter() - started) * 1000
info = hm.runtime_info()
native = sys.modules.get('holistic_motion._holistic_motion') or sys.modules.get('_holistic_motion')
loaded = getattr(info, 'collision_loaded', native is not None and 'CollisionModel' in vars(native))
row = dict(import_ms=import_ms, native_loaded=info.native_loaded,
           collision_compiled=info.collision_compiled, collision_loaded_after_import=loaded,
           import_rss_kib=memory_kib(), collision_libraries_after_import=collision_libraries(),
           first_collision_ms=None, collision_rss_kib=None)
if info.native_loaded and info.collision_compiled:
    started = perf_counter()
    model_type = hm.CollisionModel
    row['first_collision_ms'] = (perf_counter() - started) * 1000
    row['collision_rss_kib'] = memory_kib()
row['package'] = str(Path(hm.__file__).resolve())
row['python'] = sys.version
print(json.dumps(row))
"""


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def package(value):
    label, separator, paths = value.partition("=")
    if not separator or not label or not paths:
        raise argparse.ArgumentTypeError("expected LABEL=PYTHONPATH")
    resolved = []
    for item in paths.split(os.pathsep):
        path = Path(item).resolve()
        if not path.is_dir():
            raise argparse.ArgumentTypeError(f"not a directory: {path}")
        resolved.append(str(path))
    return label, os.pathsep.join(resolved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=package, action="append", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repeats", type=positive_int, default=11)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    labels = [label for label, _ in args.package]
    if len(set(labels)) != len(labels):
        parser.error("package labels must be unique")
    samples = {label: [] for label in labels}
    for repeat in range(args.repeats):
        order = args.package if repeat % 2 == 0 else reversed(args.package)
        for label, paths in order:
            environment = os.environ.copy()
            environment["PYTHONPATH"] = paths
            environment["HOLISTICMOTION_PURE_PYTHON"] = "0"
            result = subprocess.run(
                [args.python, "-c", WORKER],
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode:
                raise RuntimeError(f"{label}: {result.stderr}")
            samples[label].append(json.loads(result.stdout))
    rows = []
    for label, paths in args.package:
        values = samples[label]
        row = {"label": label, "pythonpath": paths, "samples": values}
        for metric in (
            "import_ms",
            "first_collision_ms",
            "import_rss_kib",
            "collision_rss_kib",
        ):
            available = [
                sample[metric] for sample in values if sample[metric] is not None
            ]
            row[f"median_{metric}"] = (
                statistics.median(available) if available else None
            )
        rows.append(row)
        print(
            json.dumps({key: value for key, value in row.items() if key != "samples"})
        )
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
