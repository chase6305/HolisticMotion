"""Recommend native compile jobs from visible CPU and memory headroom.

The two-GiB estimate includes headroom over measured release translation units.
This is a startup estimate; explicit JOBS remains the build script override.
"""

import os
import re
from pathlib import Path

MEMORY_PER_JOB = 2 * 1024**3


def _read(path):
    try:
        return path.read_bytes().decode("utf-8", errors="surrogateescape").rstrip("\n")
    except (OSError, UnicodeError):
        return ""


def _nonnegative(value):
    try:
        number = int(value)
        return number if number >= 0 else None
    except ValueError:
        return None


def _ancestors(root, membership):
    root = Path(os.path.abspath(root))
    directory = Path(os.path.abspath(root / (membership or "/").lstrip("/")))
    if directory != root and root not in directory.parents:
        directory = root
    while True:
        yield directory
        if directory == root:
            break
        directory = directory.parent


def _cpu_quota(quota, period):
    quota, period = _nonnegative(quota), _nonnegative(period)
    if quota is not None and period and quota > 0:
        return max(1, (quota + period - 1) // period)
    return None


def _mounted_groups(proc_root, groups):
    probes = []
    for line in _read(proc_root / "self" / "mountinfo").split("\n"):
        fields, separator, filesystem = line.partition(" - ")
        # procfs uses ASCII separators; Unicode whitespace can be part of an
        # actual mount path. Byte reads also preserve carriage returns in names.
        fields, filesystem = fields.split(" "), filesystem.split(" ")
        if not separator or len(fields) < 6 or len(filesystem) < 3:
            continue
        if filesystem[0] == "cgroup2":
            controllers = {""}
        elif filesystem[0] == "cgroup":
            controllers = {"cpu", "memory"}.intersection(filesystem[2].split(","))
        else:
            continue
        root, point = (
            re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), field)
            for field in fields[3:5]
        )
        if not Path(root).is_absolute() or not Path(point).is_absolute():
            continue
        for controller in sorted(controllers.intersection(groups)):
            membership = Path(os.path.normpath(groups[controller]))
            try:
                relative = membership.relative_to(Path(os.path.normpath(root)))
            except ValueError:
                # A cgroup namespace may expose its own root through a mount
                # made before namespace creation. Other unrelated bind mounts
                # must not constrain this process's build budget.
                if membership != Path("/"):
                    continue
                relative = Path(".")
            probes.append((Path(point), str(relative), controller or "v2"))
    return probes


def choose_jobs(cpus, memory_bytes=None):
    jobs = max(1, cpus)
    if memory_bytes is not None:
        jobs = min(jobs, max(1, memory_bytes // MEMORY_PER_JOB))
    return jobs


def recommended_jobs(proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup")):
    detected = os.cpu_count()
    cpus = [detected] if detected and detected > 0 else []
    try:
        affinity = len(os.sched_getaffinity(0))
        if affinity:
            cpus.append(affinity)
    except (AttributeError, OSError):
        pass
    if not cpus:
        cpus.append(1)
    memory = []
    for line in _read(proc_root / "meminfo").split("\n"):
        fields = line.split()
        if len(fields) == 3 and fields[0] == "MemAvailable:" and fields[2] == "kB":
            value = _nonnegative(fields[1])
            if value is not None:
                memory.append(value * 1024)
    groups = {}
    for line in _read(proc_root / "self" / "cgroup").split("\n"):
        fields = line.split(":", 2)
        if len(fields) == 3:
            for controller in fields[1].split(","):
                groups[controller] = fields[2]
    probes = _mounted_groups(proc_root, groups)
    mounted = {kind for _, _, kind in probes}
    if "v2" not in mounted:
        probes.append((cgroup_root, groups.get(""), "v2"))
    if "memory" in groups and "memory" not in mounted:
        probes.append((cgroup_root / "memory", groups["memory"], "memory"))
    if "cpu" in groups and "cpu" not in mounted:
        for mount in ("cpu", "cpu,cpuacct"):
            probes.append((cgroup_root / mount, groups["cpu"], "cpu"))
    for root, membership, kind in probes:
        for directory in _ancestors(root, membership):
            if kind == "v2":
                quota = _read(directory / "cpu.max").split()
                if len(quota) == 2:
                    value = _cpu_quota(*quota)
                    if value is not None:
                        cpus.append(value)
            elif kind == "cpu":
                value = _cpu_quota(
                    _read(directory / "cpu.cfs_quota_us"),
                    _read(directory / "cpu.cfs_period_us"),
                )
                if value is not None:
                    cpus.append(value)
            if kind in ("v2", "memory"):
                limit_name, usage_name = (
                    ("memory.max", "memory.current")
                    if kind == "v2"
                    else ("memory.limit_in_bytes", "memory.usage_in_bytes")
                )
                limit = _nonnegative(_read(directory / limit_name))
                usage = _nonnegative(_read(directory / usage_name))
                if limit is not None:
                    memory.append(max(0, limit - (usage or 0)))
    return choose_jobs(min(cpus), min(memory) if memory else None)


if __name__ == "__main__":
    print(recommended_jobs())
