"""Process-group verification shared by the local execution owners."""

import ctypes
import errno
import os
import subprocess
import sys
from functools import cache
from pathlib import Path

from .io import TOKEN_ENV


def process_has_launch_token(pid: int, token: str) -> bool:
    expected = f"{TOKEN_ENV}={token}".encode()
    if sys.platform == "darwin":
        # CTL_KERN/KERN_PROCARGS2 returns argc, executable, argv, and environment.
        sysctl = ctypes.CDLL(None).sysctl
        mib = (ctypes.c_int * 3)(1, 49, pid)
        size = ctypes.c_size_t()
        if sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0 or size.value < 4:
            return False
        buffer = ctypes.create_string_buffer(size.value)
        if sysctl(mib, 3, buffer, ctypes.byref(size), None, 0) != 0 or size.value < 4:
            return False
        argc = int.from_bytes(buffer.raw[:4], sys.byteorder, signed=True)
        arguments = buffer.raw[4 : size.value].partition(b"\0")[2].lstrip(b"\0")
        if argc < 1:
            return False
        return expected in arguments.split(b"\0")[argc:]
    try:
        return expected in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return False


def process_group_member_pids(
    process_group_id: int,
) -> tuple[tuple[int, ...], str | None]:
    if sys.platform == "darwin":
        try:
            return _darwin_process_group_members(process_group_id), None
        except OSError as exc:
            return (), f"process group membership inspection failed: {exc}"
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,pgid=,stat="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return (), f"process group membership inspection failed: {exc}"
    if result.returncode != 0:
        return (), f"process group membership inspection exited {result.returncode}"
    members = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            pid, group = (int(value) for value in fields[:2])
        except ValueError:
            continue
        if group == process_group_id and not fields[2].startswith("Z"):
            members.append(pid)
    return tuple(members), None


@cache
def _darwin_libproc() -> ctypes.CDLL:
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    libproc.proc_listpids.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    libproc.proc_listpids.restype = ctypes.c_int
    libproc.proc_pidinfo.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    libproc.proc_pidinfo.restype = ctypes.c_int
    return libproc


def _darwin_process_group_members(process_group_id: int) -> tuple[int, ...]:
    """Inspect the requested group without spawning a host-wide process listing."""
    proc_pgrp_only = 2
    proc_pidt_shortbsdinfo = 13
    zombie = 5
    libproc = _darwin_libproc()
    capacity = 16
    pid_size = ctypes.sizeof(ctypes.c_int)
    while True:
        pids = (ctypes.c_int * capacity)()
        size = ctypes.sizeof(pids)
        ctypes.set_errno(0)
        used = libproc.proc_listpids(proc_pgrp_only, process_group_id, pids, size)
        if not 0 <= used <= size or used % pid_size or (used == 0 and ctypes.get_errno()):
            error = ctypes.get_errno() or errno.EIO
            raise OSError(error, os.strerror(error), "proc_listpids")
        if used < size:
            break
        capacity *= 2

    # struct proc_bsdshortinfo starts with pid, ppid, pgid and status in sys/proc_info.h.
    info = (ctypes.c_uint32 * 16)()
    size = ctypes.sizeof(info)
    members = []
    for pid in pids[: used // pid_size]:
        ctypes.set_errno(0)
        used = libproc.proc_pidinfo(pid, proc_pidt_shortbsdinfo, 0, info, size)
        if used != size:
            error = ctypes.get_errno() or errno.EIO
            if error == errno.ESRCH:
                continue
            raise OSError(error, os.strerror(error), "proc_pidinfo")
        if info[2] == process_group_id and info[3] != zombie:
            members.append(pid)
    return tuple(members)


def verify(pgid: int, token: str) -> tuple[str, str | None]:
    """Return verified, gone, or unverifiable for the recorded worker group."""
    members, error = process_group_member_pids(pgid)
    if error is not None:
        return "unverifiable", error
    if not members:
        return "gone", None
    verified = {pid for pid in members if process_has_launch_token(pid, token)}
    if len(verified) == len(members):
        return "verified", None
    remaining, error = process_group_member_pids(pgid)
    if error is not None:
        return "unverifiable", error
    if not remaining:
        return "gone", None
    if set(remaining) <= verified:
        return "verified", None
    return "unverifiable", f"process group {pgid} has a live member without its recorded token"
