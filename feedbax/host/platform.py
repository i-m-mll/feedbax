"""Local filesystem check and file notifications for a host."""

import asyncio
import ctypes
import os
import re
import select
import sys
from collections.abc import Callable
from pathlib import Path


class DarwinStatFS(ctypes.Structure):
    _fields_ = [
        ("sizes", ctypes.c_uint32 * 2),
        ("counts", ctypes.c_uint64 * 5),
        ("fsid", ctypes.c_int32 * 2),
        ("owner", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("subtype", ctypes.c_uint32),
        ("type_name", ctypes.c_char * 16),
        ("mount_path", ctypes.c_char * 1024),
        ("source", ctypes.c_char * 1024),
        ("ext_flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 7),
    ]


def local_filesystem(path: Path) -> bool:
    if sys.platform == "darwin":
        info = DarwinStatFS()
        statfs = ctypes.CDLL(None, use_errno=True).statfs
        statfs.argtypes = [ctypes.c_char_p, ctypes.POINTER(DarwinStatFS)]
        statfs.restype = ctypes.c_int
        if statfs(os.fsencode(path), ctypes.byref(info)):
            raise OSError(ctypes.get_errno(), "Could not inspect host filesystem", path)
        return bool(info.flags & 0x1000)  # MNT_LOCAL
    if sys.platform == "linux":
        mounts = []
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            prefix, suffix = line.split(" - ", 1)
            mount_point = re.sub(
                r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), prefix.split()[4]
            )
            mount = Path(mount_point)
            if path.is_relative_to(mount):
                mounts.append((len(mount.parts), suffix.split()[0]))
        network = {"9p", "ceph", "cifs", "fuse.sshfs", "glusterfs", "nfs", "nfs4", "smb3"}
        return bool(mounts) and max(mounts)[1] not in network
    raise OSError(f"Unsupported host platform: {sys.platform}")


class FileChanges:
    """Watch delivery directories and files on macOS, with reconciliation on a timer."""

    def __init__(self, loop: asyncio.AbstractEventLoop, on_change: Callable[[], None]):
        self.loop = loop
        self.on_change = on_change
        self.queue = select.kqueue() if sys.platform == "darwin" else None
        self.fds: dict[Path, int] = {}
        if self.queue is not None:
            loop.add_reader(self.queue.fileno(), self._drain)

    def _drain(self) -> None:
        self.queue.control(None, 256, 0)
        self.on_change()

    def watch(self, paths: set[Path]) -> None:
        if self.queue is None:
            return
        for path in self.fds.keys() - paths:
            os.close(self.fds.pop(path))
        for path in paths - self.fds.keys():
            try:
                fd = os.open(path, os.O_RDONLY)
            except FileNotFoundError:
                continue
            self.queue.control(
                [
                    select.kevent(
                        fd,
                        filter=select.KQ_FILTER_VNODE,
                        flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                        fflags=select.KQ_NOTE_WRITE | select.KQ_NOTE_EXTEND,
                    )
                ],
                0,
            )
            self.fds[path] = fd

    def close(self) -> None:
        for fd in self.fds.values():
            os.close(fd)
        if self.queue is not None:
            self.loop.remove_reader(self.queue.fileno())
            self.queue.close()
