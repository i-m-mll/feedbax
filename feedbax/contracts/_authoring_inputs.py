"""Read JSON authoring inputs and guard the lifetime of their compiled plan."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from errno import ENOENT
from pathlib import Path
from typing import Protocol


class AuthoringTextReader(Protocol):
    def __call__(
        self,
        path: Path,
        *,
        resolved_path: Path | None = None,
        require_file: bool = False,
    ) -> str: ...


def read_authoring_text(
    path: Path,
    *,
    resolved_path: Path | None = None,
    require_file: bool = False,
) -> str:
    """Read UTF-8 text, optionally through a checked target or only from a file."""
    target = path if resolved_path is None else resolved_path
    _require_file(target, require_file=require_file)
    return target.read_text(encoding="utf-8")


def _require_file(path: Path, *, require_file: bool) -> None:
    if require_file and not path.is_file():
        raise FileNotFoundError(ENOENT, "No such file", path)


@dataclass(frozen=True)
class _InputRead:
    target: Path
    content: bytes | None

    @classmethod
    def observe(cls, target: Path, read: Callable[[Path], bytes]) -> _InputRead:
        try:
            _require_file(target, require_file=True)
            return cls(target, read(target))
        except FileNotFoundError:
            return cls(target, None)


class AuthoringInputs:
    """Track exact bytes and resolved targets for the paths an authoring plan reads."""

    def __init__(self, *, read: Callable[[Path], bytes] = Path.read_bytes) -> None:
        self._read = read
        self._reads: dict[Path, _InputRead] = {}

    def read_text(
        self,
        path: Path,
        *,
        resolved_path: Path | None = None,
        require_file: bool = False,
    ) -> str:
        content = self.read_bytes(path, resolved_path=resolved_path)
        # Match Path.read_text's universal newlines while retaining the raw bytes.
        return content.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")

    def read_bytes(
        self,
        path: Path,
        *,
        resolved_path: Path | None = None,
    ) -> bytes:
        path = path.absolute()
        target = path.resolve() if resolved_path is None else resolved_path
        observed = _InputRead.observe(target, self._read)
        previous = self._reads.setdefault(path, observed)
        if observed != previous:
            raise ValueError(f"Evaluation authoring input changed after it was read: {path}")
        if observed.content is None:
            raise FileNotFoundError(ENOENT, "No such file", path)
        return observed.content

    def revalidate(self) -> None:
        """Refuse effects when a consumed input's bytes or symlink target changed."""
        for path, expected in self._reads.items():
            try:
                target = path.resolve()
                if target == expected.target:
                    observed = _InputRead.observe(target, self._read)
                    if observed == expected:
                        continue
            except (OSError, RuntimeError) as exc:
                raise ValueError(
                    f"Cannot revalidate evaluation authoring input {path}: {exc}"
                ) from exc
            raise ValueError(f"Evaluation authoring input changed after it was read: {path}")
