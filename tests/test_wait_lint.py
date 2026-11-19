"""A test waits for a state, not for a time.

Every test source outside `tests/support` is read for a sleep (in code or in the source of a job it writes), a
wait bounded by a literal number, and a loop that ends on a clock. A line whose duration is itself what the test
is about carries `# real clock: <what must take, or not take, this long>`.
"""

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import re

TESTS = Path(__file__).resolve().parent
_REAL_CLOCK = re.compile(r"#\s*real clock:\s*\S")
_SLEEP_CALL = "sleep" + "("
_BOUNDED_FIRST = {"wait", "join", "result", "settimeout"}
_CLOCKS = {"monotonic", "time", "perf_counter", "monotonic_ns", "time_ns", "perf_counter_ns"}

_SLEEP = (
    "a sleep. Wait for the state with `wait_until` from tests.support.host; a job that holds a slot until"
    " released or stopped comes from tests.support.host (`held_until_released`, `HELD_UNTIL_STOPPED`)."
)
_LITERAL_TIMEOUT = "a wait bounded by a literal number. Bound it with `HANG_SECONDS` from tests.support.hang."
_CLOCK_LOOP = "a loop that ends on a clock. Wait for the state with `wait_until` from tests.support.host."
_REMEDY = (
    "If the duration is itself what the test is about, end the line with"
    " `# real clock: <what must take, or not take, this long>`."
)


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    what: str

    def __str__(self) -> str:
        return f"{self.path.relative_to(TESTS.parent)}:{self.line}: {self.what}"


def _name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _literal_number(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int | float) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp):
        return _literal_number(node.operand)
    if isinstance(node, ast.BinOp):
        return _literal_number(node.left) and _literal_number(node.right)
    return False


def _reads_clock(node: ast.AST) -> bool:
    return any(
        isinstance(call, ast.Call) and _name(call.func) in _CLOCKS for call in ast.walk(node)
    )


def _exits(body: list[ast.stmt]) -> bool:
    return any(isinstance(node, ast.Break | ast.Raise | ast.Return) for stmt in body for node in ast.walk(stmt))


def _loop_statements(loop: ast.While) -> Iterator[ast.stmt]:
    stack: list[ast.stmt] = list(loop.body)
    while stack:
        stmt = stack.pop()
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.While):
            continue
        yield stmt
        for field in ("body", "orelse", "finalbody", "handlers"):
            stack.extend(getattr(stmt, field, []))


def _timeout_argument(call: ast.Call) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == "timeout":
            return keyword.value
    name = _name(call.func)
    if name in _BOUNDED_FIRST and isinstance(call.func, ast.Attribute) and call.args:
        return call.args[0]
    if name == "wait_for" and len(call.args) > 1:
        return call.args[1]
    if (
        name == "timeout" and isinstance(call.func, ast.Attribute) and _name(call.func.value) == "asyncio"
        and call.args
    ):
        return call.args[0]
    return None


def _findings(path: Path) -> Iterator[Finding]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    for node in ast.walk(ast.parse(source, str(path))):
        if isinstance(node, ast.Call):
            if _name(node.func) == "sleep":
                yield Finding(path, node.lineno, _SLEEP)
            if (timeout := _timeout_argument(node)) is not None and _literal_number(timeout):
                yield Finding(path, timeout.lineno, _LITERAL_TIMEOUT)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and _SLEEP_CALL in node.value:
            for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                if _SLEEP_CALL in lines[line - 1]:
                    yield Finding(path, line, _SLEEP)
        elif isinstance(node, ast.While):
            if _reads_clock(node.test):
                yield Finding(path, node.lineno, _CLOCK_LOOP)
            for stmt in _loop_statements(node):
                if isinstance(stmt, ast.If) and _reads_clock(stmt.test) and _exits(stmt.body):
                    yield Finding(path, stmt.lineno, _CLOCK_LOOP)


def findings(paths: list[Path]) -> list[Finding]:
    found = set()
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        found.update(
            finding for finding in _findings(path) if not _REAL_CLOCK.search(lines[finding.line - 1])
        )
    return sorted(found, key=lambda finding: (str(finding.path), finding.line))


def test_sources_wait_for_states():
    support = TESTS / "support"
    sources = sorted(path for path in TESTS.rglob("*.py") if not path.is_relative_to(support))
    found = findings(sources)
    assert not found, f"{len(found)} test line(s) wait for a time:\n" + "\n".join(map(str, found)) + "\n" + _REMEDY
