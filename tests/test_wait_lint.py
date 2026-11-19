"""A test waits for a state, not for a time.

Every test source outside `tests/support` is read for a sleep (in code or in the source of a job it writes), a
wait bounded by a literal number, a loop that ends on a clock, and a comparison on a clock reading or a value
computed from one. A claim about time in the fast tier is an order of states. A line whose duration is itself what
a `round` test is about carries `# real clock: <what must take, or not take, this long>`; the comment is accepted
only inside a `round` test, or in a helper of a module whose tests are all `round`.
"""

import ast
import io
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import re
import tokenize

TESTS = Path(__file__).resolve().parent
_REAL_CLOCK = re.compile(r"#\s*real clock:\s*\S")
_SLEEP_CALL = "sleep" + "("
_BOUNDED_FIRST = {"wait", "join", "result", "settimeout"}
_CLOCKS = {"monotonic", "time", "perf_counter", "monotonic_ns", "time_ns", "perf_counter_ns"}

_NUMERIC = {"int", "float", "round", "abs", "min", "max"}

_SLEEP = (
    "a sleep. Wait for the state with `wait_until` from tests.support.host; a job that holds a slot until"
    " released or stopped comes from tests.support.host (`held_until_released`, `HELD_UNTIL_STOPPED`)."
)
_LITERAL_TIMEOUT = "a wait bounded by a literal number. Bound it with `HANG_SECONDS` from tests.support.hang."
_CLOCK_LOOP = "a loop that ends on a clock. Wait for the state with `wait_until` from tests.support.host."
_CLOCK_COMPARISON = (
    "a comparison on elapsed real time. State the claim as an order of states: the awaited state arrives while"
    " the competing one is held by a gate or set beyond the hang limit."
)
_FAST_REAL_CLOCK = (
    "a `# real clock:` comment outside a `round` test. State the claim as an order of states, or mark the test"
    " `round` if a duration is what it asserts."
)
_REMEDY = (
    "If a duration is itself what the test asserts, mark the test `round` with its sentence and end the line with"
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


def _clock_names(tree: ast.Module) -> set[str]:
    """Names assigned anywhere in the module from a clock reading or from another such name."""
    assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            assignments.extend((target, node.value) for target in node.targets)
        elif isinstance(node, ast.AnnAssign | ast.AugAssign | ast.NamedExpr) and node.value is not None:
            assignments.append((node.target, node.value))
    names: set[str] = set()
    while True:
        found = {
            target.id for target, value in assignments
            if isinstance(target, ast.Name) and _timed(value, names)
        }
        if found <= names:
            return names
        names |= found


def _reads_clock(node: ast.AST) -> bool:
    return any(isinstance(call, ast.Call) and _name(call.func) in _CLOCKS for call in ast.walk(node))


def _timed(node: ast.AST, names: set[str]) -> bool:
    """Whether `node` is a clock reading, or arithmetic on clock readings and names assigned from them."""
    if isinstance(node, ast.Call):
        if _name(node.func) in _CLOCKS:
            return True
        return _name(node.func) in _NUMERIC and any(_timed(arg, names) for arg in node.args)
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.BinOp):
        return _timed(node.left, names) or _timed(node.right, names)
    if isinstance(node, ast.UnaryOp):
        return _timed(node.operand, names)
    return False


def _is_round(decorators: list[ast.expr]) -> bool:
    return any(
        isinstance(node, ast.Attribute) and node.attr == "round"
        and isinstance(node.value, ast.Attribute) and node.value.attr == "mark"
        for decorator in decorators for node in ast.walk(decorator)
    )


def _round_lines(tree: ast.Module, line_count: int) -> set[int]:
    """The lines where a `# real clock:` comment is accepted."""
    module_round = any(
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets)
        and _is_round([node.value])
        for node in tree.body
    )
    tests: list[tuple[ast.AST, bool]] = []

    def visit(body: list[ast.stmt], inherited: bool) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, inherited or _is_round(node.decorator_list))
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test"):
                tests.append((node, inherited or _is_round(node.decorator_list)))

    visit(tree.body, module_round)
    lines = {line for node, marked in tests if marked for line in range(node.lineno, node.end_lineno + 1)}
    if tests and all(marked for _, marked in tests):
        in_tests = {line for node, _ in tests for line in range(node.lineno, node.end_lineno + 1)}
        lines |= set(range(1, line_count + 1)) - in_tests
    return lines


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
    tree = ast.parse(source, str(path))
    clock_names = _clock_names(tree)
    accepted = _round_lines(tree, len(lines))
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and _REAL_CLOCK.search(token.string) and token.start[0] not in accepted:
            yield Finding(path, token.start[0], _FAST_REAL_CLOCK)
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and not all(isinstance(op, ast.Is | ast.IsNot) for op in node.ops) and any(
            _timed(operand, clock_names) for operand in (node.left, *node.comparators)
        ):
            yield Finding(path, node.lineno, _CLOCK_COMPARISON)
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
            finding for finding in _findings(path)
            if finding.what == _FAST_REAL_CLOCK or not _REAL_CLOCK.search(lines[finding.line - 1])
        )
    return sorted(found, key=lambda finding: (str(finding.path), finding.line))


def test_sources_wait_for_states():
    support = TESTS / "support"
    sources = sorted(path for path in TESTS.rglob("*.py") if not path.is_relative_to(support))
    found = findings(sources)
    assert not found, f"{len(found)} test line(s) wait for a time:\n" + "\n".join(map(str, found)) + "\n" + _REMEDY
