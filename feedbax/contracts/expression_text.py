"""The field text grammar, source positions, and node identity reconciliation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import json
from math import isfinite
import re
from types import MappingProxyType
from typing import Literal

from lark import Lark, Token, Tree, UnexpectedInput

from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import (
    RESERVED_NAMES,
    CallValue,
    DrawValue,
    Expression,
    FieldTextResult,
    FieldValue,
    LiteralValue,
    LocatedSpan,
    NameValue,
    NodePath,
    Problem,
    ProblemCode,
    Site,
    SweepList,
    SweepRange,
    SweepSample,
    TextSpan,
    scope_problems,
)
from feedbax.contracts.expression_registry import POINT_INTERPOLATIONS, FunctionClass, FunctionRegistry
from feedbax.contracts.strict_json import strict_json_loads


_GRAMMAR = r"""
?field: expr
      | expr "," [expr ("," expr)*] -> sweep
rule: [term (["and"] term)*]
term: NAME "=" unary ("," unary)* -> member_term
    | NAME "=" unary ".." unary -> span_term
    | (LOWEST | HIGHEST) NUMBER (NAME | ESCAPED_STRING) ["per" NAME ("," NAME)*] -> best_term
    | disjunction -> condition_term
LOWEST: "lowest"
HIGHEST: "highest"
?expr: disjunction
?disjunction: disjunction "or" conjunction -> or_
            | conjunction
?conjunction: conjunction "and" inversion -> and_
            | inversion
?inversion: "not" inversion -> not_
          | relation
?relation: sum CMP sum -> compare
         | sum
?sum: sum ADD product -> binary
    | product
?product: product MUL unary -> binary
        | unary
?unary: "-" unary -> negative
      | "+" unary -> positive
      | indexed
?indexed: indexed "[" expr "]" -> index
        | atom
?atom: NUMBER UNIT? -> number
     | "true" -> true
     | "false" -> false
     | NAME "(" [argument ("," argument)*] ")" -> call
     | NAME -> name
     | ESCAPED_STRING -> quoted_name
     | "[" expr ("," expr)* "]" -> vector
     | "(" expr ")"
?argument: NAME "=" expr -> keyword
         | expr
CMP: "==" | "!=" | "<=" | ">=" | "<" | ">"
ADD: "+" | "-" | "−"
MUL: "*" | "/" | "·" | "%"
NUMBER: /(?:[0-9]+(?:\.(?!\.)[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?/
NAME: /NAME_PATTERN/
UNIT: /UNIT_PATTERN/
%import common.WS
%import common.ESCAPED_STRING
%ignore WS
"""
_OPERATORS = {
    "+": "add",
    "-": "sub",
    "−": "sub",
    "*": "mul",
    "·": "mul",
    "/": "div",
    "%": "mod",
    "==": "eq",
    "!=": "ne",
    "<": "lt",
    "<=": "le",
    ">": "gt",
    ">=": "ge",
}
DRAW_ALIASES = MappingProxyType(
    {
        "U": "uniform",
        "N": "normal",
        "LogU": "loguniform",
        "B": "bernoulli",
        "TExp": "truncexp",
        "choice": "choice",
        "uniform_region": "uniform_region",
    }
)
_DRAW_SPELLINGS = {family: alias for alias, family in DRAW_ALIASES.items()}
_IDENTIFIER = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*")
RULE_NAME = re.compile(r"[a-zA-Z_](?:[a-zA-Z0-9_-]*[a-zA-Z0-9_])?")
"""A name as a run rule writes it unquoted: project names take hyphens (``learning-rate``), so subtraction is spaced."""
_KEYWORDS = frozenset({"true", "false", "and", "or", "not"})


@lru_cache(maxsize=2)
def _parser(start: Literal["field", "rule"] = "field") -> Lark:
    """The parser of a field's text, or of a run rule's, where terms are separated by spaces and so take no units, and
    names are project names, which take hyphens."""
    units = sorted(filter(None, UNITS), key=len, reverse=True)
    unit_pattern = (
        "(?:" + "|".join(re.escape(unit).replace("/", r"\/") for unit in units) + r")(?![\w.])"
        if start == "field" else "(?!)_"
    )
    name_pattern = _IDENTIFIER.pattern if start == "field" else RULE_NAME.pattern
    return Lark(
        _GRAMMAR.replace("UNIT_PATTERN", unit_pattern).replace("NAME_PATTERN", name_pattern),
        start=start,
        parser="lalr",
        propagate_positions=True,
        maybe_placeholders=False,
    )


@dataclass(frozen=True)
class ParsedField:
    value: FieldValue
    spans: Mapping[NodePath, TextSpan]

    def to_record(self) -> FieldTextResult:
        return FieldTextResult(
            value=self.value,
            spans=tuple(LocatedSpan(path=path, span=span) for path, span in self.spans.items()),
        )


@dataclass(frozen=True)
class _ReadError(Exception):
    problem: Problem


def _span(node: Tree | Token) -> TextSpan:
    if isinstance(node, Token):
        return TextSpan(start=node.start_pos, end=node.end_pos)
    return TextSpan(start=node.meta.start_pos, end=node.meta.end_pos)


def _fail(node: Tree | Token, code: ProblemCode, reason: str, path: NodePath) -> None:
    raise _ReadError(Problem(code=code, reason=reason, path=path, span=_span(node)))


class _Reader:
    def __init__(self, names: Mapping[str, str] | None, functions: FunctionRegistry | None):
        """``names`` maps the names in scope to their references; ``None`` takes every name as its own reference."""
        self.names = names
        self.spans: dict[NodePath, TextSpan] = {}
        self.functions = functions

    def field(self, node: Tree) -> FieldValue:
        self.spans[()] = _span(node)
        if node.data == "sweep":
            return SweepList(
                members=tuple(
                    self.expr(item, ("members", i)) for i, item in enumerate(node.children)
                )
            )
        if node.data == "call" and node.children[0] in ("linspace", "geomspace", "sample"):
            name, *arguments = node.children
            required = 2 if name == "sample" else 3
            if len(arguments) != required:
                _fail(node, ProblemCode.ARITY, f"{name} requires {required} arguments.", ())
            count = self.expr(arguments[-1], ("count",))
            if (
                not isinstance(count, LiteralValue)
                or type(count.value) is not int
                or count.value < 1
                or count.unit is not None
            ):
                _fail(
                    arguments[-1],
                    ProblemCode.TYPE_MISMATCH,
                    "Count must be a positive integer.",
                    ("count",),
                )
            if name == "sample":
                draw = self.expr(arguments[0], ("draw",))
                if not isinstance(draw, DrawValue):
                    _fail(
                        arguments[0],
                        ProblemCode.TYPE_MISMATCH,
                        "Sample requires a draw.",
                        ("draw",),
                    )
                return SweepSample(draw=draw, count=count.value)
            return SweepRange(
                first=self.expr(arguments[0], ("first",)),
                last=self.expr(arguments[1], ("last",)),
                count=count.value,
                spacing="linear" if name == "linspace" else "log",
            )
        return self.expr(node, ())

    def expr(self, node: Tree, path: NodePath) -> Expression:
        self.spans[path] = _span(node)
        return self._expr(node, path)

    def _expr(self, node: Tree, path: NodePath) -> Expression:
        match node.data:
            case "number":
                number, *units = node.children
                value = float(number) if any(c in number for c in ".eE") else int(number)
                if isinstance(value, float) and not isfinite(value):
                    _fail(node, ProblemCode.PARSE, "A number must be finite.", path)
                return LiteralValue(value=value, unit=str(units[0]) if units else None)
            case "true" | "false":
                return LiteralValue(value=node.data == "true")
            case "name" | "quoted_name":
                spelling = str(node.children[0])
                name = strict_json_loads(spelling) if node.data == "quoted_name" else spelling
                if self.names is None:
                    return NameValue(reference=name)
                if name not in self.names:
                    code = (
                        ProblemCode.OUT_OF_SCOPE
                        if name in RESERVED_NAMES
                        else ProblemCode.UNKNOWN_NAME
                    )
                    _fail(node, code, f"{name!r} is not in scope here.", path)
                return NameValue(reference=self.names[name])
            case "negative" | "positive":
                (child,) = node.children
                if child.data == "number":
                    value = self._expr(child, path)
                    assert isinstance(value, LiteralValue)
                    return (
                        value.model_copy(update={"value": -value.value})
                        if node.data == "negative"
                        else value
                    )
                function, args = ("neg" if node.data == "negative" else "pos"), (child,)
            case "binary" | "compare":
                left, op, right = node.children
                function, args = _OPERATORS[op], (left, right)
            case "and_" | "or_" | "not_":
                function, args = node.data.removesuffix("_"), node.children
            case "index" | "vector":
                function, args = str(node.data), node.children
            case "call":
                return self._call(node, path)
            case _:
                raise AssertionError(f"Unexpected expression grammar rule {node.data!r}")
        return CallValue(
            function=function,
            args=tuple(self.expr(child, (*path, "args", i)) for i, child in enumerate(args)),
        )

    def _call(self, node: Tree, path: NodePath) -> Expression:
        name, *parts = node.children
        args = []
        kwargs = {}
        for child in parts:
            if child.data == "keyword":
                key, value = child.children
                if key in kwargs:
                    _fail(
                        key,
                        ProblemCode.ARITY,
                        f"Repeated argument {key!r}.",
                        (*path, "kwargs", str(key)),
                    )
                kwargs[str(key)] = value
            elif kwargs:
                _fail(
                    child,
                    ProblemCode.ARITY,
                    "A positional argument follows a named argument.",
                    path,
                )
            else:
                args.append(self.expr(child, (*path, "args", len(args))))
        family = DRAW_ALIASES.get(name, name if name in _DRAW_SPELLINGS else None)
        registered = self.functions.resolve(name) if self.functions else None
        if registered is not None and registered.info.display_class == FunctionClass.DRAW:
            family = str(name)
        if family is not None or "per" in kwargs:
            per = kwargs.pop("per", None)
            level = "trial"
            if per is not None:
                if per.data != "name" or per.children[0] not in ("trial", "run"):
                    _fail(
                        per, ProblemCode.STRUCTURE, "Draw per must be run or trial.", (*path, "per")
                    )
                level = str(per.children[0])
            if kwargs:
                _fail(
                    node,
                    ProblemCode.ARITY,
                    "Draws take positional arguments and optional per.",
                    path,
                )
            return DrawValue(family=family or str(name), args=tuple(args), per=level)
        if name in ("linspace", "geomspace", "sample"):
            _fail(node, ProblemCode.STRUCTURE, "A sweep must be at the field root.", path)
        return CallValue(
            function=str(name),
            args=tuple(args),
            kwargs={
                key: self._keyword(str(name), key, child, (*path, "kwargs", key))
                for key, child in sorted(kwargs.items())
            },
        )

    def _keyword(self, function: str, key: str, node: Tree, path: NodePath) -> Expression:
        if function == "points" and key == "interp" and node.data == "name":
            mode = str(node.children[0])
            self.spans[path] = _span(node)
            return CallValue(function=mode)
        return self.expr(node, path)


def parse(
    text: str, site: Site, functions: FunctionRegistry | None = None
) -> ParsedField | tuple[Problem, ...]:
    """Parse editable text against the site's current names."""
    if problems := scope_problems(site):
        return problems
    try:
        tree = _parser().parse(text)
        reader = _Reader({item.name: item.id for item in site.names}, functions)
        value = reader.field(tree)
        return ParsedField(value, MappingProxyType(reader.spans))
    except _ReadError as error:
        return (error.problem,)
    except UnexpectedInput as error:
        offset = (
            len(text)
            if error.pos_in_stream is None or error.pos_in_stream < 0
            else error.pos_in_stream
        )
        return (
            Problem(
                code=ProblemCode.PARSE,
                reason="Expected a field expression.",
                span=TextSpan(start=offset, end=min(offset + 1, len(text))),
            ),
        )
    except (ValueError, RecursionError):
        return (
            Problem(code=ProblemCode.PARSE, reason="Expression exceeds the supported structure."),
        )


@dataclass(frozen=True)
class RuleChoice:
    """``name=a,b``: the runs whose value of ``name`` is one of ``values``, each a literal or a value's name."""

    name: str
    values: tuple[LiteralValue | NameValue, ...]
    span: TextSpan


@dataclass(frozen=True)
class RuleInterval:
    """``name=first..last``: the runs whose value of ``name`` lies between the two, both included."""

    name: str
    first: LiteralValue
    last: LiteralValue
    span: TextSpan


@dataclass(frozen=True)
class RuleCondition:
    """A comparison in the expression language, whose names are axes or metrics."""

    expression: Expression
    span: TextSpan


@dataclass(frozen=True)
class RuleBest:
    """``lowest 3 loss per lr``: the ``count`` runs of each ``per`` group that rank first by ``metric``."""

    order: Literal["min", "max"]
    count: int
    metric: str
    per: tuple[str, ...]
    span: TextSpan


type RuleTerm = RuleChoice | RuleInterval | RuleCondition | RuleBest


def _rule_value(reader: _Reader, node: Tree) -> LiteralValue | NameValue:
    value = reader.expr(node, ())
    if not isinstance(value, LiteralValue | NameValue):
        _fail(node, ProblemCode.STRUCTURE, "A value is a number or a name.", ())
    return value


def _rule_terms(reader: _Reader, node: Tree) -> list[RuleTerm]:
    span = _span(node)
    match node.data:
        case "member_term":
            name, *values = node.children
            return [RuleChoice(str(name), tuple(_rule_value(reader, value) for value in values), span)]
        case "span_term":
            name, *ends = node.children
            first, last = (_rule_value(reader, end) for end in ends)
            if not isinstance(first, LiteralValue) or not isinstance(last, LiteralValue):
                _fail(node, ProblemCode.STRUCTURE, "A range runs between two numbers.", ())
            return [RuleInterval(str(name), first, last, span)]
        case "best_term":
            order, count, metric, *per = node.children
            if not count.isdigit() or int(count) < 1:
                _fail(count, ProblemCode.TYPE_MISMATCH, "Count must be a positive integer.", ())
            return [RuleBest(
                "min" if order.type == "LOWEST" else "max", int(count),
                strict_json_loads(metric) if metric.type == "ESCAPED_STRING" else str(metric), tuple(map(str, per)), span,
            )]
        case "condition_term":
            (expression,) = node.children
            conjuncts = []
            while isinstance(expression, Tree) and expression.data == "and_":
                expression, right = expression.children
                conjuncts.append(right)
            conjuncts.append(expression)
            return [RuleCondition(reader.expr(item, ()), _span(item)) for item in reversed(conjuncts)]
    raise AssertionError(f"Unexpected rule grammar rule {node.data!r}")


def parse_rule(text: str) -> tuple[RuleTerm, ...] | Problem:
    """Parse a run rule: terms separated by spaces or ``and``, every one of which a chosen run meets."""
    try:
        tree = _parser("rule").parse(text)
        reader = _Reader(None, None)
        return tuple(term for node in tree.children for term in _rule_terms(reader, node))
    except _ReadError as error:
        return error.problem
    except UnexpectedInput as error:
        offset = len(text) if error.pos_in_stream is None or error.pos_in_stream < 0 else error.pos_in_stream
        return Problem(
            code=ProblemCode.PARSE, reason="Expected a rule term.",
            span=TextSpan(start=offset, end=min(offset + 1, len(text))),
        )
    except (ValueError, RecursionError):
        return Problem(code=ProblemCode.PARSE, reason="Rule exceeds the supported structure.")


_RELATION = 4
_INFIX = {
    **{function: (5, symbol) for function, symbol in (("add", "+"), ("sub", "-"))},
    **{function: (6, symbol) for function, symbol in (("mul", "*"), ("div", "/"), ("mod", "%"))},
    **{
        function: (_RELATION, symbol)
        for function, symbol in (("eq", "=="), ("ne", "!="), ("lt", "<"), ("le", "<="), ("gt", ">"), ("ge", ">="))
    },
    "and": (2, "and"),
    "or": (1, "or"),
}
"""Each binary operator's precedence in the grammar, loosest first, and its spelling; a relation's operands are sums."""
_PREFIX = {"not": (3, "not "), "neg": (7, "-"), "pos": (7, "+")}
_UNARY, _INDEXED, _ATOM = 7, 8, 9


def _print(node: Expression, names: Mapping[str, str], least: int = 0) -> str:
    """The text of ``node`` as an operand that binds at least as tightly as ``least``, parenthesized only if it must be."""
    text, precedence = _printed(node, names)
    return f"({text})" if precedence < least else text


def _printed(node: Expression, names: Mapping[str, str]) -> tuple[str, int]:
    match node:
        case LiteralValue(value=value, unit=unit):
            text = str(value).lower() + ("°" if unit == "°" else f" {unit}" if unit else "")
            return text, _UNARY if text.startswith("-") else _ATOM
        case NameValue(reference=reference):
            name = names[reference]
            return (
                name
                if _IDENTIFIER.fullmatch(name) and name not in _KEYWORDS
                else json.dumps(name, ensure_ascii=False)
            ), _ATOM
        case DrawValue(family=family, args=args, per=per):
            parts = [_print(arg, names) for arg in args]
            if per == "run" or family not in _DRAW_SPELLINGS:
                parts.append(f"per={per}")
            return f"{_DRAW_SPELLINGS.get(family, family)}({', '.join(parts)})", _ATOM
        case CallValue(function=function, args=(left, right), kwargs=kwargs) if function in _INFIX and not kwargs:
            precedence, symbol = _INFIX[function]
            first = precedence + 1 if precedence == _RELATION else precedence
            return f"{_print(left, names, first)} {symbol} {_print(right, names, precedence + 1)}", precedence
        case CallValue(function=function, args=(operand,), kwargs=kwargs) if (
            function in _PREFIX and not kwargs and (function == "not" or not isinstance(operand, LiteralValue))
        ):
            # A sign written before a number reads back as part of the number, so a signed literal stays a call.
            precedence, symbol = _PREFIX[function]
            return f"{symbol}{_print(operand, names, precedence)}", precedence
        case CallValue(function="index", args=(base, index), kwargs=kwargs) if not kwargs:
            return f"{_print(base, names, _INDEXED)}[{_print(index, names)}]", _INDEXED
        case CallValue(function=function, args=args, kwargs=kwargs):
            parts = [_print(arg, names) for arg in args]
            if function == "vector" and args and not kwargs:
                return "[" + ", ".join(parts) + "]", _ATOM
            for key, child in sorted(kwargs.items()):
                if (
                    function == "points"
                    and key == "interp"
                    and isinstance(child, CallValue)
                    and child.function in POINT_INTERPOLATIONS
                    and not child.args
                    and not child.kwargs
                ):
                    parts.append(f"{key}={child.function}")
                else:
                    parts.append(f"{key}={_print(child, names)}")
            return f"{function}({', '.join(parts)})", _ATOM


def print_field(value: FieldValue, site: Site) -> str:
    """Render editable text with the names currently assigned to references."""
    names = {item.id: item.name for item in site.names}
    match value:
        case SweepList(members=members):
            if len(members) == 1:
                return _print(members[0], names) + ","
            return ", ".join(_print(member, names) for member in members)
        case SweepRange(first=first, last=last, count=count, spacing=spacing):
            function = "linspace" if spacing == "linear" else "geomspace"
            return f"{function}({_print(first, names)}, {_print(last, names)}, {count})"
        case SweepSample(draw=draw, count=count):
            return f"sample({_print(draw, names)}, {count})"
        case _:
            return _print(value, names)
