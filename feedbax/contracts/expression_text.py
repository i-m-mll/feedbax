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

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.expression_sequence import sequence_definition
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
    Scalar,
    SequenceValue,
    Site,
    SweepList,
    SweepRange,
    SweepSample,
    TextSpan,
    UnresolvedValue,
    ValueBase,
    normal_literal_number as _stored_number,
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
      | power
?power: indexed POW unary -> binary
      | indexed
?indexed: indexed "[" sequence_item "]" -> index
        | atom
?atom: NUMBER UNIT? -> number
     | "true" -> true
     | "false" -> false
     | NAME "(" [argument ("," argument)*] ")" -> call
     | NAME -> name
     | ESCAPED_STRING -> quoted_name
     | SINGLE_QUOTED_NAME -> single_quoted_name
     | "[" sequence_item ("," sequence_item)* "]" -> vector
     | "(" expr ")"
?argument: NAME "=" expr -> keyword
         | sequence_item
?sequence_item: sum ".." sum -> sequence_span
              | expr
CMP: "==" | "!=" | "<=" | ">=" | "<" | ">"
ADD: "+" | "-" | "−"
MUL: "*" | "/" | "·" | "%"
POW: "**"
NUMBER: /(?:[0-9]+(?:\.(?!\.)[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?/
NAME: /NAME_PATTERN/
UNIT: /UNIT_PATTERN/
SINGLE_QUOTED_NAME: /'(?:\\['\\]|[^'\\])*'/
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
    "**": "pow",
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


@lru_cache(maxsize=4)
def _parser(start: Literal["field", "rule", "expr", "sequence_item"] = "field") -> Lark:
    """The parser of a field's text, or of a run rule's, where terms are separated by spaces and so take no units, and
    names are project names, which take hyphens."""
    units = sorted(filter(None, UNITS), key=len, reverse=True)
    unit_pattern = (
        "(?:" + "|".join(re.escape(unit).replace("/", r"\/") for unit in units) + r")(?=\.\.|[^\w.]|$)"
        if start != "rule" else "(?!)_"
    )
    name_pattern = RULE_NAME.pattern if start == "rule" else _IDENTIFIER.pattern
    return Lark(
        _GRAMMAR.replace("UNIT_PATTERN", unit_pattern).replace("NAME_PATTERN", name_pattern),
        start=start,
        parser="lalr",
        propagate_positions=True,
        maybe_placeholders=False,
    )


@dataclass(frozen=True)
class ParsedField[T: FieldValue]:
    value: T
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
    def __init__(
        self, names: Mapping[str, str] | None, functions: FunctionRegistry | None, *,
        stored: bool = False, keep_unknown_names: bool = False,
    ):
        """``names`` maps the names in scope to their references; ``None`` takes every name as its own reference."""
        self.names = names
        self.spans: dict[NodePath, TextSpan] = {}
        self.functions = functions
        self.stored = stored
        self.keep_unknown_names = keep_unknown_names

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
        expression = self._expr(node, path)
        if self.stored and isinstance(expression, CallValue) and not expression.kwargs:
            match expression.function, expression.args:
                case ("neg" | "pos") as sign, (LiteralValue(value=value, unit=unit),) if type(value) is not bool:
                    self._keep_operand_spans(path)
                    return LiteralValue(value=_stored_number(-value if sign == "neg" else value), unit=unit)
                case "lag", (operand, LiteralValue(value=0, unit=None)) if type(expression.args[1].value) is int:
                    self._keep_operand_spans(path)
                    return operand
        return expression

    def _keep_operand_spans(self, path: NodePath) -> None:
        """Move descendants of a normalized first operand to its new path."""
        operand = (*path, "args", 0)
        contents = {
            parts: self.spans.pop(parts) for parts in tuple(self.spans)
            if len(parts) > len(path) and parts[:len(path)] == path
        }
        self.spans.update({
            (*path, *parts[len(operand):]): span for parts, span in contents.items()
            if len(parts) > len(operand) and parts[:len(operand)] == operand
        })

    def _expr(self, node: Tree, path: NodePath) -> Expression:
        match node.data:
            case "number":
                number, *units = node.children
                value = float(number) if any(c in number for c in ".eE") else int(number)
                if isinstance(value, float) and not isfinite(value):
                    _fail(node, ProblemCode.PARSE, "A number must be finite.", path)
                return LiteralValue(value=_stored_number(value) if self.stored else value, unit=str(units[0]) if units else None)
            case "true" | "false":
                return LiteralValue(value=node.data == "true")
            case "name" | "quoted_name" | "single_quoted_name":
                spelling = str(node.children[0])
                if node.data == "quoted_name":
                    name = strict_json_loads(spelling)
                elif node.data == "single_quoted_name":
                    name = re.sub(r"\\(['\\])", r"\1", spelling[1:-1])
                else:
                    name = spelling
                if self.names is None:
                    return NameValue(reference=name)
                if name not in self.names:
                    if self.keep_unknown_names:
                        return NameValue(reference=name)
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
            case "sequence_span":
                first, last = node.children
                return SequenceValue(
                    rule="span", first=self.expr(first, (*path, "first")), last=self.expr(last, (*path, "last")),
                )
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
        if name in ("linspace", "geomspace"):
            if len(parts) != 3 or any(child.data == "keyword" for child in parts):
                _fail(node, ProblemCode.ARITY, f"{name} requires first, last and count.", path)
            first, last, count = parts
            return SequenceValue(
                rule=str(name), first=self.expr(first, (*path, "first")),
                last=self.expr(last, (*path, "last")), count=self.expr(count, (*path, "count")),
            )
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
            weights = None
            if family == "choice" and "weights" in kwargs:
                try:
                    count = sum(sequence_definition(arg).count if isinstance(arg, SequenceValue) else 1 for arg in args)
                except ValueError:
                    count = None  # Family indices are substituted before the checker counts the members.
                weights = self._choice_weights(kwargs.pop("weights"), path, count)
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
                    "Draws take positional arguments and optional per. Only choice takes weights.",
                    path,
                )
            return DrawValue(family=family or str(name), args=tuple(args), per=level, weights=weights)
        if name == "sample":
            reason = "An authored field takes one expression." if self.stored else "A sweep must be at the field root."
            _fail(node, ProblemCode.STRUCTURE, reason, path)
        if self.stored and "weights" in kwargs:
            _fail(kwargs["weights"], ProblemCode.STRUCTURE, "Only choice takes weights.", (*path, "weights"))
        return CallValue(
            function=str(name),
            args=tuple(args),
            kwargs={
                key: self._keyword(str(name), key, child, (*path, "kwargs", key))
                for key, child in sorted(kwargs.items())
            },
        )

    def _choice_weights(self, node: Tree, path: NodePath, count: int | None) -> tuple[float, ...]:
        target = (*path, "weights")
        self.spans[target] = _span(node)
        if node.data != "vector" or count is not None and len(node.children) != count:
            _fail(node, ProblemCode.STRUCTURE, "Choice needs one weight per entry.", target)
        weights = []
        for index, child in enumerate(node.children):
            at = (*target, index)
            weight = self.expr(child, at)
            if not isinstance(weight, LiteralValue) or isinstance(weight.value, bool) or weight.unit or weight.value < 0:
                _fail(child, ProblemCode.STRUCTURE, "A choice weight must be a non-negative unitless number.", at)
            weights.append(weight.value)
        if not any(weights):
            _fail(node, ProblemCode.STRUCTURE, "At least one choice weight must be positive.", target)
        return tuple(weights)

    def _keyword(self, function: str, key: str, node: Tree, path: NodePath) -> Expression:
        if function == "points" and key == "interp" and node.data == "name":
            mode = str(node.children[0])
            self.spans[path] = _span(node)
            return CallValue(function=mode)
        return self.expr(node, path)


def parse(
    text: str, site: Site, functions: FunctionRegistry | None = None, *, keep_unknown_names: bool = False,
) -> ParsedField[FieldValue] | tuple[Problem, ...]:
    """Parse editable text against the site's current names."""
    if problems := scope_problems(site):
        return problems
    return _parse_text(text, _Reader(
        {item.name: item.id for item in site.names}, functions, keep_unknown_names=keep_unknown_names,
    ))


def _parse_text(
    text: str, reader: _Reader, *, sequence: bool = False,
) -> ParsedField[FieldValue] | tuple[Problem, ...]:
    try:
        tree = _parser("sequence_item" if sequence else "expr" if reader.stored else "field").parse(text)
        value = reader.expr(tree, ()) if reader.stored else reader.field(tree)
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


@lru_cache(maxsize=1024)
def _parsed_stored_text(text: str) -> ParsedField[FieldValue] | tuple[Problem, ...]:
    if len(text) > 8192:
        return (Problem(code=ProblemCode.PARSE, reason="An expression can have at most 8,192 characters."),)
    return _parse_text(text, _Reader(None, None, stored=True))


def parse_stored_expression(value: Scalar | str) -> ParsedField[Expression] | tuple[Problem, ...]:
    """Parse one authored expression with names as stable reference spellings."""
    if not isinstance(value, str):
        return ParsedField(LiteralValue(value=_stored_number(value)), {})
    parsed = _parsed_stored_text(value)
    if isinstance(parsed, tuple):
        return parsed
    assert not isinstance(parsed.value, (SweepList, SweepRange, SweepSample))
    return ParsedField(parsed.value.model_copy(deep=True), parsed.spans)


def parse_sequence_rule(text: str) -> ParsedField[SequenceValue] | tuple[Problem, ...]:
    """Parse the one sequence at a family's index, with key-spelled names."""
    if len(text) > 8192:
        return (Problem(code=ProblemCode.PARSE, reason="An expression can have at most 8,192 characters."),)
    parsed = _parse_text(text, _Reader(None, None, stored=True), sequence=True)
    if isinstance(parsed, tuple):
        return parsed
    if not isinstance(parsed.value, SequenceValue):
        return (Problem(code=ProblemCode.STRUCTURE, reason="Expected one sequence rule.", span=parsed.spans.get(())),)
    return ParsedField(parsed.value, parsed.spans)


def resolve_expression(
    value: Expression, site: Site, functions: FunctionRegistry,
) -> ParsedField[Expression] | tuple[Problem, ...]:
    """Read held text in the current site, without checking its type."""
    if not isinstance(value, UnresolvedValue):
        return ParsedField(value, {})
    parsed = parse(value.text, site, functions)
    if isinstance(parsed, tuple):
        return parsed
    if isinstance(parsed.value, SweepList | SweepRange | SweepSample):
        return (Problem(code=ProblemCode.STRUCTURE, reason="This field takes one expression."),)
    return ParsedField(parsed.value, parsed.spans)


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
_UNARY, _POWER, _INDEXED, _ATOM = 7, 8, 9, 10


def _literal_text(value: Scalar, unit: str | None, expected: ValueBase | None) -> str:
    number = canonical_json_v2_bytes(value).decode()
    if type(value) is float and number.lstrip("-").isdigit():
        # A decimal point or exponent keeps the legacy parser's real type.
        legacy = repr(value)
        if expected != ValueBase.REAL or abs(value) >= 2**53 or legacy == "-0.0":
            number = legacy
    return number + ("°" if unit == "°" else f" {unit}" if unit else "")


def _print(
    node: Expression, names: Mapping[str, str] | None, least: int = 0, *, expected: ValueBase | None = None,
) -> str:
    """The text of ``node`` as an operand that binds at least as tightly as ``least``, parenthesized only if it must be."""
    text, precedence = _printed(node, names, expected)
    return f"({text})" if precedence < least else text


def _printed(node: Expression, names: Mapping[str, str] | None, expected: ValueBase | None) -> tuple[str, int]:
    match node:
        case UnresolvedValue(text=text):
            return text, _ATOM
        case LiteralValue(value=value, unit=unit):
            text = _literal_text(_stored_number(value) if names is None else value, unit, expected)
            return text, _UNARY if text.startswith("-") else _ATOM
        case NameValue(reference=reference):
            name = reference if names is None else names.get(reference, reference)
            if _IDENTIFIER.fullmatch(name) and name not in _KEYWORDS:
                return name, _ATOM
            if names is None:
                escaped = name.replace("\\", "\\\\").replace("'", r"\'")
                return f"'{escaped}'", _ATOM
            return json.dumps(name, ensure_ascii=False), _ATOM
        case SequenceValue(rule="span", first=first, last=last):
            return f"{_print(first, names, 5)}..{_print(last, names, 5)}", _ATOM
        case SequenceValue(rule=rule, first=first, last=last, count=count):
            assert count is not None
            return f"{rule}({_print(first, names)}, {_print(last, names)}, {_print(count, names)})", _ATOM
        case DrawValue(family=family, args=args, per=per, weights=weights):
            parts = [_print(arg, names) for arg in args]
            if per == "run" or family not in _DRAW_SPELLINGS:
                parts.append(f"per={per}")
            if weights is not None:
                text = ", ".join(_literal_text(_stored_number(weight), None, None) for weight in weights)
                parts.append(f"weights=[{text}]")
            return f"{_DRAW_SPELLINGS.get(family, family)}({', '.join(parts)})", _ATOM
        case CallValue(function="pow", args=(base, exponent), kwargs=kwargs) if not kwargs:
            return f"{_print(base, names, _INDEXED)} ** {_print(exponent, names, _UNARY)}", _POWER
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


def print_field(value: FieldValue, site: Site, *, fragment: bool = False) -> str:
    """Render editable text, using the site's expected type only for a whole field."""
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
            return _print(value, names, expected=None if fragment else site.expected.base)


def print_stored_expression(value: Expression) -> Scalar | str:
    """Write a keyed expression as a scalar literal or canonical authored text."""
    if isinstance(value, LiteralValue) and not value.unit:
        return _stored_number(value.value)
    text = _print(value, None)
    if isinstance(value, (CallValue, UnresolvedValue)):
        # Parse through the same normal form that will be compiled on reopen.
        parsed = parse_stored_expression(text)
        if not isinstance(parsed, tuple):
            canonical = parsed.value
            if isinstance(canonical, LiteralValue) and not canonical.unit:
                return canonical.value
            return _print(canonical, None)
    return text
