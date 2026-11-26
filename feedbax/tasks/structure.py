"""A task document's structure changed so that its ids, references and cells stay consistent.

Members are addressed by id, and expressions refer to variables by id, so a rename changes no reference. A removed
variable stays referenced where it was read: the document still compiles to Problems there, and is returned.
"""

from typing import assert_never

from feedbax.contracts.expression_values import (
    RESERVED_NAMES, LiteralValue, Problem, ProblemCode, UnresolvedValue, ValueBase, ValueType,
)
from feedbax.contracts.task_document import Cell, Epoch, Length, Link, Repeat, Signal, TaskDocument, Variable
from feedbax.contracts.task_structure import (
    AddMember, MakeAbsent, MakePresent, MoveMember, RemoveMember, RenameMember, SetLink, SetSpan, SetType,
    RemoveRepeat, RepeatEpochs, SetRepeatCount, SetVariableCount, SetVariableMarker, TaskStructureChange,
)
from feedbax.tasks.document import zero

_FIELDS = {"variable": "variables", "epoch": "epochs", "signal": "signals"}
_NUMBER = ValueType(base=ValueBase.REAL)
_NEW_EPOCH = Length(length=LiteralValue(value=1, unit="st"))
"""An added epoch lasts one step, so that it fits a fixed trial length the others leave room in."""


def restructure(document: TaskDocument, change: TaskStructureChange) -> TaskDocument | Problem:
    """``document`` with ``change`` made, or the Problem that names a member the document does not have."""
    match change:
        case AddMember(member="epoch" | "signal", initializes=(_, *_)):
            return Problem(code=ProblemCode.STRUCTURE, reason="Only a variable initializes the Model.")
        case AddMember():
            return _add(document, change)
        case RemoveMember(member="epoch") if len(document.epochs) == 1:
            return Problem(code=ProblemCode.STRUCTURE, reason="A task has at least one epoch.", path=("epochs",))
        case RemoveMember() | RenameMember() | MoveMember() | SetType():
            members = getattr(document, _FIELDS[change.member])
            index = next((i for i, item in enumerate(members) if item.id == change.id), None)
            if index is None:
                return _missing(change.member, change.id)
            return _edit_member(document, change, members, index)
        case SetSpan() | MakeAbsent() | MakePresent():
            signal = next((item for item in document.signals if item.id == change.signal), None)
            if signal is None:
                return _missing("signal", change.signal)
            cells = _cells(document, signal, change)
            if isinstance(cells, Problem):
                return cells
            return document.model_copy(update={"signals": tuple(
                item.model_copy(update={"cells": cells}) if item.id == signal.id else item
                for item in document.signals
            )})
        case SetLink():
            links = tuple(link for link in document.links if link.model_field != change.model_field)
            if change.variable is not None:
                links = (*links, Link(model_field=change.model_field, variable=change.variable))
            return document.model_copy(update={"links": links})
        case RepeatEpochs():
            order = tuple(epoch.id for epoch in document.epochs)
            for endpoint in (change.first_epoch, change.last_epoch):
                if endpoint not in order:
                    return _missing("epoch", endpoint)
            first, last = sorted((order.index(change.first_epoch), order.index(change.last_epoch)))
            repeat = Repeat(
                id=_fresh("repeat", {item.id for item in document.repeats}),
                first_epoch=order[first], last_epoch=order[last], count=change.count,
            )
            return document.model_copy(update={"repeats": (*document.repeats, repeat)})
        case SetRepeatCount() | RemoveRepeat():
            if all(item.id != change.id for item in document.repeats):
                return _missing("repeat", change.id)
            repeats = tuple(
                item.model_copy(update={"count": change.count}) if item.id == change.id else item
                for item in document.repeats
                if not isinstance(change, RemoveRepeat) or item.id != change.id
            )
            return document.model_copy(update={"repeats": repeats})
        case SetVariableCount() | SetVariableMarker():
            if all(item.id != change.id for item in document.variables):
                return _missing("variable", change.id)
            update = {"count": change.count} if isinstance(change, SetVariableCount) else {"marker": change.marker}
            return document.model_copy(update={"variables": tuple(
                item.model_copy(update=update) if item.id == change.id else item
                for item in document.variables
            )})
        case _:
            assert_never(change)


def _missing(member: str, id: str) -> Problem:
    return Problem(code=ProblemCode.STRUCTURE, reason=f"The task has no {member} {id!r}.")


def _fresh(name: str, taken: set[str]) -> str:
    if name not in taken:
        return name
    count = 2
    while f"{name}-{count}" in taken:
        count += 1
    return f"{name}-{count}"


def _add(document: TaskDocument, change: AddMember) -> TaskDocument:
    field = _FIELDS[change.member]
    members = getattr(document, field)
    id = _fresh(change.name, {item.id for item in members})
    value_type = change.type or _NUMBER
    added = {
        "variable": lambda: Variable(id=id, name=change.name, type=value_type, value=zero(value_type.base, value_type.shape)),
        "epoch": lambda: Epoch(id=id, name=change.name, end=_NEW_EPOCH),
        "signal": lambda: Signal(id=id, name=change.name, type=value_type, cells=()),
    }[change.member]()
    index = len(members) if change.index is None else min(change.index, len(members))
    links = tuple(link for link in document.links if link.model_field not in change.initializes)
    return document.model_copy(update={
        field: (*members[:index], added, *members[index:]),
        "links": (*links, *(Link(model_field=address, variable=id) for address in change.initializes)),
    })


def _edit_member(
    document: TaskDocument, change: RemoveMember | RenameMember | MoveMember | SetType, members: tuple, index: int,
) -> TaskDocument | Problem:
    field = _FIELDS[change.member]
    item = members[index]
    rest = (*members[:index], *members[index + 1:])
    match change:
        case RenameMember():
            peers = rest
            if change.member == "variable":
                peers += document.signals
            elif change.member == "signal":
                peers += document.variables
            place = (field, item.id, "name")
            if change.name in RESERVED_NAMES:
                return Problem(
                    code=ProblemCode.DUPLICATE_NAME, reason=f"The name {change.name!r} is reserved.", path=place,
                )
            if any(peer.name == change.name for peer in peers):
                return Problem(
                    code=ProblemCode.DUPLICATE_NAME, reason=f"The name {change.name!r} is already in use.", path=place,
                )
            return document.model_copy(update={field: (*members[:index], item.model_copy(update={"name": change.name}),
                                                       *members[index + 1:])})
        case SetType():
            empty = zero(change.type.base, change.type.shape)
            update = (
                {"type": change.type, "value": empty} if change.member == "variable" else
                {"type": change.type, "absent": None,
                 "cells": tuple(cell.model_copy(update={"value": empty}) for cell in item.cells)}
            )
            return document.model_copy(update={field: (*members[:index], item.model_copy(update=update),
                                                       *members[index + 1:])})
        case MoveMember():
            moved = (*rest[:change.index], item, *rest[change.index:])
            if change.member != "epoch":
                return document.model_copy(update={field: moved})
            return document.model_copy(update={
                "epochs": moved, "signals": _respanned(document, moved, change.id),
                "repeats": _repeated_after_epoch_edit(document, moved, change.id),
            })
        case RemoveMember():
            update = {field: rest}
            if change.member == "epoch":
                update["signals"] = _respanned(document, rest, change.id)
                update["repeats"] = _repeated_after_epoch_edit(document, rest, change.id)
            if change.member in ("variable", "epoch"):
                key = "variables" if change.member == "variable" else "epoch_steps"
                update["validation_trials"] = tuple(
                    trial.model_copy(update={key: {k: v for k, v in getattr(trial, key).items() if k != change.id}})
                    for trial in document.validation_trials
                )
            return document.model_copy(update=update)
        case _:
            assert_never(change)


def _covered(order: tuple[str, ...], cell: Cell | Repeat) -> list[str]:
    first, last = order.index(cell.first_epoch), order.index(cell.last_epoch)
    return list(order[first:last + 1])


def _repeated_after_epoch_edit(
    document: TaskDocument, epochs: tuple[Epoch, ...], epoch: str,
) -> tuple[Repeat, ...]:
    before = tuple(item.id for item in document.epochs)
    after = tuple(item.id for item in epochs)
    repeats = []
    for repeat in document.repeats:
        if repeat.first_epoch not in before or repeat.last_epoch not in before:
            repeats.append(repeat)
            continue
        remaining = set(_covered(before, repeat)) - {epoch}
        ordered = [key for key in after if key in remaining]
        if ordered:
            repeat = repeat.model_copy(update={"first_epoch": ordered[0], "last_epoch": ordered[-1]})
        elif epoch not in after and repeat.first_epoch == repeat.last_epoch == epoch:
            continue
        repeats.append(repeat)
    return tuple(repeats)


def _respanned(document: TaskDocument, epochs: tuple[Epoch, ...], epoch: str) -> tuple[Signal, ...]:
    """The signals once ``epoch`` has left the epochs, or moved among them: a cell that spanned it with others gives it
    up, a cell of its own goes with it, and a cell it lands inside the span of takes it."""
    before = tuple(item.id for item in document.epochs)
    after = tuple(item.id for item in epochs)
    signals = []
    for signal in document.signals:
        spans = []
        for cell in signal.cells:
            covered = [id for id in _covered(before, cell) if id != epoch] or (
                [epoch] if epoch in after else []
            )
            if covered:
                indexes = sorted(after.index(id) for id in covered)
                spans.append((cell, indexes[0], indexes[-1]))
        own = [span for span in spans if span[1] == span[2] and after[span[1]] == epoch]
        shared = [span for span in spans if span not in own]
        if epoch in after and any(first <= after.index(epoch) <= last for _, first, last in shared):
            spans = shared
        cells = tuple(
            cell.model_copy(update={"first_epoch": after[first], "last_epoch": after[last]})
            for cell, first, last in sorted(spans, key=lambda span: span[1])
        )
        signals.append(signal.model_copy(update={"cells": cells}))
    return tuple(signals)


def _cells(document: TaskDocument, signal: Signal, change: SetSpan | MakeAbsent | MakePresent) -> tuple[Cell, ...] | Problem:
    order = tuple(item.id for item in document.epochs)
    match change:
        case MakeAbsent():
            if all(cell.id != change.cell for cell in signal.cells):
                return _missing("cell", change.cell)
            return tuple(cell for cell in signal.cells if cell.id != change.cell)
        case MakePresent():
            if change.epoch not in order:
                return _missing("epoch", change.epoch)
            if any(change.epoch in _covered(order, cell) for cell in signal.cells):
                return Problem(code=ProblemCode.STRUCTURE, reason=f"A cell already spans {change.epoch!r}.")
            cell = Cell(
                id=_fresh(change.epoch, {item.id for item in signal.cells}), first_epoch=change.epoch,
                last_epoch=change.epoch, value=(
                    UnresolvedValue(text=change.text) if change.text is not None
                    else signal.absent or zero(signal.type.base, signal.type.shape)
                ),
            )
            return tuple(sorted((*signal.cells, cell), key=lambda item: order.index(item.first_epoch)))
        case SetSpan():
            if all(cell.id != change.cell for cell in signal.cells):
                return _missing("cell", change.cell)
            if change.first_epoch not in order or change.last_epoch not in order:
                return _missing("epoch", change.first_epoch if change.first_epoch not in order else change.last_epoch)
            first, last = sorted((order.index(change.first_epoch), order.index(change.last_epoch)))
            cells = []
            for cell in signal.cells:
                if cell.id == change.cell:
                    cells.append(cell.model_copy(update={"first_epoch": order[first], "last_epoch": order[last]}))
                    continue
                kept = [order.index(id) for id in _covered(order, cell)]
                kept = [i for i in kept if i < first] or [i for i in kept if i > last]
                if kept:
                    cells.append(cell.model_copy(update={"first_epoch": order[kept[0]], "last_epoch": order[kept[-1]]}))
            return tuple(sorted(cells, key=lambda item: order.index(item.first_epoch)))
        case _:
            assert_never(change)
