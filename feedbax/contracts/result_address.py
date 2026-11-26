"""Result addresses, ``[card/]element[@row][:label=name,...]``, resolved by element key with the card as a qualifier.

A card is a step, addressed by its key. An operation or figure key is held by one family of Analysis
documents (a root and the documents based on it), so the element key alone names a result; a card
qualifier that no longer holds the element still resolves when exactly one card holds it.

The selector after the first ``:`` names requests and rows by what they are, never by digest: a mapped request by
its label values, by name, and a row by its coordinates (an axis's value id, its drawn value, or an input axis's member
key). A pair matches when either holds it; the pairs written may be fewer than the request or row holds. A value runs
from the first ``=`` to the next ``,``, so it may hold ``/``, ``=`` and ``--``. Without a selector an address names every
request of the element. ``@row`` names a row by its exact id.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
import re

from pydantic import JsonValue

from feedbax.contracts.analysis_document import ANALYSIS_SCHEMA as ANALYSIS_SCHEMA, ELEMENT_FAMILIES as ELEMENT_FAMILIES
from feedbax.contracts.project import Document, Project, base_chain
from feedbax.contracts.project_resolution import ProjectDiagnostic, compose_document, compose_step
from feedbax.contracts.project_sweep import SweepCoordinate, coordinate_text

_KEY = r"[A-Za-z0-9_][A-Za-z0-9_-]*"
_ROW = r"[A-Za-z0-9_][A-Za-z0-9_=-]*"
_HEAD = re.compile(rf"^(?:(?P<card>{_KEY})/)?(?P<name>{_KEY})(?:@(?P<row>{_ROW}))?$")
_PAIR = re.compile(rf"^(?P<label>{_KEY})=(?P<value>[^,]+)$")
_GENERATED = re.compile(r"value-[0-9a-f]{64}")

type Labels = tuple[tuple[str, str], ...]
"""Label or axis values by name, in the order written."""


def parse_labels(text: str) -> Labels:
    """The ``label=value`` pairs of a member id or an address's selector."""
    pairs = []
    for pair in text.split(",") if text else ():
        match = _PAIR.match(pair)
        if match is None:
            raise ValueError(f"{pair!r} is not of the form label=value")
        pairs.append((match["label"], match["value"]))
    return tuple(pairs)


def format_labels(labels: Labels) -> str:
    return ",".join(f"{label}={name}" for label, name in labels)


def selects(selector: Labels, names: Mapping[str, Collection[str]]) -> bool:
    """Whether what has ``names`` (each label's or axis's accepted values) is what ``selector`` names."""
    return all(value in names.get(label, ()) for label, value in selector)


def row_names(
    axis_keys: Mapping[str, str], coordinates: Mapping[str, SweepCoordinate],
) -> dict[str, set[str]]:
    """Each axis of a row and the values a selector may name it by: its key, value id and drawn value."""
    names: dict[str, set[str]] = {axis: {key} for axis, key in axis_keys.items()}
    for axis, coordinate in coordinates.items():
        names.setdefault(axis, set()).update((coordinate.value_id, coordinate_text(coordinate)))
    return names


def row_selector(
    row: str, axis_keys: Mapping[str, str], coordinates: Mapping[str, SweepCoordinate],
) -> Labels | None:
    """The selector that names a row whose id is not plain (it holds a digest), or None when ``@row`` reads plainly."""
    if "==" not in row and not _GENERATED.search(row):
        return None
    return tuple(
        (axis, coordinate_text(coordinates[axis]) if _GENERATED.fullmatch(key) and axis in coordinates else key)
        for axis, key in sorted(axis_keys.items())
    )


@dataclass(frozen=True)
class ResultAddress:
    """An authored address: ``name`` is an element key, or a card key when no card qualifies it."""

    card: str | None
    name: str
    row: str | None = None
    labels: Labels = ()

    @classmethod
    def parse(cls, text: str) -> ResultAddress:
        head, separator, selector = text.partition(":")
        match = _HEAD.match(head)
        try:
            if match is None:
                raise ValueError(text)
            return cls(match["card"], match["name"], match["row"], parse_labels(selector) if separator else ())
        except ValueError:
            raise ValueError(
                f"{text!r} is not an address of the form [card/]element[@row][:label=value,...]"
            ) from None


@dataclass(frozen=True)
class ResultTarget:
    """A resolved address: every result of ``card``, or of its element ``element``, at ``row`` when given."""

    card: str
    element: str | None = None
    row: str | None = None
    labels: Labels = ()

    def __str__(self) -> str:
        address = self.card if self.element is None else f"{self.card}/{self.element}"
        address = address if self.row is None else f"{address}@{self.row}"
        return f"{address}:{format_labels(self.labels)}" if self.labels else address


@dataclass(frozen=True)
class AddressProblem:
    detail: str
    missing_subject: str | None = None


def analysis_elements(payload: JsonValue) -> dict[str, JsonValue]:
    """The operation and figure members of an Analysis payload."""
    if not isinstance(payload, dict):
        return {}
    return {
        key: member for family in ELEMENT_FAMILIES
        if isinstance(members := payload.get(family), dict) for key, member in members.items()
    }


def payload_elements(payload: JsonValue) -> frozenset[str]:
    return frozenset(analysis_elements(payload))


def authored_analysis_payload(project: Project, card: str) -> JsonValue:
    template = compose_step(project, card)
    if isinstance(template, ProjectDiagnostic):
        return None
    try:
        return template.authored_payload()
    except ValueError:
        return None


@dataclass
class ElementHolders:
    """Which card holds which element, read lazily from each Analyze card's composed payload."""

    cards: tuple[str, ...]
    elements_of: Callable[[str], frozenset[str]]
    _held: dict[str, frozenset[str]] = field(default_factory=dict)

    def held(self, card: str) -> frozenset[str]:
        if card not in self._held:
            self._held[card] = self.elements_of(card) if card in self.cards else frozenset()
        return self._held[card]

    def holders(self, element: str) -> tuple[str, ...]:
        return tuple(card for card in self.cards if element in self.held(card))

    def card_of(self, card: str | None, element: str) -> str | AddressProblem:
        """The card ``card`` when it holds ``element``, else the one card that does."""
        if card is not None and element in self.held(card):
            return card
        match self.holders(element):
            case (holder,):
                return holder
            case ():
                return AddressProblem(f"No card holds {element!r}.", missing_subject=element)
            case holders:
                return AddressProblem(f"{element!r} is held by {', '.join(holders)}; qualify it by its card.")

    def resolve(self, address: ResultAddress, steps: Iterable[str]) -> ResultTarget | AddressProblem:
        if address.card is None and address.name in set(steps):
            if self.holders(address.name):
                return AddressProblem(f"{address.name!r} names a card and an element.")
            return ResultTarget(address.name, row=address.row, labels=address.labels)
        card = self.card_of(address.card, address.name)
        return card if isinstance(card, AddressProblem) else ResultTarget(card, address.name, address.row, address.labels)


def project_holders(project: Project) -> ElementHolders:
    """The element holders of ``project``'s Analyze cards, composed without external readers."""
    return ElementHolders(analyze_cards(project), lambda card: payload_elements(authored_analysis_payload(project, card)))


def analyze_cards(project: Project) -> tuple[str, ...]:
    return tuple(sorted(name for name, step in project.steps.items() if step.kind == "analyze"))


@dataclass(frozen=True)
class SharedKey:
    """An element key in ``document``'s ``family`` section, held first by the unrelated family rooted at ``holder``."""

    element: str
    family: str
    document: str
    holder: str


def shared_element_keys(project: Project) -> tuple[SharedKey, ...]:
    """Each Analysis document holding a key that an unrelated family holds first, families taken by root name."""
    families: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for name in sorted(project.documents):
        if project.documents[name].schema_id != ANALYSIS_SCHEMA:
            continue
        document = compose_document(project.documents.get, name)
        if isinstance(document, ProjectDiagnostic) or not isinstance(payload := document.payload, dict):
            continue
        keys = {
            key: family for family in ELEMENT_FAMILIES
            if isinstance(members := payload.get(family), dict) for key in members
        }
        families.setdefault(base_chain(project.documents, name)[-1], []).append((name, keys))
    owner: dict[str, str] = {}
    shared = []
    for root in sorted(families):
        for name, keys in families[root]:
            shared.extend(
                SharedKey(key, family, name, owner[key])
                for key, family in sorted(keys.items()) if owner.get(key, root) != root
            )
        for _, keys in families[root]:
            for key in keys:
                owner.setdefault(key, root)
    return tuple(shared)


def family_documents(documents: Mapping[str, Document], name: str) -> tuple[str, ...]:
    """The documents sharing ``name``'s root: ``name``'s family."""
    root = base_chain(documents, name)[-1]
    return tuple(sorted(other for other in documents if base_chain(documents, other)[-1] == root))
