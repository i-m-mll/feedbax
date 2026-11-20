"""Result addresses, ``[card/]element[@row]``, resolved by element key with the card as a qualifier.

A card is a step, addressed by its key. An operation or figure key is held by one family of Analysis
documents (a root and the documents based on it), so the element key alone names a result; a card
qualifier that no longer holds the element still resolves when exactly one card holds it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
import re

from pydantic import JsonValue

from feedbax.contracts.project import Document, Project, base_chain
from feedbax.contracts.project_resolution import ProjectDiagnostic, compose_document, compose_step

ANALYSIS_SCHEMA = "feedbax.analysis_document.v1"
ELEMENT_FAMILIES = ("operations", "views")

_KEY = r"[A-Za-z0-9_][A-Za-z0-9_-]*"
_ROW = r"[A-Za-z0-9_][A-Za-z0-9_=-]*"
_ADDRESS = re.compile(rf"^(?:(?P<card>{_KEY})/)?(?P<name>{_KEY})(?:@(?P<row>{_ROW}))?$")


@dataclass(frozen=True)
class ResultAddress:
    """An authored address: ``name`` is an element key, or a card key when no card qualifies it."""

    card: str | None
    name: str
    row: str | None = None

    @classmethod
    def parse(cls, text: str) -> ResultAddress:
        match = _ADDRESS.match(text)
        if match is None:
            raise ValueError(f"{text!r} is not an address of the form [card/]element[@row]")
        return cls(match["card"], match["name"], match["row"])


@dataclass(frozen=True)
class ResultTarget:
    """A resolved address: every result of ``card``, or of its element ``element``, at ``row`` when given."""

    card: str
    element: str | None = None
    row: str | None = None

    def __str__(self) -> str:
        address = self.card if self.element is None else f"{self.card}/{self.element}"
        return address if self.row is None else f"{address}@{self.row}"


@dataclass(frozen=True)
class AddressProblem:
    detail: str


def payload_elements(payload: JsonValue) -> frozenset[str]:
    """The operation and figure keys of an Analysis payload."""
    if not isinstance(payload, dict):
        return frozenset()
    return frozenset(
        key for family in ELEMENT_FAMILIES
        if isinstance(members := payload.get(family), dict) for key in members
    )


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
                return AddressProblem(f"No card holds {element!r}.")
            case holders:
                return AddressProblem(f"{element!r} is held by {', '.join(holders)}; qualify it by its card.")

    def resolve(self, address: ResultAddress, steps: Iterable[str]) -> ResultTarget | AddressProblem:
        if address.card is None and address.name in set(steps):
            if self.holders(address.name):
                return AddressProblem(f"{address.name!r} names a card and an element.")
            return ResultTarget(address.name, row=address.row)
        card = self.card_of(address.card, address.name)
        return card if isinstance(card, AddressProblem) else ResultTarget(card, address.name, address.row)


def project_holders(project: Project) -> ElementHolders:
    """The element holders of ``project``'s Analyze cards, composed without external readers."""

    def elements_of(card: str) -> frozenset[str]:
        template = compose_step(project, card)
        if isinstance(template, ProjectDiagnostic):
            return frozenset()
        try:
            return payload_elements(template.authored_payload())
        except ValueError:
            return frozenset()

    return ElementHolders(analyze_cards(project), elements_of)


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
