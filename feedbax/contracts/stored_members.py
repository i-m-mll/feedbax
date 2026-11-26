"""Keyed member contracts shared by the authored document kinds."""

from collections.abc import Callable, Iterable, Mapping
from hashlib import sha256
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, StringConstraints


type StoredMemberKey = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z_][A-Za-z0-9_-]*$")]


type StoredMembers[T] = dict[StoredMemberKey, T]


def scoped_set_member_id(*parts: str) -> str:
    """Identify a lowered Set member by the keys that name its authoring scope."""
    return "set-" + sha256("\0".join(parts).encode()).hexdigest()[:28]


def _canonical_model_address(text: str) -> str:
    from feedbax.contracts.model_address import ModelAddress

    ModelAddress.from_authored_text(text)
    return text


type StoredModelAddressKey = Annotated[str, StringConstraints(strict=True), AfterValidator(_canonical_model_address)]
type StoredAddressedMembers[T] = dict[StoredModelAddressKey, T]


def project_member[ModelT: BaseModel](
    source: BaseModel, target: type[ModelT], transforms: Mapping[str, Callable[[Any], Any]] | None = None,
    /, **overrides: Any,
) -> ModelT:
    """Copy authored fields, transforming collections whose typed shapes differ."""
    transforms = transforms or {}
    values = {}
    for name, field in type(source).model_fields.items():
        if name not in target.model_fields or name in overrides:
            continue
        value = getattr(source, name)
        if name in source.model_fields_set or field.is_required() or field.default_factory_takes_validated_data:
            authored = True
        else:
            default = field.get_default(call_default_factory=True)
            authored = type(value) is not type(default) or value != default
        if authored:
            values[name] = transforms[name](value) if name in transforms else value
    return target.model_validate(values | overrides)


def keyed_members[MemberT](members: Iterable[tuple[str, MemberT]]) -> dict[str, MemberT]:
    """Keep member order and reject a projection that would lose a duplicate member."""
    keyed = {}
    for key, member in members:
        if key in keyed:
            raise ValueError(f"Duplicate member key {key!r}.")
        keyed[key] = member
    return keyed
