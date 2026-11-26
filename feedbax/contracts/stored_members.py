"""Keyed member contracts shared by the authored document kinds."""

from collections.abc import Iterable
from typing import Annotated, Any

from pydantic import BaseModel, StringConstraints


type StoredMemberKey = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z_][A-Za-z0-9_-]*$")]


def project_member[ModelT: BaseModel](source: BaseModel, target: type[ModelT], **overrides: Any) -> ModelT:
    """Copy supplied fields shared by the two model shapes and validate the new shape."""
    return target.model_validate({
        **{name: getattr(source, name) for name in type(source).model_fields
           if name in source.model_fields_set and name in target.model_fields},
        **overrides,
    })


def keyed_members[MemberT](members: Iterable[tuple[str, MemberT]]) -> dict[str, MemberT]:
    """Keep member order and reject a projection that would lose a duplicate member."""
    keyed = {}
    for key, member in members:
        if key in keyed:
            raise ValueError(f"Duplicate member key {key!r}.")
        keyed[key] = member
    return keyed
