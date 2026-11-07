"""Shared physical authoring constraints."""

from typing import Annotated
from pydantic import Field

NonNegativeFloat = Annotated[float, Field(ge=0)]
