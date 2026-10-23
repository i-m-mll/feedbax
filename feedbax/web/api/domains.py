from __future__ import annotations

from fastapi import APIRouter

from feedbax.component_registry import builtin_domain_registry
from feedbax.contracts.domain import DomainRegistryPayload
from feedbax.contracts.studio_api import DomainListResponse


router = APIRouter()
registry = builtin_domain_registry()


@router.get("", response_model=DomainListResponse)
async def list_domains() -> DomainListResponse:
    return DomainListResponse(data=DomainRegistryPayload(domains=registry.list_all()))
