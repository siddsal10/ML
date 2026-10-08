from __future__ import annotations

from fastapi import APIRouter

from app.providers.registry import describe_all_providers
from app.schemas.benchmark import ProviderInfo

router = APIRouter(prefix="/api/providers", tags=["providers"])


@router.get("", response_model=list[ProviderInfo])
def list_providers() -> list[ProviderInfo]:
    return [ProviderInfo(**p) for p in describe_all_providers()]
