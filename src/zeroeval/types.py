from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Dict


@dataclass
class Prompt:
    content: str
    version: Optional[int]
    version_id: Optional[str]
    tag: Optional[str]
    is_latest: bool
    created_by: Optional[str]
    updated_by: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    metadata: Dict[str, Any]
    source: str  # Literal["server", "fallback"] but keep simple for py39 compatibility

    @staticmethod
    def from_response(data: Dict[str, Any]) -> "Prompt":
        return Prompt(
            content=str(data.get("content", "")),
            version=data.get("version"),
            version_id=data.get("version_id"),
            tag=data.get("tag"),
            is_latest=bool(data.get("is_latest", False)),
            created_by=data.get("created_by"),
            updated_by=data.get("updated_by"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            metadata=data.get("metadata", {}) or {},
            source="server",
        )


