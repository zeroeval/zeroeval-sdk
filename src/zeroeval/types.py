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
    model: Optional[str]
    created_by: Optional[str]
    updated_by: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    metadata: Dict[str, Any]
    source: str  # Literal["server", "fallback"] but keep simple for py39 compatibility

    @staticmethod
    def from_response(data: Dict[str, Any]) -> "Prompt":
        model_value = data.get("model_id") or data.get("model")
        if isinstance(model_value, str) and model_value:
            model_value = f"zeroeval/{model_value}"
        # Normalize version_id from payload or nested metadata
        version_id_value = data.get("version_id")
        if not version_id_value:
            meta = data.get("metadata") or {}
            if isinstance(meta, dict):
                version_id_value = meta.get("version_id") or meta.get("prompt_version_id")
        return Prompt(
            content=str(data.get("content", "")),
            version=data.get("version"),
            version_id=version_id_value,
            tag=data.get("tag"),
            is_latest=bool(data.get("is_latest", False)),
            model=model_value,
            created_by=data.get("created_by"),
            updated_by=data.get("updated_by"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            metadata=data.get("metadata", {}) or {},
            source="server",
        )


