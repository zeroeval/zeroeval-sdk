from __future__ import annotations

from typing import Optional


class PromptNotFoundError(Exception):
    def __init__(self, slug: str, version: Optional[int], tag: Optional[str]):
        self.slug = slug
        self.version = version
        self.tag = tag
        super().__init__(f"Prompt not found: {slug} (version={version}, tag={tag})")


class PromptRequestError(Exception):
    def __init__(self, message: str, status: Optional[int] = None):
        self.status = status
        super().__init__(message)


