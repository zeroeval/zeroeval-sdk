from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Tuple

import requests

from .cache import TTLCache
from .errors import PromptNotFoundError, PromptRequestError
from .template import render_template
from .types import Prompt


_SLUG_RE = re.compile(r"^[a-z0-9-]+$")
_TAG_RE = re.compile(r"^[a-z0-9-]+$")


class ZeroEval:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_tag: Optional[str] = None,
        timeout: float = 10.0,
        cache_ttl_seconds: float = 60.0,
    ) -> None:
        self._api_key = api_key or os.getenv("ZEROEVAL_API_KEY")
        if not self._api_key:
            raise ValueError("ZEROEVAL_API_KEY not set")
        self._base_url = (base_url or os.getenv("ZEROEVAL_BASE_URL") or os.getenv("ZEROEVAL_API_URL") or "https://api.zeroeval.com").rstrip("/")
        self._default_tag = default_tag or self._resolve_default_tag()
        self._timeout = float(timeout)
        self._cache: TTLCache[Tuple[str, str, Optional[int], Optional[str]], Prompt] = TTLCache(
            ttl_seconds=cache_ttl_seconds
        )

    def _resolve_default_tag(self) -> str:
        # Explicit override
        env_tag = os.getenv("ZEROEVAL_PROMPT_TAG")
        if env_tag:
            return env_tag
        # Environment-based mapping
        env = os.getenv("ZEROEVAL_ENV", "development").lower()
        return "production" if env == "production" else "latest"

    @staticmethod
    def _validate_slug(slug: str) -> str:
        if not _SLUG_RE.match(slug):
            raise PromptRequestError("Invalid slug format. Use single segment like 'support-triage'", status=None)
        return slug

    def get_prompt(
        self,
        slug: str,
        *,
        version: Optional[int] = None,
        tag: Optional[str] = None,
        fallback: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        render: bool = True,
        missing: str = "error",
        use_cache: bool = True,
        timeout: Optional[float] = None,
    ) -> Prompt:
        # Validate inputs
        validated_slug = self._validate_slug(slug)
        if version is not None and version < 1:
            raise PromptRequestError("version must be >= 1", status=None)
        if tag is not None and not _TAG_RE.match(tag):
            raise PromptRequestError("Invalid tag format", status=None)

        # Resolve effective tag if needed
        effective_tag = None
        if version is None:
            effective_tag = tag or self._default_tag

        # Cache key: (org_id not available in SDK; use api_key suffix to namespace)
        cache_ns = self._api_key[-6:] if self._api_key else ""
        cache_key = (cache_ns, slug, version, effective_tag)
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Build request
        params: Dict[str, Any] = {}
        if version is not None:
            params["version"] = int(version)
        elif effective_tag is not None:
            params["tag"] = effective_tag

        url = f"{self._base_url}/v1/prompts/{validated_slug}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": f"zeroeval-sdk-python/{os.getenv('ZEROEVAL_SDK_VERSION', 'unknown')}",
        }

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout or self._timeout)
        except requests.RequestException as e:
            if fallback is not None:
                prompt = Prompt(
                    content=fallback,
                    version=None,
                    tag=None,
                    is_latest=False,
                    created_by=None,
                    updated_by=None,
                    created_at=None,
                    updated_at=None,
                    metadata={},
                    source="fallback",
                )
                return self._post_process(prompt, variables, render, missing, use_cache, cache_key)
            raise PromptRequestError(str(e), status=None)

        if resp.status_code == 404:
            if fallback is not None:
                prompt = Prompt(
                    content=fallback,
                    version=None,
                    tag=None,
                    is_latest=False,
                    created_by=None,
                    updated_by=None,
                    created_at=None,
                    updated_at=None,
                    metadata={},
                    source="fallback",
                )
                return self._post_process(prompt, variables, render, missing, use_cache, cache_key)
            raise PromptNotFoundError(slug, version, effective_tag)
        if resp.status_code >= 400:
            if fallback is not None:
                prompt = Prompt(
                    content=fallback,
                    version=None,
                    tag=None,
                    is_latest=False,
                    created_by=None,
                    updated_by=None,
                    created_at=None,
                    updated_at=None,
                    metadata={},
                    source="fallback",
                )
                return self._post_process(prompt, variables, render, missing, use_cache, cache_key)
            raise PromptRequestError(
                f"Request failed with status {resp.status_code}: {resp.text}", status=resp.status_code
            )

        data = resp.json()
        prompt = Prompt.from_response(data)
        return self._post_process(prompt, variables, render, missing, use_cache, cache_key)

    def _post_process(
        self,
        prompt: Prompt,
        variables: Optional[Dict[str, Any]],
        render: bool,
        missing: str,
        use_cache: bool,
        cache_key: Tuple[str, str, Optional[int], Optional[str]],
    ) -> Prompt:
        # Templating
        if variables is not None and render:
            prompt = Prompt(
                content=render_template(prompt.content, variables, missing=missing),
                version=prompt.version,
                tag=prompt.tag,
                is_latest=prompt.is_latest,
                created_by=prompt.created_by,
                updated_by=prompt.updated_by,
                created_at=prompt.created_at,
                updated_at=prompt.updated_at,
                metadata=prompt.metadata,
                source=prompt.source,
            )
        # Cache only server-sourced
        if use_cache and prompt.source == "server":
            self._cache.set(cache_key, prompt)
        return prompt


