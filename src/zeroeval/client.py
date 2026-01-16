from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

import requests

from .cache import TTLCache
from .errors import PromptNotFoundError, PromptRequestError
from .template import render_template
from .types import Prompt
from .observability import zeroeval_prompt
from .utils.hash import normalize_prompt_text


_SLUG_RE = re.compile(r"^[a-z0-9-]+$")
_TAG_RE = re.compile(r"^[a-z0-9-]+$")

logger = logging.getLogger(__name__)


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
        # Short TTL cache for prompt_version_id -> model lookups
        self._model_cache: TTLCache[str, Optional[str]] = TTLCache(ttl_seconds=60.0)

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
        task_name: Optional[str] = None,
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
                    version_id=None,
                    tag=None,
                    is_latest=False,
                    model=None,
                    created_by=None,
                    updated_by=None,
                    created_at=None,
                    updated_at=None,
                    metadata={},
                    source="fallback",
                )
                return self._post_process(prompt, variables, task_name, render, missing, use_cache, cache_key)
            raise PromptRequestError(str(e), status=None)

        if resp.status_code == 404:
            if fallback is not None:
                prompt = Prompt(
                    content=fallback,
                    version=None,
                    version_id=None,
                    tag=None,
                    is_latest=False,
                    model=None,
                    created_by=None,
                    updated_by=None,
                    created_at=None,
                    updated_at=None,
                    metadata={},
                    source="fallback",
                )
                return self._post_process(prompt, variables, task_name, render, missing, use_cache, cache_key)
            raise PromptNotFoundError(slug, version, effective_tag)
        if resp.status_code >= 400:
            if fallback is not None:
                prompt = Prompt(
                    content=fallback,
                    version=None,
                    version_id=None,
                    tag=None,
                    is_latest=False,
                    model=None,
                    created_by=None,
                    updated_by=None,
                    created_at=None,
                    updated_at=None,
                    metadata={},
                    source="fallback",
                )
                return self._post_process(prompt, variables, task_name, render, missing, use_cache, cache_key)
            raise PromptRequestError(
                f"Request failed with status {resp.status_code}: {resp.text}", status=resp.status_code
            )

        data = resp.json()
        # Normalize version_id from payload if present in metadata
        if "version_id" not in data:
            meta = data.get("metadata") or {}
            if isinstance(meta, dict):
                vid = meta.get("version_id") or meta.get("prompt_version_id")
                if vid:
                    data["version_id"] = vid
        prompt = Prompt.from_response(data)
        return self._post_process(prompt, variables, task_name, slug, render, missing, use_cache, cache_key)

    # ---- Prompt Library: task-attached prompt helpers ----
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": f"zeroeval-sdk-python/{os.getenv('ZEROEVAL_SDK_VERSION', 'unknown')}",
            "Content-Type": "application/json",
        }

    def ensure_task_prompt_version(self, *, task_name: str, content: str, content_hash: str) -> Prompt:
        url = f"{self._base_url}/v1/tasks/{task_name}/prompt/versions/ensure"
        # Inherit model_id from the latest version if it exists; otherwise leave empty
        inherited_model_id: Optional[str] = None
        try:
            latest = self.get_task_prompt_latest(task_name=task_name)
            model_str = getattr(latest, "model", None)
            if isinstance(model_str, str) and model_str:
                # Stored as "zeroeval/<id>". Strip prefix if present to get raw model_id
                inherited_model_id = model_str.split("/", 1)[1] if model_str.startswith("zeroeval/") else model_str
        except PromptNotFoundError:
            inherited_model_id = None
        except PromptRequestError:
            # On transient errors fetching latest, proceed without inheriting
            inherited_model_id = None

        payload: Dict[str, Any] = {
            "content": normalize_prompt_text(content),
            "content_hash": content_hash,
            "metadata": None,
            "model_id": inherited_model_id,
        }
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=self._timeout)
        if resp.status_code >= 400:
            raise PromptRequestError(f"ensure_task_prompt_version failed: {resp.text}", status=resp.status_code)
        data = resp.json()
        if "version_id" not in data:
            meta = data.get("metadata") or {}
            if isinstance(meta, dict):
                vid = meta.get("version_id") or meta.get("prompt_version_id")
                if vid:
                    data["version_id"] = vid
        return Prompt.from_response(data)

    def get_task_prompt_version_by_hash(self, *, task_name: str, content_hash: str) -> Prompt:
        url = f"{self._base_url}/v1/tasks/{task_name}/prompt/versions/by-hash/{content_hash}"
        resp = requests.get(url, headers=self._headers(), timeout=self._timeout)
        if resp.status_code == 404:
            raise PromptNotFoundError(task_name, None, content_hash)
        if resp.status_code >= 400:
            raise PromptRequestError(f"get_task_prompt_version_by_hash failed: {resp.text}", status=resp.status_code)
        data = resp.json()
        if "version_id" not in data:
            meta = data.get("metadata") or {}
            if isinstance(meta, dict):
                vid = meta.get("version_id") or meta.get("prompt_version_id")
                if vid:
                    data["version_id"] = vid
        return Prompt.from_response(data)

    def get_task_prompt_latest(self, *, task_name: str) -> Prompt:
        url = f"{self._base_url}/v1/tasks/{task_name}/prompt/latest"
        resp = requests.get(url, headers=self._headers(), timeout=self._timeout)
        if resp.status_code == 404:
            # Specific guidance for latest when no versions exist
            raise PromptNotFoundError(task_name, None, "latest")
        if resp.status_code >= 400:
            raise PromptRequestError(f"get_task_prompt_latest failed: {resp.text}", status=resp.status_code)
        data = resp.json()
        if "version_id" not in data:
            meta = data.get("metadata") or {}
            if isinstance(meta, dict):
                vid = meta.get("version_id") or meta.get("prompt_version_id")
                if vid:
                    data["version_id"] = vid
        return Prompt.from_response(data)

    def get_model_for_prompt_version(self, *, prompt_version_id: str) -> Optional[str]:
        cached = self._model_cache.get(prompt_version_id)
        if cached is not None:
            return cached
        url = f"{self._base_url}/v1/prompt-versions/{prompt_version_id}/model"
        resp = requests.get(url, headers=self._headers(), timeout=self._timeout)
        if resp.status_code >= 400:
            # Cache negative to avoid hammering
            self._model_cache.set(prompt_version_id, None)
            return None
        data = resp.json() or {}
        model = data.get("model")
        if isinstance(model, str) and model:
            if not model.startswith("zeroeval/"):
                model = f"zeroeval/{model}"
        self._model_cache.set(prompt_version_id, model)
        return model

    def _post_process(
        self,
        prompt: Prompt,
        variables: Optional[Dict[str, Any]],
        task_name: Optional[str],
        prompt_slug: Optional[str],
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
                version_id=getattr(prompt, "version_id", None),
                tag=prompt.tag,
                is_latest=prompt.is_latest,
                model=prompt.model,
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
        # Decorate with ZeroEval task header (not cached)
        if task_name:
            decorated = Prompt(
                content=zeroeval_prompt(
                    name=task_name,
                    content=prompt.content,
                    variables=variables or {},
                    prompt_slug=prompt_slug,
                    prompt_version=prompt.version,
                    prompt_version_id=getattr(prompt, "version_id", None),
                ),
                version=prompt.version,
                version_id=getattr(prompt, "version_id", None),
                tag=prompt.tag,
                is_latest=prompt.is_latest,
                model=prompt.model,
                created_by=prompt.created_by,
                updated_by=prompt.updated_by,
                created_at=prompt.created_at,
                updated_at=prompt.updated_at,
                metadata=prompt.metadata,
                source=prompt.source,
            )
            return decorated
        return prompt

    # ---- New Prompt Completion and Feedback API ----

    def log_completion(
        self,
        *,
        prompt_slug: str,
        prompt_id: str,
        prompt_version_id: str,
        messages: list[dict[str, Any]],
        input_text: Optional[str] = None,
        output_text: Optional[str] = None,
        model_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        cost: Optional[float] = None,
        has_error: bool = False,
        error_message: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Log a completion for a specific prompt and version.
        This is used to track prompt usage automatically.
        
        Args:
            prompt_slug: The slug of the prompt
            prompt_id: UUID of the prompt
            prompt_version_id: UUID of the prompt version
            messages: Array of message objects in OpenAI format
            input_text: Optional text representation of input
            output_text: Optional text representation of output
            model_id: Optional model identifier used
            metadata: Optional additional metadata
            duration_ms: Optional execution duration in milliseconds
            prompt_tokens: Optional number of prompt tokens
            completion_tokens: Optional number of completion tokens
            total_tokens: Optional total token count
            cost: Optional cost in USD
            has_error: Whether the completion had an error
            error_message: Optional error message
            span_id: Optional span ID for trace linking
            
        Returns:
            The created completion record
        """
        # Extract project_id from API key context (handled by backend)
        url = f"{self._base_url}/projects/{{project_id}}/prompts/{prompt_slug}/completions"
        
        payload = {
            "prompt_id": prompt_id,
            "prompt_version_id": prompt_version_id,
            "model_id": model_id,
            "messages": messages,
            "input_text": input_text,
            "output_text": output_text,
            "metadata": metadata or {},
            "duration_ms": duration_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
            "has_error": has_error,
            "error_message": error_message,
            "span_id": span_id,
        }
        
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=self._timeout)
        if resp.status_code >= 400:
            raise PromptRequestError(
                f"log_completion failed: {resp.text}", status=resp.status_code
            )
        return resp.json()

    def send_feedback(
        self,
        *,
        prompt_slug: str,
        completion_id: str,
        thumbs_up: bool,
        reason: Optional[str] = None,
        expected_output: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Send feedback for a specific completion.
        
        Args:
            prompt_slug: The slug of the prompt
            completion_id: UUID of the completion to provide feedback on
            thumbs_up: True for positive feedback, False for negative
            reason: Optional explanation of the feedback
            expected_output: Optional description of what the expected output should be
            metadata: Optional additional metadata
            
        Returns:
            The created feedback record
        """
        url = f"{self._base_url}/v1/prompts/{prompt_slug}/completions/{completion_id}/feedback"
        
        logger.debug(
            f"[SDK] Sending feedback for completion_id={completion_id}, prompt_slug={prompt_slug}",
            extra={
                "completion_id": completion_id,
                "prompt_slug": prompt_slug,
                "thumbs_up": thumbs_up,
                "url": url
            }
        )
        
        payload = {
            "thumbs_up": thumbs_up,
        }
        
        # Add optional fields only if provided
        if reason is not None:
            payload["reason"] = reason
        if expected_output is not None:
            payload["expected_output"] = expected_output
        if metadata is not None:
            payload["metadata"] = metadata
        
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=self._timeout)
        
        logger.debug(
            f"[SDK] Feedback response status={resp.status_code}",
            extra={
                "status_code": resp.status_code,
                "response_text": resp.text[:500] if resp.text else None
            }
        )
        
        if resp.status_code >= 400:
            raise PromptRequestError(
                f"send_feedback failed: {resp.text}", status=resp.status_code
            )
        return resp.json()

    # ---- Judge Evaluations API ----

    def get_behavior_evaluations(
        self,
        project_id: str,
        judge_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        evaluation_result: Optional[bool] = None,
        feedback_state: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch paginated judge evaluations for a specific judge.

        Args:
            project_id: The project UUID.
            judge_id: The judge (signal automation) UUID.
            limit: Max results per page (1-500, default 100).
            offset: Pagination offset (default 0).
            start_date: ISO datetime string to filter evaluations created after.
            end_date: ISO datetime string to filter evaluations created before.
            evaluation_result: Filter by True (positive) or False (negative).
            feedback_state: Filter by 'with_user_feedback' or 'without_user_feedback'.

        Returns:
            Dict with keys: evaluations (list), total, offset, limit.
        """
        url = f"{self._base_url}/projects/{project_id}/judges/{judge_id}/evaluations"
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date
        if evaluation_result is not None:
            params["evaluation_result"] = str(evaluation_result).lower()
        if feedback_state is not None:
            params["feedback_state"] = feedback_state

        try:
            resp = requests.get(url, headers=self._headers(), params=params, timeout=self._timeout)
        except requests.RequestException as e:
            raise PromptRequestError(str(e), status=None)

        if resp.status_code >= 400:
            raise PromptRequestError(
                f"get_behavior_evaluations failed: {resp.text}", status=resp.status_code
            )
        return resp.json()

    def get_span_evaluations(
        self,
        project_id: str,
        span_id: str,
    ) -> Dict[str, Any]:
        """
        Fetch all judge evaluations for a specific span.

        Args:
            project_id: The project UUID.
            span_id: The span UUID.

        Returns:
            Dict with keys: span_id, evaluations (list of judge evaluation objects).
        """
        url = f"{self._base_url}/projects/{project_id}/spans/{span_id}/evaluations"

        try:
            resp = requests.get(url, headers=self._headers(), timeout=self._timeout)
        except requests.RequestException as e:
            raise PromptRequestError(str(e), status=None)

        if resp.status_code >= 400:
            raise PromptRequestError(
                f"get_span_evaluations failed: {resp.text}", status=resp.status_code
            )
        return resp.json()


