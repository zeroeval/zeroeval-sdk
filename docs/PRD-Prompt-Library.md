# PRD: Prompts with Versions & Tags — Python SDK (`@zeroeval-sdk`)

## 1) Objective

Provide a managed Prompts client in the Python SDK that lets users fetch prompt text (team-scoped) by a stable `slug` and either a specific `version` or a movable `tag` (e.g., `production`, `staging`, `dev`, `canary`). The SDK should default to an appropriate tag in each environment, support a local `fallback` on failures, and return rich metadata (version info and audit fields).

Non-goals:

- Authoring prompts from the SDK (creation and edits are UI-only).
- Adding complex templating/variables (may be future work via `metadata`).

## 2) Terminology

- Prompt: A team-scoped prompt identified by a unique `slug` within a team.
- Version: An immutable, sequentially numbered revision of a prompt.
- Tag: A free-form, named pointer mapping to a specific version of a prompt (e.g., `production`, `staging`, `dev`, `canary`).

## 3) Public API (User-Facing)

Add a method on the main client:

```python
from typing import Literal, Optional, Any
from dataclasses import dataclass

@dataclass
class Prompt:
    content: str
    version: Optional[int]
    tag: Optional[str]
    is_latest: bool
    created_by: Optional[str]
    updated_by: Optional[str]
    created_at: Optional[str]  # ISO 8601
    updated_at: Optional[str]  # ISO 8601
    metadata: dict
    source: Literal["server", "fallback"]

class ZeroEval:
    def get_prompt(
        self,
        slug: str,  # prompt slug unique within team, e.g. "support-triage"
        *,
        version: Optional[int] = None,
        tag: Optional[str] = None,
        fallback: Optional[str] = None,
        variables: Optional[dict[str, Any]] = None,
        task_name: Optional[str] = None,
        render: bool = True,
        missing: Literal["error", "leave"] = "error",
        use_cache: bool = True,
        timeout: float = 10.0,
    ) -> Prompt: ...
```

Notes:

- `version` outranks `tag`. If `version` is provided, `tag` is ignored.
- If neither `version` nor `tag` is provided, the SDK resolves a default tag (see §5).
- `slug` must be a single segment like `"support-triage"` (no library concept).
- `Prompt.source` is `"server"` when returned from the API and `"fallback"` when resolved locally.
- Runtime templating:
  - If `variables` is provided and `render=True` (default), the SDK performs client-side rendering by substituting `{{variable_name}}` with values from `variables` (see §18).
  - `missing` controls behavior when a referenced variable has no value: `"error"` raises; `"leave"` leaves the token intact.
- Task association (optional):
  - If `task_name` is provided, the returned `Prompt.content` is decorated with a ZeroEval metadata header so downstream OpenAI calls are traced against that task, mirroring `ze.prompt` behavior.
  - Format: `<zeroeval>{"task": "<task_name>", "variables": { ...optional... }}</zeroeval><content>`.
  - Included variables: when `variables` is passed, those keys are embedded in the header for observability. Rendering of `content` still follows `render`/`missing` semantics.
  - This does not change server-side storage; it only affects the returned content for observability and automatic task creation/association.

Convenience:

```python
prompt = ze.prompts.get("support-triage", tag="production")  # thin wrapper
```

## 4) HTTP Contract (SDK ↔ Backend)

Request (GET):

```
GET /v1/prompts/{prompt_slug}?version={int}&tag={string}
Authorization: Bearer <API_KEY>
```

Response 200 (JSON):

```json
{
  "prompt": "support-triage",
  "version": 7,
  "tag": "production",
  "is_latest": false,
  "content": "You are a helpful assistant...",
  "metadata": { "lang": "en" },
  "created_by": "user_123",
  "updated_by": "user_456",
  "created_at": "2025-08-25T12:00:00Z",
  "updated_at": "2025-08-27T09:30:00Z"
}
```

Errors:

- 404: slug not found OR requested `version`/`tag` not pinned.
- 400: invalid slug, invalid parameters.
- 401/403: auth failures.
- 5xx: server errors.

SDK behavior on errors: if `fallback` is provided, return `Prompt(content=fallback, version=None, tag=None, is_latest=False, ..., source="fallback")`. Otherwise raise a typed SDK exception.

## 5) Default Tag Resolution

Priority: `version` > `tag` > `default_tag`.

Client config fields:

- `default_tag`: explicit default (string). If not set, use env-based mapping below.
- Env mapping: if `ZEROEVAL_PROMPT_TAG` set → use it; else if `ZEROEVAL_ENV == "production"` → `default_tag = "production"`; else `default_tag = "latest"`.

Special handling of `latest`:

- `latest` is a computed pseudo-tag that always resolves to the highest `version` for the entry.
- It is never stored as a tag on the server.

## 6) Caching

Design goals:

- Keep hot prompts fast with a tiny in-process cache.
- Avoid correctness bugs with runtime variables by caching only the raw server content.
- Keep stale windows short and controllable (TTL).
- Be thread-safe within a process; do not share across processes.

Mechanics:

- In-memory LRU cache with TTL (default 60s; configurable):
  - Cache key: `(api_key_namespace, slug, version|None, tag|resolved_default_tag)`.
    - `api_key_namespace`: last 6 chars of the API key (or equivalent) to prevent cross-tenant collisions.
    - `resolved_default_tag`: if the caller did not pass `version` or `tag`, use the client’s resolved default tag (see §5).
  - Cache value: Raw server `Prompt` (unrendered content + metadata), `source="server"`.
    - Do NOT cache rendered variants (those depend on caller-provided `variables`).
    - Do NOT cache `fallback` results.
  - Eviction: LRU by insertion/access with bounded `maxsize` (default 512; configurable).
  - Expiration: entries older than `ttl_seconds` are treated as misses.
  - Thread-safety: `Lock` around `get/set/clear`.

Behavioral notes:

- Bypass with `use_cache=False` to force a fresh network fetch.
- Pseudo-tag `latest`: treated as a normal tag at the cache key level; short TTL keeps drift limited.
- Errors and non-200s are not cached. If `fallback` is used, return a non-cached `Prompt(source="fallback")`.
- Rendering: if `variables` are provided and `render=True`, render AFTER retrieval using the raw cached content; the rendered result is NOT stored back into the cache.
- Invalidation: provide a `.clear()` on the internal cache. Optional DX: `ZeroEval.clear_prompt_cache()` convenience method.
- `task_name` decoration: when provided, the ZeroEval header decoration is applied AFTER retrieval (and after optional rendering) and is NOT cached. The cache stores only the raw server `Prompt`.

Implementation suggestion:

- Add `zeroeval/cache.py` with a minimal TTL LRU built on `OrderedDict`.
- Store and return copies or new `Prompt` instances when applying rendering so cached raw content is never mutated.

## 7) Errors and Exceptions

Add the following exceptions in `zeroeval/errors.py`:

- `PromptNotFoundError(slug: str, version: Optional[int], tag: Optional[str])`
- `PromptRequestError(message: str, status: Optional[int])`

Raise `PromptNotFoundError` for 404s; `PromptRequestError` otherwise. On `fallback`, return without raising.

## 8) Input Validation

- Validate `slug` matches `^[a-z0-9-]+$` (single segment; no library prefix).
- Validate `version >= 1` if provided.
- Validate `tag` matches `^[a-z0-9-]+$` if provided. Reserved: `latest` is allowed but treated as pseudo-tag.

## 9) Configuration

`ZeroEval` client accepts / reads:

- `base_url` (default from `ZEROEVAL_BASE_URL` or `ZEROEVAL_API_URL` or `https://api.zeroeval.com`)
- `api_key` (from `ZEROEVAL_API_KEY`)
- `default_tag` (from `ZEROEVAL_PROMPT_TAG` or env mapping)
- `timeout` (default 10s)
- `cache_ttl_seconds` (default 60)

## 10) Implementation Plan (Files and Functions)

Edit/add the following files:

1. `src/zeroeval/client.py`

   - Add method `get_prompt(...)` implementing resolution order, request building, caching, and fallback.
   - Add helper `_validate_slug(slug: str) -> str`.
   - Add helper `_resolve_default_tag() -> str`.
   - If `variables` provided and `render=True`, call `render_template(content, variables, missing=missing)` before returning.
   - If `task_name` is provided, decorate the returned content with the ZeroEval header. Prefer reusing `zeroeval_prompt(name=task_name, content=content, variables=variables)` to ensure parity with `ze.prompt`. Ensure the cached value remains the undecorated server `Prompt`.

2. `src/zeroeval/types.py`

   - Add `@dataclass class Prompt` per §3 and a `from_response(json: dict) -> Prompt` constructor.

3. `src/zeroeval/errors.py`

   - Add `PromptNotFoundError`, `PromptRequestError`.

4. `src/zeroeval/cache.py` (new)

   - Implement `class TTLCache[K, V]` with `get`, `set`, `clear`, honoring TTL.

5. `src/zeroeval/__init__.py`

   - Export `Prompt` and `ZeroEval.get_prompt` in the public API. Pass through `task_name` from `ze.get_prompt`/`ze.prompts.get`.

6. `examples/openai_responses_parse_with_prompt.py`

   - Add usage example with `tag` and `fallback`.
   - Add example passing `variables` and demonstrating missing behavior.

7. `src/zeroeval/template.py` (new)

   - Implement templating:
     - Syntax: `{{variable_name}}` where `variable_name` matches `^[a-zA-Z_][a-zA-Z0-9_]*$`.
     - Escaping: `\{{` renders literal `{{` and `\}}` renders literal `}}`.
     - API: `def render_template(content: str, variables: dict[str, Any], *, missing: Literal["error","leave"] = "error") -> str`.
     - Behavior: replace unescaped tokens with value converted via `str(value)`; on missing per `missing`.

## 11) Networking Details

- Use `httpx` (preferred) or `requests` per project conventions. Respect `timeout` for connect+read.
- Include `User-Agent: zeroeval-sdk-python/<version>`.
- Retries: None by default; the `fallback` handles outages at callsite. Keep minimal.

## 12) Thread Safety

- The client is thread-safe if `TTLCache` uses a `threading.Lock` for `get/set`.
- Document that per-process cache is not shared across processes.

## 13) Testing Plan

- Unit tests:

  - `slug` parsing and validation.
  - Default tag resolution given env vars.
  - Caching keys and TTL expiration.
  - Error handling and `fallback` behavior.
  - Response mapping to `Prompt`.
  - Templating render: basic replacement, escaping, missing variable policies (`error`/`leave`).
  - `task_name` decoration: header presence/format, inclusion of `variables`, and that cache contents remain undecorated.

- Integration tests (requires backend dev server or mocked responses):
  - Fetch by `version`.
  - Fetch by `tag` (production, staging).
  - Fetch with `latest` (pseudo-tag) returning max version.
  - 404 with and without `fallback`.

## 14) Backwards Compatibility

- New method; no breaking changes. If a prior concept of `is_production` existed, it is fully subsumed by pinning tag `production` on backend.

## 15) Example Usage

```python
import os
import zeroeval as ze

# Latest in non-prod, production in prod:
p = ze.get_prompt("support-triage", fallback="You are a helpful assistant.")

# Explicit tag:
p = ze.get_prompt("support-triage", tag="staging")

# Explicit version (ignores tag):
p = ze.get_prompt("support-triage", version=12)

print(p.content)

# With variables and rendering
p = ze.get_prompt(
    "events-create",
    tag="production",
    variables={"event_name": "ZeroEval Launch"},
)
print(p.content)  # "You are creating an event with title ZeroEval Launch"

# Alternatively, use the namespace wrapper
p = ze.prompts.get("support-triage", tag="production")

# With task association (mirrors ze.prompt behavior in returned content)
tasked = ze.get_prompt(
    "support-triage",
    tag="production",
    variables={"customer": "Acme"},
    task_name="support-triage",
)
# Use in OpenAI call to ensure spans are linked to the task
# client.chat.completions.create(messages=[{"role": "system", "content": tasked.content}, ...])
```

## 16) Risks & Mitigations

- Tag drift across environments → allow overriding via `ZEROEVAL_PROMPT_TAG`.
- Cache staleness after tag re-pin → short TTL (60s default) and `use_cache=False` opt-out.
- Large prompts → responses are text; no change required.

## 17) Rollout

1. Ship backend endpoints and DB first.
2. Release SDK minor version with `get_prompt`.
3. Add docs and example to README/examples.

## 18) Prompt Variables and Runtime Injection

### Syntax

- Use double braces: `{{variable_name}}`.
- Allowed variable names: `^[a-zA-Z_][a-zA-Z0-9_]*$`.
- Escape `{{` as `\{{` and `}}` as `\}}` to render literal braces.
- No conditionals/loops/filters in v1. Keep templates simple and deterministic.

### Rendering (SDK-side)

- Rendering is performed client-side by default when `variables` is provided (`render=True`).
- Algorithm:
  - Scan for unescaped `{{name}}` tokens and replace with `str(variables[name])`.
  - If `name` is missing:
    - `missing="error"` (default): raise `PromptRequestError` with details.
    - `missing="leave"`: leave `{{name}}` intact in the content.
- Rendering never mutates or stores content server-side; it is purely runtime substitution.

### Validation

- SDK validates that keys in `variables` match the allowed identifier regex.
- Optional helper: `extract_variables(content: str) -> set[str]` in `template.py` used by examples/tests.

### Examples

```python
tpl = "You are creating an event with title {{event_name}}"
rendered = render_template(tpl, {"event_name": "ZeroEval Launch"})
# => "You are creating an event with title ZeroEval Launch"

# Missing handling
render_template(tpl, {}, missing="leave")  # => "You are creating an event with title {{event_name}}"
```
