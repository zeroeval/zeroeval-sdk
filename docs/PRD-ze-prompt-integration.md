## PRD: Enhance `ze.prompt` to integrate with Prompt Library

### Objective

- Expand `ze.prompt` to be version-aware via the Prompt Library and usable in three modes:
  - Create/ensure a version when given `content` (dedupe via content hash)
  - Fetch by `from = "latest"` for a task-attached prompt
  - Fetch by `from = <hash>` for a task-attached prompt
- When used in OpenAI chat/responses.parse, automatically patch the model to the one bound to that prompt version.

### Public SDK API Changes

- New signature and semantics (Python):

  - Location: `zeroeval-sdk/src/zeroeval/__init__.py`
  - Replace current alias `prompt = zeroeval_prompt` with a wrapper function:

  - Parameters:

    - `name: str` (task name; required)
    - `content: str | None = None`
    - `variables: dict | None = None`
    - `from_: str | None = None` // accepts a prompt hash or the literal string "latest"

  - Constraints:

    - Exactly one of `content` or `from_` must be provided
    - `from_ == "latest"` is valid; otherwise treat `from_` as a hex SHA-256 content hash

  - Returns:
    - A string: `<zeroeval>{metadata}</zeroeval><content>` where metadata includes:
      - `task`, `prompt_slug`, `prompt_version`, `prompt_version_id`, and `variables` (if provided)

### Content Hash Specification

- Normalization prior to hashing:
  - Convert CRLF to LF
  - Strip trailing whitespace on each line
  - Strip leading/trailing whitespace overall
  - Do not modify `{{variable}}` tokens
- Hash algorithm: SHA-256 over normalized bytes; store/transmit lowercase hex (64 chars)

### Behaviors by Mode

- `content` provided:

  - Compute `content_hash`
  - Resolve the prompt attached to task `name` (see Backend PRD for task→prompt mapping)
  - Ensure a prompt version with this `content_hash` exists; create if not
  - Decorate and return content including `prompt_slug`, `prompt_version`, `prompt_version_id`

- `from_ == "latest"`:

  - Resolve the prompt attached to task `name`
  - Fetch latest version and return decorated content

- `from_ == <hash>`:
  - Resolve the prompt attached to task `name`
  - Fetch version by `content_hash` and return decorated content; 404 if not found

### OpenAI Integration Model Patching

- Location: `zeroeval-sdk/src/zeroeval/observability/integrations/openai/integration.py`
  - In wrappers for `chat.completions.create`, `responses.create`, and `responses.parse`:
    - Extract `<zeroeval>` metadata (already implemented)
    - If `prompt_version_id` present:
      - Resolve associated model via new client helper (below)
      - Override `kwargs["model"]` if model is returned
    - Continue attaching `span_attributes["zeroeval"] = metadata`

### SDK Client Changes

- Location: `zeroeval-sdk/src/zeroeval/client.py`

- Add helpers (HTTP to backend):

  - `ensure_task_prompt_version(task_name: str, content: str, content_hash: str) -> Prompt`
  - `get_task_prompt_version_by_hash(task_name: str, content_hash: str) -> Prompt`
  - `get_task_prompt_latest(task_name: str) -> Prompt`
  - `get_model_for_prompt_version(prompt_version_id: str) -> str | None`

- Caching:

  - Reuse existing `TTLCache` for prompt reads
  - Add a short TTL cache for `prompt_version_id → model` lookups (e.g., 60s)

- Decoration:
  - Continue using `zeroeval_prompt(...)` to produce the `<zeroeval>` header with `variables`, `prompt_slug`, `prompt_version`, `prompt_version_id`

### Types and Utilities

- `zeroeval-sdk/src/zeroeval/types.py`

  - Ensure `Prompt` exposes `version_id` (fallback: infer from `metadata.prompt_version_id`)

- `zeroeval-sdk/src/zeroeval/utils/hash.py` (new)
  - Implement normalization + SHA-256 helper for prompts to guarantee consistent hashing

### Validation & Errors

- Error if both `content` and `from_` set, or both missing
- On `from_` hash not found: raise typed SDK error; do not fallback automatically

### File-Level Insertion Points

- `src/zeroeval/__init__.py`

  - Replace `prompt = zeroeval_prompt` (currently around lines ~59–61) with function implementing new logic
  - Use `_ensure_prompt_client()` to reach new client helpers

- `src/zeroeval/client.py`

  - Implement the new helper methods and small `prompt_version_id → model` cache

- `src/zeroeval/observability/integrations/openai/integration.py`
  - In existing wrappers, before invoking the original SDK call, when `prompt_version_id` exists, resolve model and override `kwargs.model`

### Examples

```python
# ensure/create a version by content
system = ze.prompt(name="support-triage", content="You are a helpful assistant for {{product}}.")

# fetch latest for this task
system = ze.prompt(name="support-triage", from_="latest")

# fetch by hash
system = ze.prompt(name="support-triage", from_="c6a7...deadbeef")
```

### Risks

- Tight coupling to backend availability; mitigate with short caches and clear errors
- Hash normalization must match backend to avoid duplicate versions
