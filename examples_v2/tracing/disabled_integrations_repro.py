"""
Repro: disabled_integrations not taking effect if integrations initialize early.

This script intentionally triggers tracing BEFORE calling ze.init(), which can
cause integrations to patch libraries early. Then it calls ze.init() with
disabled_integrations and verifies those patches are removed.

Dependencies (install locally):
  pip install openai langchain langchain-core

Run:
  uv run python examples_v2/tracing/disabled_integrations_repro.py

Optional (to actually make a live OpenAI request after init):
  export OPENAI_API_KEY="..."
  export RUN_OPENAI_CALL=1
"""

from __future__ import annotations

import os
import sys


def _patched_flag(obj) -> bool:
    return bool(getattr(obj, "__ze_patched__", False))


def _print_patch_state(*, label: str, openai_mod, lc_base_language_model, client=None) -> None:
    print(f"\n=== {label} ===")

    # OpenAIIntegration patches openai.OpenAI.__init__ (and patches instances during init)
    print(f"openai.OpenAI.__init__ patched: {_patched_flag(openai_mod.OpenAI.__init__)}")

    if client is not None:
        try:
            create_fn = client.chat.completions.create
        except Exception:
            create_fn = None
        print(f"client.chat.completions.create patched: {_patched_flag(create_fn) if create_fn else 'N/A'}")

    # LangChainIntegration patches BaseLanguageModel.generate/agenerate
    blm = lc_base_language_model.BaseLanguageModel
    for method_name in ("agenerate", "generate"):
        try:
            method = getattr(blm, method_name)
        except Exception:
            method = None
        print(
            f"langchain_core.BaseLanguageModel.{method_name} patched: "
            f"{_patched_flag(method) if method else 'N/A'}"
        )

    # LangChainIntegration also patches Runnable entrypoints (invoke/ainvoke/etc.)
    try:
        from langchain_core.runnables.base import Runnable as _Runnable  # type: ignore
        runnable_ainvoke = getattr(_Runnable, "ainvoke", None)
        print(f"langchain_core.Runnable.ainvoke patched: {_patched_flag(runnable_ainvoke) if runnable_ainvoke else 'N/A'}")
    except Exception:
        print("langchain_core.Runnable.ainvoke patched: N/A")


class _LazyZeroEval:
    def __init__(self, *, api_key: str):
        self.api_key = api_key
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily initialize ZeroEval SDK only when needed."""
        if self._initialized:
            return

        # Import zeroeval only when needed to avoid import-time patching
        import zeroeval as ze

        ze.init(
            api_key=self.api_key,
            disabled_integrations=[
                "openai",
                "anthropic",
                "gemini",
                "langchain",
                "langgraph",
                "llamaindex",
                "haystack",
                "crewai",
                "autogen",
                "bedrock",
            ],
            debug=True,
        )
        self._initialized = True


def main() -> int:
    try:
        import openai
    except ImportError as e:
        print("Missing dependency: openai. Install with: pip install openai")
        raise

    try:
        from langchain_core.language_models import base as lc_base_language_model
    except ImportError as e:
        print("Missing dependency: langchain-core. Install with: pip install langchain-core")
        raise

    print("Python:", sys.version.split()[0])

    # 1) Show baseline patch state (should be False/clean in a fresh process)
    _print_patch_state(
        label="Before any tracing / before ze.init()",
        openai_mod=openai,
        lc_base_language_model=lc_base_language_model,
    )

    # 2) Create an OpenAI client BEFORE importing zeroeval (should not be patched).
    client_before_import = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-dummy"))
    _print_patch_state(
        label="After creating OpenAI client (still before importing zeroeval)",
        openai_mod=openai,
        lc_base_language_model=lc_base_language_model,
        client=client_before_import,
    )

    # 3) Lazy init zeroeval with integrations disabled (the recommended pattern).
    lazy = _LazyZeroEval(api_key=os.environ.get("ZEROEVAL_API_KEY", "test-key"))
    lazy._ensure_initialized()

    # Import tracer only after init for visibility into active integrations.
    from zeroeval.observability.tracer import tracer

    print("\nActive integrations after lazy ze.init():", list(tracer._integrations.keys()))

    # 4) Verify new clients are not wrapped
    client_after_init = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-dummy"))

    _print_patch_state(
        label="After lazy ze.init(disabled_integrations=...)",
        openai_mod=openai,
        lc_base_language_model=lc_base_language_model,
        client=client_after_init,
    )

    # Optional: actually make a request to confirm no openai.* spans are created.
    # (This will incur network / API usage if enabled.)
    if os.environ.get("RUN_OPENAI_CALL") == "1":
        if not os.environ.get("OPENAI_API_KEY"):
            print("\nRUN_OPENAI_CALL=1 but OPENAI_API_KEY is not set; skipping call.")
        else:
            print("\nMaking a live OpenAI call (RUN_OPENAI_CALL=1)...")
            client_after_init.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": "Say 'hello'."}],
            )
            print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

