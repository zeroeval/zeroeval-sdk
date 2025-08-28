from __future__ import annotations

import re
from typing import Any, Dict, Set

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def render_template(content: str, variables: Dict[str, Any], *, missing: str = "error") -> str:
    if missing not in {"error", "leave"}:
        raise ValueError("missing must be 'error' or 'leave'")

    # Validate variable keys early
    for key in variables.keys():
        if not _IDENTIFIER_RE.match(key):
            raise ValueError(f"Invalid variable name: {key}")

    # Handle escaped braces: \{{ and \}}
    ESC_L = "__ZE_ESC_L__"
    ESC_R = "__ZE_ESC_R__"
    tmp = content.replace(r"\{{", ESC_L).replace(r"\}}", ESC_R)

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in variables:
            return str(variables[name])
        if missing == "error":
            from .errors import PromptRequestError

            raise PromptRequestError(f"Missing variable: {name}", status=None)
        return "{{" + name + "}}"

    rendered = re.sub(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}", repl, tmp)
    return rendered.replace(ESC_L, "{{").replace(ESC_R, "}}")


def extract_variables(content: str) -> Set[str]:
    names: Set[str] = set()
    # Temporarily remove escaped braces
    tmp = content.replace(r"\{{", "").replace(r"\}}", "")
    for m in re.finditer(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}", tmp):
        names.add(m.group(1))
    return names


