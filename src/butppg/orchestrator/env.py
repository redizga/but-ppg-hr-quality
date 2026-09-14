"""Minimal ``.env`` loader (dependency-free).

The OpenTSLM adapter downloads a gated Llama checkpoint from the Hugging Face
Hub, which needs a token in the environment (``HF_TOKEN``). We load it from a
project-root ``.env`` at CLI startup so the token lives in one gitignored file
instead of the shell history or the code.

Existing environment variables always win — a value already exported in the
shell is not overwritten by ``.env``.
"""

from __future__ import annotations

import os
from pathlib import Path

from butppg.paths import PROJECT_ROOT


def load_dotenv(path: str | Path | None = None) -> dict[str, str]:
    """Load ``KEY=VALUE`` lines from ``.env`` into ``os.environ``.

    Supports ``#`` comments, blank lines, an optional ``export`` prefix and
    single/double-quoted values. Returns the keys it set (for logging/tests).
    Silently does nothing if the file is absent.
    """
    path = Path(path) if path is not None else PROJECT_ROOT / ".env"
    if not path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        if key and key not in os.environ:  # shell env wins
            os.environ[key] = value
            loaded[key] = value
    return loaded
