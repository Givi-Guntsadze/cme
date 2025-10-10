from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Dict, Optional


_CACHE: Dict[str, Optional[str]] = {}
_CODESPACES_SECRET_CACHE: Optional[Dict[str, str]] = None


def _load_codespaces_secrets() -> Dict[str, str]:
    global _CODESPACES_SECRET_CACHE
    if _CODESPACES_SECRET_CACHE is not None:
        return _CODESPACES_SECRET_CACHE

    secrets: Dict[str, str] = {}
    # Preferred plaintext mapping when available
    json_path = Path("/workspaces/.codespaces/shared/user-secrets-envs.json")
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
            for key, value in data.items():
                if isinstance(key, str) and isinstance(value, str):
                    secrets[key.upper()] = value
        except Exception:
            pass

    # Backup: legacy base64-encoded env file
    env_path = Path("/workspaces/.codespaces/shared/.env-secrets")
    if env_path.exists():
        try:
            for line in env_path.read_text().splitlines():
                if not line or "=" not in line:
                    continue
                key, encoded = line.split("=", 1)
                key = key.strip()
                encoded = encoded.strip()
                if not key or not encoded:
                    continue
                try:
                    decoded = base64.b64decode(encoded).decode("utf-8")
                except Exception:
                    continue
                secrets.setdefault(key.upper(), decoded)
        except Exception:
            pass

    _CODESPACES_SECRET_CACHE = secrets
    return secrets


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieve a secret from the environment, Codespaces secret files, or fallback prefixes.
    """
    key = name.upper()
    if key in _CACHE:
        return _CACHE[key] if _CACHE[key] is not None else default

    # Direct environment lookup
    value = os.getenv(key)
    if value:
        _CACHE[key] = value
        return value

    # Alternate prefixes GitHub Codespaces might expose
    for prefix in (
        "CODESPACE_SECRET_",
        "CODESPACES_SECRET_",
        "GITHUB_CODESPACES_SECRET_",
        "GH_CODESPACES_SECRET_",
    ):
        alt_value = os.getenv(prefix + key)
        if alt_value:
            _CACHE[key] = alt_value
            return alt_value

    # Codespaces shared files
    secrets = _load_codespaces_secrets()
    if key in secrets:
        _CACHE[key] = secrets[key]
        return secrets[key]

    _CACHE[key] = None
    return default
