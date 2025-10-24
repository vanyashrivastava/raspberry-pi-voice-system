# src package initializer
# This file makes `src` a Python package. It's intentionally minimal.
# Purpose: allow imports like `from src.audio import ...` when project installed or run from repo root.
# Owner: repository (shared)

__all__ = [
    'audio',
    'email',
    'detection',
    'alerts',
    'web',
    'config',
]

# No runtime code here. Keep package import lightweight.
