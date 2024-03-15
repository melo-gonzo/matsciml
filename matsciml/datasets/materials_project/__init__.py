from __future__ import annotations

from importlib.util import find_spec


if find_spec("torch_geometric") is not None:
    pass
